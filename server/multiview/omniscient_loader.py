#!/usr/bin/env python3
"""
Omniscient Data Loader

Omniscientアプリからエクスポートされたデータを読み込むモジュール。
動画、深度画像、カメラパラメータを統合的に扱う。

使い方:
    python -m server.multiview.omniscient_loader experiments/omniscient_sample/003

出力形式（Omniscientエクスポート）:
    {session_name}/
    ├── {session_name}.abc          # カメラポーズ（Alembic形式）
    ├── {session_name}.mov          # RGB動画（1080x1920, 60fps）
    ├── {session_name}.omni         # メタデータJSON
    ├── {session_name}_depth/       # LiDAR深度フレーム（16-bit PNG, mm単位）
    ├── {session_name}_depthConfidence/  # 深度信頼度
    └── scan_*.obj                  # 3Dメッシュ
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Iterator
from dataclasses import dataclass
import numpy as np
from PIL import Image

# Alembicローダーのインポート（オプション）
HAS_ALEMBIC_LOADER = False
try:
    from server.multiview.alembic_loader import AlembicCameraLoader, transform_points_to_world
    HAS_ALEMBIC_LOADER = True
except ImportError:
    pass

# 統合カメラローダー（USD対応）
HAS_CAMERA_LOADER = False
try:
    from server.multiview.camera_loader import CameraLoader, load_camera_poses
    HAS_CAMERA_LOADER = True
except ImportError:
    pass


@dataclass
class CameraIntrinsics:
    """カメラ内部パラメータ"""
    focal_length: float  # mm (35mm換算)
    width: int           # 動画幅
    height: int          # 動画高さ
    depth_width: int     # 深度画像幅
    depth_height: int    # 深度画像高さ

    @property
    def fx(self) -> float:
        """x方向の焦点距離（ピクセル単位）"""
        # 35mm換算からピクセル単位に変換
        # センサーサイズを仮定（iPhone/iPad LiDAR: 約6.17mm x 4.55mm相当）
        sensor_width_mm = 6.17
        return self.focal_length * self.depth_width / sensor_width_mm

    @property
    def fy(self) -> float:
        """y方向の焦点距離（ピクセル単位）"""
        sensor_height_mm = 4.55
        return self.focal_length * self.depth_height / sensor_height_mm

    @property
    def cx(self) -> float:
        """主点x座標"""
        return self.depth_width / 2

    @property
    def cy(self) -> float:
        """主点y座標"""
        return self.depth_height / 2


@dataclass
class FrameData:
    """1フレームのデータ"""
    frame_index: int
    rgb_frame: Optional[np.ndarray]  # RGB画像 (H, W, 3)
    depth_map: np.ndarray            # 深度マップ (H, W), mm単位
    depth_confidence: Optional[np.ndarray]  # 深度信頼度 (H, W)
    focal_length: float              # そのフレームの焦点距離
    timestamp: Optional[float] = None


class OmniscientLoader:
    """Omniscientデータローダー"""

    def __init__(self, session_dir: str):
        """
        Args:
            session_dir: Omniscientセッションディレクトリへのパス
        """
        self.session_path = Path(session_dir)
        if not self.session_path.exists():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")

        # セッション名を取得（ディレクトリ内の.omniファイルから）
        omni_files = list(self.session_path.glob("*.omni"))
        if not omni_files:
            raise FileNotFoundError(f"No .omni file found in {session_dir}")

        self.session_name = omni_files[0].stem
        self._metadata: Optional[Dict] = None
        self._intrinsics: Optional[CameraIntrinsics] = None
        self._depth_files: Optional[List[Path]] = None
        self._confidence_files: Optional[List[Path]] = None
        self._video_capture = None
        self._camera_loader: Optional['AlembicCameraLoader'] = None

    @property
    def omni_path(self) -> Path:
        return self.session_path / f"{self.session_name}.omni"

    @property
    def video_path(self) -> Path:
        return self.session_path / f"{self.session_name}.mov"

    @property
    def abc_path(self) -> Path:
        return self.session_path / f"{self.session_name}.abc"

    @property
    def usd_path(self) -> Optional[Path]:
        """USDカメラポーズファイルパス（存在する場合）"""
        for ext in ['.usda', '.usd', '.usdc']:
            path = self.session_path / f"{self.session_name}{ext}"
            if path.exists():
                return path
        return None

    @property
    def camera_pose_path(self) -> Optional[Path]:
        """カメラポーズファイルパス（USD優先、ABCフォールバック）"""
        # USD形式を優先（ARM64でも動作する）
        if self.usd_path:
            return self.usd_path
        # Alembic形式
        if self.abc_path.exists():
            return self.abc_path
        return None

    @property
    def depth_dir(self) -> Path:
        return self.session_path / f"{self.session_name}_depth"

    @property
    def confidence_dir(self) -> Path:
        return self.session_path / f"{self.session_name}_depthConfidence"

    @property
    def metadata(self) -> Dict:
        """メタデータを読み込み"""
        if self._metadata is None:
            with open(self.omni_path, 'r') as f:
                self._metadata = json.load(f)
        return self._metadata

    @property
    def video_fps(self) -> int:
        """動画のFPS"""
        return self.metadata.get("data", {}).get("video", {}).get("fps", 60)

    @property
    def video_resolution(self) -> Tuple[int, int]:
        """動画の解像度 (width, height)"""
        video_data = self.metadata.get("data", {}).get("video", {}).get("resolution", {})
        return video_data.get("width", 1080), video_data.get("height", 1920)

    @property
    def num_depth_frames(self) -> int:
        """深度フレーム数"""
        return len(self.depth_files)

    @property
    def depth_files(self) -> List[Path]:
        """深度ファイルリスト（ソート済み）"""
        if self._depth_files is None:
            self._depth_files = sorted(self.depth_dir.glob("*.png"))
        return self._depth_files

    @property
    def confidence_files(self) -> List[Path]:
        """深度信頼度ファイルリスト（ソート済み）"""
        if self._confidence_files is None:
            self._confidence_files = sorted(self.confidence_dir.glob("*.png"))
        return self._confidence_files

    @property
    def camera_frames(self) -> List[Dict]:
        """各フレームのカメラパラメータ"""
        return self.metadata.get("data", {}).get("camera", {}).get("frames", [])

    @property
    def camera_loader(self):
        """カメラローダー（USD優先、Alembicフォールバック）"""
        if self._camera_loader is None:
            pose_path = self.camera_pose_path
            if pose_path:
                try:
                    # 統合CameraLoaderを優先（USD対応）
                    if HAS_CAMERA_LOADER:
                        self._camera_loader = CameraLoader(str(pose_path))
                        if self._camera_loader.num_frames > 0:
                            print(f"Loaded camera poses from: {pose_path}")
                        else:
                            self._camera_loader = None
                    # フォールバック: AlembicCameraLoader
                    elif HAS_ALEMBIC_LOADER and pose_path.suffix == '.abc':
                        self._camera_loader = AlembicCameraLoader(str(pose_path))
                except Exception as e:
                    print(f"Warning: Failed to load camera poses: {e}")
        return self._camera_loader

    @property
    def has_camera_poses(self) -> bool:
        """カメラポーズが利用可能か"""
        return self.camera_loader is not None and self.camera_loader.num_frames > 0

    @property
    def num_camera_poses(self) -> int:
        """カメラポーズのフレーム数"""
        if self.camera_loader:
            return self.camera_loader.num_frames
        return 0

    def get_intrinsics(self, frame_index: int = 0) -> CameraIntrinsics:
        """カメラ内部パラメータを取得"""
        # 深度画像のサイズを取得
        if self.depth_files:
            sample_depth = Image.open(self.depth_files[0])
            depth_width, depth_height = sample_depth.size
        else:
            depth_width, depth_height = 144, 256

        # 焦点距離を取得
        camera_frames = self.camera_frames
        if camera_frames and frame_index < len(camera_frames):
            focal_length = camera_frames[frame_index].get("focal_length", 28.6)
        else:
            focal_length = 28.6  # デフォルト値

        video_width, video_height = self.video_resolution

        return CameraIntrinsics(
            focal_length=focal_length,
            width=video_width,
            height=video_height,
            depth_width=depth_width,
            depth_height=depth_height
        )

    def load_depth(self, frame_index: int) -> np.ndarray:
        """
        深度画像を読み込み

        Args:
            frame_index: フレームインデックス

        Returns:
            depth_map: 深度マップ (H, W)、単位はメートル
        """
        if frame_index >= len(self.depth_files):
            raise IndexError(f"Frame index {frame_index} out of range (max: {len(self.depth_files)-1})")

        depth_path = self.depth_files[frame_index]
        depth_img = Image.open(depth_path)
        depth_arr = np.array(depth_img, dtype=np.float32)

        # mm → メートルに変換
        depth_meters = depth_arr / 1000.0

        return depth_meters

    def load_depth_confidence(self, frame_index: int) -> Optional[np.ndarray]:
        """深度信頼度を読み込み"""
        if frame_index >= len(self.confidence_files):
            return None

        conf_path = self.confidence_files[frame_index]
        conf_img = Image.open(conf_path)
        return np.array(conf_img)

    def load_video_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """
        動画からフレームを抽出

        Args:
            frame_index: フレームインデックス

        Returns:
            rgb_frame: RGB画像 (H, W, 3) または None
        """
        try:
            import cv2
        except ImportError:
            print("Warning: OpenCV not available. Video frames will not be loaded.")
            return None

        if self._video_capture is None:
            if not self.video_path.exists():
                return None
            self._video_capture = cv2.VideoCapture(str(self.video_path))

        # フレーム位置を設定
        self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self._video_capture.read()

        if not ret:
            return None

        # BGR → RGB変換
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_frame(self, frame_index: int, load_rgb: bool = True) -> FrameData:
        """
        指定フレームの全データを取得

        Args:
            frame_index: フレームインデックス
            load_rgb: RGB画像も読み込むか

        Returns:
            FrameData: フレームデータ
        """
        depth_map = self.load_depth(frame_index)
        depth_confidence = self.load_depth_confidence(frame_index)
        rgb_frame = self.load_video_frame(frame_index) if load_rgb else None

        # カメラパラメータ
        camera_frames = self.camera_frames
        if camera_frames and frame_index < len(camera_frames):
            focal_length = camera_frames[frame_index].get("focal_length", 28.6)
        else:
            focal_length = 28.6

        return FrameData(
            frame_index=frame_index,
            rgb_frame=rgb_frame,
            depth_map=depth_map,
            depth_confidence=depth_confidence,
            focal_length=focal_length
        )

    def iter_frames(
        self,
        start: int = 0,
        end: Optional[int] = None,
        step: int = 1,
        load_rgb: bool = True
    ) -> Iterator[FrameData]:
        """
        フレームをイテレート

        Args:
            start: 開始フレーム
            end: 終了フレーム（None=最後まで）
            step: ステップ数
            load_rgb: RGB画像も読み込むか

        Yields:
            FrameData: 各フレームのデータ
        """
        if end is None:
            end = self.num_depth_frames

        for i in range(start, end, step):
            yield self.get_frame(i, load_rgb=load_rgb)

    def depth_to_pointcloud(
        self,
        depth_map: np.ndarray,
        intrinsics: CameraIntrinsics,
        max_depth: float = 10.0,
        min_depth: float = 0.1
    ) -> np.ndarray:
        """
        深度マップから点群を生成（カメラ座標系）

        Args:
            depth_map: 深度マップ (H, W)、メートル単位
            intrinsics: カメラ内部パラメータ
            max_depth: 最大深度（メートル）
            min_depth: 最小深度（メートル）

        Returns:
            points: 点群 (N, 3)、カメラ座標系
        """
        height, width = depth_map.shape

        # ピクセルグリッドを作成
        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v)

        # 有効な深度のマスク
        valid_mask = (depth_map > min_depth) & (depth_map < max_depth) & np.isfinite(depth_map)

        # 3D座標に変換（ピンホールカメラモデル）
        z = depth_map
        x = (u - intrinsics.cx) * z / intrinsics.fx
        y = (v - intrinsics.cy) * z / intrinsics.fy

        # 有効な点のみ抽出
        points = np.stack([
            x[valid_mask],
            y[valid_mask],
            z[valid_mask]
        ], axis=-1)

        return points

    def depth_to_pointcloud_world(
        self,
        depth_map: np.ndarray,
        intrinsics: CameraIntrinsics,
        frame_index: int,
        max_depth: float = 10.0,
        min_depth: float = 0.1
    ) -> np.ndarray:
        """
        深度マップからワールド座標系の点群を生成

        Args:
            depth_map: 深度マップ (H, W)、メートル単位
            intrinsics: カメラ内部パラメータ
            frame_index: フレームインデックス（カメラポーズ用）
            max_depth: 最大深度（メートル）
            min_depth: 最小深度（メートル）

        Returns:
            points: 点群 (N, 3)、ワールド座標系
        """
        # カメラ座標系の点群を生成
        points_camera = self.depth_to_pointcloud(depth_map, intrinsics, max_depth, min_depth)

        if len(points_camera) == 0:
            return points_camera

        # カメラポーズが利用可能な場合、ワールド座標に変換
        if self.has_camera_poses and frame_index < self.num_camera_poses:
            camera_matrix = self.camera_loader.get_transform(frame_index)
            points_world = transform_points_to_world(points_camera, camera_matrix, flip_yz=True)
            return points_world

        return points_camera

    def get_camera_transform(self, frame_index: int) -> Optional[np.ndarray]:
        """
        特定フレームのカメラ変換行列を取得

        Args:
            frame_index: フレームインデックス

        Returns:
            4x4 変換行列、または None
        """
        if self.has_camera_poses and frame_index < self.num_camera_poses:
            return self.camera_loader.get_transform(frame_index)
        return None

    def get_camera_position(self, frame_index: int) -> Optional[np.ndarray]:
        """
        特定フレームのカメラ位置を取得

        Args:
            frame_index: フレームインデックス

        Returns:
            (3,) 位置ベクトル、または None
        """
        if self.has_camera_poses and frame_index < self.num_camera_poses:
            return self.camera_loader.get_position(frame_index)
        return None

    def get_mesh_path(self) -> Optional[Path]:
        """スキャンメッシュのパスを取得"""
        obj_files = list(self.session_path.glob("scan_*.obj"))
        return obj_files[0] if obj_files else None

    def summary(self) -> Dict:
        """セッション情報のサマリー"""
        intrinsics = self.get_intrinsics()

        # カメラポーズ情報
        camera_poses_info = {
            "available": self.has_camera_poses,
            "num_frames": self.num_camera_poses
        }
        if self.has_camera_poses and self.num_camera_poses > 0:
            first_pos = self.get_camera_position(0)
            last_pos = self.get_camera_position(self.num_camera_poses - 1)
            camera_poses_info["first_position"] = first_pos.tolist() if first_pos is not None else None
            camera_poses_info["last_position"] = last_pos.tolist() if last_pos is not None else None

        return {
            "session_name": self.session_name,
            "session_path": str(self.session_path),
            "video": {
                "path": str(self.video_path) if self.video_path.exists() else None,
                "fps": self.video_fps,
                "resolution": self.video_resolution
            },
            "depth": {
                "num_frames": self.num_depth_frames,
                "resolution": (intrinsics.depth_width, intrinsics.depth_height),
                "unit": "meters"
            },
            "camera_intrinsics": {
                "focal_length_mm": intrinsics.focal_length,
                "fx_px": intrinsics.fx,
                "fy_px": intrinsics.fy,
                "cx": intrinsics.cx,
                "cy": intrinsics.cy
            },
            "camera_poses": camera_poses_info,
            "files": {
                "omni": str(self.omni_path),
                "abc": str(self.abc_path) if self.abc_path.exists() else None,
                "mesh": str(self.get_mesh_path()) if self.get_mesh_path() else None
            }
        }

    def __del__(self):
        """リソースの解放"""
        if self._video_capture is not None:
            self._video_capture.release()


def main():
    parser = argparse.ArgumentParser(
        description="Omniscientデータローダー",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # セッション情報を表示
  python -m server.multiview.omniscient_loader experiments/omniscient_sample/003

  # 点群を生成して保存
  python -m server.multiview.omniscient_loader experiments/omniscient_sample/003 --export-ply

  # 特定フレームの深度を表示
  python -m server.multiview.omniscient_loader experiments/omniscient_sample/003 --frame 100
        """
    )

    parser.add_argument("session_dir", help="Omniscientセッションディレクトリ")
    parser.add_argument("--frame", "-f", type=int, help="特定フレームの情報を表示")
    parser.add_argument("--export-ply", action="store_true", help="点群をPLY形式で出力")
    parser.add_argument("--output", "-o", help="出力ファイルパス")
    parser.add_argument("--step", type=int, default=10, help="フレーム間隔（点群統合時）")

    args = parser.parse_args()

    # ローダーを初期化
    loader = OmniscientLoader(args.session_dir)

    # サマリーを表示
    print("=== Omniscient Session Summary ===")
    summary = loader.summary()
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # 特定フレームの情報
    if args.frame is not None:
        print(f"\n=== Frame {args.frame} ===")
        frame = loader.get_frame(args.frame, load_rgb=False)
        print(f"Depth shape: {frame.depth_map.shape}")
        print(f"Depth range: {frame.depth_map[frame.depth_map > 0].min():.3f}m - {frame.depth_map.max():.3f}m")
        print(f"Focal length: {frame.focal_length}mm")

        # 点群を生成
        intrinsics = loader.get_intrinsics(args.frame)
        points = loader.depth_to_pointcloud(frame.depth_map, intrinsics)
        print(f"Point cloud: {len(points)} points")

    # PLYエクスポート
    if args.export_ply:
        use_world = loader.has_camera_poses
        coord_system = "world" if use_world else "camera"
        print(f"\n=== Exporting Point Cloud (step={args.step}, {coord_system} coords) ===")

        if use_world:
            print(f"Camera poses available: {loader.num_camera_poses} frames")

        all_points = []
        max_frame = min(loader.num_depth_frames, loader.num_camera_poses) if use_world else loader.num_depth_frames

        for frame_idx in range(0, max_frame, args.step):
            frame = loader.get_frame(frame_idx, load_rgb=False)
            intrinsics = loader.get_intrinsics(frame.frame_index)

            if use_world:
                points = loader.depth_to_pointcloud_world(
                    frame.depth_map, intrinsics, frame.frame_index
                )
                pos = loader.get_camera_position(frame.frame_index)
                print(f"Frame {frame.frame_index}: {len(points)} points, camera=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            else:
                points = loader.depth_to_pointcloud(frame.depth_map, intrinsics)
                print(f"Frame {frame.frame_index}: {len(points)} points")

            all_points.append(points)

        merged_points = np.vstack(all_points)
        print(f"\nTotal points: {len(merged_points)}")

        # PLY保存
        output_path = args.output or str(loader.session_path / "pointcloud_world.ply")
        save_ply(merged_points, output_path)
        print(f"Saved to: {output_path}")


def save_ply(points: np.ndarray, filepath: str):
    """PLYファイルを保存"""
    with open(filepath, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for point in points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")


if __name__ == "__main__":
    main()
