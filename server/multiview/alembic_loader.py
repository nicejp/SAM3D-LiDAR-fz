#!/usr/bin/env python3
"""
Alembic Camera Pose Loader

Alembic (.abc) ファイルからカメラポーズ（位置・回転）を抽出する。
BlenderのPython API (bpy) を使用。

使い方:
    python -m server.multiview.alembic_loader experiments/omniscient_sample/003/003.abc

出力:
    カメラポーズをJSONで出力、または各フレームの変換行列を返す
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

# bpyのインポート（インストールされている場合）
HAS_BPY = False
try:
    import bpy
    from mathutils import Matrix, Vector, Euler
    HAS_BPY = True
except ImportError:
    pass


def load_alembic_camera_poses(abc_path: str, output_json: Optional[str] = None) -> List[Dict]:
    """
    Alembicファイルからカメラポーズを抽出

    Args:
        abc_path: Alembicファイルパス
        output_json: 出力JSONファイルパス（オプション）

    Returns:
        カメラポーズのリスト（各フレームの変換行列を含む）
    """
    if not HAS_BPY:
        print("Warning: bpy (Blender Python) is not installed. Camera poses will not be available.")
        print("         Run: pip install bpy")
        return []  # 空のリストを返す（エラーではなく警告）

    abc_path = Path(abc_path)
    if not abc_path.exists():
        raise FileNotFoundError(f"Alembic file not found: {abc_path}")

    # 新しいBlenderシーンを作成
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Alembicをインポート
    print(f"Loading Alembic: {abc_path}")
    bpy.ops.wm.alembic_import(filepath=str(abc_path))

    # カメラオブジェクトを検索
    camera = None
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            camera = obj
            print(f"Found camera: {obj.name}")
            break

    if camera is None:
        # カメラがない場合、他のオブジェクトを探す
        for obj in bpy.data.objects:
            print(f"  Object: {obj.name} (type: {obj.type})")
        raise RuntimeError("No camera found in Alembic file")

    # フレーム範囲を取得
    scene = bpy.context.scene
    start_frame = scene.frame_start
    end_frame = scene.frame_end

    # Alembicのフレーム範囲を検出
    # アクションからフレーム範囲を取得
    if camera.animation_data and camera.animation_data.action:
        action = camera.animation_data.action
        start_frame = int(action.frame_range[0])
        end_frame = int(action.frame_range[1])
    else:
        # デフォルトのフレーム範囲を使用（Omniscientは通常60fps）
        # タイムライン全体をスキャン
        end_frame = 500  # 最大500フレームまで探索

    print(f"Frame range: {start_frame} - {end_frame}")

    poses = []

    for frame in range(start_frame, end_frame + 1):
        scene.frame_set(frame)

        # depsgraphを更新して評価済みオブジェクトを取得
        # （Transform Cacheコンストレイントからアニメーションを読み込むために必要）
        depsgraph = bpy.context.evaluated_depsgraph_get()
        camera_eval = camera.evaluated_get(depsgraph)

        # ワールド変換行列を取得
        world_matrix = camera_eval.matrix_world.copy()

        # 4x4行列をリストに変換
        matrix_list = [list(row) for row in world_matrix]

        # 位置を抽出
        location = world_matrix.translation.copy()

        # 回転を抽出（オイラー角）
        rotation = world_matrix.to_euler()

        pose = {
            "frame": frame,
            "matrix_world": matrix_list,
            "location": {
                "x": location.x,
                "y": location.y,
                "z": location.z
            },
            "rotation_euler": {
                "x": rotation.x,
                "y": rotation.y,
                "z": rotation.z,
                "order": rotation.order
            }
        }
        poses.append(pose)

        if frame % 100 == 0:
            print(f"  Frame {frame}: pos=({location.x:.3f}, {location.y:.3f}, {location.z:.3f})")

    # 同じ行列が続く場合は終端を検出
    # 最後の有効フレームを見つける
    valid_poses = []
    last_matrix = None
    repeated_count = 0

    for pose in poses:
        current_matrix = tuple(tuple(row) for row in pose["matrix_world"])

        if last_matrix is not None and current_matrix == last_matrix:
            repeated_count += 1
            if repeated_count > 10:  # 10フレーム以上同じなら終了とみなす
                break
        else:
            repeated_count = 0

        valid_poses.append(pose)
        last_matrix = current_matrix

    print(f"Valid frames: {len(valid_poses)}")

    # JSON出力
    if output_json:
        with open(output_json, 'w') as f:
            json.dump({
                "abc_file": str(abc_path),
                "num_frames": len(valid_poses),
                "poses": valid_poses
            }, f, indent=2)
        print(f"Saved to: {output_json}")

    return valid_poses


def poses_to_transforms(poses: List[Dict]) -> np.ndarray:
    """
    ポーズリストを変換行列の配列に変換

    Args:
        poses: load_alembic_camera_poses()の出力

    Returns:
        transforms: (N, 4, 4) の変換行列配列
    """
    transforms = []
    for pose in poses:
        matrix = np.array(pose["matrix_world"])
        transforms.append(matrix)
    return np.array(transforms)


def get_camera_position(pose: Dict) -> np.ndarray:
    """ポーズからカメラ位置を取得"""
    loc = pose["location"]
    return np.array([loc["x"], loc["y"], loc["z"]])


def transform_points_to_world(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    flip_yz: bool = True
) -> np.ndarray:
    """
    カメラ座標系の点群をワールド座標系に変換

    Args:
        points: カメラ座標系の点群 (N, 3)
        camera_matrix: 4x4 ワールド変換行列
        flip_yz: Y/Z軸を反転（Blender座標系対応）

    Returns:
        world_points: ワールド座標系の点群 (N, 3)
    """
    if flip_yz:
        # カメラ座標系（Y下、Z奥）からBlender座標系（Z上、Y奥）に変換
        points_flipped = points.copy()
        points_flipped[:, 1] = -points[:, 1]  # Y反転
        points_flipped[:, 2] = -points[:, 2]  # Z反転
        points = points_flipped

    # 同次座標に変換
    ones = np.ones((points.shape[0], 1))
    points_homo = np.hstack([points, ones])

    # ワールド座標に変換
    world_points = (camera_matrix @ points_homo.T).T[:, :3]

    return world_points


class AlembicCameraLoader:
    """Alembicファイルからカメラポーズを読み込むクラス"""

    def __init__(self, abc_path: str):
        """
        Args:
            abc_path: Alembicファイルパス
        """
        self.abc_path = Path(abc_path)
        self._poses: Optional[List[Dict]] = None
        self._transforms: Optional[np.ndarray] = None

    @property
    def poses(self) -> List[Dict]:
        """カメラポーズを読み込み（遅延読み込み）"""
        if self._poses is None:
            self._poses = load_alembic_camera_poses(str(self.abc_path))
        return self._poses

    @property
    def transforms(self) -> np.ndarray:
        """変換行列の配列を取得"""
        if self._transforms is None:
            self._transforms = poses_to_transforms(self.poses)
        return self._transforms

    @property
    def num_frames(self) -> int:
        """フレーム数"""
        return len(self.poses)

    def get_transform(self, frame_index: int) -> np.ndarray:
        """
        特定フレームの変換行列を取得

        Args:
            frame_index: フレームインデックス

        Returns:
            4x4 変換行列
        """
        if frame_index >= self.num_frames:
            raise IndexError(f"Frame index {frame_index} out of range (max: {self.num_frames-1})")
        return self.transforms[frame_index]

    def get_position(self, frame_index: int) -> np.ndarray:
        """特定フレームのカメラ位置を取得"""
        return get_camera_position(self.poses[frame_index])

    def transform_points(
        self,
        points: np.ndarray,
        frame_index: int,
        flip_yz: bool = True
    ) -> np.ndarray:
        """
        点群をワールド座標系に変換

        Args:
            points: カメラ座標系の点群 (N, 3)
            frame_index: フレームインデックス
            flip_yz: Y/Z軸を反転

        Returns:
            ワールド座標系の点群
        """
        camera_matrix = self.get_transform(frame_index)
        return transform_points_to_world(points, camera_matrix, flip_yz)

    def save_poses_json(self, output_path: str):
        """ポーズをJSONファイルに保存"""
        with open(output_path, 'w') as f:
            json.dump({
                "abc_file": str(self.abc_path),
                "num_frames": self.num_frames,
                "poses": self.poses
            }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Alembicファイルからカメラポーズを抽出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # カメラポーズを表示
  python -m server.multiview.alembic_loader experiments/omniscient_sample/003/003.abc

  # JSONに保存
  python -m server.multiview.alembic_loader experiments/omniscient_sample/003/003.abc -o poses.json
        """
    )

    parser.add_argument("abc_file", help="Alembicファイルパス")
    parser.add_argument("--output", "-o", help="出力JSONファイルパス")
    parser.add_argument("--frame", "-f", type=int, help="特定フレームの情報を表示")

    args = parser.parse_args()

    if not HAS_BPY:
        print("Error: bpy (Blender Python) is not installed.")
        print("Install with: pip install bpy")
        sys.exit(1)

    # ポーズを読み込み
    loader = AlembicCameraLoader(args.abc_file)

    print(f"=== Alembic Camera Poses ===")
    print(f"File: {args.abc_file}")
    print(f"Frames: {loader.num_frames}")

    if args.frame is not None:
        print(f"\n=== Frame {args.frame} ===")
        pose = loader.poses[args.frame]
        print(f"Position: ({pose['location']['x']:.4f}, {pose['location']['y']:.4f}, {pose['location']['z']:.4f})")
        print(f"Rotation: ({pose['rotation_euler']['x']:.4f}, {pose['rotation_euler']['y']:.4f}, {pose['rotation_euler']['z']:.4f})")
        print(f"Matrix:\n{np.array(pose['matrix_world'])}")
    else:
        # 最初と最後の数フレームを表示
        for i in [0, 1, 2, loader.num_frames // 2, loader.num_frames - 1]:
            if i < loader.num_frames:
                pos = loader.get_position(i)
                print(f"  Frame {i}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    if args.output:
        loader.save_poses_json(args.output)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
