#!/usr/bin/env python3
"""
Camera Pose Loader

複数フォーマット（USD, Alembic）からカメラポーズを読み込む統合ローダー。
ARM64環境ではUSD形式を推奨（usd-coreがpipでインストール可能）。

対応フォーマット:
- .usda, .usd, .usdc (USD) - pxr経由
- .abc (Alembic) - Blender subprocess経由

使い方:
    python -m server.multiview.camera_loader path/to/camera.usda
    python -m server.multiview.camera_loader path/to/camera.abc
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import numpy as np

# USDサポート確認
HAS_USD = False
try:
    from pxr import Usd, UsdGeom, Gf
    HAS_USD = True
except ImportError:
    pass

# Alembic loader（Blender経由）
from server.multiview.alembic_loader import (
    load_alembic_camera_poses,
    BLENDER_PATH,
    HAS_BPY
)


def load_usd_camera_poses(usd_path: str) -> List[Dict]:
    """
    USDファイルからカメラポーズを抽出

    Args:
        usd_path: USD/USDA/USDCファイルパス

    Returns:
        カメラポーズのリスト
    """
    if not HAS_USD:
        print("Warning: pxr (USD) not available. Install with: pip install usd-core")
        return []

    usd_path = str(Path(usd_path).absolute())

    # USDステージを開く
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"Warning: Failed to open USD file: {usd_path}")
        return []

    # カメラを検索
    camera_prim = None
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            camera_prim = prim
            print(f"Found camera: {prim.GetPath()}")
            break

    if camera_prim is None:
        # カメラがない場合、Xformを探す（Omniscientはカメラをトランスフォームノードとして出力することがある）
        print("No Camera prim found. Looking for Xform prims...")
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Xform):
                # カメラっぽい名前のXformを探す
                name = prim.GetName().lower()
                if 'camera' in name or 'cam' in name:
                    camera_prim = prim
                    print(f"Found camera Xform: {prim.GetPath()}")
                    break

    if camera_prim is None:
        print("Warning: No camera found in USD file")
        # 利用可能なプリムを表示
        print("Available prims:")
        for prim in stage.Traverse():
            print(f"  {prim.GetPath()} ({prim.GetTypeName()})")
        return []

    # UsdGeomXformableとして取得
    xformable = UsdGeom.Xformable(camera_prim)

    # タイムコードを取得
    time_codes = stage.GetTimeSamples(xformable.GetXformOpOrderAttr())
    if not time_codes:
        # 静的なシーンの場合、デフォルトのタイムコードを使用
        time_codes = [Usd.TimeCode.Default()]

    print(f"Time samples: {len(time_codes)}")

    poses = []
    for i, time_code in enumerate(time_codes):
        # ローカル変換行列を取得
        transform = xformable.ComputeLocalToWorldTransform(time_code)

        # GfMatrix4d → numpy array
        matrix_list = [[transform[row][col] for col in range(4)] for row in range(4)]
        matrix_np = np.array(matrix_list)

        # 位置を抽出
        location = transform.ExtractTranslation()

        # 回転を抽出（クォータニオン → オイラー角）
        rotation = transform.ExtractRotation()
        decomposed = rotation.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))

        pose = {
            "frame": i if time_code == Usd.TimeCode.Default() else int(time_code),
            "matrix_world": matrix_list,
            "location": {
                "x": location[0],
                "y": location[1],
                "z": location[2]
            },
            "rotation_euler": {
                "x": decomposed[0],
                "y": decomposed[1],
                "z": decomposed[2],
                "order": "XYZ"
            }
        }
        poses.append(pose)

        if i % 100 == 0:
            print(f"  Frame {pose['frame']}: pos=({location[0]:.3f}, {location[1]:.3f}, {location[2]:.3f})")

    print(f"Loaded {len(poses)} camera poses from USD")
    return poses


def load_camera_poses(
    file_path: str,
    output_json: Optional[str] = None
) -> List[Dict]:
    """
    カメラポーズを読み込む（統合関数）

    フォーマットを自動検出し、適切なローダーを使用。

    Args:
        file_path: カメラポーズファイル（.usda, .usd, .usdc, .abc）
        output_json: 出力JSONファイルパス（オプション）

    Returns:
        カメラポーズのリスト
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    poses = []

    if ext in ['.usda', '.usd', '.usdc']:
        # USD形式
        if not HAS_USD:
            print("Error: USD format requires pxr. Install with: pip install usd-core")
            return []
        poses = load_usd_camera_poses(file_path)

    elif ext == '.abc':
        # Alembic形式
        if not HAS_BPY and BLENDER_PATH is None:
            print("Error: Alembic format requires Blender or bpy.")
            print("       On ARM64 Linux, consider using USD format instead:")
            print("       1. Re-export from Omniscient with USD format")
            print("       2. pip install usd-core")
            return []
        poses = load_alembic_camera_poses(file_path)

    else:
        print(f"Error: Unsupported format: {ext}")
        print("       Supported formats: .usda, .usd, .usdc, .abc")
        return []

    # JSON出力
    if output_json and poses:
        with open(output_json, 'w') as f:
            json.dump({
                "source_file": str(file_path),
                "format": ext,
                "num_frames": len(poses),
                "poses": poses
            }, f, indent=2)
        print(f"Saved to: {output_json}")

    return poses


def poses_to_transforms(poses: List[Dict]) -> np.ndarray:
    """
    ポーズリストを変換行列の配列に変換
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


class CameraLoader:
    """統合カメラポーズローダー"""

    def __init__(self, file_path: str):
        """
        Args:
            file_path: カメラポーズファイル（.usda, .usd, .usdc, .abc）
        """
        self.file_path = Path(file_path)
        self._poses: Optional[List[Dict]] = None
        self._transforms: Optional[np.ndarray] = None

    @property
    def poses(self) -> List[Dict]:
        """カメラポーズを読み込み（遅延読み込み）"""
        if self._poses is None:
            self._poses = load_camera_poses(str(self.file_path))
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
        """特定フレームの変換行列を取得"""
        if frame_index >= self.num_frames:
            raise IndexError(f"Frame index {frame_index} out of range (max: {self.num_frames-1})")
        return self.transforms[frame_index]

    def get_position(self, frame_index: int) -> np.ndarray:
        """特定フレームのカメラ位置を取得"""
        return get_camera_position(self.poses[frame_index])

    def save_poses_json(self, output_path: str):
        """ポーズをJSONファイルに保存"""
        with open(output_path, 'w') as f:
            json.dump({
                "source_file": str(self.file_path),
                "num_frames": self.num_frames,
                "poses": self.poses
            }, f, indent=2)


def check_capabilities():
    """利用可能なフォーマットを表示"""
    print("=== Camera Pose Loader Capabilities ===")
    print(f"USD  (.usda/.usd/.usdc): {'✅ Available' if HAS_USD else '❌ Not available (pip install usd-core)'}")
    print(f"Alembic (.abc):")
    print(f"  - bpy (direct):        {'✅ Available' if HAS_BPY else '❌ Not available'}")
    print(f"  - Blender (subprocess): {'✅ Available at ' + BLENDER_PATH if BLENDER_PATH else '❌ Not available'}")

    if BLENDER_PATH:
        # Alembicサポート確認
        import subprocess
        try:
            result = subprocess.run(
                [BLENDER_PATH, '--background', '--python-expr',
                 "import bpy; print('ALEMBIC_IMPORT:', 'alembic_import' in dir(bpy.ops.wm))"],
                capture_output=True, text=True, timeout=30
            )
            if 'ALEMBIC_IMPORT: True' in result.stdout:
                print("    (Alembic import: ✅)")
            else:
                print("    (Alembic import: ❌ - Blender built without Alembic support)")
        except:
            pass

    # ARM64環境の推奨
    import platform
    if platform.machine() in ['aarch64', 'arm64']:
        print("\n⚠️  ARM64環境を検出しました。")
        print("   Blender公式ビルドはx86_64のみです。")
        print("   USD形式の使用を推奨します:")
        print("   1. Omniscientから.usda形式でエクスポート")
        print("   2. pip install usd-core")


def main():
    parser = argparse.ArgumentParser(
        description="カメラポーズを読み込む（USD/Alembic対応）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # USDファイルからカメラポーズを読み込み
  python -m server.multiview.camera_loader path/to/camera.usda

  # Alembicファイルからカメラポーズを読み込み
  python -m server.multiview.camera_loader path/to/camera.abc

  # JSONに保存
  python -m server.multiview.camera_loader path/to/camera.usda -o poses.json

  # 対応フォーマットを確認
  python -m server.multiview.camera_loader --check
        """
    )

    parser.add_argument("file", nargs='?', help="カメラポーズファイル（.usda, .usd, .usdc, .abc）")
    parser.add_argument("--output", "-o", help="出力JSONファイルパス")
    parser.add_argument("--frame", "-f", type=int, help="特定フレームの情報を表示")
    parser.add_argument("--check", action="store_true", help="対応フォーマットを確認")

    args = parser.parse_args()

    if args.check:
        check_capabilities()
        return

    if not args.file:
        parser.print_help()
        return

    # ポーズを読み込み
    loader = CameraLoader(args.file)

    print(f"\n=== Camera Poses ===")
    print(f"File: {args.file}")
    print(f"Frames: {loader.num_frames}")

    if loader.num_frames == 0:
        print("No camera poses found.")
        sys.exit(1)

    if args.frame is not None:
        print(f"\n=== Frame {args.frame} ===")
        pose = loader.poses[args.frame]
        print(f"Position: ({pose['location']['x']:.4f}, {pose['location']['y']:.4f}, {pose['location']['z']:.4f})")
        rot = pose['rotation_euler']
        print(f"Rotation: ({rot['x']:.4f}, {rot['y']:.4f}, {rot['z']:.4f})")
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
