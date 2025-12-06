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
import subprocess
import tempfile
import shutil
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

# Blenderの実行ファイルパス
BLENDER_PATH = shutil.which("blender")


def _create_blender_script(abc_path: str, output_json_path: str) -> str:
    """Blenderで実行するPythonスクリプトを生成"""
    return f'''
import bpy
import json
import sys

abc_path = r"{abc_path}"
output_path = r"{output_json_path}"

# 新しいBlenderシーンを作成
bpy.ops.wm.read_factory_settings(use_empty=True)

# Alembicをインポート
print(f"Loading Alembic: {{abc_path}}", file=sys.stderr)
bpy.ops.wm.alembic_import(filepath=abc_path)

# カメラオブジェクトを検索
camera = None
for obj in bpy.data.objects:
    if obj.type == 'CAMERA':
        camera = obj
        print(f"Found camera: {{obj.name}}", file=sys.stderr)
        break

if camera is None:
    # カメラがない場合、他のオブジェクトを探す
    print("Objects in scene:", file=sys.stderr)
    for obj in bpy.data.objects:
        print(f"  {{obj.name}} (type: {{obj.type}})", file=sys.stderr)
    result = {{"error": "No camera found in Alembic file", "poses": []}}
    with open(output_path, 'w') as f:
        json.dump(result, f)
    sys.exit(0)

# フレーム範囲を取得
scene = bpy.context.scene
start_frame = scene.frame_start
end_frame = scene.frame_end

# Alembicのフレーム範囲を検出
if camera.animation_data and camera.animation_data.action:
    action = camera.animation_data.action
    start_frame = int(action.frame_range[0])
    end_frame = int(action.frame_range[1])
else:
    end_frame = 500  # 最大500フレームまで探索

print(f"Frame range: {{start_frame}} - {{end_frame}}", file=sys.stderr)

poses = []

for frame in range(start_frame, end_frame + 1):
    scene.frame_set(frame)

    # depsgraphを更新
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

    pose = {{
        "frame": frame,
        "matrix_world": matrix_list,
        "location": {{
            "x": location.x,
            "y": location.y,
            "z": location.z
        }},
        "rotation_euler": {{
            "x": rotation.x,
            "y": rotation.y,
            "z": rotation.z,
            "order": rotation.order
        }}
    }}
    poses.append(pose)

    if frame % 100 == 0:
        print(f"  Frame {{frame}}: pos=({{location.x:.3f}}, {{location.y:.3f}}, {{location.z:.3f}})", file=sys.stderr)

# 同じ行列が続く場合は終端を検出
valid_poses = []
last_matrix = None
repeated_count = 0

for pose in poses:
    current_matrix = tuple(tuple(row) for row in pose["matrix_world"])

    if last_matrix is not None and current_matrix == last_matrix:
        repeated_count += 1
        if repeated_count > 10:
            break
    else:
        repeated_count = 0

    valid_poses.append(pose)
    last_matrix = current_matrix

print(f"Valid frames: {{len(valid_poses)}}", file=sys.stderr)

# JSON出力
result = {{
    "abc_file": abc_path,
    "num_frames": len(valid_poses),
    "poses": valid_poses
}}
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f"Saved to: {{output_path}}", file=sys.stderr)
'''


def load_alembic_camera_poses_subprocess(abc_path: str) -> List[Dict]:
    """
    Blender subprocessを使用してAlembicファイルからカメラポーズを抽出

    Args:
        abc_path: Alembicファイルパス

    Returns:
        カメラポーズのリスト
    """
    if BLENDER_PATH is None:
        print("Warning: Blender is not installed. Camera poses will not be available.")
        print("         Run: apt-get install blender")
        return []

    abc_path = str(Path(abc_path).absolute())

    # 一時ファイルを作成
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as json_file:
            script_path = script_file.name
            json_path = json_file.name

    try:
        # Blenderスクリプトを書き込み
        script_content = _create_blender_script(abc_path, json_path)
        with open(script_path, 'w') as f:
            f.write(script_content)

        # Blenderをバックグラウンドで実行
        print(f"Loading Alembic via Blender: {abc_path}")
        result = subprocess.run(
            [BLENDER_PATH, '--background', '--python', script_path],
            capture_output=True,
            text=True,
            timeout=120
        )

        # stderrに進捗が出力される
        if result.stderr:
            for line in result.stderr.split('\n'):
                if line.strip() and not line.startswith('Blender'):
                    print(line)

        # JSONを読み込み
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)

            if "error" in data:
                print(f"Warning: {data['error']}")
                return []

            return data.get("poses", [])
        else:
            print("Warning: Blender output not found")
            return []

    except subprocess.TimeoutExpired:
        print("Warning: Blender process timed out")
        return []
    except Exception as e:
        print(f"Warning: Failed to load Alembic via Blender: {e}")
        return []
    finally:
        # 一時ファイルを削除
        if os.path.exists(script_path):
            os.unlink(script_path)
        if os.path.exists(json_path):
            os.unlink(json_path)


def load_alembic_camera_poses(abc_path: str, output_json: Optional[str] = None) -> List[Dict]:
    """
    Alembicファイルからカメラポーズを抽出

    Args:
        abc_path: Alembicファイルパス
        output_json: 出力JSONファイルパス（オプション）

    Returns:
        カメラポーズのリスト（各フレームの変換行列を含む）
    """
    # bpyが直接使えない場合はsubprocessを使用
    if not HAS_BPY:
        poses = load_alembic_camera_poses_subprocess(abc_path)

        if output_json and poses:
            with open(output_json, 'w') as f:
                json.dump({
                    "abc_file": str(abc_path),
                    "num_frames": len(poses),
                    "poses": poses
                }, f, indent=2)
            print(f"Saved to: {output_json}")

        return poses

    # bpyが使える場合は直接使用
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
        self._frame_to_index: Optional[Dict[int, int]] = None

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
    def frame_to_index(self) -> Dict[int, int]:
        """フレーム番号→配列インデックスのマッピング"""
        if self._frame_to_index is None:
            self._frame_to_index = {
                pose["frame"]: idx for idx, pose in enumerate(self.poses)
            }
        return self._frame_to_index

    @property
    def num_frames(self) -> int:
        """フレーム数"""
        return len(self.poses)

    @property
    def max_frame(self) -> int:
        """最大フレーム番号"""
        if not self.poses:
            return 0
        return max(pose["frame"] for pose in self.poses)

    def get_transform(self, frame_number: int) -> np.ndarray:
        """
        特定フレームの変換行列を取得

        Args:
            frame_number: 元のフレーム番号（配列インデックスではない）

        Returns:
            4x4 変換行列
        """
        if frame_number in self.frame_to_index:
            idx = self.frame_to_index[frame_number]
            return self.transforms[idx]

        # フレーム番号が存在しない場合、最も近いフレームを使用
        available_frames = sorted(self.frame_to_index.keys())
        if not available_frames:
            raise IndexError("No camera poses available")

        # 最も近いフレームを探す
        closest_frame = min(available_frames, key=lambda f: abs(f - frame_number))
        idx = self.frame_to_index[closest_frame]
        return self.transforms[idx]

    def has_frame(self, frame_number: int) -> bool:
        """指定フレーム番号のポーズが存在するか"""
        return frame_number in self.frame_to_index

    def get_position(self, frame_number: int) -> np.ndarray:
        """特定フレームのカメラ位置を取得"""
        if frame_number in self.frame_to_index:
            idx = self.frame_to_index[frame_number]
            return get_camera_position(self.poses[idx])
        # 最も近いフレームを使用
        available_frames = sorted(self.frame_to_index.keys())
        closest_frame = min(available_frames, key=lambda f: abs(f - frame_number))
        idx = self.frame_to_index[closest_frame]
        return get_camera_position(self.poses[idx])

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

    if not HAS_BPY and BLENDER_PATH is None:
        print("Error: Neither bpy nor Blender is available.")
        print("Install Blender with: apt-get install blender")
        sys.exit(1)

    # ポーズを読み込み
    loader = AlembicCameraLoader(args.abc_file)

    print(f"=== Alembic Camera Poses ===")
    print(f"File: {args.abc_file}")
    print(f"Frames: {loader.num_frames}")

    if loader.num_frames == 0:
        print("No camera poses found.")
        sys.exit(1)

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
