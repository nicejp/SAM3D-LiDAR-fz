#!/usr/bin/env python3
"""カメラポーズのデバッグスクリプト"""

import sys
import json
import numpy as np

# セッションディレクトリを引数から取得
if len(sys.argv) < 2:
    print("Usage: python debug_camera_poses.py <session_dir>")
    print("Example: python debug_camera_poses.py experiments/omniscient_sample/003")
    sys.exit(1)

session_dir = sys.argv[1]

from server.multiview.omniscient_loader import OmniscientLoader

# ローダーを初期化
print(f"Loading session: {session_dir}")
loader = OmniscientLoader(session_dir)

print("\n" + "="*60)
print("SESSION SUMMARY")
print("="*60)
print(f"Session name: {loader.session_name}")
print(f"Video FPS: {loader.video_fps}")
print(f"Depth frames: {loader.num_depth_frames}")
print(f"Camera poses: {loader.num_camera_poses}")
print(f"Has camera poses: {loader.has_camera_poses}")

if loader.num_depth_frames > 0 and loader.num_camera_poses > 0:
    ratio = loader.num_camera_poses / loader.num_depth_frames
    print(f"Frame ratio (camera/depth): {ratio:.4f}")
    print(f"Scaling enabled: {ratio < 0.9 or ratio > 1.1}")

print("\n" + "="*60)
print("CAMERA POSE DETAILS")
print("="*60)

if loader.camera_loader:
    cam = loader.camera_loader
    print(f"Camera loader type: {type(cam).__name__}")
    print(f"Number of poses: {cam.num_frames}")

    # フレーム番号の範囲を確認
    if hasattr(cam, 'frame_to_index') and cam.frame_to_index:
        frames = sorted(cam.frame_to_index.keys())
        print(f"Frame numbers: {frames[0]} to {frames[-1]}")
        print(f"First 10 frames: {frames[:10]}")
        print(f"Last 10 frames: {frames[-10:]}")

    # 最初と最後のカメラ位置
    print("\nCamera positions:")
    for i in [0, 1, 2, cam.num_frames // 2, cam.num_frames - 2, cam.num_frames - 1]:
        if i < cam.num_frames:
            pos = cam.get_position(i)
            frame_num = list(cam.frame_to_index.keys())[i] if hasattr(cam, 'frame_to_index') else i
            print(f"  Pose {i} (frame {frame_num}): ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

print("\n" + "="*60)
print("FRAME MAPPING TEST")
print("="*60)

# ビデオフレーム→カメラフレームのマッピングをテスト
test_frames = [0, 10, 50, 100, 150, 200, 250, 300, 350, 400, loader.num_depth_frames - 1]
print("\nVideo frame -> Camera frame mapping:")
for vf in test_frames:
    if vf < loader.num_depth_frames:
        cf = loader.video_frame_to_camera_frame(vf)
        pos = loader.get_camera_position(vf)
        pos_str = f"({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})" if pos is not None else "N/A"
        print(f"  Video {vf:4d} -> Camera {cf:4d} | Position: {pos_str}")

print("\n" + "="*60)
print("CAMERA MATRIX ANALYSIS")
print("="*60)

# カメラ行列の分析
if loader.camera_loader:
    print("\nFirst frame camera matrix:")
    matrix = loader.get_camera_transform(0)
    if matrix is not None:
        print(matrix)
        print(f"\nTranslation (position): {matrix[:3, 3]}")
        print(f"Rotation matrix determinant: {np.linalg.det(matrix[:3, :3]):.4f}")

    print("\nMiddle frame camera matrix:")
    mid_frame = loader.num_depth_frames // 2
    matrix = loader.get_camera_transform(mid_frame)
    if matrix is not None:
        print(matrix)
        print(f"\nTranslation (position): {matrix[:3, 3]}")

    print("\nLast frame camera matrix:")
    matrix = loader.get_camera_transform(loader.num_depth_frames - 1)
    if matrix is not None:
        print(matrix)
        print(f"\nTranslation (position): {matrix[:3, 3]}")

print("\n" + "="*60)
print("DEPTH FILE MAPPING")
print("="*60)

# 深度ファイルマッピング
depth_map = loader.depth_frame_map
if depth_map:
    frames = sorted(depth_map.keys())
    print(f"Depth frame map entries: {len(frames)}")
    print(f"Frame range: {frames[0]} to {frames[-1]}")
    print(f"First 10: {frames[:10]}")
    print(f"Last 10: {frames[-10:]}")

    # サンプルファイル名
    print("\nSample depth files:")
    for f in [0, 10, 100, 200, frames[-1]]:
        if f in depth_map:
            print(f"  Frame {f}: {depth_map[f].name}")

print("\n" + "="*60)
print("COORDINATE SYSTEM TEST")
print("="*60)

# 座標変換のテスト
intrinsics = loader.get_intrinsics()
print(f"Intrinsics: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, cx={intrinsics.cx:.2f}, cy={intrinsics.cy:.2f}")

# 単純なテストポイント（カメラの1m前の中心点）
test_point = np.array([[0, 0, 1.0]])  # カメラ座標系: 1m前方
print(f"\nTest point in camera coords: {test_point[0]}")

for frame_idx in [0, loader.num_depth_frames // 2, loader.num_depth_frames - 1]:
    matrix = loader.get_camera_transform(frame_idx)
    if matrix is not None:
        from server.multiview.alembic_loader import transform_points_to_world
        world_point = transform_points_to_world(test_point, matrix, flip_yz=True)
        cam_pos = loader.get_camera_position(frame_idx)
        print(f"Frame {frame_idx}: camera at {cam_pos}, test point at {world_point[0]}")
