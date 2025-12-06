#!/usr/bin/env python3
"""座標変換のデバッグ - flip_yzの効果を確認"""

import sys
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python debug_coordinate_test.py <session_dir>")
    sys.exit(1)

session_dir = sys.argv[1]

from server.multiview.omniscient_loader import OmniscientLoader
from server.multiview.alembic_loader import transform_points_to_world

loader = OmniscientLoader(session_dir)

print("="*60)
print("COORDINATE TRANSFORM COMPARISON")
print("="*60)

# テスト用の点（カメラの前方0.5m、中央）
test_points = np.array([
    [0, 0, 0.5],      # カメラ中心から0.5m前方
    [0.1, 0, 0.5],    # 少し右
    [0, 0.1, 0.5],    # 少し下（Y-down座標系では）
])

print(f"\nTest points in camera coords (X-right, Y-down, Z-forward):")
for i, p in enumerate(test_points):
    print(f"  Point {i}: {p}")

# 複数フレームで比較
test_frames = [0, 50, 100, 150, 200]

print("\n" + "-"*60)
print("Comparing flip_yz=True vs flip_yz=False")
print("-"*60)

for frame_idx in test_frames:
    if frame_idx >= loader.num_depth_frames:
        continue

    camera_frame = loader.video_frame_to_camera_frame(frame_idx)
    matrix = loader.camera_loader.get_transform(camera_frame)
    cam_pos = loader.get_camera_position(frame_idx)

    # flip_yz=True
    world_flip = transform_points_to_world(test_points, matrix, flip_yz=True)

    # flip_yz=False
    world_noflip = transform_points_to_world(test_points, matrix, flip_yz=False)

    print(f"\nFrame {frame_idx} (camera frame {camera_frame}):")
    print(f"  Camera position: ({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f})")
    print(f"  Point 0 (0.5m forward):")
    print(f"    flip_yz=True:  ({world_flip[0,0]:.3f}, {world_flip[0,1]:.3f}, {world_flip[0,2]:.3f})")
    print(f"    flip_yz=False: ({world_noflip[0,0]:.3f}, {world_noflip[0,1]:.3f}, {world_noflip[0,2]:.3f})")

# スキャンメッシュの重心を取得して比較
print("\n" + "="*60)
print("MESH CENTROID COMPARISON")
print("="*60)

mesh_path = loader.get_mesh_path()
if mesh_path:
    print(f"Loading mesh: {mesh_path}")

    # OBJファイルから頂点を読み込み
    vertices = []
    with open(mesh_path) as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    vertices = np.array(vertices)
    centroid = vertices.mean(axis=0)
    print(f"Mesh vertices: {len(vertices)}")
    print(f"Mesh centroid: ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")
    print(f"Mesh bounds: X=[{vertices[:,0].min():.3f}, {vertices[:,0].max():.3f}]")
    print(f"            Y=[{vertices[:,1].min():.3f}, {vertices[:,1].max():.3f}]")
    print(f"            Z=[{vertices[:,2].min():.3f}, {vertices[:,2].max():.3f}]")

    # カメラ位置との関係
    cam_pos_first = loader.get_camera_position(0)
    cam_pos_last = loader.get_camera_position(loader.num_depth_frames - 1)
    print(f"\nCamera trajectory:")
    print(f"  First frame: ({cam_pos_first[0]:.3f}, {cam_pos_first[1]:.3f}, {cam_pos_first[2]:.3f})")
    print(f"  Last frame:  ({cam_pos_last[0]:.3f}, {cam_pos_last[1]:.3f}, {cam_pos_last[2]:.3f})")

    # カメラからメッシュ重心への方向
    direction = centroid - cam_pos_first
    distance = np.linalg.norm(direction)
    print(f"  Distance to mesh centroid: {distance:.3f}m")
else:
    print("No mesh file found")

# 単一フレームの点群を生成して比較
print("\n" + "="*60)
print("SINGLE FRAME POINTCLOUD TEST")
print("="*60)

frame_idx = 0
intrinsics = loader.get_intrinsics(frame_idx)
depth = loader.load_depth(frame_idx)

print(f"Frame {frame_idx}:")
print(f"  Depth shape: {depth.shape}")
print(f"  Depth range: {depth[depth > 0].min():.3f} - {depth.max():.3f} m")

# 中心ピクセルの深度
cy, cx = depth.shape[0] // 2, depth.shape[1] // 2
center_depth = depth[cy, cx]
print(f"  Center pixel depth: {center_depth:.3f} m")

# その点のカメラ座標
x_cam = (cx - intrinsics.cx) * center_depth / intrinsics.fx
y_cam = (cy - intrinsics.cy) * center_depth / intrinsics.fy
z_cam = center_depth
print(f"  Center point in camera coords: ({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f})")

# ワールド座標に変換
center_point = np.array([[x_cam, y_cam, z_cam]])
matrix = loader.get_camera_transform(frame_idx)

world_flip = transform_points_to_world(center_point, matrix, flip_yz=True)
world_noflip = transform_points_to_world(center_point, matrix, flip_yz=False)

print(f"  Center point in world coords:")
print(f"    flip_yz=True:  ({world_flip[0,0]:.3f}, {world_flip[0,1]:.3f}, {world_flip[0,2]:.3f})")
print(f"    flip_yz=False: ({world_noflip[0,0]:.3f}, {world_noflip[0,1]:.3f}, {world_noflip[0,2]:.3f})")

if mesh_path:
    dist_flip = np.linalg.norm(world_flip[0] - centroid)
    dist_noflip = np.linalg.norm(world_noflip[0] - centroid)
    print(f"  Distance to mesh centroid:")
    print(f"    flip_yz=True:  {dist_flip:.3f} m")
    print(f"    flip_yz=False: {dist_noflip:.3f} m")
    print(f"\n  --> {'flip_yz=True' if dist_flip < dist_noflip else 'flip_yz=False'} is closer to mesh!")
