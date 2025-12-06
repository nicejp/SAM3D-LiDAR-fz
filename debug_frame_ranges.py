#!/usr/bin/env python3
"""フレーム範囲ごとの点群位置を確認"""

import sys
import numpy as np
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: python debug_frame_ranges.py <session_dir> <mask_dir>")
    print("Example: python debug_frame_ranges.py experiments/downloads/003_extracted/003 experiments/downloads/003_extracted/003/output/masks")
    sys.exit(1)

session_dir = sys.argv[1]
mask_dir = sys.argv[2]

from server.multiview.omniscient_loader import OmniscientLoader
from server.multiview.pointcloud_fusion import MultiViewPointCloudFusion
from PIL import Image

loader = OmniscientLoader(session_dir)
fusion = MultiViewPointCloudFusion(loader)

# マスクファイルを取得
mask_path = Path(mask_dir)
mask_files = sorted(mask_path.glob("mask_*.png"))

print(f"Found {len(mask_files)} mask files")

# フレーム番号を抽出
frame_numbers = []
for mf in mask_files:
    parts = mf.stem.split("_")
    if len(parts) >= 2:
        try:
            frame_numbers.append(int(parts[1]))
        except ValueError:
            pass

frame_numbers = sorted(set(frame_numbers))
print(f"Unique frames with masks: {len(frame_numbers)}")
print(f"Frame range: {frame_numbers[0]} to {frame_numbers[-1]}")

# フレーム範囲を分割して点群の重心を計算
ranges = [
    (0, 100),
    (100, 200),
    (200, 300),
    (300, 426),
]

print("\n" + "="*60)
print("POINTCLOUD CENTROIDS BY FRAME RANGE")
print("="*60)

intrinsics = loader.get_intrinsics()

for start, end in ranges:
    frames_in_range = [f for f in frame_numbers if start <= f < end]
    if not frames_in_range:
        print(f"\nFrames {start}-{end}: No masks")
        continue

    all_points = []
    for frame_idx in frames_in_range[:5]:  # 各範囲から最大5フレーム
        # マスクを探す
        mask_file = None
        for mf in mask_files:
            if f"mask_{frame_idx:06d}" in mf.name:
                mask_file = mf
                break

        if mask_file is None:
            continue

        mask = np.array(Image.open(mask_file)) > 127

        # 点群を抽出
        points, _ = fusion.extract_masked_pointcloud(
            frame_idx, mask,
            use_world_coords=True,
            max_depth=5.0,
            min_depth=0.1
        )

        if len(points) > 0:
            all_points.append(points)

    if all_points:
        merged = np.vstack(all_points)
        centroid = merged.mean(axis=0)
        cam_frame = loader.video_frame_to_camera_frame(frames_in_range[0])
        cam_pos = loader.get_camera_position(frames_in_range[0])

        print(f"\nFrames {start}-{end}:")
        print(f"  Masks used: {len(all_points)} frames")
        print(f"  Total points: {len(merged)}")
        print(f"  Centroid: ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")
        print(f"  Camera frame: {cam_frame}, Camera pos: ({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f})")
        print(f"  Point bounds: X=[{merged[:,0].min():.3f}, {merged[:,0].max():.3f}]")
        print(f"               Y=[{merged[:,1].min():.3f}, {merged[:,1].max():.3f}]")
        print(f"               Z=[{merged[:,2].min():.3f}, {merged[:,2].max():.3f}]")
    else:
        print(f"\nFrames {start}-{end}: No points extracted")

# 個別フレームの詳細分析
print("\n" + "="*60)
print("INDIVIDUAL FRAME ANALYSIS")
print("="*60)

test_frames = [frame_numbers[0], frame_numbers[len(frame_numbers)//4],
               frame_numbers[len(frame_numbers)//2], frame_numbers[-1]]

for frame_idx in test_frames:
    mask_file = None
    for mf in mask_files:
        if f"mask_{frame_idx:06d}" in mf.name:
            mask_file = mf
            break

    if mask_file is None:
        continue

    mask = np.array(Image.open(mask_file)) > 127

    # 深度情報
    depth = loader.load_depth(frame_idx)
    masked_depth = depth.copy()
    if mask.shape != depth.shape:
        mask_resized = np.array(Image.fromarray(mask.astype(np.uint8)*255).resize(
            (depth.shape[1], depth.shape[0]), Image.NEAREST)) > 127
    else:
        mask_resized = mask
    masked_depth[~mask_resized] = 0

    valid_depths = masked_depth[masked_depth > 0]

    cam_frame = loader.video_frame_to_camera_frame(frame_idx)
    cam_pos = loader.get_camera_position(frame_idx)
    cam_matrix = loader.get_camera_transform(frame_idx)

    print(f"\nFrame {frame_idx} (camera frame {cam_frame}):")
    print(f"  Mask pixels: {mask_resized.sum()}")
    print(f"  Valid depth pixels: {len(valid_depths)}")
    if len(valid_depths) > 0:
        print(f"  Depth range: {valid_depths.min():.3f} - {valid_depths.max():.3f} m")
        print(f"  Depth mean: {valid_depths.mean():.3f} m")
    print(f"  Camera position: ({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f})")

    # カメラの向き（Z軸）
    cam_z = cam_matrix[:3, 2]  # カメラのローカルZ軸（後方）
    print(f"  Camera -Z direction (forward): ({-cam_z[0]:.3f}, {-cam_z[1]:.3f}, {-cam_z[2]:.3f})")
