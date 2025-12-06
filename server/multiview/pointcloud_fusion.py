#!/usr/bin/env python3
"""
Multi-view Point Cloud Fusion

複数フレームからセグメントされた点群を統合する。
SAM 3のマスクとカメラポーズを使用して、高密度な点群を生成。

使い方:
    python -m server.multiview.pointcloud_fusion \\
        experiments/omniscient_sample/003 \\
        --masks output/masks \\
        -o fused_pointcloud.ply
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image


@dataclass
class FusedPointCloud:
    """統合された点群"""
    points: np.ndarray      # (N, 3) 座標
    colors: Optional[np.ndarray] = None  # (N, 3) RGB色
    normals: Optional[np.ndarray] = None  # (N, 3) 法線
    frame_indices: Optional[np.ndarray] = None  # 元フレームのインデックス


class MultiViewPointCloudFusion:
    """多視点点群統合"""

    def __init__(self, omniscient_loader):
        """
        Args:
            omniscient_loader: OmniscientLoaderインスタンス
        """
        self.loader = omniscient_loader

    def load_mask(self, mask_path: str) -> np.ndarray:
        """
        マスクを読み込み

        Args:
            mask_path: マスクファイルパス（.png or .npy）

        Returns:
            mask: (H, W) bool配列
        """
        mask_path = Path(mask_path)

        if mask_path.suffix == ".npy":
            return np.load(mask_path).astype(bool)
        else:
            img = Image.open(mask_path)
            return np.array(img) > 127

    def extract_masked_pointcloud(
        self,
        frame_index: int,
        mask: np.ndarray,
        use_world_coords: bool = True,
        max_depth: float = 10.0,
        min_depth: float = 0.1,
        mask_dilation: int = 0
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        マスクを適用して点群を抽出

        Args:
            frame_index: フレームインデックス
            mask: セグメンテーションマスク
            use_world_coords: ワールド座標系に変換するか
            max_depth: 最大深度
            min_depth: 最小深度
            mask_dilation: マスク膨張ピクセル数

        Returns:
            points: (N, 3) 点群
            colors: (N, 3) 色（RGBが利用可能な場合）
        """
        # フレームデータを取得
        frame = self.loader.get_frame(frame_index, load_rgb=True)
        intrinsics = self.loader.get_intrinsics(frame_index)

        # マスクを深度画像サイズにリサイズ
        depth_shape = frame.depth_map.shape
        if mask.shape != depth_shape:
            mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_pil = mask_pil.resize(
                (depth_shape[1], depth_shape[0]),
                Image.NEAREST
            )
            mask = np.array(mask_pil) > 127

        # マスク膨張（オプション）
        if mask_dilation > 0:
            from scipy.ndimage import binary_dilation
            mask = binary_dilation(mask, iterations=mask_dilation)

        # マスクを適用した深度マップを作成
        masked_depth = frame.depth_map.copy()
        masked_depth[~mask] = 0

        # 点群を生成
        if use_world_coords and self.loader.has_camera_poses:
            points = self.loader.depth_to_pointcloud_world(
                masked_depth, intrinsics, frame_index,
                max_depth=max_depth, min_depth=min_depth
            )
        else:
            points = self.loader.depth_to_pointcloud(
                masked_depth, intrinsics,
                max_depth=max_depth, min_depth=min_depth
            )

        # 色情報を取得
        colors = None
        if frame.rgb_frame is not None:
            # RGBを深度サイズにリサイズ
            rgb_pil = Image.fromarray(frame.rgb_frame)
            rgb_pil = rgb_pil.resize((depth_shape[1], depth_shape[0]), Image.BILINEAR)
            rgb_resized = np.array(rgb_pil)

            # 有効な点のマスク
            height, width = depth_shape
            u = np.arange(width)
            v = np.arange(height)
            u, v = np.meshgrid(u, v)

            valid_mask = (
                mask &
                (masked_depth > min_depth) &
                (masked_depth < max_depth) &
                np.isfinite(masked_depth)
            )

            colors = rgb_resized[valid_mask]

        return points, colors

    def fuse_frames(
        self,
        frame_indices: List[int],
        masks: Dict[int, np.ndarray],
        use_world_coords: bool = True,
        max_depth: float = 10.0,
        min_depth: float = 0.1,
        voxel_downsample: Optional[float] = None
    ) -> FusedPointCloud:
        """
        複数フレームの点群を統合

        Args:
            frame_indices: 統合するフレームのインデックス
            masks: フレームインデックス→マスクの辞書
            use_world_coords: ワールド座標系に変換するか
            max_depth: 最大深度
            min_depth: 最小深度
            voxel_downsample: ボクセルダウンサンプリングのサイズ（Noneで無効）

        Returns:
            FusedPointCloud: 統合された点群
        """
        all_points = []
        all_colors = []
        all_frame_indices = []

        for frame_idx in frame_indices:
            if frame_idx not in masks:
                print(f"Warning: No mask for frame {frame_idx}, skipping")
                continue

            mask = masks[frame_idx]
            points, colors = self.extract_masked_pointcloud(
                frame_idx, mask,
                use_world_coords=use_world_coords,
                max_depth=max_depth,
                min_depth=min_depth
            )

            if len(points) > 0:
                all_points.append(points)
                if colors is not None:
                    all_colors.append(colors)
                all_frame_indices.append(
                    np.full(len(points), frame_idx, dtype=np.int32)
                )

            # カメラフレームマッピング情報を表示
            camera_info = ""
            if hasattr(self.loader, 'video_frame_to_camera_frame'):
                camera_frame = self.loader.video_frame_to_camera_frame(frame_idx)
                if camera_frame != frame_idx:
                    camera_info = f" -> camera frame {camera_frame}"
            print(f"Frame {frame_idx}{camera_info}: {len(points)} points")

        if not all_points:
            return FusedPointCloud(points=np.zeros((0, 3)))

        # 統合
        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors) if all_colors else None
        merged_frame_indices = np.concatenate(all_frame_indices)

        print(f"Total before downsampling: {len(merged_points)} points")

        # ボクセルダウンサンプリング
        if voxel_downsample is not None and voxel_downsample > 0:
            merged_points, keep_indices = self._voxel_downsample(
                merged_points, voxel_downsample
            )
            if merged_colors is not None:
                merged_colors = merged_colors[keep_indices]
            merged_frame_indices = merged_frame_indices[keep_indices]
            print(f"After downsampling (voxel={voxel_downsample}): {len(merged_points)} points")

        return FusedPointCloud(
            points=merged_points,
            colors=merged_colors,
            frame_indices=merged_frame_indices
        )

    def _voxel_downsample(
        self,
        points: np.ndarray,
        voxel_size: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ボクセルダウンサンプリング

        Args:
            points: (N, 3) 点群
            voxel_size: ボクセルサイズ

        Returns:
            downsampled_points: ダウンサンプリングされた点群
            keep_indices: 保持されたインデックス
        """
        # ボクセルインデックスを計算
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)

        # ユニークなボクセルを取得
        unique_voxels, inverse_indices = np.unique(
            voxel_indices, axis=0, return_inverse=True
        )

        # 各ボクセルから1点を選択（最初の点）
        keep_indices = []
        for i in range(len(unique_voxels)):
            first_idx = np.where(inverse_indices == i)[0][0]
            keep_indices.append(first_idx)

        keep_indices = np.array(keep_indices)
        return points[keep_indices], keep_indices

    def get_object_ids_from_masks(self, mask_dir: str) -> List[int]:
        """
        マスクディレクトリからオブジェクトIDのリストを取得

        Args:
            mask_dir: マスクファイルのディレクトリ

        Returns:
            オブジェクトIDのソート済みリスト
        """
        import re
        mask_path = Path(mask_dir)
        mask_files = list(mask_path.glob("mask_*.png")) + list(mask_path.glob("mask_*.npy"))

        object_ids = set()
        for mf in mask_files:
            # _obj{数字} パターンを検索
            match = re.search(r'_obj(\d+)', mf.name)
            if match:
                object_ids.add(int(match.group(1)))

        return sorted(object_ids)

    def fuse_from_mask_directory(
        self,
        mask_dir: str,
        frame_step: int = 1,
        use_world_coords: bool = True,
        max_depth: float = 10.0,
        min_depth: float = 0.1,
        voxel_downsample: Optional[float] = None,
        object_id: Optional[int] = None
    ) -> FusedPointCloud:
        """
        マスクディレクトリから点群を統合

        Args:
            mask_dir: マスクファイルのディレクトリ
            frame_step: フレーム間隔
            use_world_coords: ワールド座標系に変換するか
            max_depth: 最大深度
            min_depth: 最小深度
            voxel_downsample: ボクセルダウンサンプリングのサイズ
            object_id: 特定のオブジェクトIDのみを使用（None=全オブジェクト）

        Returns:
            FusedPointCloud: 統合された点群
        """
        mask_path = Path(mask_dir)

        # マスクファイルを検索
        mask_files = sorted(mask_path.glob("mask_*.png")) + sorted(mask_path.glob("mask_*.npy"))

        if not mask_files:
            raise FileNotFoundError(f"No mask files found in {mask_dir}")

        # オブジェクトIDでフィルタリング
        if object_id is not None:
            filtered_files = []
            for mf in mask_files:
                if f"_obj{object_id}." in mf.name or f"_obj{object_id}_" in mf.name:
                    filtered_files.append(mf)
            mask_files = filtered_files
            print(f"Filtered to object_id={object_id}: {len(mask_files)} masks")

        # フレームインデックスとマスクを取得
        masks = {}
        frame_indices = []

        for mask_file in mask_files:
            # ファイル名からフレームインデックスを抽出
            # 形式: mask_000000_obj0.png or mask_000000.png
            name = mask_file.stem
            parts = name.split("_")
            if len(parts) >= 2:
                try:
                    frame_idx = int(parts[1])
                    if frame_idx % frame_step == 0:
                        masks[frame_idx] = self.load_mask(str(mask_file))
                        frame_indices.append(frame_idx)
                except ValueError:
                    continue

        frame_indices = sorted(set(frame_indices))
        print(f"Found masks for {len(frame_indices)} frames")

        return self.fuse_frames(
            frame_indices, masks,
            use_world_coords=use_world_coords,
            max_depth=max_depth,
            min_depth=min_depth,
            voxel_downsample=voxel_downsample
        )

    def fuse_all_objects(
        self,
        mask_dir: str,
        frame_step: int = 1,
        use_world_coords: bool = True,
        max_depth: float = 10.0,
        min_depth: float = 0.1,
        voxel_downsample: Optional[float] = None
    ) -> Dict[int, FusedPointCloud]:
        """
        全オブジェクトの点群を個別に統合

        Args:
            mask_dir: マスクファイルのディレクトリ
            frame_step: フレーム間隔
            use_world_coords: ワールド座標系に変換するか
            max_depth: 最大深度
            min_depth: 最小深度
            voxel_downsample: ボクセルダウンサンプリングのサイズ

        Returns:
            オブジェクトID -> FusedPointCloud の辞書
        """
        object_ids = self.get_object_ids_from_masks(mask_dir)

        if not object_ids:
            # オブジェクトIDがない場合は全マスクを1つのオブジェクトとして処理
            print("No object IDs found in mask files, treating all masks as one object")
            result = self.fuse_from_mask_directory(
                mask_dir,
                frame_step=frame_step,
                use_world_coords=use_world_coords,
                max_depth=max_depth,
                min_depth=min_depth,
                voxel_downsample=voxel_downsample
            )
            return {0: result}

        print(f"Found {len(object_ids)} objects: {object_ids}")
        results = {}

        for obj_id in object_ids:
            print(f"\n--- Processing object {obj_id} ---")
            result = self.fuse_from_mask_directory(
                mask_dir,
                frame_step=frame_step,
                use_world_coords=use_world_coords,
                max_depth=max_depth,
                min_depth=min_depth,
                voxel_downsample=voxel_downsample,
                object_id=obj_id
            )
            results[obj_id] = result
            print(f"Object {obj_id}: {len(result.points)} points")

        return results


def save_ply(
    points: np.ndarray,
    filepath: str,
    colors: Optional[np.ndarray] = None
):
    """PLYファイルを保存"""
    with open(filepath, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        for i, point in enumerate(points):
            if colors is not None:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                       f"{int(colors[i][0])} {int(colors[i][1])} {int(colors[i][2])}\n")
            else:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")


def main():
    parser = argparse.ArgumentParser(
        description="多視点点群統合",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # マスクディレクトリから点群を統合
  python -m server.multiview.pointcloud_fusion \\
      experiments/omniscient_sample/003 \\
      --masks output/masks \\
      -o fused_pointcloud.ply

  # ダウンサンプリング付き
  python -m server.multiview.pointcloud_fusion \\
      experiments/omniscient_sample/003 \\
      --masks output/masks \\
      --voxel 0.01 \\
      -o fused_pointcloud.ply
        """
    )

    parser.add_argument("session_dir", help="Omniscientセッションディレクトリ")
    parser.add_argument("--masks", "-m", required=True, help="マスクディレクトリ")
    parser.add_argument("--output", "-o", required=True, help="出力PLYファイル")
    parser.add_argument("--step", type=int, default=1, help="フレーム間隔")
    parser.add_argument("--voxel", type=float, help="ボクセルダウンサンプリングサイズ")
    parser.add_argument("--max-depth", type=float, default=10.0, help="最大深度")
    parser.add_argument("--min-depth", type=float, default=0.1, help="最小深度")
    parser.add_argument("--object-id", type=int, help="特定のオブジェクトIDのみを使用（例: 0, 3）")

    args = parser.parse_args()

    # Omniscientローダーを初期化
    from server.multiview.omniscient_loader import OmniscientLoader
    loader = OmniscientLoader(args.session_dir)

    print("=== Multi-view Point Cloud Fusion ===")
    print(f"Session: {args.session_dir}")
    print(f"Masks: {args.masks}")
    print(f"Camera poses available: {loader.has_camera_poses}")

    # 点群統合
    fusion = MultiViewPointCloudFusion(loader)

    if args.object_id is not None:
        # 特定のオブジェクトIDのみ処理
        print(f"Object ID filter: {args.object_id}")
        result = fusion.fuse_from_mask_directory(
            args.masks,
            frame_step=args.step,
            use_world_coords=loader.has_camera_poses,
            max_depth=args.max_depth,
            min_depth=args.min_depth,
            voxel_downsample=args.voxel,
            object_id=args.object_id
        )
        print(f"\nFused point cloud: {len(result.points)} points")
        save_ply(result.points, args.output, result.colors)
        print(f"Saved to: {args.output}")
    else:
        # 全オブジェクトを個別に処理
        results = fusion.fuse_all_objects(
            args.masks,
            frame_step=args.step,
            use_world_coords=loader.has_camera_poses,
            max_depth=args.max_depth,
            min_depth=args.min_depth,
            voxel_downsample=args.voxel
        )

        # 各オブジェクトを個別のファイルに保存
        output_path = Path(args.output)
        base_name = output_path.stem
        suffix = output_path.suffix

        print(f"\n=== Saving {len(results)} point clouds ===")
        for obj_id, result in results.items():
            if len(result.points) > 0:
                output_file = output_path.parent / f"{base_name}_obj{obj_id}{suffix}"
                save_ply(result.points, str(output_file), result.colors)
                print(f"Object {obj_id}: {len(result.points)} points -> {output_file}")


if __name__ == "__main__":
    main()
