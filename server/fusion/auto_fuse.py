#!/usr/bin/env python3
"""
SAM 3D + LiDAR 自動融合プログラム

SAM 3D Objects出力（Gaussian Splat）とLiDAR点群を自動で融合する。
Open3Dを使用せず、SciPyベースで実装（DGX Sparkでのクラッシュを回避）。

処理フロー:
1. SAM 3D Gaussian Splatを読み込み（位置抽出）
2. LiDAR点群を読み込み
3. 両データを正規化（中心0、スケール1）
4. 正規化空間でKDTreeマッチング
5. 近い点をスナップ（融合）
6. LiDARスケールに変換
7. Gaussian Splatの位置を更新して保存

使い方:
    python -m server.fusion.auto_fuse \\
        --sam3d sam3d_output.ply \\
        --lidar lidar_points.ply \\
        -o fused_output.ply

    # スナップ閾値を変更
    python -m server.fusion.auto_fuse \\
        --sam3d sam3d_output.ply \\
        --lidar lidar_points.ply \\
        --threshold 0.5 \\
        -o fused_output.ply
"""

import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import argparse
import time


class GaussianSplatIO:
    """Gaussian Splat PLY入出力"""

    @staticmethod
    def load(path: str) -> Tuple[np.ndarray, str, int]:
        """
        Gaussian Splat PLYを読み込み

        Args:
            path: PLYファイルパス

        Returns:
            (data, header, num_properties)
            - data: 全データ (N, num_properties)
            - header: ヘッダー文字列
            - num_properties: プロパティ数
        """
        with open(path, 'rb') as f:
            header_lines = []
            num_vertices = 0
            num_properties = 0

            while True:
                line = f.readline().decode('ascii', errors='ignore')
                header_lines.append(line)

                if line.strip().startswith('element vertex'):
                    num_vertices = int(line.strip().split()[-1])
                elif line.strip().startswith('property'):
                    num_properties += 1
                elif line.strip() == 'end_header':
                    break

            header = ''.join(header_lines)

            # バイナリデータを読み込み
            data = np.frombuffer(f.read(), dtype=np.float32).copy()
            actual_props = len(data) // num_vertices
            data = data.reshape(num_vertices, actual_props)

        return data, header, actual_props

    @staticmethod
    def save(path: str, data: np.ndarray, header: str):
        """
        Gaussian Splat PLYを保存

        Args:
            path: 出力ファイルパス
            data: 全データ (N, num_properties)
            header: ヘッダー文字列
        """
        with open(path, 'wb') as f:
            f.write(header.encode('ascii'))
            f.write(data.astype(np.float32).tobytes())

    @staticmethod
    def get_positions(data: np.ndarray) -> np.ndarray:
        """位置(XYZ)を取得"""
        return data[:, :3].copy()

    @staticmethod
    def set_positions(data: np.ndarray, positions: np.ndarray):
        """位置(XYZ)を設定"""
        data[:, :3] = positions


class PointCloudIO:
    """点群PLY入出力"""

    @staticmethod
    def load(path: str) -> np.ndarray:
        """
        点群PLYを読み込み（ASCII/バイナリ両対応）

        Args:
            path: PLYファイルパス

        Returns:
            点群座標 (N, 3)
        """
        with open(path, 'rb') as f:
            num_vertices = 0
            is_binary = False
            num_properties = 0

            while True:
                line = f.readline().decode('ascii', errors='ignore').strip()

                if line.startswith('element vertex'):
                    num_vertices = int(line.split()[-1])
                elif line.startswith('format binary'):
                    is_binary = True
                elif line.startswith('property'):
                    num_properties += 1
                elif line == 'end_header':
                    break

            if is_binary:
                # バイナリ形式
                # プロパティがfloat(4bytes)とuchar(1byte)混在の可能性
                # まずfloat32で試す
                try:
                    data = np.frombuffer(f.read(num_vertices * 3 * 4), dtype=np.float32)
                    return data.reshape(-1, 3)
                except ValueError:
                    # ファイルを再読み込みして別の方法を試す
                    pass

            # ASCII形式
            f.seek(0)
            in_data = False
            points = []

            for line in f:
                try:
                    line_str = line.decode('ascii', errors='ignore').strip()
                except:
                    continue

                if in_data:
                    parts = line_str.split()
                    if len(parts) >= 3:
                        try:
                            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        except ValueError:
                            continue
                elif line_str == 'end_header':
                    in_data = True

            return np.array(points)

    @staticmethod
    def save(path: str, points: np.ndarray, colors: Optional[np.ndarray] = None):
        """
        点群PLYを保存（ASCII形式）

        Args:
            path: 出力ファイルパス
            points: 点群座標 (N, 3)
            colors: 色 (N, 3) オプション、0-255のuchar
        """
        with open(path, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(points)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')

            if colors is not None:
                f.write('property uchar red\n')
                f.write('property uchar green\n')
                f.write('property uchar blue\n')

            f.write('end_header\n')

            for i, p in enumerate(points):
                if colors is not None:
                    c = colors[i]
                    f.write(f'{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n')
                else:
                    f.write(f'{p[0]} {p[1]} {p[2]}\n')


class AutoFusion:
    """自動融合クラス"""

    def __init__(
        self,
        snap_threshold: float = 1.0,
        output_scale: str = 'lidar'
    ):
        """
        Args:
            snap_threshold: 正規化空間でのスナップ閾値（デフォルト1.0）
            output_scale: 出力スケール ('lidar' or 'sam3d')
        """
        self.snap_threshold = snap_threshold
        self.output_scale = output_scale

    def normalize(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        点群を正規化（中心0、スケール1）

        Args:
            points: 点群座標 (N, 3)

        Returns:
            (normalized_points, center, scale)
        """
        center = points.mean(axis=0)
        scale = points.std()

        if scale < 1e-10:
            scale = 1.0

        normalized = (points - center) / scale
        return normalized, center, scale

    def denormalize(
        self,
        points: np.ndarray,
        center: np.ndarray,
        scale: float
    ) -> np.ndarray:
        """
        正規化を解除

        Args:
            points: 正規化された点群 (N, 3)
            center: 中心座標
            scale: スケール

        Returns:
            元のスケールの点群
        """
        return points * scale + center

    def fuse(
        self,
        sam3d_points: np.ndarray,
        lidar_points: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        SAM 3DとLiDARを融合

        Args:
            sam3d_points: SAM 3D点群 (N, 3)
            lidar_points: LiDAR点群 (M, 3)

        Returns:
            (fused_points, stats)
            - fused_points: 融合後の点群（LiDARスケール）
            - stats: 統計情報
        """
        # 正規化
        sam3d_norm, sam3d_center, sam3d_scale = self.normalize(sam3d_points)
        lidar_norm, lidar_center, lidar_scale = self.normalize(lidar_points)

        # KDTreeでマッチング
        tree = cKDTree(lidar_norm)
        distances, indices = tree.query(sam3d_norm, k=1)

        # スナップ
        mask = distances < self.snap_threshold
        snapped_norm = sam3d_norm.copy()
        snapped_norm[mask] = lidar_norm[indices[mask]]

        # 出力スケールに変換
        if self.output_scale == 'lidar':
            fused_points = self.denormalize(snapped_norm, lidar_center, lidar_scale)
        else:
            fused_points = self.denormalize(snapped_norm, sam3d_center, sam3d_scale)

        # 統計情報
        stats = {
            'sam3d_points': len(sam3d_points),
            'lidar_points': len(lidar_points),
            'snapped_points': int(mask.sum()),
            'snap_ratio': float(mask.sum() / len(sam3d_points)),
            'mean_distance': float(distances.mean()),
            'median_distance': float(np.median(distances)),
            'sam3d_center': sam3d_center.tolist(),
            'sam3d_scale': float(sam3d_scale),
            'lidar_center': lidar_center.tolist(),
            'lidar_scale': float(lidar_scale),
        }

        return fused_points, stats

    def process(
        self,
        sam3d_path: str,
        lidar_path: str,
        output_path: str,
        save_pointcloud: bool = True
    ) -> Dict[str, Any]:
        """
        融合処理を実行

        Args:
            sam3d_path: SAM 3D Gaussian Splat PLYファイルパス
            lidar_path: LiDAR点群PLYファイルパス
            output_path: 出力Gaussian Splat PLYファイルパス
            save_pointcloud: 点群PLYも保存するか

        Returns:
            統計情報
        """
        start_time = time.time()

        print("=" * 60)
        print("SAM 3D + LiDAR Auto Fusion")
        print("=" * 60)

        # SAM 3D Gaussian Splatを読み込み
        print(f"\n[1/5] Loading SAM 3D Gaussian Splat: {sam3d_path}")
        gs_data, gs_header, gs_props = GaussianSplatIO.load(sam3d_path)
        sam3d_points = GaussianSplatIO.get_positions(gs_data)
        print(f"      Points: {len(sam3d_points)}, Properties: {gs_props}")
        print(f"      X: [{sam3d_points[:,0].min():.4f}, {sam3d_points[:,0].max():.4f}]")
        print(f"      Y: [{sam3d_points[:,1].min():.4f}, {sam3d_points[:,1].max():.4f}]")
        print(f"      Z: [{sam3d_points[:,2].min():.4f}, {sam3d_points[:,2].max():.4f}]")

        # LiDAR点群を読み込み
        print(f"\n[2/5] Loading LiDAR point cloud: {lidar_path}")
        lidar_points = PointCloudIO.load(lidar_path)
        print(f"      Points: {len(lidar_points)}")
        print(f"      X: [{lidar_points[:,0].min():.4f}, {lidar_points[:,0].max():.4f}]")
        print(f"      Y: [{lidar_points[:,1].min():.4f}, {lidar_points[:,1].max():.4f}]")
        print(f"      Z: [{lidar_points[:,2].min():.4f}, {lidar_points[:,2].max():.4f}]")

        # 融合
        print(f"\n[3/5] Fusing point clouds (threshold: {self.snap_threshold})...")
        fused_points, stats = self.fuse(sam3d_points, lidar_points)
        print(f"      Snapped: {stats['snapped_points']} / {stats['sam3d_points']} ({stats['snap_ratio']*100:.1f}%)")
        print(f"      Mean distance: {stats['mean_distance']:.4f}")
        print(f"      Median distance: {stats['median_distance']:.4f}")

        # Gaussian Splatを更新して保存
        print(f"\n[4/5] Updating Gaussian Splat positions...")
        GaussianSplatIO.set_positions(gs_data, fused_points)

        output_path = Path(output_path)
        GaussianSplatIO.save(str(output_path), gs_data, gs_header)
        print(f"      Saved: {output_path}")
        print(f"      X: [{fused_points[:,0].min():.4f}, {fused_points[:,0].max():.4f}]")
        print(f"      Y: [{fused_points[:,1].min():.4f}, {fused_points[:,1].max():.4f}]")
        print(f"      Z: [{fused_points[:,2].min():.4f}, {fused_points[:,2].max():.4f}]")

        # 点群PLYも保存
        if save_pointcloud:
            print(f"\n[5/5] Saving point cloud PLY...")
            pcd_path = output_path.with_suffix('.points.ply')
            PointCloudIO.save(str(pcd_path), fused_points)
            print(f"      Saved: {pcd_path}")

        elapsed_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("Fusion Complete!")
        print("=" * 60)
        print(f"  Time: {elapsed_time:.2f} seconds")
        print(f"  Output: {output_path}")

        stats['elapsed_time'] = elapsed_time
        stats['output_path'] = str(output_path)

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="SAM 3D + LiDAR Auto Fusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 基本的な使い方
    python -m server.fusion.auto_fuse \\
        --sam3d sam3d_output.ply \\
        --lidar lidar_points.ply \\
        -o fused_output.ply

    # スナップ閾値を変更（正規化空間での距離）
    python -m server.fusion.auto_fuse \\
        --sam3d sam3d_output.ply \\
        --lidar lidar_points.ply \\
        --threshold 0.5 \\
        -o fused_output.ply

    # SAM 3Dスケールで出力
    python -m server.fusion.auto_fuse \\
        --sam3d sam3d_output.ply \\
        --lidar lidar_points.ply \\
        --scale sam3d \\
        -o fused_output.ply
"""
    )

    parser.add_argument('--sam3d', required=True,
                        help='SAM 3D Gaussian Splat PLY file')
    parser.add_argument('--lidar', required=True,
                        help='LiDAR point cloud PLY file')
    parser.add_argument('-o', '--output', required=True,
                        help='Output Gaussian Splat PLY file')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Snap threshold in normalized space (default: 1.0)')
    parser.add_argument('--scale', choices=['lidar', 'sam3d'], default='lidar',
                        help='Output scale (default: lidar)')
    parser.add_argument('--no-pointcloud', action='store_true',
                        help='Do not save separate point cloud PLY')

    args = parser.parse_args()

    fusion = AutoFusion(
        snap_threshold=args.threshold,
        output_scale=args.scale
    )

    fusion.process(
        sam3d_path=args.sam3d,
        lidar_path=args.lidar,
        output_path=args.output,
        save_pointcloud=not args.no_pointcloud
    )


if __name__ == "__main__":
    main()
