#!/usr/bin/env python3
"""
Gaussian Splat変換ユーティリティ

SAM 3D Objects出力のGaussian Splat PLYファイルを処理する。

機能:
1. Gaussian Splatから位置(XYZ)のみを抽出
2. 融合後の位置でGaussian Splatを更新
3. 通常の点群PLYに変換

使い方:
    # 位置を抽出して通常の点群PLYに変換
    python -m server.fusion.gaussian_splat extract input.ply -o output.ply

    # 融合後の位置でGaussian Splatを更新
    python -m server.fusion.gaussian_splat update original.ply fused.ply -o updated.ply
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import struct


class GaussianSplatConverter:
    """Gaussian Splat PLY変換クラス"""

    # Gaussian Splatのプロパティ定義（SAM 3D Objects出力形式）
    PROPERTIES = [
        ('x', 'f'),
        ('y', 'f'),
        ('z', 'f'),
        ('nx', 'f'),
        ('ny', 'f'),
        ('nz', 'f'),
        ('f_dc_0', 'f'),
        ('f_dc_1', 'f'),
        ('f_dc_2', 'f'),
        ('opacity', 'f'),
        ('scale_0', 'f'),
        ('scale_1', 'f'),
        ('scale_2', 'f'),
        ('rot_0', 'f'),
        ('rot_1', 'f'),
        ('rot_2', 'f'),
        ('rot_3', 'f'),
    ]

    def __init__(self):
        self.header = None
        self.data = None
        self.num_points = 0
        self.num_properties = 0

    def load(self, path: str) -> np.ndarray:
        """
        Gaussian Splat PLYを読み込み

        Args:
            path: PLYファイルパス

        Returns:
            データ配列 (N, num_properties)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, 'rb') as f:
            # ヘッダーを解析
            header_lines = []
            self.num_properties = 0

            while True:
                line = f.readline().decode('ascii', errors='ignore')
                header_lines.append(line)

                if line.startswith('element vertex'):
                    self.num_points = int(line.split()[-1])
                elif line.startswith('property'):
                    self.num_properties += 1
                elif line.strip() == 'end_header':
                    break

            self.header = ''.join(header_lines)

            # バイナリデータを読み込み
            # 全プロパティがfloat32と仮定
            data = np.frombuffer(f.read(), dtype=np.float32)

            if len(data) != self.num_points * self.num_properties:
                # プロパティ数を再計算
                self.num_properties = len(data) // self.num_points

            self.data = data.reshape(self.num_points, self.num_properties)

        print(f"Loaded Gaussian Splat: {path}")
        print(f"  Points: {self.num_points}")
        print(f"  Properties: {self.num_properties}")

        return self.data

    def get_positions(self) -> np.ndarray:
        """位置(XYZ)を取得"""
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        return self.data[:, :3].copy()

    def set_positions(self, positions: np.ndarray):
        """位置(XYZ)を設定"""
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        if len(positions) != self.num_points:
            raise ValueError(f"Position count mismatch: {len(positions)} vs {self.num_points}")

        self.data[:, :3] = positions

    def save(self, path: str):
        """
        Gaussian Splat PLYを保存

        Args:
            path: 出力ファイルパス
        """
        if self.data is None or self.header is None:
            raise ValueError("No data to save. Call load() first.")

        path = Path(path)

        with open(path, 'wb') as f:
            f.write(self.header.encode('ascii'))
            f.write(self.data.astype(np.float32).tobytes())

        print(f"Saved Gaussian Splat: {path}")

    def extract_to_pointcloud(self, output_path: str, include_color: bool = True):
        """
        通常の点群PLYとして保存（位置とオプションで色）

        Args:
            output_path: 出力ファイルパス
            include_color: 色を含めるか
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        positions = self.get_positions()

        with open(output_path, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {self.num_points}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')

            if include_color and self.num_properties >= 9:
                f.write('property uchar red\n')
                f.write('property uchar green\n')
                f.write('property uchar blue\n')

            f.write('end_header\n')

            for i in range(self.num_points):
                x, y, z = positions[i]

                if include_color and self.num_properties >= 9:
                    # f_dc_0/1/2 をRGBに変換（球面調和の0次項）
                    # SH係数からRGBへの変換: color = 0.5 + SH_C0 * f_dc
                    SH_C0 = 0.28209479177387814
                    r = int(np.clip((0.5 + SH_C0 * self.data[i, 6]) * 255, 0, 255))
                    g = int(np.clip((0.5 + SH_C0 * self.data[i, 7]) * 255, 0, 255))
                    b = int(np.clip((0.5 + SH_C0 * self.data[i, 8]) * 255, 0, 255))
                    f.write(f'{x} {y} {z} {r} {g} {b}\n')
                else:
                    f.write(f'{x} {y} {z}\n')

        print(f"Extracted point cloud: {output_path}")
        print(f"  Points: {self.num_points}")

    def update_from_pointcloud(self, pointcloud_path: str) -> np.ndarray:
        """
        点群ファイルから位置を更新

        Args:
            pointcloud_path: 融合後の点群PLYファイルパス

        Returns:
            更新された位置配列
        """
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(pointcloud_path)
        new_positions = np.asarray(pcd.points)

        if len(new_positions) != self.num_points:
            print(f"Warning: Point count mismatch ({len(new_positions)} vs {self.num_points})")
            print("Using nearest neighbor mapping...")

            # 最近傍マッピング
            from scipy.spatial import cKDTree
            tree = cKDTree(new_positions)
            old_positions = self.get_positions()
            _, indices = tree.query(old_positions)
            new_positions = new_positions[indices]

        self.set_positions(new_positions)
        return new_positions

    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        if self.data is None:
            return {}

        positions = self.get_positions()
        return {
            'num_points': self.num_points,
            'num_properties': self.num_properties,
            'bounds': {
                'min': positions.min(axis=0).tolist(),
                'max': positions.max(axis=0).tolist(),
            },
            'center': positions.mean(axis=0).tolist(),
        }


def extract_positions(input_path: str, output_path: str, include_color: bool = True):
    """
    Gaussian Splatから位置を抽出して点群PLYとして保存

    Args:
        input_path: 入力Gaussian Splat PLY
        output_path: 出力点群PLY
        include_color: 色を含めるか
    """
    converter = GaussianSplatConverter()
    converter.load(input_path)
    converter.extract_to_pointcloud(output_path, include_color)

    stats = converter.get_stats()
    print(f"\nStatistics:")
    print(f"  Center: {stats['center']}")
    print(f"  Bounds: {stats['bounds']}")


def update_positions(original_path: str, fused_path: str, output_path: str):
    """
    融合後の位置でGaussian Splatを更新

    Args:
        original_path: 元のGaussian Splat PLY
        fused_path: 融合後の点群PLY
        output_path: 出力Gaussian Splat PLY
    """
    converter = GaussianSplatConverter()
    converter.load(original_path)

    print(f"\nUpdating positions from: {fused_path}")
    converter.update_from_pointcloud(fused_path)

    converter.save(output_path)

    stats = converter.get_stats()
    print(f"\nUpdated statistics:")
    print(f"  Center: {stats['center']}")
    print(f"  Bounds: {stats['bounds']}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Gaussian Splat PLY Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Gaussian Splatから点群を抽出
    python -m server.fusion.gaussian_splat extract sam3d_output.ply -o points.ply

    # 融合後の位置でGaussian Splatを更新
    python -m server.fusion.gaussian_splat update sam3d_output.ply fused.ply -o updated.ply

    # 統計情報を表示
    python -m server.fusion.gaussian_splat info sam3d_output.ply
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # extract コマンド
    extract_parser = subparsers.add_parser('extract', help='Extract positions to point cloud PLY')
    extract_parser.add_argument('input', help='Input Gaussian Splat PLY')
    extract_parser.add_argument('-o', '--output', required=True, help='Output point cloud PLY')
    extract_parser.add_argument('--no-color', action='store_true', help='Do not include color')

    # update コマンド
    update_parser = subparsers.add_parser('update', help='Update Gaussian Splat positions')
    update_parser.add_argument('original', help='Original Gaussian Splat PLY')
    update_parser.add_argument('fused', help='Fused point cloud PLY')
    update_parser.add_argument('-o', '--output', required=True, help='Output Gaussian Splat PLY')

    # info コマンド
    info_parser = subparsers.add_parser('info', help='Show Gaussian Splat info')
    info_parser.add_argument('input', help='Input Gaussian Splat PLY')

    args = parser.parse_args()

    if args.command == 'extract':
        extract_positions(args.input, args.output, not args.no_color)

    elif args.command == 'update':
        update_positions(args.original, args.fused, args.output)

    elif args.command == 'info':
        converter = GaussianSplatConverter()
        converter.load(args.input)
        stats = converter.get_stats()
        print(f"\nStatistics:")
        print(f"  Points: {stats['num_points']}")
        print(f"  Properties: {stats['num_properties']}")
        print(f"  Center: {stats['center']}")
        print(f"  Bounds min: {stats['bounds']['min']}")
        print(f"  Bounds max: {stats['bounds']['max']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
