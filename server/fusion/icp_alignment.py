#!/usr/bin/env python3
"""
ICP位置合わせモジュール

SAM 3Dで生成した3DモデルとLiDAR点群を位置合わせする。

処理フロー:
1. 大まかな位置合わせ (Coarse Alignment)
   - 重心合わせ
   - スケール合わせ（バウンディングボックス比）
   - PCA主軸合わせ
2. 精密な位置合わせ (Rigid ICP)
   - Open3D ICPで位置・回転を微調整

使い方:
    from server.fusion.icp_alignment import ICPAligner

    aligner = ICPAligner()
    result = aligner.align(sam3d_ply_path, lidar_ply_path)

    # 変換行列を取得
    transform = result['transformation']

    # 位置合わせ済み点群を保存
    result['aligned_source'].save("aligned.ply")
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Open3D is required. Install with: pip install open3d")


class ICPAligner:
    """ICP位置合わせクラス"""

    def __init__(
        self,
        voxel_size: float = 0.02,
        icp_threshold: float = 0.05,
        max_iteration: int = 100
    ):
        """
        Args:
            voxel_size: ダウンサンプリング用ボクセルサイズ
            icp_threshold: ICP収束閾値
            max_iteration: ICP最大反復回数
        """
        self.voxel_size = voxel_size
        self.icp_threshold = icp_threshold
        self.max_iteration = max_iteration

    def load_pointcloud(self, path: str) -> o3d.geometry.PointCloud:
        """点群ファイルを読み込む"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() == '.ply':
            pcd = o3d.io.read_point_cloud(str(path))
        elif path.suffix.lower() == '.pcd':
            pcd = o3d.io.read_point_cloud(str(path))
        elif path.suffix.lower() == '.xyz':
            pcd = o3d.io.read_point_cloud(str(path))
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        if len(pcd.points) == 0:
            raise ValueError(f"Empty point cloud: {path}")

        return pcd

    def preprocess(
        self,
        pcd: o3d.geometry.PointCloud,
        voxel_size: Optional[float] = None
    ) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        """
        点群を前処理（ダウンサンプリング + 法線計算 + FPFH特徴量計算）

        Args:
            pcd: 入力点群
            voxel_size: ボクセルサイズ（Noneの場合はself.voxel_sizeを使用）

        Returns:
            (ダウンサンプリング済み点群, FPFH特徴量)
        """
        if voxel_size is None:
            voxel_size = self.voxel_size

        # ダウンサンプリング
        pcd_down = pcd.voxel_down_sample(voxel_size)

        # 法線推定
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )

        # FPFH特徴量計算
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
        )

        return pcd_down, fpfh

    def compute_center_and_scale(
        self,
        pcd: o3d.geometry.PointCloud
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        点群の重心、バウンディングボックスサイズ、主軸を計算

        Returns:
            (center, size, principal_axes)
        """
        points = np.asarray(pcd.points)

        # 重心
        center = np.mean(points, axis=0)

        # バウンディングボックス
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        size = max_bound - min_bound

        # PCA主軸
        centered = points - center
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # 固有値の大きい順にソート
        idx = np.argsort(eigenvalues)[::-1]
        principal_axes = eigenvectors[:, idx]

        return center, size, principal_axes

    def coarse_alignment(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud
    ) -> np.ndarray:
        """
        大まかな位置合わせ（重心、スケール、主軸）

        Args:
            source: ソース点群（SAM 3D生成モデル）
            target: ターゲット点群（LiDAR点群）

        Returns:
            4x4変換行列
        """
        # 重心とスケールを計算
        src_center, src_size, src_axes = self.compute_center_and_scale(source)
        tgt_center, tgt_size, tgt_axes = self.compute_center_and_scale(target)

        # スケール係数
        scale = np.mean(tgt_size / (src_size + 1e-6))
        scale = np.clip(scale, 0.1, 10.0)  # 極端なスケールを制限

        # 回転行列（主軸を合わせる）
        # ソースの主軸をターゲットの主軸に合わせる
        R = tgt_axes @ src_axes.T

        # 行列式が-1の場合（反転している場合）は修正
        if np.linalg.det(R) < 0:
            tgt_axes[:, 2] = -tgt_axes[:, 2]
            R = tgt_axes @ src_axes.T

        # 4x4変換行列を構築
        # T = T_target * S * R * T_source^-1
        T = np.eye(4)

        # 1. ソースを原点に移動
        T_src_inv = np.eye(4)
        T_src_inv[:3, 3] = -src_center

        # 2. 回転
        T_rot = np.eye(4)
        T_rot[:3, :3] = R

        # 3. スケール
        T_scale = np.eye(4)
        T_scale[:3, :3] *= scale

        # 4. ターゲット重心に移動
        T_tgt = np.eye(4)
        T_tgt[:3, 3] = tgt_center

        # 合成
        T = T_tgt @ T_scale @ T_rot @ T_src_inv

        return T

    def refine_alignment(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        initial_transform: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        精密な位置合わせ（ICP）

        Args:
            source: ソース点群
            target: ターゲット点群
            initial_transform: 初期変換行列

        Returns:
            (最終変換行列, fitness, inlier_rmse)
        """
        # Point-to-Point ICP
        result = o3d.pipelines.registration.registration_icp(
            source, target,
            self.icp_threshold,
            initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.max_iteration
            )
        )

        return result.transformation, result.fitness, result.inlier_rmse

    def align(
        self,
        source_path: str,
        target_path: str,
        use_coarse: bool = True,
        use_fpfh: bool = False
    ) -> Dict[str, Any]:
        """
        点群を位置合わせ

        Args:
            source_path: ソース点群ファイルパス（SAM 3D生成モデル）
            target_path: ターゲット点群ファイルパス（LiDAR点群）
            use_coarse: 大まかな位置合わせを使用するか
            use_fpfh: FPFH特徴量ベースの位置合わせを使用するか

        Returns:
            {
                'transformation': 4x4変換行列,
                'fitness': ICPフィットネススコア,
                'inlier_rmse': インライアRMSE,
                'source': 元のソース点群,
                'target': ターゲット点群,
                'aligned_source': 位置合わせ済みソース点群
            }
        """
        print(f"Loading source: {source_path}")
        source = self.load_pointcloud(source_path)
        print(f"  Points: {len(source.points)}")

        print(f"Loading target: {target_path}")
        target = self.load_pointcloud(target_path)
        print(f"  Points: {len(target.points)}")

        # 初期変換
        if use_coarse:
            print("Coarse alignment...")
            initial_transform = self.coarse_alignment(source, target)
        else:
            initial_transform = np.eye(4)

        # FPFH特徴量ベースの位置合わせ（オプション）
        if use_fpfh:
            print("FPFH-based alignment...")
            source_down, source_fpfh = self.preprocess(source)
            target_down, target_fpfh = self.preprocess(target)

            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down, target_down,
                source_fpfh, target_fpfh,
                mutual_filter=True,
                max_correspondence_distance=self.voxel_size * 1.5,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=4,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.voxel_size * 1.5)
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
            )
            initial_transform = result.transformation

        # ICP精密位置合わせ
        print("ICP refinement...")
        final_transform, fitness, inlier_rmse = self.refine_alignment(
            source, target, initial_transform
        )

        print(f"  Fitness: {fitness:.4f}")
        print(f"  Inlier RMSE: {inlier_rmse:.6f}")

        # 位置合わせ済み点群を作成
        aligned_source = source.transform(final_transform)

        return {
            'transformation': final_transform,
            'fitness': fitness,
            'inlier_rmse': inlier_rmse,
            'source': source,
            'target': target,
            'aligned_source': aligned_source
        }

    def visualize(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        aligned_source: Optional[o3d.geometry.PointCloud] = None
    ):
        """
        位置合わせ結果を可視化

        Args:
            source: 元のソース点群（赤）
            target: ターゲット点群（緑）
            aligned_source: 位置合わせ済みソース点群（青）
        """
        geometries = []

        # ソース（赤）
        source_vis = o3d.geometry.PointCloud(source)
        source_vis.paint_uniform_color([1, 0, 0])
        geometries.append(source_vis)

        # ターゲット（緑）
        target_vis = o3d.geometry.PointCloud(target)
        target_vis.paint_uniform_color([0, 1, 0])
        geometries.append(target_vis)

        # 位置合わせ済み（青）
        if aligned_source is not None:
            aligned_vis = o3d.geometry.PointCloud(aligned_source)
            aligned_vis.paint_uniform_color([0, 0, 1])
            geometries.append(aligned_vis)

        o3d.visualization.draw_geometries(geometries)

    def save_result(
        self,
        result: Dict[str, Any],
        output_path: str,
        save_transformation: bool = True
    ):
        """
        位置合わせ結果を保存

        Args:
            result: align()の戻り値
            output_path: 出力ファイルパス（.ply）
            save_transformation: 変換行列も保存するか
        """
        output_path = Path(output_path)

        # 位置合わせ済み点群を保存
        o3d.io.write_point_cloud(str(output_path), result['aligned_source'])
        print(f"Saved aligned point cloud: {output_path}")

        # 変換行列を保存
        if save_transformation:
            transform_path = output_path.with_suffix('.npy')
            np.save(str(transform_path), result['transformation'])
            print(f"Saved transformation: {transform_path}")

            # JSONでも保存（可読性のため）
            import json
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump({
                    'transformation': result['transformation'].tolist(),
                    'fitness': result['fitness'],
                    'inlier_rmse': result['inlier_rmse']
                }, f, indent=2)
            print(f"Saved metadata: {json_path}")


def main():
    """コマンドライン実行用"""
    import argparse

    parser = argparse.ArgumentParser(description="ICP Point Cloud Alignment")
    parser.add_argument("source", help="Source point cloud (SAM 3D output)")
    parser.add_argument("target", help="Target point cloud (LiDAR)")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--voxel-size", type=float, default=0.02,
                        help="Voxel size for downsampling")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="ICP threshold")
    parser.add_argument("--no-coarse", action="store_true",
                        help="Skip coarse alignment")
    parser.add_argument("--fpfh", action="store_true",
                        help="Use FPFH feature-based alignment")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize result")

    args = parser.parse_args()

    aligner = ICPAligner(
        voxel_size=args.voxel_size,
        icp_threshold=args.threshold
    )

    result = aligner.align(
        args.source,
        args.target,
        use_coarse=not args.no_coarse,
        use_fpfh=args.fpfh
    )

    print("\n=== Result ===")
    print(f"Fitness: {result['fitness']:.4f}")
    print(f"Inlier RMSE: {result['inlier_rmse']:.6f}")
    print("Transformation matrix:")
    print(result['transformation'])

    if args.output:
        aligner.save_result(result, args.output)

    if args.visualize:
        aligner.visualize(
            result['source'],
            result['target'],
            result['aligned_source']
        )


if __name__ == "__main__":
    main()
