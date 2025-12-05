#!/usr/bin/env python3
"""
SAM 3D + LiDAR 融合パイプライン

SAM 3Dで生成した3DモデルとLiDAR点群を融合して高精度3Dモデルを生成する。

処理フロー:
1. ICP位置合わせ: SAM 3DメッシュをLiDAR点群に位置合わせ
2. 可視判定: カメラから見える頂点を特定
3. Shrinkwrap: 可視頂点をLiDAR点群にスナップ
4. 結果出力: 融合済みメッシュを保存

使い方:
    python -m server.fusion.run \\
        --sam3d sam3d_output.ply \\
        --lidar lidar_points.ply \\
        --camera 0 0 2 \\
        --output fused_output.ply

    # セッションディレクトリから自動検出
    python -m server.fusion.run experiments/session_xxx
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import time

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Open3D is required. Install with: pip install open3d")

from .icp_alignment import ICPAligner
from .visibility_check import VisibilityChecker
from .shrinkwrap import ShrinkwrapProcessor


class FusionPipeline:
    """SAM 3D + LiDAR 融合パイプライン"""

    def __init__(
        self,
        voxel_size: float = 0.02,
        icp_threshold: float = 0.05,
        snap_distance: float = 0.1,
        smoothing_iterations: int = 3,
        visibility_method: str = "raycast"
    ):
        """
        Args:
            voxel_size: ICPダウンサンプリング用ボクセルサイズ
            icp_threshold: ICP収束閾値
            snap_distance: Shrinkwrapスナップ距離
            smoothing_iterations: スムージング反復回数
            visibility_method: 可視判定方法 ("raycast" or "normal")
        """
        self.aligner = ICPAligner(
            voxel_size=voxel_size,
            icp_threshold=icp_threshold
        )
        self.visibility_checker = VisibilityChecker()
        self.shrinkwrap = ShrinkwrapProcessor(
            snap_distance=snap_distance,
            smoothing_iterations=smoothing_iterations
        )
        self.visibility_method = visibility_method

    def find_session_files(self, session_dir: str) -> Dict[str, str]:
        """
        セッションディレクトリからファイルを自動検出

        Args:
            session_dir: セッションディレクトリパス

        Returns:
            {
                'sam3d_mesh': SAM 3Dメッシュパス,
                'lidar_pcd': LiDAR点群パス,
                'camera_params': カメラパラメータパス（オプション）
            }
        """
        session_dir = Path(session_dir)

        if not session_dir.exists():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")

        result = {}

        # SAM 3Dメッシュを検索
        sam3d_patterns = [
            "sam3d_*.ply",
            "*_sam3d.ply",
            "output/sam3d_*.ply",
            "generation/*.ply"
        ]
        for pattern in sam3d_patterns:
            matches = list(session_dir.glob(pattern))
            if matches:
                result['sam3d_mesh'] = str(matches[0])
                break

        # LiDAR点群を検索
        lidar_patterns = [
            "lidar_*.ply",
            "*_lidar.ply",
            "pointcloud/*.ply",
            "depth/*.ply"
        ]
        for pattern in lidar_patterns:
            matches = list(session_dir.glob(pattern))
            if matches:
                result['lidar_pcd'] = str(matches[0])
                break

        # カメラパラメータを検索
        camera_patterns = [
            "camera/*.json",
            "camera_params.json",
            "metadata.json"
        ]
        for pattern in camera_patterns:
            matches = list(session_dir.glob(pattern))
            if matches:
                result['camera_params'] = str(matches[0])
                break

        return result

    def load_camera_positions(self, camera_path: str) -> np.ndarray:
        """
        カメラパラメータファイルからカメラ位置を読み込む

        Args:
            camera_path: カメラパラメータファイルパス（JSON）

        Returns:
            カメラ位置の配列 (N, 3)
        """
        with open(camera_path, 'r') as f:
            data = json.load(f)

        # 形式に応じて解析
        if isinstance(data, list):
            # フレームごとのカメラパラメータのリスト
            positions = []
            for frame in data:
                if 'position' in frame:
                    positions.append(frame['position'])
                elif 'camera_position' in frame:
                    positions.append(frame['camera_position'])
                elif 'transform' in frame:
                    # 4x4変換行列から位置を抽出
                    transform = np.array(frame['transform'])
                    positions.append(transform[:3, 3].tolist())
            return np.array(positions)
        elif 'position' in data:
            return np.array([data['position']])
        elif 'camera_positions' in data:
            return np.array(data['camera_positions'])
        else:
            raise ValueError(f"Unknown camera parameter format: {camera_path}")

    def run(
        self,
        sam3d_mesh_path: str,
        lidar_pcd_path: str,
        camera_positions: np.ndarray,
        output_path: Optional[str] = None,
        skip_icp: bool = False,
        skip_visibility: bool = False,
        skip_shrinkwrap: bool = False
    ) -> Dict[str, Any]:
        """
        融合パイプラインを実行

        Args:
            sam3d_mesh_path: SAM 3Dメッシュファイルパス
            lidar_pcd_path: LiDAR点群ファイルパス
            camera_positions: カメラ位置の配列 (N, 3) または (3,)
            output_path: 出力ファイルパス
            skip_icp: ICP位置合わせをスキップ
            skip_visibility: 可視判定をスキップ
            skip_shrinkwrap: Shrinkwrapをスキップ

        Returns:
            {
                'mesh': 融合済みメッシュ,
                'transformation': ICP変換行列,
                'visible_mask': 可視マスク,
                'icp_fitness': ICPフィットネス,
                'snapped_count': スナップされた頂点数
            }
        """
        start_time = time.time()
        result = {}

        print("=" * 60)
        print("SAM 3D + LiDAR Fusion Pipeline")
        print("=" * 60)

        # Step 1: ICP位置合わせ
        if not skip_icp:
            print("\n[Step 1] ICP Alignment")
            print("-" * 40)

            icp_result = self.aligner.align(
                sam3d_mesh_path,
                lidar_pcd_path,
                use_coarse=True
            )

            result['transformation'] = icp_result['transformation']
            result['icp_fitness'] = icp_result['fitness']
            result['icp_rmse'] = icp_result['inlier_rmse']

            # 位置合わせ済みメッシュを使用
            aligned_mesh = self.visibility_checker.load_mesh(sam3d_mesh_path)
            aligned_mesh.transform(icp_result['transformation'])

            print(f"  ICP Fitness: {icp_result['fitness']:.4f}")
            print(f"  Inlier RMSE: {icp_result['inlier_rmse']:.6f}")
        else:
            print("\n[Step 1] ICP Alignment - SKIPPED")
            aligned_mesh = self.visibility_checker.load_mesh(sam3d_mesh_path)
            result['transformation'] = np.eye(4)

        # Step 2: 可視判定
        if not skip_visibility:
            print("\n[Step 2] Visibility Check")
            print("-" * 40)

            visible_mask = self.visibility_checker.compute_visibility(
                aligned_mesh,
                camera_positions
            )

            result['visible_mask'] = visible_mask
            visible_count = np.sum(visible_mask)
            total_count = len(aligned_mesh.vertices)

            print(f"  Visible vertices: {visible_count} / {total_count}")
            print(f"  Visibility ratio: {visible_count / total_count * 100:.1f}%")
        else:
            print("\n[Step 2] Visibility Check - SKIPPED")
            visible_mask = None

        # Step 3: Shrinkwrap
        if not skip_shrinkwrap:
            print("\n[Step 3] Shrinkwrap")
            print("-" * 40)

            # 一時ファイルに位置合わせ済みメッシュを保存
            temp_mesh_path = "/tmp/aligned_mesh.ply"
            o3d.io.write_triangle_mesh(temp_mesh_path, aligned_mesh)

            shrinkwrap_result = self.shrinkwrap.process(
                temp_mesh_path,
                lidar_pcd_path,
                visible_mask=visible_mask,
                output_path=output_path
            )

            result['mesh'] = shrinkwrap_result['mesh']
            result['snapped_count'] = shrinkwrap_result['snapped_count']

            print(f"  Snapped vertices: {shrinkwrap_result['snapped_count']}")
        else:
            print("\n[Step 3] Shrinkwrap - SKIPPED")
            result['mesh'] = aligned_mesh

            if output_path:
                o3d.io.write_triangle_mesh(output_path, aligned_mesh)
                print(f"  Saved mesh: {output_path}")

        # 完了
        elapsed_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        print(f"  Total time: {elapsed_time:.2f} seconds")

        if output_path:
            print(f"  Output: {output_path}")

        return result

    def run_from_session(
        self,
        session_dir: str,
        output_path: Optional[str] = None,
        default_camera: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        セッションディレクトリから自動検出して実行

        Args:
            session_dir: セッションディレクトリパス
            output_path: 出力ファイルパス（Noneの場合は自動生成）
            default_camera: カメラパラメータが見つからない場合のデフォルト位置

        Returns:
            run()の戻り値
        """
        print(f"Scanning session directory: {session_dir}")

        files = self.find_session_files(session_dir)

        if 'sam3d_mesh' not in files:
            raise FileNotFoundError("SAM 3D mesh not found in session directory")

        if 'lidar_pcd' not in files:
            raise FileNotFoundError("LiDAR point cloud not found in session directory")

        print(f"  SAM 3D mesh: {files.get('sam3d_mesh')}")
        print(f"  LiDAR point cloud: {files.get('lidar_pcd')}")
        print(f"  Camera params: {files.get('camera_params', 'Not found')}")

        # カメラ位置を取得
        if 'camera_params' in files:
            camera_positions = self.load_camera_positions(files['camera_params'])
        elif default_camera:
            camera_positions = np.array(default_camera)
        else:
            # デフォルト: Z軸方向から
            camera_positions = np.array([0, 0, 2])

        # 出力パスを自動生成
        if output_path is None:
            session_path = Path(session_dir)
            output_path = str(session_path / "fused_output.ply")

        return self.run(
            files['sam3d_mesh'],
            files['lidar_pcd'],
            camera_positions,
            output_path
        )


def main():
    """コマンドライン実行用"""
    import argparse

    parser = argparse.ArgumentParser(
        description="SAM 3D + LiDAR Fusion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 個別ファイルを指定
    python -m server.fusion.run --sam3d mesh.ply --lidar points.ply --camera 0 0 2

    # セッションディレクトリから自動検出
    python -m server.fusion.run experiments/session_xxx

    # ステップをスキップ
    python -m server.fusion.run --sam3d mesh.ply --lidar points.ply --skip-icp
"""
    )

    parser.add_argument("session_dir", nargs="?",
                        help="Session directory (auto-detect files)")
    parser.add_argument("--sam3d", help="SAM 3D mesh file path")
    parser.add_argument("--lidar", help="LiDAR point cloud file path")
    parser.add_argument("--camera", type=float, nargs=3,
                        help="Camera position (x y z)")
    parser.add_argument("-o", "--output", help="Output file path")

    # パイプラインパラメータ
    parser.add_argument("--voxel-size", type=float, default=0.02,
                        help="Voxel size for ICP downsampling")
    parser.add_argument("--icp-threshold", type=float, default=0.05,
                        help="ICP convergence threshold")
    parser.add_argument("--snap-distance", type=float, default=0.1,
                        help="Shrinkwrap snap distance")
    parser.add_argument("--smoothing", type=int, default=3,
                        help="Smoothing iterations")

    # スキップオプション
    parser.add_argument("--skip-icp", action="store_true",
                        help="Skip ICP alignment")
    parser.add_argument("--skip-visibility", action="store_true",
                        help="Skip visibility check")
    parser.add_argument("--skip-shrinkwrap", action="store_true",
                        help="Skip shrinkwrap")

    # 可視化
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize result")

    args = parser.parse_args()

    # パイプラインを作成
    pipeline = FusionPipeline(
        voxel_size=args.voxel_size,
        icp_threshold=args.icp_threshold,
        snap_distance=args.snap_distance,
        smoothing_iterations=args.smoothing
    )

    # 実行
    if args.session_dir:
        # セッションディレクトリから自動検出
        result = pipeline.run_from_session(
            args.session_dir,
            output_path=args.output,
            default_camera=args.camera
        )
    elif args.sam3d and args.lidar:
        # 個別ファイルを指定
        camera_pos = np.array(args.camera) if args.camera else np.array([0, 0, 2])

        result = pipeline.run(
            args.sam3d,
            args.lidar,
            camera_pos,
            output_path=args.output,
            skip_icp=args.skip_icp,
            skip_visibility=args.skip_visibility,
            skip_shrinkwrap=args.skip_shrinkwrap
        )
    else:
        parser.error("Either session_dir or both --sam3d and --lidar are required")

    # 可視化
    if args.visualize and 'mesh' in result:
        print("\nVisualizing result...")
        lidar_pcd = o3d.io.read_point_cloud(args.lidar if args.lidar else
                                            pipeline.find_session_files(args.session_dir)['lidar_pcd'])

        geometries = [result['mesh'], lidar_pcd]

        # LiDAR点群を白色に
        lidar_pcd.paint_uniform_color([1, 1, 1])

        o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    main()
