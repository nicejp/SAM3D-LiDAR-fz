#!/usr/bin/env python3
"""
Multi-view LiDAR Fusion Pipeline

Omniscientデータから高密度点群を生成する統合パイプライン。

ワークフロー:
1. Omniscientデータを読み込み
2. SAM 3でオブジェクトをセグメント・トラッキング
3. 各フレームの点群を抽出
4. カメラポーズで位置合わせ・統合
5. 高密度点群を出力

使い方:
    # テキストプロンプトで追跡・統合
    python -m server.multiview.run experiments/omniscient_sample/003 --text "chair"

    # クリックプロンプトで追跡・統合
    python -m server.multiview.run experiments/omniscient_sample/003 --click 512,384

    # マスクがすでにある場合（SAM 3なしで統合のみ）
    python -m server.multiview.run experiments/omniscient_sample/003 --masks output/masks
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np

from server.multiview.omniscient_loader import OmniscientLoader
from server.multiview.pointcloud_fusion import MultiViewPointCloudFusion, save_ply

# SAM 3トラッカー（オプション）
HAS_SAM3_TRACKER = False
try:
    from server.multiview.sam3_video_tracker import SAM3VideoTracker, HAS_SAM3
    HAS_SAM3_TRACKER = HAS_SAM3
except ImportError:
    pass


class MultiViewPipeline:
    """多視点点群生成パイプライン"""

    def __init__(self, session_dir: str):
        """
        Args:
            session_dir: Omniscientセッションディレクトリ
        """
        self.session_dir = Path(session_dir)
        self.loader = OmniscientLoader(session_dir)
        self.output_dir = self.session_dir / "output"
        self.output_dir.mkdir(exist_ok=True)

    def run_tracking(
        self,
        prompt_type: str = "text",
        prompt_frame: int = 0,
        point_coords: Optional[List[tuple]] = None,
        text: Optional[str] = None
    ) -> Path:
        """
        SAM 3でオブジェクトをトラッキング

        Args:
            prompt_type: "click" or "text"
            prompt_frame: プロンプトフレーム
            point_coords: クリック座標
            text: テキストプロンプト

        Returns:
            マスク出力ディレクトリ
        """
        if not HAS_SAM3_TRACKER:
            raise RuntimeError(
                "SAM 3 is not available. Please run inside Docker container or "
                "provide pre-computed masks with --masks option."
            )

        tracker = SAM3VideoTracker()
        results = tracker.track_object(
            video_path=str(self.loader.video_path),
            prompt_type=prompt_type,
            prompt_frame=prompt_frame,
            point_coords=point_coords,
            text=text
        )

        # マスクを保存
        mask_dir = self.output_dir / "masks"
        tracker.save_masks(results, str(mask_dir))

        return mask_dir

    def run_fusion(
        self,
        mask_dir: str,
        frame_step: int = 1,
        voxel_size: Optional[float] = None,
        max_depth: float = 10.0,
        min_depth: float = 0.1,
        smooth: bool = False,
        statistical_outlier_removal: bool = False,
        sor_neighbors: int = 20,
        sor_std_ratio: float = 2.0,
        radius_outlier_removal: bool = False,
        ror_radius: float = 0.05,
        ror_min_neighbors: int = 5
    ) -> List[Path]:
        """
        マスクから点群を統合（複数オブジェクト対応）

        Args:
            mask_dir: マスクディレクトリ
            frame_step: フレーム間隔
            voxel_size: ダウンサンプリングサイズ
            max_depth: 最大深度
            min_depth: 最小深度
            smooth: ボクセル平均化（表面を滑らかに）
            statistical_outlier_removal: 統計的外れ値除去
            sor_neighbors: 近傍点数
            sor_std_ratio: 標準偏差倍率
            radius_outlier_removal: 半径フィルタリング
            ror_radius: 検索半径
            ror_min_neighbors: 最小近傍点数

        Returns:
            出力PLYファイルパスのリスト
        """
        fusion = MultiViewPointCloudFusion(self.loader)

        # 全オブジェクトを個別に処理
        results = fusion.fuse_all_objects(
            mask_dir,
            frame_step=frame_step,
            use_world_coords=self.loader.has_camera_poses,
            max_depth=max_depth,
            min_depth=min_depth,
            voxel_downsample=voxel_size,
            smooth=smooth,
            statistical_outlier_removal=statistical_outlier_removal,
            sor_neighbors=sor_neighbors,
            sor_std_ratio=sor_std_ratio,
            radius_outlier_removal=radius_outlier_removal,
            ror_radius=ror_radius,
            ror_min_neighbors=ror_min_neighbors
        )

        # 各オブジェクトを個別のファイルに保存
        output_paths = []
        for obj_id, result in results.items():
            if len(result.points) > 0:
                output_path = self.output_dir / f"fused_pointcloud_obj{obj_id}.ply"
                save_ply(result.points, str(output_path), result.colors)
                output_paths.append(output_path)
                print(f"Object {obj_id}: {len(result.points)} points -> {output_path}")

        return output_paths

    def run_full_pipeline(
        self,
        prompt_type: str = "text",
        prompt_frame: int = 0,
        point_coords: Optional[List[tuple]] = None,
        text: Optional[str] = None,
        frame_step: int = 1,
        voxel_size: Optional[float] = None,
        max_depth: float = 10.0,
        min_depth: float = 0.1,
        smooth: bool = False,
        statistical_outlier_removal: bool = False,
        sor_neighbors: int = 20,
        sor_std_ratio: float = 2.0,
        radius_outlier_removal: bool = False,
        ror_radius: float = 0.05,
        ror_min_neighbors: int = 5
    ) -> Dict:
        """
        フルパイプラインを実行

        Args:
            prompt_type: "click" or "text"
            prompt_frame: プロンプトフレーム
            point_coords: クリック座標
            text: テキストプロンプト
            frame_step: フレーム間隔
            voxel_size: ダウンサンプリングサイズ
            max_depth: 最大深度
            min_depth: 最小深度
            smooth: ボクセル平均化
            statistical_outlier_removal: 統計的外れ値除去
            sor_neighbors: 近傍点数
            sor_std_ratio: 標準偏差倍率
            radius_outlier_removal: 半径フィルタリング
            ror_radius: 検索半径
            ror_min_neighbors: 最小近傍点数

        Returns:
            結果の辞書
        """
        result = {
            "session_dir": str(self.session_dir),
            "output_dir": str(self.output_dir)
        }

        # Step 1: トラッキング
        print("\n=== Step 1: SAM 3 Tracking ===")
        mask_dir = self.run_tracking(
            prompt_type=prompt_type,
            prompt_frame=prompt_frame,
            point_coords=point_coords,
            text=text
        )
        result["mask_dir"] = str(mask_dir)

        # Step 2: 点群統合
        print("\n=== Step 2: Point Cloud Fusion ===")
        output_plys = self.run_fusion(
            str(mask_dir),
            frame_step=frame_step,
            voxel_size=voxel_size,
            max_depth=max_depth,
            min_depth=min_depth,
            smooth=smooth,
            statistical_outlier_removal=statistical_outlier_removal,
            sor_neighbors=sor_neighbors,
            sor_std_ratio=sor_std_ratio,
            radius_outlier_removal=radius_outlier_removal,
            ror_radius=ror_radius,
            ror_min_neighbors=ror_min_neighbors
        )
        result["output_plys"] = [str(p) for p in output_plys]
        # 後方互換性のため最初のファイルを output_ply にも設定
        if output_plys:
            result["output_ply"] = str(output_plys[0])

        return result

    def summary(self) -> Dict:
        """セッション情報のサマリー"""
        return {
            "session": self.loader.summary(),
            "output_dir": str(self.output_dir),
            "sam3_available": HAS_SAM3_TRACKER
        }


def main():
    parser = argparse.ArgumentParser(
        description="多視点LiDAR融合パイプライン",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # テキストプロンプトで追跡・統合（SAM 3が必要）
  python -m server.multiview.run experiments/omniscient_sample/003 --text "chair"

  # クリックプロンプトで追跡・統合（SAM 3が必要）
  python -m server.multiview.run experiments/omniscient_sample/003 --click 512,384

  # マスクがすでにある場合（SAM 3なしで統合のみ）
  python -m server.multiview.run experiments/omniscient_sample/003 --masks output/masks

  # セッション情報のみ表示
  python -m server.multiview.run experiments/omniscient_sample/003 --info

注意: SAM 3トラッキングはDockerコンテナ内で実行する必要があります
        """
    )

    parser.add_argument("session_dir", help="Omniscientセッションディレクトリ")
    parser.add_argument("--click", "-c", help="クリック座標 (x,y)")
    parser.add_argument("--text", "-t", help="テキストプロンプト")
    parser.add_argument("--masks", "-m", help="既存のマスクディレクトリ（トラッキングをスキップ）")
    parser.add_argument("--frame", "-f", type=int, default=0, help="プロンプトフレーム")
    parser.add_argument("--step", type=int, default=1, help="フレーム間隔")
    parser.add_argument("--voxel", type=float, help="ボクセルダウンサンプリングサイズ")
    parser.add_argument("--max-depth", type=float, default=10.0, help="最大深度")
    parser.add_argument("--min-depth", type=float, default=0.1, help="最小深度")
    parser.add_argument("--output", "-o", help="出力PLYファイル")
    parser.add_argument("--info", action="store_true", help="セッション情報のみ表示")

    args = parser.parse_args()

    # パイプラインを初期化
    pipeline = MultiViewPipeline(args.session_dir)

    # 情報のみ表示
    if args.info:
        print("=== Session Information ===")
        print(json.dumps(pipeline.summary(), indent=2, ensure_ascii=False))
        return

    # マスクが指定されている場合、統合のみ実行
    if args.masks:
        print("=== Point Cloud Fusion (masks provided) ===")
        output_ply = pipeline.run_fusion(
            args.masks,
            frame_step=args.step,
            voxel_size=args.voxel,
            max_depth=args.max_depth,
            min_depth=args.min_depth
        )

        if args.output:
            import shutil
            shutil.move(str(output_ply), args.output)
            print(f"Moved to: {args.output}")

        return

    # プロンプトを確認
    if not args.click and not args.text:
        print("Error: Either --click, --text, or --masks is required")
        print("\nUse --info to see session information")
        print("Use --masks to provide pre-computed masks (no SAM 3 required)")
        return

    # SAM 3が利用可能か確認
    if not HAS_SAM3_TRACKER:
        print("Error: SAM 3 is not available in this environment.")
        print("\nOptions:")
        print("1. Run inside Docker container:")
        print("   docker run --gpus all --ipc=host --network=host \\")
        print("     -v ~/SAM3D-LiDAR-fz:/workspace \\")
        print("     -it lidar-llm-mcp:sam3-tested")
        print("")
        print("2. Use pre-computed masks with --masks option")
        return

    # フルパイプライン実行
    prompt_type = "click" if args.click else "text"
    point_coords = None
    text = None

    if args.click:
        x, y = map(int, args.click.split(","))
        point_coords = [(x, y)]
    else:
        text = args.text

    print("=== Multi-view LiDAR Fusion Pipeline ===")
    result = pipeline.run_full_pipeline(
        prompt_type=prompt_type,
        prompt_frame=args.frame,
        point_coords=point_coords,
        text=text,
        frame_step=args.step,
        voxel_size=args.voxel,
        max_depth=args.max_depth,
        min_depth=args.min_depth
    )

    print("\n=== Result ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
