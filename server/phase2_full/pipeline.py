#!/usr/bin/env python3
"""
Phase 2 Full Pipeline
SAM 3 + Claude統合パイプライン

使い方:
    # クリック座標でセグメント → Claude生成
    python -m server.phase2_full.pipeline session_dir --click 512,384 --prompt "椅子を作成"

    # セグメントなしでClaude生成（全点群使用）
    python -m server.phase2_full.pipeline session_dir --prompt "椅子を作成" --no-segment
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from .sam3_segmentation import process_session_with_sam3, HAS_SAM3
from .claude_interface import (
    generate_blender_code_claude,
    get_session_images,
    check_api_available as check_claude
)

# Phase 1モジュールをインポート
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from server.phase1_minimal.pointcloud_gen import process_session, pointcloud_to_statistics
from server.phase1_minimal.blender_executor import execute_blender_code, find_blender
from server.phase1_minimal.llm_interface import check_ollama_status, generate_blender_code


def run_phase2_pipeline(
    session_dir: str,
    user_prompt: str,
    point_coords: Optional[List[tuple]] = None,
    use_sam3: bool = True,
    use_claude: bool = True,
    output_dir: Optional[str] = None,
    initial_frame: int = 0
) -> dict:
    """
    Phase 2パイプラインを実行

    Args:
        session_dir: セッションディレクトリ
        user_prompt: ユーザープロンプト
        point_coords: クリック座標（SAM 3用）
        use_sam3: SAM 3セグメンテーションを使用
        use_claude: Claudeを使用（FalseならOllama）
        output_dir: 出力ディレクトリ
        initial_frame: 初期フレームインデックス

    Returns:
        実行結果
    """
    result = {
        "status": "running",
        "session_dir": session_dir,
        "user_prompt": user_prompt,
        "timestamp": datetime.now().isoformat(),
        "pipeline": "phase2"
    }

    # 出力ディレクトリ
    if output_dir is None:
        output_dir = str(Path(session_dir) / "output" / "phase2")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result["output_dir"] = output_dir

    print("=" * 60)
    print("Phase 2 Full Pipeline")
    print("=" * 60)

    # Step 1: 環境チェック
    print("\n[Step 1] 環境チェック...")

    # Blender確認
    blender = find_blender()
    if blender is None:
        result["status"] = "error"
        result["error"] = "Blenderが見つかりません"
        print(f"  Error: {result['error']}")
        return result
    print(f"  Blender: OK ({blender})")

    # LLM確認
    if use_claude:
        if not check_claude():
            print("  Warning: Claude API not available, falling back to Ollama")
            use_claude = False

    if use_claude:
        print("  LLM: Claude API")
    else:
        if not check_ollama_status():
            result["status"] = "error"
            result["error"] = "Ollamaサーバーが起動していません"
            print(f"  Error: {result['error']}")
            return result
        print("  LLM: Ollama (gpt-oss 120B)")

    # SAM 3確認
    if use_sam3:
        if not HAS_SAM3:
            print("  Warning: SAM 3 not available, using full point cloud")
            use_sam3 = False
        else:
            print("  SAM 3: OK")

    result["use_claude"] = use_claude
    result["use_sam3"] = use_sam3

    # Step 2: 点群生成 / セグメンテーション
    print("\n[Step 2] 点群生成...")

    if use_sam3 and point_coords:
        # SAM 3でセグメント
        print(f"  SAM 3 segmentation with click: {point_coords}")
        seg_result = process_session_with_sam3(
            session_dir,
            point_coords=point_coords,
            initial_frame=initial_frame,
            output_dir=str(Path(output_dir) / "segmented")
        )

        if seg_result.get("status") == "success":
            # セグメントされた点群を読み込んで統計を計算
            ply_path = seg_result.get("output_ply")
            stats = {"num_points": seg_result.get("num_points", 0)}
            result["segmentation"] = seg_result
            print(f"  Segmented points: {stats['num_points']}")
        else:
            print(f"  Warning: Segmentation failed: {seg_result.get('error')}")
            print("  Falling back to full point cloud")
            use_sam3 = False
    else:
        use_sam3 = False

    if not use_sam3:
        # 全点群を使用
        pc_output = str(Path(output_dir) / "pointcloud")
        pc_result = process_session(session_dir, pc_output)
        stats = pc_result.get("merged_stats", {})
        print(f"  Full point cloud: {stats.get('num_points', 0)} points")

    result["pointcloud_stats"] = stats
    print(f"  Dimensions: {stats.get('dimensions', [])}")

    # Step 3: Blenderコード生成
    print("\n[Step 3] Blenderコード生成...")

    if use_claude:
        # Claude（マルチモーダル）
        rgb_images = get_session_images(session_dir, max_images=3)
        print(f"  Using Claude with {len(rgb_images)} reference images")

        blender_code = generate_blender_code_claude(
            stats,
            user_prompt,
            rgb_images=rgb_images
        )
    else:
        # Ollama
        print("  Using Ollama (gpt-oss 120B)")
        blender_code = generate_blender_code(stats, user_prompt)

    result["blender_code"] = blender_code
    print(f"  Code generated: {len(blender_code)} chars")

    # コード保存
    code_path = Path(output_dir) / "generated_code.py"
    with open(code_path, "w") as f:
        f.write(blender_code)
    print(f"  Saved: {code_path}")

    # Step 4: Blender実行
    print("\n[Step 4] Blender実行...")
    blend_path = str(Path(output_dir) / "result.blend")

    success, output, saved_path = execute_blender_code(
        blender_code,
        output_path=blend_path
    )

    result["blender_success"] = success
    result["blender_output"] = output

    if success:
        result["status"] = "success"
        result["output_file"] = saved_path
        print(f"  Success! Output: {saved_path}")
    else:
        result["status"] = "blender_error"
        print(f"  Error: {output[:200]}...")

    # 結果保存
    result_path = Path(output_dir) / "pipeline_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Pipeline completed: {result['status']}")
    print(f"Result: {result_path}")
    print("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 Full Pipeline: SAM 3 + Claude統合",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # SAM 3セグメント + Claude生成
  python -m server.phase2_full.pipeline experiments/session_xxx --click 512,384 --prompt "椅子を作成"

  # セグメントなし + Claude生成
  python -m server.phase2_full.pipeline experiments/session_xxx --prompt "椅子を作成" --no-segment

  # Ollama使用（Claude APIキーがない場合）
  python -m server.phase2_full.pipeline experiments/session_xxx --prompt "椅子を作成" --use-ollama
        """
    )

    parser.add_argument("session_dir", help="セッションディレクトリ")
    parser.add_argument("--prompt", "-p", required=True, help="ユーザープロンプト")
    parser.add_argument("--click", "-c", help="クリック座標 (x,y)")
    parser.add_argument("--frame", "-f", type=int, default=0, help="初期フレーム")
    parser.add_argument("--output", "-o", help="出力ディレクトリ")
    parser.add_argument("--no-segment", action="store_true", help="SAM 3セグメントをスキップ")
    parser.add_argument("--use-ollama", action="store_true", help="ClaudeではなくOllamaを使用")

    args = parser.parse_args()

    # クリック座標をパース
    point_coords = None
    if args.click:
        x, y = map(int, args.click.split(","))
        point_coords = [(x, y)]

    result = run_phase2_pipeline(
        session_dir=args.session_dir,
        user_prompt=args.prompt,
        point_coords=point_coords,
        use_sam3=not args.no_segment,
        use_claude=not args.use_ollama,
        output_dir=args.output,
        initial_frame=args.frame
    )

    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    exit(main())
