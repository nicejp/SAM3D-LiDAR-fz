#!/usr/bin/env python3
"""
SAM 3D Objects Web UI

WSL2上で動作するSAM 3D ObjectsをラップするGradio Web UI。
DGX Sparkなど他のホストから、RGBAファイルをアップロードして3D生成し、
PLYファイルをダウンロードできる。

使い方:
    cd ~/sam-3d-objects
    conda activate sam3d
    python /path/to/sam3d_web_ui.py --port 8000

ブラウザでアクセス:
    http://<WSL2のIP>:8000
"""

import os
import sys
import datetime
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

# SAM 3D Objectsのパスを追加
SAM3D_PATH = os.environ.get("SAM3D_PATH", os.path.expanduser("~/sam-3d-objects"))
sys.path.insert(0, SAM3D_PATH)
sys.path.insert(0, os.path.join(SAM3D_PATH, "notebook"))

# 出力ディレクトリ
OUTPUT_DIR = os.environ.get("SAM3D_OUTPUT_DIR", os.path.expanduser("~/sam3d_outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# グローバル変数でモデルをキャッシュ
_inference = None


def load_model():
    """SAM 3D Objectsモデルを読み込む（遅延読み込み）"""
    global _inference
    if _inference is None:
        from inference import Inference

        config_path = os.path.join(SAM3D_PATH, "checkpoints/hf/pipeline.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config not found: {config_path}\n"
                "Please download checkpoints first:\n"
                "  huggingface-cli download facebook/sam-3d-objects --local-dir checkpoints/hf"
            )

        print("Loading SAM 3D Objects model...")
        _inference = Inference(config_path, compile=False)
        print("Model loaded successfully!")

    return _inference


def generate_3d(
    image: np.ndarray,
    seed: int = 42,
    progress=gr.Progress()
) -> tuple:
    """
    RGBA画像から3Dオブジェクトを生成

    Args:
        image: RGBA画像 (numpy array)
        seed: ランダムシード
        progress: Gradio progress bar

    Returns:
        (ply_path, status_message, file_path_str)
    """
    if image is None:
        return None, "画像をアップロードしてください", ""

    try:
        progress(0.1, desc="モデルを読み込み中...")
        inference = load_model()

        # RGBA画像を処理
        if len(image.shape) == 3 and image.shape[-1] == 4:
            # アルファチャンネルからマスクを作成
            rgb = image[:, :, :3]
            alpha = image[:, :, 3]
            mask = (alpha > 128).astype(np.uint8)
        elif len(image.shape) == 3 and image.shape[-1] == 3:
            # RGBのみの場合、全体をマスクとする
            rgb = image
            mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            return None, f"サポートされていない画像形式: {image.shape}"

        # マスクが空でないか確認
        if mask.sum() == 0:
            return None, "マスクが空です。アルファチャンネルにオブジェクト領域が含まれていません。"

        progress(0.3, desc="3Dオブジェクトを生成中...")

        # 3D生成
        output = inference(rgb, mask, seed=seed)

        progress(0.8, desc="PLYファイルを保存中...")

        # 出力ファイル名を生成
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ply_filename = f"sam3d_{timestamp}_seed{seed}.ply"
        ply_path = os.path.join(OUTPUT_DIR, ply_filename)

        # PLYファイルを保存
        output["gs"].save_ply(ply_path)

        progress(1.0, desc="完了!")

        # ファイルサイズを取得
        file_size = os.path.getsize(ply_path) / 1024  # KB

        status = f"生成完了!\n" \
                 f"ファイル: {ply_filename}\n" \
                 f"サイズ: {file_size:.1f} KB\n" \
                 f"シード: {seed}"

        return ply_path, status, ply_path

    except Exception as e:
        import traceback
        error_msg = f"エラー: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg, ""


def create_ui():
    """Gradio UIを作成"""

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # SAM 3D Objects Web UI

            RGBA画像（背景透明PNG）をアップロードして、3Dオブジェクトを生成します。

            **使い方:**
            1. RGBA画像（背景透明のPNG）をアップロード
            2. シード値を設定（オプション）
            3. 「3D生成」ボタンをクリック
            4. 生成されたPLYファイルをダウンロード
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # 入力
                input_image = gr.Image(
                    label="RGBA画像（背景透明PNG）",
                    type="numpy",
                    image_mode="RGBA"
                )

                seed_input = gr.Number(
                    label="シード値",
                    value=42,
                    precision=0
                )

                generate_btn = gr.Button(
                    "3D生成",
                    variant="primary"
                )

            with gr.Column(scale=1):
                # 出力
                status_output = gr.Textbox(
                    label="ステータス",
                    lines=6,
                    interactive=False
                )

                ply_output = gr.File(
                    label="生成されたPLYファイル（クリックでダウンロード）"
                )

                # ダウンロードパス表示
                download_path = gr.Textbox(
                    label="出力ファイルパス",
                    interactive=False
                )

        # サンプル画像セクション
        gr.Markdown("---")
        gr.Markdown("### サンプル画像")

        sample_dir = os.path.join(SAM3D_PATH, "demo/example_images")
        if os.path.exists(sample_dir):
            sample_images = list(Path(sample_dir).glob("*.png"))[:4]
            if sample_images:
                gr.Examples(
                    examples=[[str(img)] for img in sample_images],
                    inputs=[input_image],
                    label="サンプル画像をクリックして使用"
                )

        # イベントハンドラ
        generate_btn.click(
            fn=generate_3d,
            inputs=[input_image, seed_input],
            outputs=[ply_output, status_output, download_path]
        )

        # フッター
        gr.Markdown(
            f"""
            ---
            **出力先:** `{OUTPUT_DIR}`

            **SAM 3D Objects** by Meta AI |
            [GitHub](https://github.com/facebookresearch/sam-3d-objects)
            """
        )

    return demo


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SAM 3D Objects Web UI")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind (default: 8000)")
    parser.add_argument("--share", action="store_true",
                        help="Create public Gradio link")
    parser.add_argument("--sam3d-path", type=str, default=None,
                        help="Path to sam-3d-objects directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for PLY files")

    args = parser.parse_args()

    # 環境変数を設定
    if args.sam3d_path:
        os.environ["SAM3D_PATH"] = args.sam3d_path
        global SAM3D_PATH
        SAM3D_PATH = args.sam3d_path
        sys.path.insert(0, SAM3D_PATH)
        sys.path.insert(0, os.path.join(SAM3D_PATH, "notebook"))

    if args.output_dir:
        os.environ["SAM3D_OUTPUT_DIR"] = args.output_dir
        global OUTPUT_DIR
        OUTPUT_DIR = args.output_dir
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║              SAM 3D Objects Web UI                            ║
╠═══════════════════════════════════════════════════════════════╣
║  SAM3D Path:  {SAM3D_PATH:<47} ║
║  Output Dir:  {OUTPUT_DIR:<47} ║
║  Server:      http://{args.host}:{args.port:<38} ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    # UIを作成して起動
    demo = create_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
