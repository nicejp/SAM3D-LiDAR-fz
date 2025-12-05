#!/usr/bin/env python3
"""
SAM 3D Objects Web UI

WSL2上で動作するSAM 3D ObjectsをラップするGradio Web UI。
ローカルファイルパスを指定して3D生成し、PLYファイルをダウンロードできる。

使い方:
    cd ~/sam-3d-objects
    conda activate sam3d
    python /path/to/sam3d_web_ui.py --port 8000

ブラウザでアクセス（WSL2内から）:
    http://localhost:8000
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

# 入力用ディレクトリ（DGX Sparkからscpで転送する先）
INPUT_DIR = os.environ.get("SAM3D_INPUT_DIR", os.path.expanduser("~/sam3d_inputs"))
os.makedirs(INPUT_DIR, exist_ok=True)

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


def list_input_files():
    """入力ディレクトリのPNGファイル一覧を取得"""
    files = list(Path(INPUT_DIR).glob("*.png"))
    files.extend(Path(INPUT_DIR).glob("*.PNG"))
    return sorted([str(f) for f in files], key=os.path.getmtime, reverse=True)


def generate_3d_from_path(
    file_path: str,
    seed: int = 42,
    progress=gr.Progress()
) -> tuple:
    """
    ファイルパスから3Dオブジェクトを生成

    Args:
        file_path: RGBA画像のパス
        seed: ランダムシード
        progress: Gradio progress bar

    Returns:
        (ply_path, status_message, preview_image)
    """
    if not file_path or not file_path.strip():
        return None, "ファイルパスを入力してください", None

    file_path = file_path.strip()

    if not os.path.exists(file_path):
        return None, f"ファイルが見つかりません: {file_path}", None

    try:
        progress(0.1, desc="画像を読み込み中...")

        # 画像を読み込み
        image = Image.open(file_path)
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        image_np = np.array(image)

        # プレビュー用に画像を保持
        preview = image_np.copy()

        progress(0.2, desc="モデルを読み込み中...")
        inference = load_model()

        # RGBA画像を処理
        rgb = image_np[:, :, :3]
        alpha = image_np[:, :, 3]
        mask = (alpha > 128).astype(np.uint8)

        # マスクが空でないか確認
        if mask.sum() == 0:
            return None, "マスクが空です。アルファチャンネルにオブジェクト領域が含まれていません。", preview

        progress(0.3, desc="3Dオブジェクトを生成中...")

        # 3D生成
        output = inference(rgb, mask, seed=seed)

        progress(0.8, desc="PLYファイルを保存中...")

        # 出力ファイル名を生成
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        input_name = Path(file_path).stem
        ply_filename = f"sam3d_{input_name}_{timestamp}_seed{seed}.ply"
        ply_path = os.path.join(OUTPUT_DIR, ply_filename)

        # PLYファイルを保存
        output["gs"].save_ply(ply_path)

        progress(1.0, desc="完了!")

        # ファイルサイズを取得
        file_size = os.path.getsize(ply_path) / 1024  # KB

        status = f"生成完了!\n" \
                 f"出力: {ply_path}\n" \
                 f"サイズ: {file_size:.1f} KB\n" \
                 f"シード: {seed}"

        return ply_path, status, preview

    except Exception as e:
        import traceback
        error_msg = f"エラー: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg, None


def refresh_file_list():
    """ファイル一覧を更新"""
    files = list_input_files()
    if files:
        return gr.update(choices=files, value=files[0])
    return gr.update(choices=[], value="")


def create_ui():
    """Gradio UIを作成"""

    with gr.Blocks() as demo:
        gr.Markdown(
            f"""
            # SAM 3D Objects Web UI

            RGBA画像（背景透明PNG）から3Dオブジェクトを生成します。

            ## 使い方

            ### DGX Sparkからファイルを転送:
            ```bash
            scp rgba_image.png nicejp@<WSL2のIP>:{INPUT_DIR}/
            ```

            ### または直接ファイルパスを入力

            ---
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # 入力ファイル選択
                gr.Markdown("### 入力")

                file_dropdown = gr.Dropdown(
                    label=f"入力ファイル ({INPUT_DIR})",
                    choices=list_input_files(),
                    value=list_input_files()[0] if list_input_files() else "",
                    allow_custom_value=True
                )

                refresh_btn = gr.Button("🔄 ファイル一覧を更新")

                gr.Markdown("または直接パスを入力:")
                file_path_input = gr.Textbox(
                    label="ファイルパス",
                    placeholder="/path/to/rgba_image.png",
                    value=""
                )

                seed_input = gr.Number(
                    label="シード値",
                    value=42,
                    precision=0,
                    minimum=0,
                    maximum=2147483647
                )

                generate_btn = gr.Button(
                    "3D生成",
                    variant="primary"
                )

            with gr.Column(scale=1):
                # プレビュー
                preview_image = gr.Image(
                    label="入力画像プレビュー",
                    type="numpy"
                )

        with gr.Row():
            with gr.Column():
                # 出力
                status_output = gr.Textbox(
                    label="ステータス",
                    lines=5,
                    interactive=False
                )

                ply_output = gr.File(
                    label="生成されたPLYファイル"
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

        # イベントハンドラ
        def get_path(dropdown, text_input):
            """ドロップダウンまたはテキスト入力からパスを取得"""
            if text_input and text_input.strip():
                return text_input.strip()
            return dropdown

        def on_generate(dropdown, text_input, seed, progress=gr.Progress()):
            path = get_path(dropdown, text_input)
            return generate_3d_from_path(path, int(seed), progress)

        generate_btn.click(
            fn=on_generate,
            inputs=[file_dropdown, file_path_input, seed_input],
            outputs=[ply_output, status_output, preview_image]
        )

        refresh_btn.click(
            fn=refresh_file_list,
            outputs=[file_dropdown]
        )

        # ドロップダウン選択時にプレビュー表示
        def preview_selected(path):
            if path and os.path.exists(path):
                try:
                    img = Image.open(path)
                    return np.array(img)
                except:
                    pass
            return None

        file_dropdown.change(
            fn=preview_selected,
            inputs=[file_dropdown],
            outputs=[preview_image]
        )

    return demo


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SAM 3D Objects Web UI")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind (default: 8000)")
    parser.add_argument("--sam3d-path", type=str, default=None,
                        help="Path to sam-3d-objects directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for PLY files")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Input directory for RGBA images")

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

    if args.input_dir:
        os.environ["SAM3D_INPUT_DIR"] = args.input_dir
        global INPUT_DIR
        INPUT_DIR = args.input_dir
        os.makedirs(INPUT_DIR, exist_ok=True)

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║              SAM 3D Objects Web UI                            ║
╠═══════════════════════════════════════════════════════════════╣
║  SAM3D Path:  {SAM3D_PATH:<47} ║
║  Input Dir:   {INPUT_DIR:<47} ║
║  Output Dir:  {OUTPUT_DIR:<47} ║
║  Server:      http://localhost:{args.port:<35} ║
╠═══════════════════════════════════════════════════════════════╣
║  DGX Sparkからファイルを転送:                                 ║
║    scp rgba_image.png nicejp@<IP>:{INPUT_DIR}/                ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    # UIを作成して起動
    demo = create_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False
    )


if __name__ == "__main__":
    main()
