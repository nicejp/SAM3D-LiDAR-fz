#!/usr/bin/env python3
"""
SAM 3D Objects API Server

シンプルなFastAPIベースのAPIサーバー。
curlでRGBA画像をアップロードして3D生成し、PLYファイルをダウンロードできる。

使い方:
    cd ~/sam-3d-objects
    conda activate sam3d
    python /path/to/sam3d_api_server.py --port 8000

API:
    POST /generate
        - file: RGBA画像ファイル (PNG)
        - seed: シード値 (オプション、デフォルト: 42)

    レスポンス: PLYファイル

curlでの使用例:
    curl -X POST -F "file=@input.png" -F "seed=42" \
         http://<WSL2のIP>:8000/generate -o output.ply
"""

import os
import sys
import tempfile
import datetime
from pathlib import Path

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


def create_app():
    """FastAPIアプリを作成"""
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse
    import numpy as np
    from PIL import Image
    import io

    app = FastAPI(title="SAM 3D Objects API")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """シンプルなHTMLフォーム"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SAM 3D Objects API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
                h1 { color: #333; }
                .form-group { margin: 20px 0; }
                label { display: block; margin-bottom: 5px; font-weight: bold; }
                input[type="file"], input[type="number"] { padding: 10px; margin: 5px 0; }
                button { background: #007bff; color: white; padding: 15px 30px; border: none; cursor: pointer; font-size: 16px; }
                button:hover { background: #0056b3; }
                pre { background: #f4f4f4; padding: 15px; overflow-x: auto; }
                .status { margin-top: 20px; padding: 15px; border-radius: 5px; }
                .success { background: #d4edda; color: #155724; }
                .error { background: #f8d7da; color: #721c24; }
            </style>
        </head>
        <body>
            <h1>SAM 3D Objects API</h1>

            <h2>Web UI</h2>
            <form action="/generate" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <label>RGBA画像 (背景透明PNG):</label>
                    <input type="file" name="file" accept=".png" required>
                </div>
                <div class="form-group">
                    <label>シード値:</label>
                    <input type="number" name="seed" value="42" min="0">
                </div>
                <button type="submit">3D生成</button>
            </form>

            <hr>

            <h2>curl での使用方法</h2>
            <pre>curl -X POST -F "file=@input.png" -F "seed=42" \\
     http://localhost:8000/generate -o output.ply</pre>

            <h2>API エンドポイント</h2>
            <ul>
                <li><code>GET /</code> - このページ</li>
                <li><code>GET /health</code> - ヘルスチェック</li>
                <li><code>POST /generate</code> - 3D生成</li>
            </ul>
        </body>
        </html>
        """

    @app.get("/health")
    async def health():
        """ヘルスチェック"""
        return {"status": "ok", "sam3d_path": SAM3D_PATH, "output_dir": OUTPUT_DIR}

    @app.post("/generate")
    async def generate(
        file: UploadFile = File(...),
        seed: int = Form(42)
    ):
        """RGBA画像から3Dオブジェクトを生成"""
        try:
            # ファイルを読み込み
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))

            # RGBA形式に変換
            if image.mode != "RGBA":
                image = image.convert("RGBA")

            image_np = np.array(image)

            # アルファチャンネルからマスクを作成
            rgb = image_np[:, :, :3]
            alpha = image_np[:, :, 3]
            mask = (alpha > 128).astype(np.uint8)

            # マスクが空でないか確認
            if mask.sum() == 0:
                raise HTTPException(
                    status_code=400,
                    detail="マスクが空です。アルファチャンネルにオブジェクト領域が含まれていません。"
                )

            # モデルを読み込み
            inference = load_model()

            # 3D生成
            print(f"Generating 3D object (seed={seed})...")
            output = inference(rgb, mask, seed=seed)

            # 出力ファイル名を生成
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ply_filename = f"sam3d_{timestamp}_seed{seed}.ply"
            ply_path = os.path.join(OUTPUT_DIR, ply_filename)

            # PLYファイルを保存
            output["gs"].save_ply(ply_path)
            print(f"Saved to {ply_path}")

            # PLYファイルを返す
            return FileResponse(
                ply_path,
                media_type="application/octet-stream",
                filename=ply_filename
            )

        except HTTPException:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    return app


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="SAM 3D Objects API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind (default: 8000)")
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
║              SAM 3D Objects API Server                        ║
╠═══════════════════════════════════════════════════════════════╣
║  SAM3D Path:  {SAM3D_PATH:<47} ║
║  Output Dir:  {OUTPUT_DIR:<47} ║
║  Server:      http://{args.host}:{args.port:<38} ║
╠═══════════════════════════════════════════════════════════════╣
║  Usage:                                                       ║
║    curl -X POST -F "file=@input.png" -F "seed=42" \\           ║
║         http://localhost:{args.port}/generate -o output.ply     ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
