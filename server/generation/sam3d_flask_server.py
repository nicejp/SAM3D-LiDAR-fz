#!/usr/bin/env python3
"""
SAM 3D Objects Flask Server

シンプルなFlaskベースのWebサーバー。
ブラウザから画像をアップロードして3D生成できる。

使い方:
    cd ~/sam-3d-objects
    conda activate sam3d
    pip install flask  # 初回のみ
    python /path/to/sam3d_flask_server.py --port 8000

ブラウザでアクセス:
    http://<WSL2のIP>:8000
"""

import os
import sys
import datetime
import tempfile
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


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SAM 3D Objects</title>
    <meta charset="utf-8">
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #333; text-align: center; }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .form-group { margin: 20px 0; }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
        }
        input[type="file"] {
            width: 100%;
            padding: 15px;
            border: 2px dashed #ccc;
            border-radius: 8px;
            background: #fafafa;
            cursor: pointer;
        }
        input[type="file"]:hover { border-color: #007bff; }
        input[type="number"] {
            width: 150px;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 16px;
        }
        button {
            background: #007bff;
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            cursor: pointer;
            width: 100%;
            margin-top: 20px;
        }
        button:hover { background: #0056b3; }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .preview {
            max-width: 100%;
            max-height: 400px;
            margin: 20px 0;
            border-radius: 8px;
            display: none;
        }
        .status {
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            display: none;
        }
        .status.loading {
            display: block;
            background: #fff3cd;
            color: #856404;
        }
        .status.success {
            display: block;
            background: #d4edda;
            color: #155724;
        }
        .status.error {
            display: block;
            background: #f8d7da;
            color: #721c24;
        }
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #007bff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
            vertical-align: middle;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .download-link {
            display: inline-block;
            background: #28a745;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            text-decoration: none;
            margin-top: 10px;
        }
        .download-link:hover { background: #218838; }
        .info {
            background: #e7f3ff;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 14px;
        }
        .info code {
            background: #d4e5f7;
            padding: 2px 6px;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>SAM 3D Objects</h1>

        <div class="info">
            <strong>使い方:</strong> RGBA画像（背景透明のPNG）をアップロードして3Dオブジェクトを生成します。<br>
            出力先: <code>{{output_dir}}</code>
        </div>

        <form id="uploadForm" enctype="multipart/form-data">
            <div class="form-group">
                <label for="file">RGBA画像 (PNG)</label>
                <input type="file" id="file" name="file" accept=".png,.PNG" required>
            </div>

            <img id="preview" class="preview">

            <div class="form-group">
                <label for="seed">シード値</label>
                <input type="number" id="seed" name="seed" value="42" min="0" max="2147483647">
            </div>

            <button type="submit" id="submitBtn">3D生成</button>
        </form>

        <div id="status" class="status"></div>
        <div id="result"></div>
    </div>

    <script>
        // プレビュー表示
        document.getElementById('file').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const preview = document.getElementById('preview');
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });

        // フォーム送信
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const fileInput = document.getElementById('file');
            const seedInput = document.getElementById('seed');
            const submitBtn = document.getElementById('submitBtn');
            const status = document.getElementById('status');
            const result = document.getElementById('result');

            if (!fileInput.files[0]) {
                alert('ファイルを選択してください');
                return;
            }

            // UI更新
            submitBtn.disabled = true;
            submitBtn.textContent = '生成中...';
            status.className = 'status loading';
            status.innerHTML = '<span class="spinner"></span>3Dオブジェクトを生成中... (数分かかる場合があります)';
            result.innerHTML = '';

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('seed', seedInput.value);

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const data = await response.json();
                    status.className = 'status success';
                    status.innerHTML = '生成完了!';
                    result.innerHTML = `
                        <p><strong>出力ファイル:</strong> ${data.filename}</p>
                        <p><strong>サイズ:</strong> ${data.size_kb.toFixed(1)} KB</p>
                        <a href="/download/${data.filename}" class="download-link">PLYファイルをダウンロード</a>
                    `;
                } else {
                    const error = await response.json();
                    status.className = 'status error';
                    status.innerHTML = 'エラー: ' + (error.error || '不明なエラー');
                }
            } catch (err) {
                status.className = 'status error';
                status.innerHTML = '通信エラー: ' + err.message;
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = '3D生成';
            }
        });
    </script>
</body>
</html>
"""


def create_app():
    from flask import Flask, request, jsonify, send_file, render_template_string
    import numpy as np
    from PIL import Image
    import io

    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE, output_dir=OUTPUT_DIR)

    @app.route('/health')
    def health():
        return jsonify({"status": "ok", "sam3d_path": SAM3D_PATH, "output_dir": OUTPUT_DIR})

    @app.route('/generate', methods=['POST'])
    def generate():
        try:
            if 'file' not in request.files:
                return jsonify({"error": "ファイルがありません"}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "ファイルが選択されていません"}), 400

            seed = int(request.form.get('seed', 42))

            # 画像を読み込み
            image = Image.open(file.stream)
            if image.mode != "RGBA":
                image = image.convert("RGBA")

            image_np = np.array(image)

            # アルファチャンネルからマスクを作成
            rgb = image_np[:, :, :3]
            alpha = image_np[:, :, 3]
            mask = (alpha > 128).astype(np.uint8)

            if mask.sum() == 0:
                return jsonify({"error": "マスクが空です。アルファチャンネルにオブジェクト領域が含まれていません。"}), 400

            # モデルを読み込み
            print(f"Generating 3D object (seed={seed})...")
            inference = load_model()

            # 3D生成
            output = inference(rgb, mask, seed=seed)

            # 出力ファイル名を生成
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            original_name = Path(file.filename).stem
            ply_filename = f"sam3d_{original_name}_{timestamp}_seed{seed}.ply"
            ply_path = os.path.join(OUTPUT_DIR, ply_filename)

            # PLYファイルを保存
            output["gs"].save_ply(ply_path)
            print(f"Saved to {ply_path}")

            file_size = os.path.getsize(ply_path) / 1024

            return jsonify({
                "success": True,
                "filename": ply_filename,
                "path": ply_path,
                "size_kb": file_size
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.route('/download/<filename>')
    def download(filename):
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        return jsonify({"error": "ファイルが見つかりません"}), 404

    return app


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SAM 3D Objects Flask Server")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind (default: 8000)")
    parser.add_argument("--sam3d-path", type=str, default=None,
                        help="Path to sam-3d-objects directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for PLY files")

    args = parser.parse_args()

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
║              SAM 3D Objects Flask Server                      ║
╠═══════════════════════════════════════════════════════════════╣
║  SAM3D Path:  {SAM3D_PATH:<47} ║
║  Output Dir:  {OUTPUT_DIR:<47} ║
║  Server:      http://{args.host}:{args.port:<38} ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    app = create_app()
    # threaded=Trueでタイムアウトを緩和
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
