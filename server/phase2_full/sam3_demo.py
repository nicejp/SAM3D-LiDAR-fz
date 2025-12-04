#!/usr/bin/env python3
"""
SAM 3 Interactive Demo
Meta AIのデモサイトのようなインタラクティブセグメンテーション

使い方:
    # Dockerコンテナ内で実行
    export PYTHONPATH=/workspace:/workspace/sam3:$PYTHONPATH
    cd /workspace
    python -m server.phase2_full.sam3_demo

    # ブラウザで http://<サーバーIP>:7860 にアクセス
"""

import os
import sys

# PYTHONPATHの設定
if "/workspace/sam3" not in sys.path:
    sys.path.insert(0, "/workspace/sam3")
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import numpy as np
from PIL import Image
import torch

# Gradioのインポート
try:
    import gradio as gr
except ImportError:
    print("Gradioがインストールされていません。以下を実行してください:")
    print("  pip install gradio")
    sys.exit(1)

# SAM3のインポート
try:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    import sam3
    HAS_SAM3 = True
except ImportError:
    HAS_SAM3 = False
    print("SAM 3が見つかりません。Dockerコンテナ内で実行してください。")
    sys.exit(1)


class SAM3Demo:
    """SAM3インタラクティブデモ"""

    def __init__(self):
        self.model = None
        self.processor = None
        self.current_image = None
        self.current_image_path = None  # 画像パスを保存
        self.inference_state = None
        self.click_points = []
        self.click_labels = []

    def load_model(self):
        """モデルを読み込み"""
        if self.model is not None:
            return

        print("SAM 3モデルを読み込み中...")

        # デバイス設定
        if torch.cuda.is_available():
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        # BPEパスを取得
        sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
        bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")

        # モデルをビルド
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            eval_mode=True,
            enable_inst_interactivity=True,
            load_from_HF=True
        )

        # プロセッサを作成
        self.processor = Sam3Processor(self.model)
        print("SAM 3モデル読み込み完了!")

    def set_image(self, image):
        """画像を設定"""
        if image is None:
            return None, "画像をアップロードしてください"

        try:
            self.load_model()

            self.current_image = image.copy()
            self.click_points = []
            self.click_labels = []

            # PIL Imageに変換
            pil_image = Image.fromarray(image)
            self.inference_state = self.processor.set_image(pil_image)

            h, w = image.shape[:2]
            return image, f"画像を読み込みました ({w}x{h})。クリックしてセグメントしてください。"
        except Exception as e:
            return image, f"エラー: {str(e)}"

    def load_from_path(self, file_path):
        """ファイルパスから画像を読み込み"""
        if not file_path or not file_path.strip():
            return None, "ファイルパスを入力してください", ""

        file_path = file_path.strip()
        if not os.path.exists(file_path):
            return None, f"ファイルが見つかりません: {file_path}", ""

        try:
            self.load_model()

            # 画像を読み込み
            pil_image = Image.open(file_path).convert("RGB")
            image = np.array(pil_image)

            self.current_image = image.copy()
            self.current_image_path = file_path  # パスを保存
            self.click_points = []
            self.click_labels = []

            self.inference_state = self.processor.set_image(pil_image)

            h, w = image.shape[:2]

            # 深度マップパスを自動推測
            depth_path = self._guess_depth_path(file_path)

            return image, f"画像を読み込みました ({w}x{h})。出力画像をクリックしてセグメントしてください。", depth_path
        except Exception as e:
            return None, f"エラー: {str(e)}", ""

    def _guess_depth_path(self, rgb_path):
        """RGBパスから深度マップパスを推測"""
        # /workspace/experiments/session_xxx/rgb/frame_000002.jpg
        # → /workspace/experiments/session_xxx/depth/frame_000002.npy
        if "/rgb/" in rgb_path:
            depth_path = rgb_path.replace("/rgb/", "/depth/")
            # 拡張子を .npy に変更
            base = os.path.splitext(depth_path)[0]
            depth_path = base + ".npy"
            if os.path.exists(depth_path):
                return depth_path
        return ""

    def segment_click(self, image, evt: gr.SelectData):
        """クリック位置でセグメント"""
        if self.inference_state is None:
            return image, "先に画像をアップロードしてください"

        try:
            # クリック座標を追加
            x, y = evt.index
            self.click_points.append([x, y])
            self.click_labels.append(1)  # 前景

            # セグメンテーション実行
            masks, scores, _ = self.model.predict_inst(
                self.inference_state,
                point_coords=np.array(self.click_points),
                point_labels=np.array(self.click_labels),
                multimask_output=True
            )

            # ベストマスクを選択
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            best_score = scores[best_idx]

            # マスクをブール型に変換（NumPyインデックスに必要）
            if best_mask.dtype != bool:
                best_mask = best_mask > 0.5

            # マスクをオーバーレイ
            overlay = self.current_image.copy()

            # 青色のマスクオーバーレイ
            mask_color = np.array([30, 144, 255], dtype=np.float32)
            overlay = overlay.astype(np.float32)
            overlay[best_mask] = overlay[best_mask] * 0.5 + mask_color * 0.5

            # クリックポイントを描画
            for i, (px, py) in enumerate(self.click_points):
                # 緑色の点
                cv_y, cv_x = int(py), int(px)
                for dy in range(-5, 6):
                    for dx in range(-5, 6):
                        if dy*dy + dx*dx <= 25:  # 半径5の円
                            ny, nx = cv_y + dy, cv_x + dx
                            if 0 <= ny < overlay.shape[0] and 0 <= nx < overlay.shape[1]:
                                overlay[ny, nx] = [0, 255, 0]

            overlay = overlay.astype(np.uint8)

            info = f"スコア: {best_score:.4f} | クリック数: {len(self.click_points)} | マスク面積: {best_mask.sum() / best_mask.size * 100:.1f}%"

            return overlay, info
        except Exception as e:
            return image, f"エラー: {str(e)}"

    def add_negative_point(self, image, evt: gr.SelectData):
        """背景ポイントを追加（右クリック相当）"""
        if self.inference_state is None:
            return image, "先に画像をアップロードしてください"

        # クリック座標を追加（背景として）
        x, y = evt.index
        self.click_points.append([x, y])
        self.click_labels.append(0)  # 背景

        # セグメンテーション実行
        masks, scores, _ = self.model.predict_inst(
            self.inference_state,
            point_coords=np.array(self.click_points),
            point_labels=np.array(self.click_labels),
            multimask_output=True
        )

        # ベストマスクを選択
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]

        # マスクをブール型に変換（NumPyインデックスに必要）
        if best_mask.dtype != bool:
            best_mask = best_mask > 0.5

        # マスクをオーバーレイ
        overlay = self.current_image.copy()
        mask_color = np.array([30, 144, 255], dtype=np.float32)
        overlay = overlay.astype(np.float32)
        overlay[best_mask] = overlay[best_mask] * 0.5 + mask_color * 0.5

        # クリックポイントを描画
        for i, ((px, py), label) in enumerate(zip(self.click_points, self.click_labels)):
            cv_y, cv_x = int(py), int(px)
            color = [0, 255, 0] if label == 1 else [255, 0, 0]  # 緑=前景, 赤=背景
            for dy in range(-5, 6):
                for dx in range(-5, 6):
                    if dy*dy + dx*dx <= 25:
                        ny, nx = cv_y + dy, cv_x + dx
                        if 0 <= ny < overlay.shape[0] and 0 <= nx < overlay.shape[1]:
                            overlay[ny, nx] = color

        overlay = overlay.astype(np.uint8)

        fg_count = sum(self.click_labels)
        bg_count = len(self.click_labels) - fg_count
        info = f"スコア: {best_score:.4f} | 前景: {fg_count} | 背景: {bg_count}"

        return overlay, info

    def reset(self):
        """リセット"""
        self.click_points = []
        self.click_labels = []
        if self.current_image is not None:
            return self.current_image, "リセットしました。クリックしてセグメントしてください。"
        return None, "画像をアップロードしてください"

    def save_mask(self):
        """マスクを保存"""
        if self.inference_state is None or not self.click_points:
            return "マスクがありません"

        masks, scores, _ = self.model.predict_inst(
            self.inference_state,
            point_coords=np.array(self.click_points),
            point_labels=np.array(self.click_labels),
            multimask_output=True
        )

        best_mask = masks[np.argmax(scores)]

        # マスクをブール型に変換
        if best_mask.dtype != bool:
            best_mask = best_mask > 0.5

        # 保存
        save_path = "/workspace/experiments/sam3_demo_mask.npy"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, best_mask)

        # PNG画像も保存
        mask_img = Image.fromarray((best_mask.astype(np.uint8) * 255))
        mask_img.save("/workspace/experiments/sam3_demo_mask.png")

        return f"マスクを保存しました: {save_path}"

    def export_3d(self, depth_path):
        """マスク領域を3D点群として出力"""
        if self.inference_state is None or not self.click_points:
            return "マスクがありません。先に画像をクリックしてセグメントしてください。"

        if not depth_path or not depth_path.strip():
            return "深度マップのパスを入力してください"

        depth_path = depth_path.strip()
        if not os.path.exists(depth_path):
            return f"深度マップが見つかりません: {depth_path}"

        try:
            # マスクを取得
            masks, scores, _ = self.model.predict_inst(
                self.inference_state,
                point_coords=np.array(self.click_points),
                point_labels=np.array(self.click_labels),
                multimask_output=True
            )
            best_mask = masks[np.argmax(scores)]
            if best_mask.dtype != bool:
                best_mask = best_mask > 0.5

            # マスクを一時保存
            mask_path = "/workspace/experiments/sam3_demo_mask.npy"
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            np.save(mask_path, best_mask)

            # 3D変換
            from server.phase2_full.mask_to_3d import mask_to_3d
            output_path = "/workspace/experiments/segmented_object.ply"

            result = mask_to_3d(
                mask_path=mask_path,
                depth_path=depth_path,
                rgb_path=self.current_image_path,
                output_path=output_path,
            )

            return f"3D点群を出力しました!\n  点の数: {result['n_points']}\n  出力: {output_path}\n\nBlenderでインポート: File > Import > Stanford (.ply)"
        except Exception as e:
            return f"エラー: {str(e)}"


def get_sample_images():
    """サンプル画像のパスを取得"""
    import glob
    samples = []

    # 複数のディレクトリを検索
    search_dirs = [
        "/workspace/experiments",
        "/workspace/datasets",
    ]

    for sample_dir in search_dirs:
        if not os.path.exists(sample_dir):
            continue
        for pattern in ["**/rgb/*.jpg", "**/rgb/*.png", "**/*.jpg", "**/*.png"]:
            files = glob.glob(os.path.join(sample_dir, pattern), recursive=True)
            # マスク画像を除外
            files = [f for f in files if "mask" not in f.lower() and "depth" not in f.lower()]
            samples.extend(files[:10])

    # 重複を除去して最初の10個を返す
    seen = set()
    unique = []
    for s in samples:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique[:10] if unique else None


def create_demo():
    """Gradioデモを作成"""
    demo_app = SAM3Demo()
    sample_images = get_sample_images()

    with gr.Blocks(title="SAM 3 Interactive Demo") as demo:
        gr.Markdown("""
        # SAM 3 Interactive Demo

        Meta AIのSegment Anything Model 3をインタラクティブにテストできます。

        ## 使い方
        1. 「ファイルパス」にコンテナ内の画像パスを入力
        2. 「パスから読み込む」ボタンをクリック
        3. 出力画像をクリックしてセグメント
        4. 背景モードをONにして除外したい領域をクリック
        """)

        with gr.Row():
            with gr.Column():
                file_path_input = gr.Textbox(
                    label="RGBファイルパス",
                    placeholder="/workspace/experiments/sample.jpg",
                    info="コンテナ内の画像ファイルパスを入力"
                )
                load_path_btn = gr.Button("パスから読み込む", variant="primary")

                with gr.Row():
                    reset_btn = gr.Button("リセット", variant="secondary")
                    save_btn = gr.Button("マスク保存", variant="secondary")
                bg_mode = gr.Checkbox(label="背景モード（チェックすると次のクリックは背景として扱う）", value=False)

                # 3D出力セクション
                gr.Markdown("---\n#### 3D出力（オプション）")
                depth_path_input = gr.Textbox(
                    label="深度マップパス",
                    placeholder="/workspace/experiments/depth.npy",
                    info="深度マップファイル（.npy or .png）"
                )
                export_3d_btn = gr.Button("3D点群を出力 (PLY)", variant="secondary")

            with gr.Column():
                output_image = gr.Image(label="セグメント結果（ここをクリック）")
                info_text = gr.Textbox(label="情報", interactive=False)
                save_result = gr.Textbox(label="保存結果", interactive=False, lines=4)

        # サンプル画像のパスを表示
        if sample_images:
            gr.Markdown("### サンプル画像パス（コピーして上のファイルパス欄に貼り付け）")
            sample_list = "\n".join([f"- `{s}`" for s in sample_images])
            gr.Markdown(sample_list)

        # イベントハンドラ
        # ファイルパスから読み込み（深度パスも自動設定）
        load_path_btn.click(
            demo_app.load_from_path,
            inputs=file_path_input,
            outputs=[output_image, info_text, depth_path_input]
        )

        def handle_click(bg_mode, evt: gr.SelectData):
            if bg_mode:
                return demo_app.add_negative_point(None, evt)
            else:
                return demo_app.segment_click(None, evt)

        output_image.select(
            handle_click,
            inputs=[bg_mode],
            outputs=[output_image, info_text]
        )

        reset_btn.click(
            demo_app.reset,
            outputs=[output_image, info_text]
        )

        save_btn.click(
            demo_app.save_mask,
            outputs=save_result
        )

        export_3d_btn.click(
            demo_app.export_3d,
            inputs=depth_path_input,
            outputs=save_result
        )

        gr.Markdown("""
        ---
        ## ヒント
        - 複数回クリックすると、セグメントが洗練されます
        - 背景モードをONにして、除外したい部分をクリックすると精度が上がります
        - マスク保存で `/workspace/experiments/sam3_demo_mask.png` に保存されます
        - **3D出力**: 深度マップがあれば、セグメント領域を3D点群（PLY）として出力できます
        - **Blenderでインポート**: File > Import > Stanford (.ply)
        """)

    return demo


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="SAM 3 Interactive Demo")
    parser.add_argument("--port", type=int, default=7860, help="ポート番号")
    parser.add_argument("--share", action="store_true", help="公開リンクを生成")
    args = parser.parse_args()

    print("=" * 50)
    print("SAM 3 Interactive Demo")
    print("=" * 50)
    print(f"ポート: {args.port}")
    print(f"URL: http://0.0.0.0:{args.port}")
    print("=" * 50)

    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        max_file_size="50mb",  # 最大ファイルサイズを増やす
    )


if __name__ == "__main__":
    main()
