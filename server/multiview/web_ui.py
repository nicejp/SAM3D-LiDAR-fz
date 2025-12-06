#!/usr/bin/env python3
"""
Multi-view LiDAR Fusion Web UI

Gradio-based web interface for:
- Downloading Omniscient data from URL
- Selecting local session folders
- Clicking on video frames to select objects
- Running SAM 3 video tracking and point cloud fusion
"""

import os
import sys
import json
import shutil
import tempfile
import zipfile
import tarfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import urllib.request
import urllib.parse

import numpy as np

# Try to import gradio
try:
    import gradio as gr
except ImportError:
    print("Error: gradio not installed. Run: pip install gradio")
    sys.exit(1)

# Try to import PIL
try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Error: Pillow not installed. Run: pip install Pillow")
    sys.exit(1)

# Try to import cv2 for video frame extraction
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Video frame extraction will be limited.")


# Constants
DEFAULT_EXPERIMENTS_DIR = "experiments"
SUPPORTED_ARCHIVE_EXTENSIONS = ['.zip', '.tar.gz', '.tgz', '.tar']


@dataclass
class SessionState:
    """Holds the current session state"""
    session_path: Optional[str] = None
    video_path: Optional[str] = None
    total_frames: int = 0
    current_frame: int = 0
    click_point: Optional[Tuple[int, int]] = None
    text_prompt: Optional[str] = None
    loader: Any = None


# Global state
state = SessionState()


def get_available_sessions(experiments_dir: str = DEFAULT_EXPERIMENTS_DIR) -> List[str]:
    """Get list of available Omniscient sessions"""
    sessions = []

    if not os.path.exists(experiments_dir):
        return sessions

    # Look for Omniscient session directories
    for root, dirs, files in os.walk(experiments_dir):
        # Check if this directory contains Omniscient data
        omni_files = [f for f in files if f.endswith('.omni')]
        if omni_files:
            rel_path = os.path.relpath(root, experiments_dir)
            sessions.append(rel_path)

    return sorted(sessions)


def download_from_url(url: str, progress=gr.Progress()) -> Tuple[str, str]:
    """Download data from URL and extract if archive"""
    if not url.strip():
        return "", "URLを入力してください"

    try:
        progress(0, desc="ダウンロード準備中...")

        # Create download directory
        download_dir = os.path.join(DEFAULT_EXPERIMENTS_DIR, "downloads")
        os.makedirs(download_dir, exist_ok=True)

        temp_path = None

        # Handle Google Drive links
        if 'drive.google.com' in url:
            progress(0.1, desc="Google Driveからダウンロード中...")

            # Extract file ID from Google Drive URL
            if '/file/d/' in url:
                file_id = url.split('/file/d/')[1].split('/')[0]
            elif 'id=' in url:
                file_id = url.split('id=')[1].split('&')[0]
            else:
                return "", "Google DriveのURLが正しくありません"

            # Use gdown for Google Drive (handles large files better)
            try:
                import gdown
                download_url = f"https://drive.google.com/uc?id={file_id}"
                temp_path = os.path.join(download_dir, f"gdrive_{file_id}")
                gdown.download(download_url, temp_path, quiet=False, fuzzy=True)
            except ImportError:
                return "", "gdownがインストールされていません。pip install gdown を実行してください"
        else:
            # Regular URL download
            parsed = urllib.parse.urlparse(url)
            filename = os.path.basename(parsed.path) or "downloaded_data"
            temp_path = os.path.join(download_dir, filename)

            progress(0.1, desc=f"ダウンロード中: {filename}")
            urllib.request.urlretrieve(url, temp_path)

        if not temp_path or not os.path.exists(temp_path):
            return "", "ダウンロードに失敗しました"

        progress(0.5, desc="ダウンロード完了、ファイルタイプを確認中...")

        # Detect file type and extract
        extract_dir = None

        # Try to open as ZIP (check magic bytes)
        try:
            with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                extract_dir = temp_path + "_extracted"
                os.makedirs(extract_dir, exist_ok=True)
                progress(0.6, desc="ZIPファイルを展開中...")
                zip_ref.extractall(extract_dir)
                os.remove(temp_path)
        except zipfile.BadZipFile:
            # Not a ZIP file, try tar.gz
            try:
                with tarfile.open(temp_path, 'r:gz') as tar_ref:
                    extract_dir = temp_path + "_extracted"
                    os.makedirs(extract_dir, exist_ok=True)
                    progress(0.6, desc="tar.gzファイルを展開中...")
                    tar_ref.extractall(extract_dir)
                    os.remove(temp_path)
            except:
                # Not an archive, use download directory
                extract_dir = download_dir

        progress(0.8, desc="Omniscientセッションを検索中...")

        # Find Omniscient session in extracted directory
        session_path = None
        for root, dirs, files in os.walk(extract_dir):
            if any(f.endswith('.omni') for f in files):
                session_path = root
                break

        if session_path:
            progress(1.0, desc="完了!")
            rel_path = os.path.relpath(session_path, DEFAULT_EXPERIMENTS_DIR)
            return rel_path, f"ダウンロード成功: {rel_path}"
        else:
            # List what was extracted for debugging
            extracted_files = []
            for root, dirs, files in os.walk(extract_dir):
                for f in files[:10]:  # Limit to 10 files
                    extracted_files.append(os.path.join(root, f))
            files_info = "\n".join(extracted_files[:5]) if extracted_files else "なし"
            return "", f"Omniscientセッション(.omniファイル)が見つかりません。\n展開先: {extract_dir}\n展開されたファイル:\n{files_info}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return "", f"エラー: {str(e)}"


def load_session(session_name: str) -> Tuple[Optional[Image.Image], str, int]:
    """Load a session and return first frame"""
    global state

    if not session_name:
        return None, "セッションを選択してください", 0

    session_path = os.path.join(DEFAULT_EXPERIMENTS_DIR, session_name)

    if not os.path.exists(session_path):
        return None, f"セッションが見つかりません: {session_path}", 0

    try:
        # Import loader
        from server.multiview.omniscient_loader import OmniscientLoader

        loader = OmniscientLoader(session_path)
        state.loader = loader
        state.session_path = session_path

        # Get video path
        video_path = loader.video_path
        state.video_path = video_path

        # Get frame count
        if CV2_AVAILABLE and video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            state.total_frames = total_frames
        else:
            state.total_frames = loader.num_depth_frames

        # Get first frame
        frame_image = get_video_frame(0)

        summary = loader.summary()
        info_text = f"""
セッション: {session_name}
動画: {os.path.basename(video_path) if video_path else 'N/A'}
深度フレーム数: {loader.num_depth_frames}
カメラポーズ数: {summary.get('camera_poses', {}).get('num_frames', 'N/A')}
"""

        return frame_image, info_text, state.total_frames

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"読み込みエラー: {str(e)}", 0


def get_video_frame(frame_idx: int) -> Optional[Image.Image]:
    """Extract a specific frame from video"""
    global state

    if not state.video_path or not os.path.exists(state.video_path):
        return None

    if not CV2_AVAILABLE:
        return None

    try:
        cap = cv2.VideoCapture(state.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # Draw click point if exists
            if state.click_point and state.current_frame == frame_idx:
                draw = ImageDraw.Draw(image)
                x, y = state.click_point
                radius = 15
                draw.ellipse([x-radius, y-radius, x+radius, y+radius],
                           outline='red', width=3)
                draw.ellipse([x-3, y-3, x+3, y+3], fill='red')

            return image
        return None

    except Exception as e:
        print(f"Frame extraction error: {e}")
        return None


def update_frame(frame_idx: int) -> Optional[Image.Image]:
    """Update displayed frame"""
    global state
    state.current_frame = int(frame_idx)
    return get_video_frame(state.current_frame)


def on_image_click(evt: gr.SelectData, frame_idx: int) -> Tuple[Optional[Image.Image], str]:
    """Handle click on image to select object"""
    global state

    x, y = evt.index
    state.click_point = (x, y)
    state.current_frame = int(frame_idx)

    # Redraw frame with click marker
    image = get_video_frame(state.current_frame)

    info = f"選択位置: ({x}, {y}) @ フレーム {state.current_frame}"

    return image, info


def clear_selection() -> Tuple[Optional[Image.Image], str]:
    """Clear click selection"""
    global state
    state.click_point = None
    image = get_video_frame(state.current_frame)
    return image, "選択をクリアしました"


def run_fusion(
    text_prompt: str,
    frame_step: int,
    voxel_size: float,
    min_depth: float,
    max_depth: float,
    progress=gr.Progress()
) -> Tuple[str, Optional[str]]:
    """Run the fusion pipeline"""
    global state

    if not state.session_path:
        return "セッションが選択されていません", None

    # Determine prompt type
    prompt_type = None
    prompt_value = None

    if state.click_point:
        prompt_type = "click"
        prompt_value = f"{state.click_point[0]},{state.click_point[1]}"
    elif text_prompt.strip():
        prompt_type = "text"
        prompt_value = text_prompt.strip()
    else:
        return "クリックまたはテキストプロンプトを指定してください", None

    try:
        progress(0.1, desc="パイプライン初期化中...")

        from server.multiview.run import MultiViewPipeline

        pipeline = MultiViewPipeline(state.session_path)

        progress(0.2, desc="点群統合を実行中...")

        # Check if SAM 3 is available
        try:
            from sam3.model_builder import build_sam3_video_predictor
            sam3_available = True
        except ImportError:
            sam3_available = False

        if sam3_available:
            # Full pipeline with SAM 3
            result = pipeline.run_full_pipeline(
                prompt_type=prompt_type,
                prompt_frame=state.current_frame,
                point_coords=state.click_point if prompt_type == "click" else None,
                text=prompt_value if prompt_type == "text" else None,
                frame_step=frame_step,
                voxel_size=voxel_size if voxel_size > 0 else None,
                min_depth=min_depth,
                max_depth=max_depth
            )

            output_path = result.get('fused_pointcloud')

        else:
            # Without SAM 3, just generate full point cloud
            progress(0.3, desc="SAM 3が利用不可のため、全点群を生成中...")

            from server.multiview.omniscient_loader import OmniscientLoader

            loader = OmniscientLoader(state.session_path)

            all_points = []
            num_frames = loader.num_depth_frames
            frames_to_process = list(range(0, num_frames, frame_step))

            for i, frame_idx in enumerate(frames_to_process):
                progress(0.3 + 0.6 * (i / len(frames_to_process)),
                        desc=f"フレーム {frame_idx}/{num_frames} を処理中...")

                depth = loader.load_depth(frame_idx)
                if depth is None:
                    continue

                intrinsics = loader.get_intrinsics(frame_idx)
                points = loader.depth_to_pointcloud_world(
                    depth, intrinsics, frame_idx,
                    min_depth=min_depth, max_depth=max_depth
                )

                if points is not None and len(points) > 0:
                    all_points.append(points)

            if all_points:
                merged = np.vstack(all_points)

                # Apply voxel downsampling if specified
                if voxel_size > 0:
                    try:
                        import open3d as o3d
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(merged)
                        pcd = pcd.voxel_down_sample(voxel_size)
                        merged = np.asarray(pcd.points)
                    except ImportError:
                        pass

                # Save point cloud
                output_dir = os.path.join(state.session_path, "output")
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, "fused_pointcloud.ply")

                # Write PLY
                with open(output_path, 'w') as f:
                    f.write("ply\n")
                    f.write("format ascii 1.0\n")
                    f.write(f"element vertex {len(merged)}\n")
                    f.write("property float x\n")
                    f.write("property float y\n")
                    f.write("property float z\n")
                    f.write("end_header\n")
                    for p in merged:
                        f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            else:
                return "点群の生成に失敗しました", None

        progress(1.0, desc="完了!")

        result_text = f"""
処理完了!

出力ファイル: {output_path}
点群数: {len(merged) if 'merged' in dir() else 'N/A'}
プロンプト: {prompt_type} = {prompt_value}
フレーム間隔: {frame_step}
"""

        return result_text, output_path

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"エラー: {str(e)}", None


def create_ui():
    """Create Gradio UI"""

    with gr.Blocks(title="Multi-view LiDAR Fusion") as demo:
        gr.Markdown("# 多視点LiDAR融合 Web UI")
        gr.Markdown("Omniscientデータから多視点点群を統合します")

        with gr.Tabs():
            # Tab 1: Data Source
            with gr.TabItem("1. データソース"):
                gr.Markdown("### データの取得方法を選択")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### URLからダウンロード")
                        url_input = gr.Textbox(
                            label="データURL",
                            placeholder="https://... または Google Drive URL",
                            lines=1
                        )
                        download_btn = gr.Button("ダウンロード", variant="primary")
                        download_status = gr.Textbox(label="ステータス", interactive=False)

                    with gr.Column(scale=1):
                        gr.Markdown("#### ローカルフォルダから選択")
                        session_dropdown = gr.Dropdown(
                            label="セッション選択",
                            choices=get_available_sessions(),
                            interactive=True,
                            allow_custom_value=True
                        )
                        refresh_btn = gr.Button("リスト更新")

                load_btn = gr.Button("セッションを読み込む", variant="primary", size="lg")
                session_info = gr.Textbox(label="セッション情報", lines=6, interactive=False)

            # Tab 2: Object Selection
            with gr.TabItem("2. オブジェクト選択"):
                gr.Markdown("### ビデオフレームをクリックしてオブジェクトを選択")

                with gr.Row():
                    with gr.Column(scale=2):
                        frame_image = gr.Image(
                            label="ビデオフレーム",
                            interactive=True,
                            type="pil",
                            height=600
                        )

                        with gr.Row():
                            frame_slider = gr.Slider(
                                label="フレーム",
                                minimum=0,
                                maximum=100,
                                step=1,
                                value=0
                            )

                        selection_info = gr.Textbox(
                            label="選択状態",
                            interactive=False,
                            value="画像をクリックしてオブジェクトを選択"
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("#### プロンプト設定")

                        text_prompt = gr.Textbox(
                            label="テキストプロンプト（オプション）",
                            placeholder="例: chair, 椅子",
                            lines=1
                        )

                        gr.Markdown("*クリックまたはテキストのどちらかを使用*")

                        clear_btn = gr.Button("選択をクリア")

                        gr.Markdown("---")
                        gr.Markdown("#### 処理パラメータ")

                        frame_step = gr.Slider(
                            label="フレーム間隔",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=10
                        )

                        voxel_size = gr.Slider(
                            label="ボクセルサイズ（0=無効）",
                            minimum=0,
                            maximum=0.1,
                            step=0.005,
                            value=0.01
                        )

                        min_depth = gr.Slider(
                            label="最小深度（m）",
                            minimum=0,
                            maximum=5,
                            step=0.1,
                            value=0.1
                        )

                        max_depth = gr.Slider(
                            label="最大深度（m）",
                            minimum=1,
                            maximum=20,
                            step=0.5,
                            value=10.0
                        )

            # Tab 3: Run Fusion
            with gr.TabItem("3. 実行"):
                gr.Markdown("### 点群統合を実行")

                run_btn = gr.Button("点群統合を実行", variant="primary", size="lg")

                result_text = gr.Textbox(
                    label="実行結果",
                    lines=10,
                    interactive=False
                )

                output_file = gr.File(label="出力ファイル")

        # Event handlers
        download_btn.click(
            fn=download_from_url,
            inputs=[url_input],
            outputs=[session_dropdown, download_status]
        )

        refresh_btn.click(
            fn=lambda: gr.update(choices=get_available_sessions()),
            outputs=[session_dropdown]
        )

        load_btn.click(
            fn=load_session,
            inputs=[session_dropdown],
            outputs=[frame_image, session_info, frame_slider]
        ).then(
            fn=lambda x: gr.update(maximum=x if x > 0 else 100),
            inputs=[frame_slider],
            outputs=[frame_slider]
        )

        frame_slider.change(
            fn=update_frame,
            inputs=[frame_slider],
            outputs=[frame_image]
        )

        frame_image.select(
            fn=on_image_click,
            inputs=[frame_slider],
            outputs=[frame_image, selection_info]
        )

        clear_btn.click(
            fn=clear_selection,
            outputs=[frame_image, selection_info]
        )

        run_btn.click(
            fn=run_fusion,
            inputs=[text_prompt, frame_step, voxel_size, min_depth, max_depth],
            outputs=[result_text, output_file]
        )

    return demo


def main():
    """Main entry point"""
    global DEFAULT_EXPERIMENTS_DIR

    import argparse

    parser = argparse.ArgumentParser(description="Multi-view LiDAR Fusion Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7861, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--experiments-dir", default="experiments",
                       help="Directory containing experiments")

    args = parser.parse_args()

    DEFAULT_EXPERIMENTS_DIR = args.experiments_dir

    print(f"""
╔════════════════════════════════════════════════════════════╗
║         Multi-view LiDAR Fusion Web UI                     ║
╠════════════════════════════════════════════════════════════╣
║  URL: http://{args.host}:{args.port}                              ║
║  Experiments: {args.experiments_dir:<40} ║
╚════════════════════════════════════════════════════════════╝
""")

    demo = create_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
