#!/usr/bin/env python3
"""
Click Selector Tool
画像上でクリックして座標を取得するツール

使い方:
    # セッションの最初のフレームを表示
    python -m server.phase2_full.click_selector session_dir

    # 特定のフレームを表示
    python -m server.phase2_full.click_selector session_dir --frame 10

    # 画像ファイルを直接指定
    python -m server.phase2_full.click_selector image.jpg
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import json

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# クリック座標を保存
click_points: List[Tuple[int, int]] = []


def mouse_callback(event, x, y, flags, param):
    """マウスクリックコールバック"""
    global click_points
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((x, y))
        print(f"Click: ({x}, {y}) - Total: {len(click_points)} points")
        # 画像に点を描画
        image = param
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Click to Select Object", image)
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 右クリックで最後の点を削除
        if click_points:
            removed = click_points.pop()
            print(f"Removed: {removed} - Total: {len(click_points)} points")


def select_clicks(image_path: str, window_name: str = "Click to Select Object") -> List[Tuple[int, int]]:
    """
    画像を表示してクリック座標を取得

    Args:
        image_path: 画像ファイルパス

    Returns:
        クリック座標のリスト [(x, y), ...]
    """
    global click_points
    click_points = []

    if not HAS_CV2:
        print("Error: OpenCV (cv2) is required")
        return []

    # 画像読み込み
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image: {image_path}")
        return []

    # ウィンドウ作成
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setMouseCallback(window_name, mouse_callback, image)

    print("\n" + "="*50)
    print("Click Selector")
    print("="*50)
    print("Left click:  Add point (foreground)")
    print("Right click: Remove last point")
    print("Enter/Space: Confirm selection")
    print("ESC:         Cancel")
    print("="*50 + "\n")

    cv2.imshow(window_name, image)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            print("Cancelled")
            click_points = []
            break
        elif key in [13, 32]:  # Enter or Space
            if click_points:
                print(f"\nConfirmed {len(click_points)} points:")
                for i, (x, y) in enumerate(click_points):
                    print(f"  {i+1}. ({x}, {y})")
            break

    cv2.destroyAllWindows()
    return click_points


def get_session_frame(session_dir: str, frame_idx: int = 0) -> str:
    """セッションからフレーム画像パスを取得"""
    session_path = Path(session_dir)
    rgb_dir = session_path / "rgb"

    if not rgb_dir.exists():
        print(f"Error: RGB directory not found: {rgb_dir}")
        return ""

    rgb_files = sorted(rgb_dir.glob("frame_*.jpg"))
    if not rgb_files:
        print(f"Error: No RGB files found in {rgb_dir}")
        return ""

    if frame_idx >= len(rgb_files):
        print(f"Warning: Frame {frame_idx} not found, using last frame")
        frame_idx = len(rgb_files) - 1

    return str(rgb_files[frame_idx])


def main():
    parser = argparse.ArgumentParser(
        description="画像上でクリックして座標を取得",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # セッションの最初のフレームを表示
  python -m server.phase2_full.click_selector experiments/session_xxx

  # 特定フレームを表示
  python -m server.phase2_full.click_selector experiments/session_xxx --frame 10

  # 画像ファイルを直接指定
  python -m server.phase2_full.click_selector path/to/image.jpg

SAM 3と組み合わせて使用:
  # 1. クリック座標を取得
  python -m server.phase2_full.click_selector experiments/session_xxx
  # → Click: (512, 384)

  # 2. SAM 3でセグメント
  python -m server.phase2_full.sam3_segmentation experiments/session_xxx --click 512,384
        """
    )

    parser.add_argument("path", help="セッションディレクトリまたは画像ファイル")
    parser.add_argument("--frame", "-f", type=int, default=0, help="フレームインデックス")
    parser.add_argument("--output", "-o", help="座標をJSONファイルに保存")

    args = parser.parse_args()

    if not HAS_CV2:
        print("Error: OpenCV (cv2) is required")
        print("Install: pip install opencv-python")
        return

    # パスを解決
    path = Path(args.path)
    if path.is_dir():
        image_path = get_session_frame(str(path), args.frame)
    elif path.is_file() and path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        image_path = str(path)
    else:
        print(f"Error: Invalid path: {path}")
        return

    if not image_path:
        return

    print(f"Loading: {image_path}")

    # クリック選択
    points = select_clicks(image_path)

    if points:
        # コマンドライン用に出力
        print("\nFor SAM 3:")
        coords = ",".join([f"{x},{y}" for x, y in points])
        print(f"  python -m server.phase2_full.sam3_segmentation {args.path} --click {points[0][0]},{points[0][1]}")

        # JSONに保存
        if args.output:
            with open(args.output, "w") as f:
                json.dump({"points": points}, f)
            print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
