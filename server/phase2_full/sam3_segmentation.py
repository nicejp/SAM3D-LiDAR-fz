#!/usr/bin/env python3
"""
SAM 3 Segmentation Module
セグメンテーションによるオブジェクト抽出

使い方:
    # セッションのRGB画像からマスクを生成
    python -m server.phase2_full.sam3_segmentation session_dir --click 100,200

    # テキストプロンプトでセグメント（未対応）
    python -m server.phase2_full.sam3_segmentation session_dir --text "椅子"
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from PIL import Image

# SAM 3 imports (Dockerコンテナ内で利用可能)
HAS_SAM3 = False
try:
    import torch
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    import sam3
    HAS_SAM3 = True
except ImportError:
    print("Warning: SAM 3 not available. Run inside Docker container: lidar-llm-mcp:sam3-ready")


class SAM3Segmentor:
    """SAM 3を使用したセグメンテーション"""

    def __init__(self, device: str = "cuda"):
        """
        Args:
            device: "cuda" or "cpu"
        """
        if not HAS_SAM3:
            raise RuntimeError("SAM 3 is not installed. Please run inside Docker container.")

        self.device = device
        self.model = None
        self.processor = None
        self._is_loaded = False

    def load_model(self):
        """モデルを読み込み"""
        if self._is_loaded:
            return

        print("Loading SAM 3 model...")

        # デバイス設定
        if self.device == "cuda" and torch.cuda.is_available():
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        # BPEパスを取得
        sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
        bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")

        # モデルをビルド
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=self.device,
            eval_mode=True,
            enable_inst_interactivity=True,  # SAM1タスク（クリックセグメント）有効
            load_from_HF=True
        )

        # プロセッサを作成
        self.processor = Sam3Processor(self.model)
        self._is_loaded = True
        print("SAM 3 model loaded!")

    def segment_image(
        self,
        image: np.ndarray,
        point_coords: Optional[List[Tuple[int, int]]] = None,
        point_labels: Optional[List[int]] = None,
        box: Optional[Tuple[int, int, int, int]] = None,
        multimask_output: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        単一画像をセグメント

        Args:
            image: RGB画像 (H, W, 3) または PIL Image
            point_coords: クリック座標のリスト [(x, y), ...]
            point_labels: 各座標のラベル (1=前景, 0=背景)
            box: バウンディングボックス (x1, y1, x2, y2)
            multimask_output: 複数マスクを出力するか

        Returns:
            masks: マスク配列 (N, H, W)
            scores: 各マスクのスコア (N,)
            logits: 低解像度マスクlogits (N, 256, 256)
        """
        self.load_model()

        # PIL Imageに変換（必要な場合）
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # 画像を設定してinference_stateを取得
        inference_state = self.processor.set_image(pil_image)

        # プロンプトを準備
        input_point = None
        input_label = None
        input_box = None

        if point_coords is not None:
            input_point = np.array(point_coords)
            input_label = np.array(point_labels or [1] * len(point_coords))

        if box is not None:
            input_box = np.array(box)

        # 予測
        masks, scores, logits = self.model.predict_inst(
            inference_state,
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=multimask_output,
        )

        return masks, scores, logits

    def segment_image_batch(
        self,
        images: List[np.ndarray],
        point_coords_batch: Optional[List[List[Tuple[int, int]]]] = None,
        point_labels_batch: Optional[List[List[int]]] = None,
        box_batch: Optional[List[Tuple[int, int, int, int]]] = None,
        multimask_output: bool = True
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        複数画像をバッチでセグメント

        Args:
            images: RGB画像のリスト
            point_coords_batch: 各画像のクリック座標
            point_labels_batch: 各画像の座標ラベル
            box_batch: 各画像のバウンディングボックス
            multimask_output: 複数マスクを出力するか

        Returns:
            masks_batch, scores_batch, logits_batch
        """
        self.load_model()

        # バッチ設定
        inference_state = self.processor.set_image_batch(images)

        # プロンプト準備
        pts_batch = None
        labels_batch = None

        if point_coords_batch is not None:
            pts_batch = [np.array(pts) for pts in point_coords_batch]
            labels_batch = [
                np.array(labels or [1] * len(pts))
                for pts, labels in zip(point_coords_batch, point_labels_batch or [None] * len(point_coords_batch))
            ]

        boxes = None
        if box_batch is not None:
            boxes = [np.array(box) for box in box_batch]

        # バッチ予測
        masks_batch, scores_batch, logits_batch = self.model.predict_inst_batch(
            inference_state,
            pts_batch,
            labels_batch,
            box_batch=boxes,
            multimask_output=multimask_output
        )

        return masks_batch, scores_batch, logits_batch


def apply_mask_to_depth(
    depth_map: np.ndarray,
    mask: np.ndarray,
    dilate_pixels: int = 0
) -> np.ndarray:
    """
    マスクを深度マップに適用して対象物のみを抽出

    Args:
        depth_map: 深度マップ (H, W)
        mask: セグメンテーションマスク (H, W) bool
        dilate_pixels: マスクを膨張させるピクセル数

    Returns:
        masked_depth: マスクされた深度マップ
    """
    # マスクのリサイズ（深度マップと同じサイズに）
    if mask.shape != depth_map.shape:
        from PIL import Image as PILImage
        mask_pil = PILImage.fromarray(mask.astype(np.uint8) * 255)
        mask_pil = mask_pil.resize((depth_map.shape[1], depth_map.shape[0]), PILImage.NEAREST)
        mask = np.array(mask_pil) > 127

    # 膨張処理
    if dilate_pixels > 0:
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(mask, iterations=dilate_pixels)

    # マスク適用
    masked_depth = depth_map.copy()
    masked_depth[~mask] = 0  # マスク外は0

    return masked_depth


def masked_depth_to_pointcloud(
    depth_map: np.ndarray,
    mask: np.ndarray,
    intrinsics: dict,
    rgb_image: Optional[np.ndarray] = None,
    transform: Optional[list] = None,
    max_depth: float = 5.0
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    マスクされた深度マップから点群を生成

    Args:
        depth_map: 深度マップ (H, W)
        mask: セグメンテーションマスク (H, W)
        intrinsics: カメラ内部パラメータ
        rgb_image: RGB画像（色付き点群用）
        transform: ワールド座標変換行列
        max_depth: 最大深度

    Returns:
        points: 点群 (N, 3)
        colors: 色情報 (N, 3) または None
    """
    # マスクを深度マップサイズにリサイズ
    if mask.shape != depth_map.shape:
        from PIL import Image as PILImage
        mask_pil = PILImage.fromarray(mask.astype(np.uint8) * 255)
        mask_pil = mask_pil.resize((depth_map.shape[1], depth_map.shape[0]), PILImage.NEAREST)
        mask = np.array(mask_pil) > 127

    height, width = depth_map.shape

    # カメラ内部パラメータ
    fx = intrinsics.get("fx", 500.0)
    fy = intrinsics.get("fy", 500.0)
    cx = intrinsics.get("cx", width / 2)
    cy = intrinsics.get("cy", height / 2)

    # ピクセルグリッド
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)

    # 有効マスク（マスク + 有効深度）
    z = depth_map
    valid_mask = mask & (z > 0) & (z < max_depth) & np.isfinite(z)

    # 3D座標変換
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # 有効点を抽出
    points = np.stack([
        x[valid_mask],
        y[valid_mask],
        z[valid_mask]
    ], axis=-1)

    # 色情報
    colors = None
    if rgb_image is not None:
        # RGBを深度マップサイズにリサイズ
        if rgb_image.shape[:2] != depth_map.shape:
            from PIL import Image as PILImage
            rgb_pil = PILImage.fromarray(rgb_image)
            rgb_pil = rgb_pil.resize((depth_map.shape[1], depth_map.shape[0]), PILImage.BILINEAR)
            rgb_image = np.array(rgb_pil)

        colors = rgb_image[valid_mask]

    # ワールド座標変換
    if transform is not None and len(points) > 0:
        transform_matrix = np.array(transform).reshape(4, 4)
        ones = np.ones((points.shape[0], 1))
        points_homo = np.hstack([points, ones])
        points = (transform_matrix @ points_homo.T).T[:, :3]

    return points, colors


def process_session_with_sam3(
    session_dir: str,
    point_coords: Optional[List[Tuple[int, int]]] = None,
    box: Optional[Tuple[int, int, int, int]] = None,
    initial_frame: int = 0,
    output_dir: Optional[str] = None,
    device: str = "cuda"
) -> dict:
    """
    セッションをSAM 3でセグメントして点群を生成

    Args:
        session_dir: セッションディレクトリ
        point_coords: クリック座標
        box: バウンディングボックス (x1, y1, x2, y2)
        initial_frame: 初期フレームインデックス
        output_dir: 出力ディレクトリ
        device: デバイス ("cuda" or "cpu")

    Returns:
        処理結果
    """
    session_path = Path(session_dir)

    if output_dir is None:
        output_path = session_path / "output" / "segmented"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 結果
    result = {
        "session_dir": str(session_path),
        "output_dir": str(output_path),
        "point_coords": point_coords,
        "box": box,
    }

    # RGBファイルを読み込み
    rgb_files = sorted((session_path / "rgb").glob("frame_*.jpg"))
    depth_files = sorted((session_path / "depth").glob("frame_*.npy"))

    if not rgb_files:
        result["error"] = "No RGB files found"
        return result

    print(f"Found {len(rgb_files)} RGB frames")

    # SAM 3でセグメント
    if not HAS_SAM3:
        result["error"] = "SAM 3 not available"
        return result

    segmentor = SAM3Segmentor(device=device)

    # 初期フレームを読み込み
    initial_rgb_file = rgb_files[initial_frame]
    initial_image = np.array(Image.open(initial_rgb_file))
    print(f"Processing initial frame: {initial_rgb_file.name}")
    print(f"Image size: {initial_image.shape}")

    print(f"Segmenting with {'click' if point_coords else 'box'} prompt...")

    # セグメンテーション実行
    masks, scores, logits = segmentor.segment_image(
        initial_image,
        point_coords=point_coords,
        box=box,
        multimask_output=True
    )

    # ベストマスクを選択
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]
    best_score = scores[best_idx]

    print(f"Generated mask with score: {best_score:.4f}")
    print(f"Mask coverage: {best_mask.sum() / best_mask.size * 100:.2f}%")

    # フレームマスク（現在は初期フレームのみ）
    frame_masks = {initial_frame: best_mask}

    # マスクを保存
    mask_dir = output_path / "masks"
    mask_dir.mkdir(exist_ok=True)

    for frame_idx, mask in frame_masks.items():
        mask_path = mask_dir / f"mask_{frame_idx:06d}.npy"
        np.save(mask_path, mask)

        # PNG画像としても保存
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(mask_dir / f"mask_{frame_idx:06d}.png")

    result["best_score"] = float(best_score)
    result["mask_coverage"] = float(best_mask.sum() / best_mask.size * 100)

    # マスク適用した点群を生成（深度データがある場合）
    if depth_files:
        all_points = []
        all_colors = []

        for frame_idx, mask in frame_masks.items():
            # 対応する深度ファイルを探す
            depth_file = session_path / "depth" / f"frame_{frame_idx:06d}.npy"
            if not depth_file.exists():
                continue

            # データ読み込み
            depth = np.load(depth_file)

            camera_file = session_path / "camera" / f"frame_{frame_idx:06d}.json"
            if camera_file.exists():
                with open(camera_file) as f:
                    camera = json.load(f)
            else:
                camera = {}

            rgb_file = session_path / "rgb" / f"frame_{frame_idx:06d}.jpg"
            rgb = np.array(Image.open(rgb_file)) if rgb_file.exists() else None

            # マスク適用した点群生成
            points, colors = masked_depth_to_pointcloud(
                depth,
                mask,
                camera.get("intrinsics", {}),
                rgb,
                camera.get("transform")
            )

            if len(points) > 0:
                all_points.append(points)
                if colors is not None:
                    all_colors.append(colors)

        if all_points:
            merged_points = np.vstack(all_points)
            merged_colors = np.vstack(all_colors) if all_colors else None

            # PLY保存
            ply_path = output_path / "segmented_object.ply"
            save_colored_ply(merged_points, ply_path, merged_colors)

            result["output_ply"] = str(ply_path)
            result["num_points"] = len(merged_points)

            print(f"Saved segmented point cloud: {ply_path}")
            print(f"Points: {len(merged_points)}")

    result["status"] = "success"
    return result


def save_colored_ply(points: np.ndarray, filepath: str, colors: Optional[np.ndarray] = None):
    """色付きPLY保存"""
    with open(filepath, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        for i, point in enumerate(points):
            if colors is not None:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                       f"{int(colors[i][0])} {int(colors[i][1])} {int(colors[i][2])}\n")
            else:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")


def main():
    parser = argparse.ArgumentParser(
        description="SAM 3によるセグメンテーション",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # クリック座標でセグメント
  python -m server.phase2_full.sam3_segmentation session_dir --click 100,200

  # バウンディングボックスでセグメント
  python -m server.phase2_full.sam3_segmentation session_dir --box 100,100,400,400
        """
    )

    parser.add_argument("session_dir", help="セッションディレクトリ")
    parser.add_argument("--click", "-c", help="クリック座標 (x,y)")
    parser.add_argument("--box", "-b", help="バウンディングボックス (x1,y1,x2,y2)")
    parser.add_argument("--frame", "-f", type=int, default=0, help="初期フレーム")
    parser.add_argument("--output", "-o", help="出力ディレクトリ")
    parser.add_argument("--device", "-d", default="cuda", help="デバイス (cuda/cpu)")

    args = parser.parse_args()

    # クリック座標をパース
    point_coords = None
    if args.click:
        x, y = map(int, args.click.split(","))
        point_coords = [(x, y)]

    # ボックスをパース
    box = None
    if args.box:
        box = tuple(map(int, args.box.split(",")))

    result = process_session_with_sam3(
        args.session_dir,
        point_coords=point_coords,
        box=box,
        initial_frame=args.frame,
        output_dir=args.output,
        device=args.device
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
