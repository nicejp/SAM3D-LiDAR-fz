#!/usr/bin/env python3
"""
SAM 3 Video Tracker

SAM 3のビデオトラッキング機能を使用して、複数フレームにわたって
オブジェクトをセグメントする。

使い方（Dockerコンテナ内で実行）:
    # クリックプロンプトで追跡
    python -m server.multiview.sam3_video_tracker video.mp4 --click 512,384

    # テキストプロンプトで追跡
    python -m server.multiview.sam3_video_tracker video.mp4 --text "椅子"

    # Omniscientセッションから実行
    python -m server.multiview.sam3_video_tracker experiments/omniscient_sample/003 --text "chair"

注意:
    SAM 3はDockerコンテナ内で実行する必要があります:
    docker run --gpus all --ipc=host --network=host \\
        -v ~/SAM3D-LiDAR-fz:/workspace \\
        -it lidar-llm-mcp:sam3-tested
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
import numpy as np

# SAM 3 imports
HAS_SAM3 = False
try:
    import torch
    from sam3.model_builder import build_sam3_video_predictor
    HAS_SAM3 = True
except ImportError:
    pass


@dataclass
class TrackingResult:
    """トラッキング結果"""
    frame_index: int
    mask: np.ndarray  # (H, W) bool
    score: float
    object_id: int


class SAM3VideoTracker:
    """SAM 3を使用したビデオトラッキング"""

    def __init__(self, device: str = "cuda"):
        """
        Args:
            device: "cuda" or "cpu"
        """
        if not HAS_SAM3:
            raise RuntimeError(
                "SAM 3 is not installed. Please run inside Docker container:\n"
                "docker run --gpus all --ipc=host --network=host \\\n"
                "  -v ~/SAM3D-LiDAR-fz:/workspace \\\n"
                "  -it lidar-llm-mcp:sam3-tested"
            )

        self.device = device
        self.predictor = None
        self._is_loaded = False
        self._session_id = None
        self._video_path = None

    def load_model(self):
        """モデルを読み込み"""
        if self._is_loaded:
            return

        print("Loading SAM 3 video model...")

        if self.device == "cuda" and torch.cuda.is_available():
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        # SAM 3 API: no arguments needed, auto-detects GPU and loads from HF
        self.predictor = build_sam3_video_predictor()

        self._is_loaded = True
        print("SAM 3 video model loaded!")

    def start_session(self, video_path: str) -> str:
        """
        ビデオセッションを開始

        Args:
            video_path: 動画ファイルパス（MP4）またはJPEGフォルダ

        Returns:
            session_id: セッションID
        """
        self.load_model()

        video_path = Path(video_path)

        # Omniscientセッションの場合、動画ファイルを探す
        if video_path.is_dir():
            # .movファイルを探す
            mov_files = list(video_path.glob("*.mov"))
            if mov_files:
                video_path = mov_files[0]
            else:
                # JPEGフォルダとして扱う
                pass

        self._video_path = str(video_path)
        print(f"Starting session with: {self._video_path}")

        # セッションを開始 (handle_request形式)
        response = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=self._video_path,
            )
        )
        self._session_id = response['session_id']

        # セッションをリセット（キャッシュを初期化）
        _ = self.predictor.handle_request(
            request=dict(
                type="reset_session",
                session_id=self._session_id,
            )
        )

        return self._session_id

    def _build_cache_for_frame(self, frame_index: int):
        """
        特定フレームまでの特徴キャッシュを構築

        点ベースのプロンプトを使用する前に必要

        Args:
            frame_index: キャッシュを構築するフレーム
        """
        print(f"Building feature cache to frame {frame_index}...")

        # 前方向に伝播してキャッシュを構築
        try:
            response = self.predictor.handle_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=self._session_id,
                    start_frame_idx=0,
                    max_frame_num_to_track=frame_index + 1,
                    propagation_direction="forward",
                )
            )
            print(f"Cache built to frame {frame_index}")
        except Exception as e:
            # handle_requestで失敗したら直接メソッドを試す
            print(f"handle_request failed, trying direct method: {e}")
            try:
                # propagate_in_videoを直接呼ぶ
                for result in self.predictor.propagate_in_video(
                    session_id=self._session_id,
                    start_frame_idx=0,
                    max_frame_num_to_track=frame_index + 1,
                    propagation_direction="forward",
                ):
                    pass  # 結果を消費してキャッシュを構築
                print(f"Cache built to frame {frame_index} (direct method)")
            except Exception as e2:
                print(f"Direct propagation also failed: {e2}")
                # キャッシュ構築に失敗してもプロンプト追加を試みる

    def add_click_prompt(
        self,
        frame_index: int,
        point_coords: List[Tuple[int, int]],
        point_labels: Optional[List[int]] = None,
        object_id: int = 0,
        img_width: int = 1920,
        img_height: int = 1080
    ) -> Dict:
        """
        クリックプロンプトを追加

        Args:
            frame_index: 対象フレーム
            point_coords: クリック座標 [(x, y), ...] (絶対座標)
            point_labels: 各座標のラベル (1=前景, 0=背景)
            object_id: オブジェクトID
            img_width: 画像の幅（相対座標変換用）
            img_height: 画像の高さ（相対座標変換用）

        Returns:
            初期フレームのセグメント結果
        """
        if self._session_id is None:
            raise RuntimeError("Session not started. Call start_session() first.")

        if point_labels is None:
            point_labels = [1] * len(point_coords)

        # 点プロンプトの前に特徴キャッシュを構築（必須）
        self._build_cache_for_frame(frame_index)

        # 座標形式: SAM 3は (y, x) の順序を期待する可能性
        # 多くのモデルは (row, col) = (y, x) 形式を使用
        points_yx = [[float(y), float(x)] for x, y in point_coords]

        print(f"Adding click prompt: frame={frame_index}")
        print(f"  Original coords (x, y): {point_coords}")
        print(f"  Converted coords (y, x): {points_yx}")
        print(f"  Image size: {img_width}x{img_height}")

        # テンソルに変換
        points_tensor = torch.tensor(points_yx, dtype=torch.float32)
        labels_tensor = torch.tensor(point_labels, dtype=torch.int32)

        # handle_requestを使用（新API）
        response = self.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=self._session_id,
                frame_index=frame_index,
                points=points_tensor,
                point_labels=labels_tensor,
                obj_id=object_id,
            )
        )

        # デバッグ: レスポンスの内容を確認
        print(f"add_prompt response keys: {response.keys() if response else 'None'}")
        if response:
            for key, value in response.items():
                if key == "outputs":
                    print(f"  outputs type: {type(value)}")
                    if isinstance(value, dict):
                        for k, v in value.items():
                            print(f"    outputs[{k}] type: {type(v)}, shape: {v.shape if hasattr(v, 'shape') else 'N/A'}")
                else:
                    print(f"  {key}: {type(value)}")

        # 結果からマスクを取得
        mask = None
        if response and "outputs" in response:
            outputs = response["outputs"]
            if isinstance(outputs, dict):
                # デバッグ: out_probs を確認
                if "out_probs" in outputs:
                    print(f"  out_probs: {outputs['out_probs']}")

                # 新API形式: out_binary_masks にマスクが格納される
                if "out_binary_masks" in outputs:
                    binary_masks = outputs["out_binary_masks"]
                    if binary_masks is not None and len(binary_masks) > 0:
                        # shape: (num_objects, height, width) -> 最初のマスクを取得
                        mask = binary_masks[0]
                        total_pixels = mask.size
                        mask_sum = mask.sum()
                        print(f"Extracted mask shape: {mask.shape}")
                        print(f"  Mask sum: {mask_sum} / {total_pixels} ({100*mask_sum/total_pixels:.1f}%)")
                        print(f"  Mask unique values: {np.unique(mask)}")

                        # マスクが半分以上の場合は反転が必要かもしれない
                        if mask_sum > total_pixels * 0.5:
                            print("  WARNING: Mask covers >50% of image, may be inverted")
                else:
                    # 旧API形式: obj_id をキーとする辞書
                    obj_output = outputs.get(object_id)
                    if obj_output is None and len(outputs) > 0:
                        first_key = list(outputs.keys())[0]
                        obj_output = outputs[first_key]

                    if obj_output is not None:
                        if isinstance(obj_output, np.ndarray):
                            mask = obj_output
                        elif hasattr(obj_output, 'cpu'):
                            mask = obj_output.cpu().numpy()
                        elif isinstance(obj_output, dict):
                            mask = obj_output.get("mask")
                            if mask is not None and hasattr(mask, 'cpu'):
                                mask = mask.cpu().numpy()

        return {
            "frame_index": frame_index,
            "object_id": object_id,
            "mask": mask,
            "raw_result": response
        }

    def add_text_prompt(
        self,
        frame_index: int,
        text: str,
        object_id: int = 0
    ) -> Dict:
        """
        テキストプロンプトを追加

        Args:
            frame_index: 対象フレーム（-1で全フレーム自動検出）
            text: テキスト記述（例: "yellow chair"）
            object_id: オブジェクトID

        Returns:
            初期フレームのセグメント結果
        """
        if self._session_id is None:
            raise RuntimeError("Session not started. Call start_session() first.")

        # テキストプロンプトを追加 (handle_request形式)
        response = self.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=self._session_id,
                frame_index=frame_index,
                text=text,
            )
        )

        # 結果からマスクを取得
        mask = None
        if response and "outputs" in response:
            outputs = response["outputs"]
            if isinstance(outputs, dict):
                # 最初のオブジェクトのマスクを取得
                for obj_id, obj_data in outputs.items():
                    if "mask" in obj_data:
                        mask = obj_data["mask"]
                        if hasattr(mask, 'cpu'):
                            mask = mask.cpu().numpy()
                        break

        return {
            "frame_index": frame_index,
            "object_id": object_id,
            "text": text,
            "mask": mask,
            "raw_result": response
        }

    def propagate(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        reverse: bool = False
    ) -> List[TrackingResult]:
        """
        マスクを他のフレームに伝播

        Args:
            start_frame: 開始フレーム
            end_frame: 終了フレーム（None=最後まで）
            reverse: 逆方向に伝播するか

        Returns:
            各フレームのトラッキング結果
        """
        if self._session_id is None:
            raise RuntimeError("Session not started. Call start_session() first.")

        results = []

        # 新API: propagation_direction = "backward" or "forward"
        propagation_direction = "backward" if reverse else "forward"

        # ビデオを伝播 (新API)
        propagation_result = self.predictor.propagate_in_video(
            session_id=self._session_id,
            propagation_direction=propagation_direction,
            start_frame_idx=start_frame,
            max_frame_num_to_track=end_frame
        )

        # 結果を処理（新APIの戻り値形式に対応）
        if propagation_result:
            if isinstance(propagation_result, dict):
                # 辞書形式の場合
                for frame_idx, frame_data in propagation_result.items():
                    if isinstance(frame_data, dict) and "masks" in frame_data:
                        for obj_id, mask in enumerate(frame_data["masks"]):
                            results.append(TrackingResult(
                                frame_index=int(frame_idx),
                                mask=mask,
                                score=1.0,
                                object_id=obj_id
                            ))
            elif hasattr(propagation_result, '__iter__'):
                # イテレータ形式の場合（旧API互換）
                for item in propagation_result:
                    if isinstance(item, tuple) and len(item) >= 3:
                        frame_idx, out_obj_ids, out_mask_logits = item[:3]
                        for obj_id, mask_logits in zip(out_obj_ids, out_mask_logits):
                            mask = (mask_logits > 0).cpu().numpy().squeeze()
                            score = float(torch.sigmoid(mask_logits).max().cpu().numpy())
                            results.append(TrackingResult(
                                frame_index=frame_idx,
                                mask=mask,
                                score=score,
                                object_id=obj_id
                            ))
                    if isinstance(item, tuple) and len(item) >= 1:
                        frame_idx = item[0] if isinstance(item[0], int) else 0
                        if frame_idx % 50 == 0:
                            print(f"  Propagated to frame {frame_idx}")

        return results

    def track_object(
        self,
        video_path: str,
        prompt_type: str = "click",
        prompt_frame: int = 0,
        point_coords: Optional[List[Tuple[int, int]]] = None,
        text: Optional[str] = None,
        propagate_forward: bool = True,
        propagate_backward: bool = True
    ) -> List[TrackingResult]:
        """
        オブジェクトを追跡（セッション開始からトラッキングまで一括実行）

        Args:
            video_path: 動画ファイルパス
            prompt_type: "click" or "text"
            prompt_frame: プロンプトを追加するフレーム
            point_coords: クリック座標（prompt_type="click"の場合）
            text: テキスト記述（prompt_type="text"の場合）
            propagate_forward: 前方向に伝播
            propagate_backward: 後方向に伝播

        Returns:
            全フレームのトラッキング結果
        """
        # セッション開始
        self.start_session(video_path)

        # プロンプト追加
        if prompt_type == "click":
            if point_coords is None:
                raise ValueError("point_coords is required for click prompt")
            self.add_click_prompt(prompt_frame, point_coords)
        elif prompt_type == "text":
            if text is None:
                raise ValueError("text is required for text prompt")
            self.add_text_prompt(prompt_frame, text)
        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")

        # 伝播
        all_results = []

        if propagate_forward:
            print("Propagating forward...")
            forward_results = self.propagate(start_frame=prompt_frame, reverse=False)
            all_results.extend(forward_results)

        if propagate_backward and prompt_frame > 0:
            print("Propagating backward...")
            backward_results = self.propagate(start_frame=prompt_frame, reverse=True)
            all_results.extend(backward_results)

        # フレーム順にソート
        all_results.sort(key=lambda r: r.frame_index)

        return all_results

    def save_masks(
        self,
        results: List[TrackingResult],
        output_dir: str,
        save_png: bool = True,
        save_npy: bool = True
    ):
        """
        マスクをファイルに保存

        Args:
            results: トラッキング結果
            output_dir: 出力ディレクトリ
            save_png: PNG画像として保存
            save_npy: NumPy配列として保存
        """
        from PIL import Image

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for result in results:
            base_name = f"mask_{result.frame_index:06d}_obj{result.object_id}"

            if save_png:
                mask_img = Image.fromarray((result.mask * 255).astype(np.uint8))
                mask_img.save(output_path / f"{base_name}.png")

            if save_npy:
                np.save(output_path / f"{base_name}.npy", result.mask)

        print(f"Saved {len(results)} masks to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="SAM 3ビデオトラッキング",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # クリックプロンプトで追跡
  python -m server.multiview.sam3_video_tracker video.mp4 --click 512,384

  # テキストプロンプトで追跡
  python -m server.multiview.sam3_video_tracker video.mp4 --text "椅子"

  # Omniscientセッションから実行
  python -m server.multiview.sam3_video_tracker experiments/omniscient_sample/003 --text "chair"

注意: SAM 3はDockerコンテナ内で実行する必要があります
        """
    )

    parser.add_argument("video_path", help="動画ファイルまたはOmniscientセッションディレクトリ")
    parser.add_argument("--click", "-c", help="クリック座標 (x,y)")
    parser.add_argument("--text", "-t", help="テキストプロンプト")
    parser.add_argument("--frame", "-f", type=int, default=0, help="プロンプトフレーム")
    parser.add_argument("--output", "-o", help="マスク出力ディレクトリ")
    parser.add_argument("--device", "-d", default="cuda", help="デバイス (cuda/cpu)")

    args = parser.parse_args()

    if not HAS_SAM3:
        print("Error: SAM 3 is not installed.")
        print("Please run inside Docker container:")
        print("  docker run --gpus all --ipc=host --network=host \\")
        print("    -v ~/SAM3D-LiDAR-fz:/workspace \\")
        print("    -it lidar-llm-mcp:sam3-tested")
        return

    # トラッカーを初期化
    tracker = SAM3VideoTracker(device=args.device)

    # プロンプトを設定
    prompt_type = None
    point_coords = None
    text = None

    if args.click:
        prompt_type = "click"
        x, y = map(int, args.click.split(","))
        point_coords = [(x, y)]
    elif args.text:
        prompt_type = "text"
        text = args.text
    else:
        print("Error: Either --click or --text is required")
        return

    # トラッキング実行
    print(f"=== SAM 3 Video Tracking ===")
    print(f"Video: {args.video_path}")
    print(f"Prompt: {prompt_type} ({point_coords or text})")
    print(f"Frame: {args.frame}")

    results = tracker.track_object(
        video_path=args.video_path,
        prompt_type=prompt_type,
        prompt_frame=args.frame,
        point_coords=point_coords,
        text=text
    )

    print(f"\nTracked {len(results)} frames")

    # マスク保存
    if args.output:
        tracker.save_masks(results, args.output)
    else:
        # デフォルト出力先
        video_path = Path(args.video_path)
        if video_path.is_dir():
            output_dir = video_path / "output" / "masks"
        else:
            output_dir = video_path.parent / "output" / "masks"
        tracker.save_masks(results, str(output_dir))


if __name__ == "__main__":
    main()
