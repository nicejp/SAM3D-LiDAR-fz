#!/usr/bin/env python3
"""
WebSocket Server for receiving LiDAR + RGB data from iPad
iPad DisasterScanner アプリからのデータを受信するWebSocketサーバー
"""

import asyncio
import json
import base64
import os
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import websockets
except ImportError:
    print("websocketsをインストールしてください: pip install websockets")
    exit(1)

try:
    from PIL import Image
    import io
except ImportError:
    print("Pillowをインストールしてください: pip install Pillow")
    exit(1)


class DataReceiver:
    """iPadからのデータを受信・保存するクラス"""

    def __init__(self, save_dir: str = "~/SAM3D-LiDAR-fz/experiments"):
        self.save_dir = Path(save_dir).expanduser()
        self.session_dir = None
        self.frame_count = 0

    def start_session(self) -> Path:
        """新しいセッションを開始"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.save_dir / f"session_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "rgb").mkdir(exist_ok=True)
        (self.session_dir / "depth").mkdir(exist_ok=True)
        (self.session_dir / "camera").mkdir(exist_ok=True)
        self.frame_count = 0
        print(f"セッション開始: {self.session_dir}")
        return self.session_dir

    def save_frame(self, data: dict) -> bool:
        """1フレームのデータを保存"""
        if self.session_dir is None:
            self.start_session()

        try:
            packet_type = data.get("type", "unknown")
            frame_id = data.get("frame_id", self.frame_count)

            # mesh_verticesパケット: リアルタイム表示用（ログのみ）
            if packet_type == "mesh_vertices":
                vertex_count = data.get("vertex_count", 0)
                total_vertices = data.get("total_vertices", 0)
                print(f"  メッシュ頂点受信: {vertex_count}/{total_vertices} 頂点")
                return True

            # frame_dataパケット: RGB/深度/カメラパラメータを保存
            if packet_type == "frame_data":
                # RGB画像を保存 (JPEG)
                if "rgb" in data and data["rgb"]:
                    rgb_bytes = base64.b64decode(data["rgb"])
                    rgb_path = self.session_dir / "rgb" / f"frame_{frame_id:06d}.jpg"
                    with open(rgb_path, "wb") as f:
                        f.write(rgb_bytes)
                    print(f"  RGB画像保存: {rgb_path.name}")

                # 深度マップを保存 (NumPy)
                if "depth" in data and data["depth"]:
                    depth_bytes = base64.b64decode(data["depth"])
                    width = data.get("depth_width", 256)
                    height = data.get("depth_height", 192)
                    depth_array = np.frombuffer(depth_bytes, dtype=np.float32)
                    depth_array = depth_array.reshape((height, width))
                    depth_path = self.session_dir / "depth" / f"frame_{frame_id:06d}.npy"
                    np.save(depth_path, depth_array)
                    print(f"  深度マップ保存: {depth_path.name}")

                # カメラパラメータを保存 (JSON)
                camera_data = {
                    "frame_id": frame_id,
                    "timestamp": data.get("timestamp", 0),
                    "intrinsics": data.get("intrinsics", {}),
                    "transform": data.get("transform", []),
                    "camera_position": data.get("camera_position", [])
                }
                camera_path = self.session_dir / "camera" / f"frame_{frame_id:06d}.json"
                with open(camera_path, "w") as f:
                    json.dump(camera_data, f, indent=2)

                self.frame_count += 1
                return True

            # 旧形式のパケット（後方互換性）
            if "rgb" in data or "depth" in data:
                # RGB画像を保存 (JPEG)
                if "rgb" in data and data["rgb"]:
                    rgb_bytes = base64.b64decode(data["rgb"])
                    rgb_path = self.session_dir / "rgb" / f"frame_{frame_id:06d}.jpg"
                    with open(rgb_path, "wb") as f:
                        f.write(rgb_bytes)

                # 深度マップを保存 (NumPy)
                if "depth" in data and data["depth"]:
                    depth_bytes = base64.b64decode(data["depth"])
                    width = data.get("depth_width", 256)
                    height = data.get("depth_height", 192)
                    depth_array = np.frombuffer(depth_bytes, dtype=np.float32)
                    depth_array = depth_array.reshape((height, width))
                    depth_path = self.session_dir / "depth" / f"frame_{frame_id:06d}.npy"
                    np.save(depth_path, depth_array)

                # カメラパラメータを保存 (JSON)
                camera_data = {
                    "frame_id": frame_id,
                    "timestamp": data.get("timestamp", 0),
                    "intrinsics": data.get("intrinsics", {}),
                    "transform": data.get("transform", [])
                }
                camera_path = self.session_dir / "camera" / f"frame_{frame_id:06d}.json"
                with open(camera_path, "w") as f:
                    json.dump(camera_data, f, indent=2)

                self.frame_count += 1
                return True

            print(f"  不明なパケットタイプ: {packet_type}")
            return True

        except Exception as e:
            print(f"フレーム保存エラー: {e}")
            return False

    def end_session(self) -> dict:
        """セッションを終了してメタデータを保存"""
        if self.session_dir is None:
            return {}

        metadata = {
            "session_dir": str(self.session_dir),
            "total_frames": self.frame_count,
            "end_time": datetime.now().isoformat()
        }

        metadata_path = self.session_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"セッション終了: {self.frame_count} フレーム保存")
        return metadata


# グローバルなレシーバーインスタンス
receiver = DataReceiver()


async def handle_client(websocket):
    """クライアント接続を処理"""
    client_addr = websocket.remote_address
    print(f"クライアント接続: {client_addr}")

    # 接続確認メッセージを送信
    await websocket.send(json.dumps({"status": "connected", "message": "DGX Spark Server Ready"}))

    receiver.start_session()

    try:
        async for message in websocket:
            try:
                data = json.loads(message)

                # フレームデータを保存
                if receiver.save_frame(data):
                    frame_id = data.get("frame_id", receiver.frame_count)
                    print(f"  フレーム {frame_id} 受信・保存完了")

                    # 確認応答を送信
                    await websocket.send(json.dumps({
                        "status": "ok",
                        "frame_id": frame_id
                    }))
                else:
                    await websocket.send(json.dumps({
                        "status": "error",
                        "message": "Frame save failed"
                    }))

            except json.JSONDecodeError as e:
                print(f"JSONパースエラー: {e}")
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": "Invalid JSON"
                }))

    except websockets.exceptions.ConnectionClosed:
        print(f"クライアント切断: {client_addr}")
    finally:
        metadata = receiver.end_session()
        print(f"セッション完了: {metadata}")


async def main(host: str = "0.0.0.0", port: int = 8765):
    """WebSocketサーバーを起動"""
    print(f"=" * 50)
    print(f"SAM3D-LiDAR-fz WebSocket Server")
    print(f"=" * 50)
    print(f"ホスト: {host}")
    print(f"ポート: {port}")
    print(f"保存先: {receiver.save_dir}")
    print(f"=" * 50)
    print(f"iPadアプリから接続してください...")
    print(f"")

    async with websockets.serve(handle_client, host, port):
        await asyncio.Future()  # 永久に実行


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="iPad LiDAR データ受信サーバー")
    parser.add_argument("--host", default="0.0.0.0", help="ホストアドレス (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="ポート番号 (default: 8765)")
    args = parser.parse_args()

    try:
        asyncio.run(main(args.host, args.port))
    except KeyboardInterrupt:
        print("\nサーバー停止")
