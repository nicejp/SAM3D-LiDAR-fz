#!/usr/bin/env python3
"""
Realtime Point Cloud Viewer
iPadからのデータをリアルタイムで点群として可視化

使い方:
    # WebSocketサーバー + リアルタイム可視化を同時起動
    python -m server.visualization.realtime_viewer

    # ポート指定
    python -m server.visualization.realtime_viewer --port 8765
"""

import asyncio
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
import threading
import queue
import time
import io

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Error: open3d is required. Run: pip install open3d")

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    print("Error: websockets is required. Run: pip install websockets")

from PIL import Image


class RealtimePointCloudViewer:
    """リアルタイム点群ビューア"""

    def __init__(self, port: int = 8765, max_points: int = 500000):
        self.port = port
        self.max_points = max_points

        # 点群データ
        self.all_points = []
        self.all_colors = []
        self.point_queue = queue.Queue()

        # 統計
        self.frame_count = 0
        self.total_points = 0

        # 制御
        self.running = True
        self.websocket_thread = None

    def depth_to_points(
        self,
        depth_data: np.ndarray,
        intrinsics: dict,
        rgb_data: Optional[np.ndarray] = None
    ) -> tuple:
        """深度マップから点群を生成"""
        height, width = depth_data.shape

        fx = intrinsics.get("fx", 500.0)
        fy = intrinsics.get("fy", 500.0)
        cx = intrinsics.get("cx", width / 2)
        cy = intrinsics.get("cy", height / 2)

        # ピクセルグリッド
        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v)

        z = depth_data
        valid = (z > 0.1) & (z < 5.0) & np.isfinite(z)

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points = np.stack([x[valid], y[valid], z[valid]], axis=-1)

        # 色
        colors = None
        if rgb_data is not None:
            if rgb_data.shape[:2] != depth_data.shape:
                rgb_pil = Image.fromarray(rgb_data)
                rgb_pil = rgb_pil.resize((width, height), Image.BILINEAR)
                rgb_data = np.array(rgb_pil)
            colors = rgb_data[valid] / 255.0

        return points, colors

    async def handle_client(self, websocket):
        """WebSocketクライアントを処理"""
        client_addr = websocket.remote_address
        print(f"[WS] Client connected: {client_addr}")

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    await self.process_frame(message)
                else:
                    # JSONメッセージ
                    try:
                        data = json.loads(message)
                        if data.get("type") == "end_session":
                            print(f"[WS] Session ended")
                    except json.JSONDecodeError:
                        pass

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            print(f"[WS] Client disconnected: {client_addr}")

    async def process_frame(self, data: bytes):
        """フレームデータを処理"""
        try:
            # バイナリフォーマット:
            # [4 bytes: header_size][header_json][rgb_jpeg][depth_float32]
            header_size = int.from_bytes(data[:4], 'little')
            header_json = data[4:4+header_size].decode('utf-8')
            header = json.loads(header_json)

            rgb_size = header.get("rgb_size", 0)
            depth_size = header.get("depth_size", 0)

            offset = 4 + header_size
            rgb_data = data[offset:offset+rgb_size]
            offset += rgb_size
            depth_data = data[offset:offset+depth_size]

            # RGB画像をデコード
            rgb_image = np.array(Image.open(io.BytesIO(rgb_data)))

            # 深度マップをデコード
            depth_shape = (header.get("depth_height", 192), header.get("depth_width", 256))
            depth_map = np.frombuffer(depth_data, dtype=np.float32).reshape(depth_shape)

            # カメラパラメータ
            intrinsics = header.get("intrinsics", {})

            # 点群に変換
            points, colors = self.depth_to_points(depth_map, intrinsics, rgb_image)

            if len(points) > 0:
                # サブサンプリング（リアルタイム用に間引き）
                if len(points) > 5000:
                    indices = np.random.choice(len(points), 5000, replace=False)
                    points = points[indices]
                    if colors is not None:
                        colors = colors[indices]

                # キューに追加
                self.point_queue.put((points, colors))
                self.frame_count += 1

                if self.frame_count % 10 == 0:
                    print(f"[Frame {self.frame_count}] +{len(points)} points, Total: {self.total_points}")

        except Exception as e:
            print(f"[Error] Frame processing failed: {e}")

    async def run_websocket_server(self):
        """WebSocketサーバーを実行"""
        print(f"[WS] Starting server on 0.0.0.0:{self.port}")

        async with websockets.serve(
            self.handle_client,
            "0.0.0.0",
            self.port,
            max_size=50 * 1024 * 1024  # 50MB
        ):
            while self.running:
                await asyncio.sleep(0.1)

    def websocket_thread_func(self):
        """WebSocketスレッド"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.run_websocket_server())
        except Exception as e:
            print(f"[WS] Server error: {e}")
        finally:
            loop.close()

    def run_visualization(self):
        """Open3D可視化を実行（メインスレッド）"""
        if not HAS_OPEN3D:
            print("Error: Open3D is required for visualization")
            return

        # ビジュアライザー作成
        vis = o3d.visualization.Visualizer()
        vis.create_window("Realtime Point Cloud", width=1280, height=720)

        # 初期点群（空）
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        pcd.colors = o3d.utility.Vector3dVector(np.array([[0.5, 0.5, 0.5]]))
        vis.add_geometry(pcd)

        # 座標軸
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        vis.add_geometry(coord_frame)

        # ビュー設定
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.1, 0.1, 0.1])

        print("\n" + "=" * 50)
        print("Realtime Point Cloud Viewer")
        print("=" * 50)
        print(f"WebSocket: ws://0.0.0.0:{self.port}")
        print("Waiting for iPad connection...")
        print("Press Q or close window to exit")
        print("=" * 50 + "\n")

        last_update = time.time()
        update_interval = 0.1  # 100ms

        while self.running:
            # 新しい点群をチェック
            new_points = []
            new_colors = []

            while not self.point_queue.empty():
                try:
                    points, colors = self.point_queue.get_nowait()
                    new_points.append(points)
                    if colors is not None:
                        new_colors.append(colors)
                except queue.Empty:
                    break

            # 点群を更新
            if new_points and (time.time() - last_update) > update_interval:
                # 新しい点を追加
                for i, pts in enumerate(new_points):
                    self.all_points.append(pts)
                    if i < len(new_colors):
                        self.all_colors.append(new_colors[i])

                # 最大点数を制限
                all_pts = np.vstack(self.all_points) if self.all_points else np.zeros((1, 3))
                if len(all_pts) > self.max_points:
                    # 古い点を削除
                    all_pts = all_pts[-self.max_points:]
                    self.all_points = [all_pts]

                    if self.all_colors:
                        all_cols = np.vstack(self.all_colors)[-self.max_points:]
                        self.all_colors = [all_cols]

                self.total_points = len(all_pts)

                # 点群を更新
                pcd.points = o3d.utility.Vector3dVector(all_pts)

                if self.all_colors:
                    all_cols = np.vstack(self.all_colors)
                    if len(all_cols) == len(all_pts):
                        pcd.colors = o3d.utility.Vector3dVector(all_cols)

                vis.update_geometry(pcd)
                last_update = time.time()

            # イベント処理
            if not vis.poll_events():
                self.running = False
                break

            vis.update_renderer()
            time.sleep(0.01)

        vis.destroy_window()
        print("\nViewer closed")

    def run(self):
        """メイン実行"""
        if not HAS_OPEN3D or not HAS_WEBSOCKETS:
            print("Error: Required libraries not installed")
            print("  pip install open3d websockets")
            return

        # WebSocketサーバーをバックグラウンドで起動
        self.websocket_thread = threading.Thread(
            target=self.websocket_thread_func,
            daemon=True
        )
        self.websocket_thread.start()

        # メインスレッドで可視化
        try:
            self.run_visualization()
        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            self.running = False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="iPadからのデータをリアルタイムで点群可視化"
    )
    parser.add_argument("--port", "-p", type=int, default=8765, help="WebSocketポート")
    parser.add_argument("--max-points", type=int, default=500000, help="最大点数")

    args = parser.parse_args()

    viewer = RealtimePointCloudViewer(
        port=args.port,
        max_points=args.max_points
    )
    viewer.run()


if __name__ == "__main__":
    main()
