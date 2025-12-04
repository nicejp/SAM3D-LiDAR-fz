#!/usr/bin/env python3
"""
Point Cloud Viewer
点群データをOpen3Dで可視化

使い方:
    # PLYファイルを表示
    python -m server.visualization.pointcloud_viewer path/to/pointcloud.ply

    # セッションディレクトリを表示
    python -m server.visualization.pointcloud_viewer --session path/to/session_dir

    # リアルタイム可視化（WebSocket受信と同時表示）
    python -m server.visualization.pointcloud_viewer --realtime
"""

from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Any, TYPE_CHECKING
import threading
import time

# Open3Dは遅延インポート
HAS_OPEN3D = False
o3d: Any = None

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    pass


def load_ply(filepath: str) -> Optional[Any]:
    """PLYファイルを読み込み"""
    if not HAS_OPEN3D:
        return None

    pcd = o3d.io.read_point_cloud(filepath)
    if len(pcd.points) == 0:
        print(f"Warning: Empty point cloud: {filepath}")
        return None
    return pcd


def view_pointcloud(pcd: Any, window_name: str = "Point Cloud Viewer"):
    """点群を表示（ブロッキング）"""
    if not HAS_OPEN3D:
        print("Error: open3d is required for visualization")
        return

    # 座標軸を追加
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=[0, 0, 0]
    )

    # 点群の情報を表示
    print(f"Points: {len(pcd.points)}")
    if len(pcd.points) > 0:
        points = np.asarray(pcd.points)
        print(f"Bounds: {points.min(axis=0)} to {points.max(axis=0)}")
        print(f"Center: {points.mean(axis=0)}")

    # 可視化
    o3d.visualization.draw_geometries(
        [pcd, coord_frame],
        window_name=window_name,
        width=1280,
        height=720,
        point_show_normal=False
    )


def view_session(session_dir: str, merged_only: bool = True):
    """セッションディレクトリの点群を表示"""
    session_path = Path(session_dir)

    # merged.plyを優先
    merged_path = session_path / "output" / "pointcloud" / "merged.ply"
    if not merged_path.exists():
        merged_path = session_path / "pointcloud" / "merged.ply"

    if merged_path.exists() and merged_only:
        print(f"Loading merged point cloud: {merged_path}")
        pcd = load_ply(str(merged_path))
        if pcd:
            view_pointcloud(pcd, f"Session: {session_path.name}")
        return

    # 個別フレームを読み込み
    pc_dir = session_path / "output" / "pointcloud"
    if not pc_dir.exists():
        pc_dir = session_path / "pointcloud"

    if not pc_dir.exists():
        print(f"No pointcloud directory found in {session_dir}")
        return

    ply_files = sorted(pc_dir.glob("frame_*.ply"))
    if not ply_files:
        print(f"No PLY files found in {pc_dir}")
        return

    print(f"Found {len(ply_files)} frames")

    # 全フレームを統合
    all_points = []
    for ply_file in ply_files:
        pcd = load_ply(str(ply_file))
        if pcd:
            all_points.append(np.asarray(pcd.points))

    if all_points:
        merged_points = np.vstack(all_points)
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
        merged_pcd.paint_uniform_color([0.5, 0.5, 1.0])  # Light blue
        view_pointcloud(merged_pcd, f"Session: {session_path.name} ({len(ply_files)} frames)")


class RealtimeViewer:
    """リアルタイム点群ビューア（WebSocketと連携）"""

    def __init__(self):
        if not HAS_OPEN3D:
            raise RuntimeError("open3d is required")

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Realtime Point Cloud", width=1280, height=720)

        # 初期の空の点群
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        self.vis.add_geometry(self.pcd)

        # 座標軸
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        self.vis.add_geometry(self.coord_frame)

        self.running = True
        self.points_buffer = []
        self.lock = threading.Lock()

    def add_points(self, points: np.ndarray):
        """点群を追加"""
        with self.lock:
            self.points_buffer.append(points)

    def update(self):
        """ビューを更新"""
        with self.lock:
            if self.points_buffer:
                # 新しい点群を統合
                all_points = np.vstack(self.points_buffer) if self.points_buffer else np.zeros((1, 3))
                self.pcd.points = o3d.utility.Vector3dVector(all_points)
                self.pcd.paint_uniform_color([0.3, 0.7, 0.3])  # Green
                self.vis.update_geometry(self.pcd)

        self.vis.poll_events()
        self.vis.update_renderer()
        return self.running

    def run(self):
        """メインループ"""
        while self.running:
            if not self.update():
                break
            time.sleep(0.03)  # ~30 FPS

        self.vis.destroy_window()

    def stop(self):
        """停止"""
        self.running = False


def view_ply_file(filepath: str):
    """PLYファイルを直接表示"""
    if not Path(filepath).exists():
        print(f"File not found: {filepath}")
        return

    pcd = load_ply(filepath)
    if pcd:
        view_pointcloud(pcd, f"PLY: {Path(filepath).name}")


def print_stats(filepath: str):
    """点群の統計情報を表示（GUI不要）"""
    path = Path(filepath)

    if path.is_dir():
        # セッションディレクトリ
        merged_path = path / "output" / "pointcloud" / "merged.ply"
        if not merged_path.exists():
            merged_path = path / "pointcloud" / "merged.ply"
        filepath = str(merged_path)

    if not Path(filepath).exists():
        print(f"File not found: {filepath}")
        return

    # NumPyだけで読み込み（open3d不要）
    points = []
    with open(filepath, 'r') as f:
        in_header = True
        for line in f:
            if in_header:
                if line.strip() == "end_header":
                    in_header = False
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])

    if not points:
        print("No points found")
        return

    points = np.array(points)

    print(f"\n{'='*50}")
    print(f"Point Cloud Statistics: {Path(filepath).name}")
    print(f"{'='*50}")
    print(f"Total points: {len(points):,}")
    print(f"\nBounds:")
    print(f"  X: {points[:, 0].min():.3f} to {points[:, 0].max():.3f} (range: {points[:, 0].ptp():.3f}m)")
    print(f"  Y: {points[:, 1].min():.3f} to {points[:, 1].max():.3f} (range: {points[:, 1].ptp():.3f}m)")
    print(f"  Z: {points[:, 2].min():.3f} to {points[:, 2].max():.3f} (range: {points[:, 2].ptp():.3f}m)")
    print(f"\nCenter: ({points[:, 0].mean():.3f}, {points[:, 1].mean():.3f}, {points[:, 2].mean():.3f})")
    print(f"\nDimensions: {points[:, 0].ptp():.2f}m x {points[:, 1].ptp():.2f}m x {points[:, 2].ptp():.2f}m")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="点群可視化ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # PLYファイルを表示
  python -m server.visualization.pointcloud_viewer experiment/session_xxx/output/pointcloud/merged.ply

  # セッションディレクトリを表示
  python -m server.visualization.pointcloud_viewer --session experiment/session_xxx

  # 統計情報のみ表示（GUI不要）
  python -m server.visualization.pointcloud_viewer --stats experiment/session_xxx
        """
    )

    parser.add_argument("path", nargs="?", help="PLYファイルまたはセッションディレクトリのパス")
    parser.add_argument("--session", "-s", help="セッションディレクトリのパス")
    parser.add_argument("--stats", action="store_true", help="統計情報のみ表示（GUI不要）")
    parser.add_argument("--all-frames", action="store_true", help="全フレームを個別に読み込み")

    args = parser.parse_args()

    if args.stats:
        # 統計情報のみ
        target = args.path or args.session
        if target:
            print_stats(target)
        else:
            parser.print_help()
        return

    if not HAS_OPEN3D:
        print("Error: open3d is required for visualization")
        print("Install: pip install open3d")
        return

    if args.session:
        view_session(args.session, merged_only=not args.all_frames)
    elif args.path:
        path = Path(args.path)
        if path.is_dir():
            view_session(str(path), merged_only=not args.all_frames)
        elif path.suffix == ".ply":
            view_ply_file(str(path))
        else:
            print(f"Unknown file type: {path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
