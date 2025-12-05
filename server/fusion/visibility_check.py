#!/usr/bin/env python3
"""
可視判定モジュール

カメラから見える頂点を判定する。
LiDARは「オモテ面」のデータしか持たないため、
SAM 3Dメッシュのどの頂点がカメラから見えるかを判定する。

処理フロー:
1. カメラ位置からメッシュの各頂点へのレイを作成
2. レイキャストで遮蔽判定
3. 見える頂点のマスクを返す

使い方:
    from server.fusion.visibility_check import VisibilityChecker

    checker = VisibilityChecker()
    visible_mask = checker.compute_visibility(mesh, camera_positions)
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Open3D is required. Install with: pip install open3d")

try:
    import trimesh
except ImportError:
    trimesh = None


class VisibilityChecker:
    """可視判定クラス"""

    def __init__(self, use_trimesh: bool = True):
        """
        Args:
            use_trimesh: trimeshを使用するか（高速なレイキャスト）
        """
        self.use_trimesh = use_trimesh and trimesh is not None
        if self.use_trimesh:
            print("Using trimesh for ray casting")
        else:
            print("Using Open3D for ray casting")

    def load_mesh(self, path: str) -> o3d.geometry.TriangleMesh:
        """メッシュファイルを読み込む"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        mesh = o3d.io.read_triangle_mesh(str(path))

        if len(mesh.vertices) == 0:
            raise ValueError(f"Empty mesh: {path}")

        # 法線を計算
        mesh.compute_vertex_normals()

        return mesh

    def compute_visibility_open3d(
        self,
        mesh: o3d.geometry.TriangleMesh,
        camera_positions: np.ndarray,
        offset: float = 0.001
    ) -> np.ndarray:
        """
        Open3Dを使用して可視判定（やや遅い）

        Args:
            mesh: 入力メッシュ
            camera_positions: カメラ位置の配列 (N, 3) または (3,)
            offset: レイ開始点のオフセット（自己交差回避）

        Returns:
            visible_mask: 各頂点の可視性 (num_vertices,)
        """
        vertices = np.asarray(mesh.vertices)
        num_vertices = len(vertices)

        # カメラ位置が1つの場合は配列に変換
        if camera_positions.ndim == 1:
            camera_positions = camera_positions.reshape(1, -1)

        # RaycastingSceneを作成
        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(mesh_t)

        # 各カメラ位置からの可視性を計算
        visible_count = np.zeros(num_vertices, dtype=int)

        for cam_pos in camera_positions:
            # 各頂点へのレイを作成
            directions = vertices - cam_pos
            distances = np.linalg.norm(directions, axis=1, keepdims=True)
            directions = directions / (distances + 1e-6)

            # レイの開始点を少しオフセット
            origins = cam_pos + directions * offset

            # テンソルに変換
            rays = np.concatenate([origins, directions], axis=1).astype(np.float32)
            rays_t = o3d.core.Tensor(rays)

            # レイキャスト
            result = scene.cast_rays(rays_t)
            t_hit = result['t_hit'].numpy()

            # 頂点までの距離より手前で交差した場合は遮蔽されている
            visible = t_hit >= (distances.flatten() - offset * 2)
            visible_count += visible.astype(int)

        # 1つ以上のカメラから見える場合は可視
        visible_mask = visible_count > 0

        return visible_mask

    def compute_visibility_trimesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        camera_positions: np.ndarray,
        offset: float = 0.001
    ) -> np.ndarray:
        """
        trimeshを使用して可視判定（高速）

        Args:
            mesh: 入力メッシュ (Open3D形式)
            camera_positions: カメラ位置の配列 (N, 3) または (3,)
            offset: レイ開始点のオフセット

        Returns:
            visible_mask: 各頂点の可視性 (num_vertices,)
        """
        if trimesh is None:
            raise ImportError("trimesh is required. Install with: pip install trimesh")

        # Open3DからtrimeshへMesh変換
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        tm_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

        num_vertices = len(vertices)

        # カメラ位置が1つの場合は配列に変換
        if camera_positions.ndim == 1:
            camera_positions = camera_positions.reshape(1, -1)

        # 各カメラ位置からの可視性を計算
        visible_count = np.zeros(num_vertices, dtype=int)

        for cam_pos in camera_positions:
            # 各頂点へのレイを作成
            directions = vertices - cam_pos
            distances = np.linalg.norm(directions, axis=1)
            directions = directions / (distances[:, np.newaxis] + 1e-6)

            # レイの開始点を少しオフセット
            origins = cam_pos + directions * offset

            # レイキャスト
            locations, index_ray, index_tri = tm_mesh.ray.intersects_location(
                ray_origins=origins,
                ray_directions=directions,
                multiple_hits=False
            )

            # 交差した頂点を特定
            if len(index_ray) > 0:
                # 交差点までの距離を計算
                hit_distances = np.linalg.norm(locations - origins[index_ray], axis=1)

                # 頂点までの距離より手前で交差した場合は遮蔽
                occluded = hit_distances < (distances[index_ray] - offset * 2)

                # 遮蔽されていない頂点を可視としてカウント
                visible = np.ones(num_vertices, dtype=bool)
                visible[index_ray[occluded]] = False
            else:
                visible = np.ones(num_vertices, dtype=bool)

            visible_count += visible.astype(int)

        # 1つ以上のカメラから見える場合は可視
        visible_mask = visible_count > 0

        return visible_mask

    def compute_visibility(
        self,
        mesh: Union[str, o3d.geometry.TriangleMesh],
        camera_positions: np.ndarray,
        offset: float = 0.001
    ) -> np.ndarray:
        """
        可視判定を実行

        Args:
            mesh: メッシュファイルパスまたはOpen3Dメッシュ
            camera_positions: カメラ位置の配列 (N, 3) または (3,)
            offset: レイ開始点のオフセット

        Returns:
            visible_mask: 各頂点の可視性 (num_vertices,)
        """
        if isinstance(mesh, str):
            mesh = self.load_mesh(mesh)

        if self.use_trimesh:
            return self.compute_visibility_trimesh(mesh, camera_positions, offset)
        else:
            return self.compute_visibility_open3d(mesh, camera_positions, offset)

    def compute_visibility_from_normals(
        self,
        mesh: Union[str, o3d.geometry.TriangleMesh],
        camera_positions: np.ndarray,
        angle_threshold: float = 90.0
    ) -> np.ndarray:
        """
        法線ベースの簡易可視判定（高速だが不正確）

        カメラ方向と頂点法線の角度が閾値以下なら可視とみなす。
        遮蔽は考慮しない。

        Args:
            mesh: メッシュファイルパスまたはOpen3Dメッシュ
            camera_positions: カメラ位置の配列 (N, 3) または (3,)
            angle_threshold: 可視とみなす角度閾値（度）

        Returns:
            visible_mask: 各頂点の可視性 (num_vertices,)
        """
        if isinstance(mesh, str):
            mesh = self.load_mesh(mesh)

        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)

        if len(normals) == 0:
            mesh.compute_vertex_normals()
            normals = np.asarray(mesh.vertex_normals)

        # カメラ位置が1つの場合は配列に変換
        if camera_positions.ndim == 1:
            camera_positions = camera_positions.reshape(1, -1)

        # 角度閾値をラジアンに変換
        cos_threshold = np.cos(np.radians(angle_threshold))

        # 各カメラ位置からの可視性を計算
        visible_count = np.zeros(len(vertices), dtype=int)

        for cam_pos in camera_positions:
            # カメラへの方向ベクトル
            to_camera = cam_pos - vertices
            to_camera = to_camera / (np.linalg.norm(to_camera, axis=1, keepdims=True) + 1e-6)

            # 法線との内積（cosθ）
            cos_angles = np.sum(normals * to_camera, axis=1)

            # 閾値以上なら可視
            visible = cos_angles >= cos_threshold
            visible_count += visible.astype(int)

        # 1つ以上のカメラから見える場合は可視
        visible_mask = visible_count > 0

        return visible_mask

    def get_visible_vertices(
        self,
        mesh: Union[str, o3d.geometry.TriangleMesh],
        camera_positions: np.ndarray,
        method: str = "raycast"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        可視頂点のインデックスと座標を取得

        Args:
            mesh: メッシュ
            camera_positions: カメラ位置
            method: "raycast" または "normal"

        Returns:
            (visible_indices, visible_vertices)
        """
        if isinstance(mesh, str):
            mesh = self.load_mesh(mesh)

        if method == "raycast":
            visible_mask = self.compute_visibility(mesh, camera_positions)
        elif method == "normal":
            visible_mask = self.compute_visibility_from_normals(mesh, camera_positions)
        else:
            raise ValueError(f"Unknown method: {method}")

        vertices = np.asarray(mesh.vertices)
        visible_indices = np.where(visible_mask)[0]
        visible_vertices = vertices[visible_mask]

        return visible_indices, visible_vertices

    def visualize(
        self,
        mesh: o3d.geometry.TriangleMesh,
        visible_mask: np.ndarray,
        camera_positions: Optional[np.ndarray] = None
    ):
        """
        可視判定結果を可視化

        Args:
            mesh: メッシュ
            visible_mask: 可視マスク
            camera_positions: カメラ位置（オプション）
        """
        # メッシュの頂点カラーを設定
        colors = np.zeros((len(mesh.vertices), 3))
        colors[visible_mask] = [0, 1, 0]  # 可視: 緑
        colors[~visible_mask] = [1, 0, 0]  # 非可視: 赤
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        geometries = [mesh]

        # カメラ位置を球で表示
        if camera_positions is not None:
            if camera_positions.ndim == 1:
                camera_positions = camera_positions.reshape(1, -1)

            for cam_pos in camera_positions:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                sphere.translate(cam_pos)
                sphere.paint_uniform_color([0, 0, 1])  # 青
                geometries.append(sphere)

        o3d.visualization.draw_geometries(geometries)


def main():
    """コマンドライン実行用"""
    import argparse

    parser = argparse.ArgumentParser(description="Mesh Visibility Check")
    parser.add_argument("mesh", help="Input mesh file (PLY/OBJ)")
    parser.add_argument("--camera", type=float, nargs=3, default=[0, 0, 2],
                        help="Camera position (x y z)")
    parser.add_argument("--method", choices=["raycast", "normal"], default="raycast",
                        help="Visibility check method")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize result")
    parser.add_argument("--output", help="Output visible vertices file")

    args = parser.parse_args()

    checker = VisibilityChecker()
    camera_pos = np.array(args.camera)

    print(f"Loading mesh: {args.mesh}")
    mesh = checker.load_mesh(args.mesh)
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Triangles: {len(mesh.triangles)}")

    print(f"\nComputing visibility (method: {args.method})...")
    visible_indices, visible_vertices = checker.get_visible_vertices(
        mesh, camera_pos, method=args.method
    )

    print(f"\n=== Result ===")
    print(f"Visible vertices: {len(visible_indices)} / {len(mesh.vertices)}")
    print(f"Visibility ratio: {len(visible_indices) / len(mesh.vertices) * 100:.1f}%")

    if args.output:
        # 可視頂点を保存
        visible_pcd = o3d.geometry.PointCloud()
        visible_pcd.points = o3d.utility.Vector3dVector(visible_vertices)
        o3d.io.write_point_cloud(args.output, visible_pcd)
        print(f"Saved visible vertices: {args.output}")

        # インデックスも保存
        np.save(args.output.replace('.ply', '_indices.npy'), visible_indices)

    if args.visualize:
        visible_mask = np.zeros(len(mesh.vertices), dtype=bool)
        visible_mask[visible_indices] = True
        checker.visualize(mesh, visible_mask, camera_pos)


if __name__ == "__main__":
    main()
