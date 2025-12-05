#!/usr/bin/env python3
"""
Shrinkwrapモジュール

SAM 3Dメッシュの可視頂点をLiDAR点群にスナップする。

処理フロー:
1. SAM 3Dメッシュを読み込み
2. 可視マスクを適用（可視頂点のみ対象）
3. 可視頂点をLiDAR点群の最近傍点にスナップ
4. 補正済みメッシュを出力

Blender APIを使用する方法と、Open3Dのみを使用する方法の2種類を提供。

使い方:
    # Open3D版（Blender不要）
    from server.fusion.shrinkwrap import ShrinkwrapProcessor

    processor = ShrinkwrapProcessor()
    result = processor.process(
        mesh_path="sam3d_output.ply",
        lidar_path="lidar_points.ply",
        visible_mask=visible_mask
    )

    # Blender版（Blenderスクリプトとして実行）
    blender --background --python shrinkwrap.py -- input.ply lidar.ply output.ply
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
import json

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Open3D is required. Install with: pip install open3d")

# Blender bpy はオプション（Blender環境でのみ使用）
try:
    import bpy
    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False


class ShrinkwrapProcessor:
    """Open3Dベースの Shrinkwrap 処理クラス"""

    def __init__(
        self,
        snap_distance: float = 0.1,
        smoothing_iterations: int = 3,
        smoothing_lambda: float = 0.5
    ):
        """
        Args:
            snap_distance: スナップする最大距離（これ以上離れた点はスナップしない）
            smoothing_iterations: スナップ後のスムージング反復回数
            smoothing_lambda: スムージング係数（0-1、大きいほど強い）
        """
        self.snap_distance = snap_distance
        self.smoothing_iterations = smoothing_iterations
        self.smoothing_lambda = smoothing_lambda

    def load_mesh(self, path: str) -> o3d.geometry.TriangleMesh:
        """メッシュファイルを読み込む"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        mesh = o3d.io.read_triangle_mesh(str(path))

        if len(mesh.vertices) == 0:
            raise ValueError(f"Empty mesh: {path}")

        mesh.compute_vertex_normals()
        return mesh

    def load_pointcloud(self, path: str) -> o3d.geometry.PointCloud:
        """点群ファイルを読み込む"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        pcd = o3d.io.read_point_cloud(str(path))

        if len(pcd.points) == 0:
            raise ValueError(f"Empty point cloud: {path}")

        return pcd

    def snap_vertices_to_pointcloud(
        self,
        vertices: np.ndarray,
        target_pcd: o3d.geometry.PointCloud,
        visible_mask: Optional[np.ndarray] = None,
        max_distance: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        頂点を点群の最近傍点にスナップ

        Args:
            vertices: 頂点座標 (N, 3)
            target_pcd: ターゲット点群
            visible_mask: 可視頂点マスク (N,)、Noneの場合は全頂点対象
            max_distance: スナップする最大距離

        Returns:
            (snapped_vertices, snap_distances)
        """
        if max_distance is None:
            max_distance = self.snap_distance

        # KD木を構築
        target_tree = o3d.geometry.KDTreeFlann(target_pcd)
        target_points = np.asarray(target_pcd.points)

        snapped_vertices = vertices.copy()
        snap_distances = np.zeros(len(vertices))

        # スナップ対象の頂点インデックス
        if visible_mask is not None:
            target_indices = np.where(visible_mask)[0]
        else:
            target_indices = np.arange(len(vertices))

        for idx in target_indices:
            vertex = vertices[idx]

            # 最近傍点を検索
            [k, nn_idx, nn_dist] = target_tree.search_knn_vector_3d(vertex, 1)

            if k > 0:
                distance = np.sqrt(nn_dist[0])
                snap_distances[idx] = distance

                # 距離が閾値以内ならスナップ
                if distance <= max_distance:
                    snapped_vertices[idx] = target_points[nn_idx[0]]

        return snapped_vertices, snap_distances

    def smooth_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        fixed_mask: Optional[np.ndarray] = None,
        iterations: Optional[int] = None,
        lambda_val: Optional[float] = None
    ) -> o3d.geometry.TriangleMesh:
        """
        メッシュをスムージング（ラプラシアン平滑化）

        固定マスクで指定した頂点は動かさない。

        Args:
            mesh: 入力メッシュ
            fixed_mask: 固定する頂点のマスク (N,)
            iterations: 反復回数
            lambda_val: スムージング係数

        Returns:
            スムージング済みメッシュ
        """
        if iterations is None:
            iterations = self.smoothing_iterations
        if lambda_val is None:
            lambda_val = self.smoothing_lambda

        if iterations <= 0:
            return mesh

        vertices = np.asarray(mesh.vertices).copy()
        triangles = np.asarray(mesh.triangles)
        num_vertices = len(vertices)

        # 隣接頂点リストを構築
        adjacency = [set() for _ in range(num_vertices)]
        for tri in triangles:
            adjacency[tri[0]].add(tri[1])
            adjacency[tri[0]].add(tri[2])
            adjacency[tri[1]].add(tri[0])
            adjacency[tri[1]].add(tri[2])
            adjacency[tri[2]].add(tri[0])
            adjacency[tri[2]].add(tri[1])

        # ラプラシアン平滑化
        for _ in range(iterations):
            new_vertices = vertices.copy()

            for i in range(num_vertices):
                # 固定頂点はスキップ
                if fixed_mask is not None and fixed_mask[i]:
                    continue

                neighbors = list(adjacency[i])
                if len(neighbors) == 0:
                    continue

                # 隣接頂点の重心
                centroid = np.mean(vertices[neighbors], axis=0)

                # 重心に向かって移動
                new_vertices[i] = vertices[i] + lambda_val * (centroid - vertices[i])

            vertices = new_vertices

        # 新しいメッシュを作成
        smoothed_mesh = o3d.geometry.TriangleMesh()
        smoothed_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        smoothed_mesh.triangles = o3d.utility.Vector3iVector(triangles)

        if mesh.has_vertex_colors():
            smoothed_mesh.vertex_colors = mesh.vertex_colors

        smoothed_mesh.compute_vertex_normals()

        return smoothed_mesh

    def process(
        self,
        mesh_path: str,
        lidar_path: str,
        visible_mask: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
        smooth_boundary: bool = True
    ) -> Dict[str, Any]:
        """
        Shrinkwrap処理を実行

        Args:
            mesh_path: SAM 3Dメッシュファイルパス
            lidar_path: LiDAR点群ファイルパス
            visible_mask: 可視頂点マスク（Noneの場合は全頂点対象）
            output_path: 出力ファイルパス（Noneの場合は保存しない）
            smooth_boundary: 境界部分をスムージングするか

        Returns:
            {
                'mesh': 補正済みメッシュ,
                'snap_distances': 各頂点のスナップ距離,
                'snapped_count': スナップされた頂点数,
                'visible_count': 可視頂点数
            }
        """
        print(f"Loading mesh: {mesh_path}")
        mesh = self.load_mesh(mesh_path)
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Triangles: {len(mesh.triangles)}")

        print(f"Loading LiDAR point cloud: {lidar_path}")
        lidar_pcd = self.load_pointcloud(lidar_path)
        print(f"  Points: {len(lidar_pcd.points)}")

        vertices = np.asarray(mesh.vertices)

        # 可視マスクの処理
        if visible_mask is not None:
            visible_count = np.sum(visible_mask)
            print(f"Visible vertices: {visible_count} / {len(vertices)}")
        else:
            visible_count = len(vertices)
            print("No visibility mask provided, processing all vertices")

        # 頂点をスナップ
        print(f"Snapping vertices (max distance: {self.snap_distance})...")
        snapped_vertices, snap_distances = self.snap_vertices_to_pointcloud(
            vertices, lidar_pcd, visible_mask
        )

        # スナップされた頂点数をカウント
        if visible_mask is not None:
            snapped_mask = (snap_distances > 0) & (snap_distances <= self.snap_distance) & visible_mask
        else:
            snapped_mask = (snap_distances > 0) & (snap_distances <= self.snap_distance)
        snapped_count = np.sum(snapped_mask)
        print(f"  Snapped vertices: {snapped_count}")

        # メッシュを更新
        mesh.vertices = o3d.utility.Vector3dVector(snapped_vertices)

        # 境界スムージング
        if smooth_boundary and self.smoothing_iterations > 0:
            print(f"Smoothing boundary (iterations: {self.smoothing_iterations})...")
            # スナップされた頂点を固定して、境界部分のみスムージング
            fixed_mask = snapped_mask if visible_mask is not None else None
            mesh = self.smooth_mesh(mesh, fixed_mask=fixed_mask)

        # 法線を再計算
        mesh.compute_vertex_normals()

        # 保存
        if output_path:
            output_path = Path(output_path)
            o3d.io.write_triangle_mesh(str(output_path), mesh)
            print(f"Saved result: {output_path}")

            # メタデータを保存
            meta_path = output_path.with_suffix('.json')
            with open(meta_path, 'w') as f:
                json.dump({
                    'snap_distance': self.snap_distance,
                    'snapped_count': int(snapped_count),
                    'visible_count': int(visible_count),
                    'total_vertices': len(vertices),
                    'mean_snap_distance': float(np.mean(snap_distances[snapped_mask])) if snapped_count > 0 else 0.0
                }, f, indent=2)
            print(f"Saved metadata: {meta_path}")

        return {
            'mesh': mesh,
            'snap_distances': snap_distances,
            'snapped_count': snapped_count,
            'visible_count': visible_count
        }

    def visualize(
        self,
        original_mesh: o3d.geometry.TriangleMesh,
        processed_mesh: o3d.geometry.TriangleMesh,
        lidar_pcd: o3d.geometry.PointCloud,
        visible_mask: Optional[np.ndarray] = None
    ):
        """
        処理結果を可視化

        Args:
            original_mesh: 元のメッシュ
            processed_mesh: 処理済みメッシュ
            lidar_pcd: LiDAR点群
            visible_mask: 可視マスク
        """
        geometries = []

        # 元のメッシュ（半透明赤）
        original_vis = o3d.geometry.TriangleMesh(original_mesh)
        original_vis.paint_uniform_color([1, 0.5, 0.5])
        original_vis.translate([0, 0, -0.5])  # オフセット
        geometries.append(original_vis)

        # 処理済みメッシュ（可視部分を緑、非可視部分を青）
        processed_vis = o3d.geometry.TriangleMesh(processed_mesh)
        if visible_mask is not None:
            colors = np.zeros((len(processed_mesh.vertices), 3))
            colors[visible_mask] = [0, 1, 0]  # 可視: 緑
            colors[~visible_mask] = [0, 0, 1]  # 非可視: 青
            processed_vis.vertex_colors = o3d.utility.Vector3dVector(colors)
        else:
            processed_vis.paint_uniform_color([0, 1, 0])
        geometries.append(processed_vis)

        # LiDAR点群（白）
        lidar_vis = o3d.geometry.PointCloud(lidar_pcd)
        lidar_vis.paint_uniform_color([1, 1, 1])
        geometries.append(lidar_vis)

        o3d.visualization.draw_geometries(geometries)


class BlenderShrinkwrap:
    """
    Blender APIを使用したShrinkwrap処理

    Blender環境でのみ使用可能。
    BlenderのShrinkwrapモディファイアを使用してより高品質な結果を得る。
    """

    def __init__(self):
        if not HAS_BLENDER:
            raise ImportError("Blender (bpy) is not available. Run this script in Blender.")

    def import_mesh(self, path: str, name: str = "SAM3D_Mesh") -> 'bpy.types.Object':
        """メッシュをBlenderにインポート"""
        path = Path(path)

        if path.suffix.lower() == '.ply':
            bpy.ops.wm.ply_import(filepath=str(path))
        elif path.suffix.lower() == '.obj':
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

        # インポートされたオブジェクトを取得
        obj = bpy.context.selected_objects[0]
        obj.name = name

        return obj

    def import_pointcloud_as_mesh(
        self,
        path: str,
        name: str = "LiDAR_Target"
    ) -> 'bpy.types.Object':
        """
        点群をメッシュとしてインポート（Shrinkwrapターゲット用）

        点群からボクセルメッシュまたは球メッシュを生成してターゲットとする。
        """
        # Open3Dで点群を読み込み
        pcd = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pcd.points)

        # 一時ファイルとして保存（PLY形式）
        temp_path = Path("/tmp/lidar_target.ply")

        # 点群からメッシュを生成（Ball Pivoting法）
        pcd.estimate_normals()
        radii = [0.005, 0.01, 0.02, 0.04]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )

        if len(mesh.triangles) == 0:
            # Ball Pivotingが失敗した場合、Poisson再構成を試行
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=8
            )

        o3d.io.write_triangle_mesh(str(temp_path), mesh)

        # Blenderにインポート
        bpy.ops.wm.ply_import(filepath=str(temp_path))
        obj = bpy.context.selected_objects[0]
        obj.name = name

        return obj

    def apply_shrinkwrap(
        self,
        source_obj: 'bpy.types.Object',
        target_obj: 'bpy.types.Object',
        visible_mask: Optional[np.ndarray] = None,
        wrap_method: str = 'NEAREST_SURFACEPOINT',
        offset: float = 0.0
    ):
        """
        Shrinkwrapモディファイアを適用

        Args:
            source_obj: ソースメッシュオブジェクト
            target_obj: ターゲットオブジェクト
            visible_mask: 可視頂点マスク（頂点グループで制御）
            wrap_method: ラップ方法 (NEAREST_SURFACEPOINT, PROJECT, NEAREST_VERTEX, TARGET_PROJECT)
            offset: オフセット距離
        """
        # 頂点グループを作成（可視マスクがある場合）
        if visible_mask is not None:
            # 頂点グループを作成
            vg = source_obj.vertex_groups.new(name="Visible")

            # 可視頂点をグループに追加
            visible_indices = np.where(visible_mask)[0].tolist()
            vg.add(visible_indices, 1.0, 'REPLACE')

        # Shrinkwrapモディファイアを追加
        modifier = source_obj.modifiers.new(name="Shrinkwrap", type='SHRINKWRAP')
        modifier.target = target_obj
        modifier.wrap_method = wrap_method
        modifier.offset = offset

        # 頂点グループで制限
        if visible_mask is not None:
            modifier.vertex_group = "Visible"

        # モディファイアを適用
        bpy.context.view_layer.objects.active = source_obj
        bpy.ops.object.modifier_apply(modifier="Shrinkwrap")

    def export_mesh(self, obj: 'bpy.types.Object', output_path: str):
        """メッシュをエクスポート"""
        output_path = Path(output_path)

        # オブジェクトを選択
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        if output_path.suffix.lower() == '.ply':
            bpy.ops.wm.ply_export(filepath=str(output_path))
        elif output_path.suffix.lower() == '.obj':
            bpy.ops.wm.obj_export(filepath=str(output_path))
        else:
            raise ValueError(f"Unsupported format: {output_path.suffix}")

    def process(
        self,
        mesh_path: str,
        lidar_path: str,
        output_path: str,
        visible_mask: Optional[np.ndarray] = None
    ):
        """
        Blender Shrinkwrap処理を実行

        Args:
            mesh_path: SAM 3Dメッシュファイルパス
            lidar_path: LiDAR点群ファイルパス
            output_path: 出力ファイルパス
            visible_mask: 可視頂点マスク
        """
        # シーンをクリア
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # メッシュをインポート
        print(f"Importing mesh: {mesh_path}")
        source_obj = self.import_mesh(mesh_path, "SAM3D_Mesh")

        # LiDAR点群をターゲットメッシュとしてインポート
        print(f"Importing LiDAR as target: {lidar_path}")
        target_obj = self.import_pointcloud_as_mesh(lidar_path, "LiDAR_Target")

        # Shrinkwrapを適用
        print("Applying Shrinkwrap...")
        self.apply_shrinkwrap(source_obj, target_obj, visible_mask)

        # エクスポート
        print(f"Exporting result: {output_path}")
        self.export_mesh(source_obj, output_path)

        print("Done!")


def main():
    """コマンドライン実行用"""
    import argparse

    parser = argparse.ArgumentParser(description="Shrinkwrap SAM 3D mesh to LiDAR point cloud")
    parser.add_argument("mesh", help="Input SAM 3D mesh file (PLY/OBJ)")
    parser.add_argument("lidar", help="LiDAR point cloud file (PLY/PCD)")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--snap-distance", type=float, default=0.1,
                        help="Maximum snap distance")
    parser.add_argument("--smoothing", type=int, default=3,
                        help="Smoothing iterations (0 to disable)")
    parser.add_argument("--visibility-mask", help="Visibility mask file (NPY)")
    parser.add_argument("--use-blender", action="store_true",
                        help="Use Blender for processing (requires Blender environment)")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize result")

    args = parser.parse_args()

    # 可視マスクを読み込み
    visible_mask = None
    if args.visibility_mask:
        visible_mask = np.load(args.visibility_mask)
        print(f"Loaded visibility mask: {np.sum(visible_mask)} visible vertices")

    if args.use_blender:
        if not HAS_BLENDER:
            print("Error: Blender (bpy) is not available.")
            print("Run this script in Blender: blender --background --python shrinkwrap.py -- ...")
            return

        processor = BlenderShrinkwrap()
        processor.process(
            args.mesh,
            args.lidar,
            args.output or "shrinkwrap_result.ply",
            visible_mask
        )
    else:
        processor = ShrinkwrapProcessor(
            snap_distance=args.snap_distance,
            smoothing_iterations=args.smoothing
        )

        result = processor.process(
            args.mesh,
            args.lidar,
            visible_mask=visible_mask,
            output_path=args.output
        )

        print(f"\n=== Result ===")
        print(f"Snapped vertices: {result['snapped_count']} / {result['visible_count']}")

        if args.visualize:
            original_mesh = processor.load_mesh(args.mesh)
            lidar_pcd = processor.load_pointcloud(args.lidar)
            processor.visualize(original_mesh, result['mesh'], lidar_pcd, visible_mask)


if __name__ == "__main__":
    main()
