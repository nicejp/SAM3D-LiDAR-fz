#!/usr/bin/env python3
"""
Mesh to Blender Exporter
保存されたメッシュデータをBlenderで読み込める形式に変換

使い方:
    # PLYファイルにエクスポート
    python -m server.visualization.mesh_to_blender experiments/session_xxx

    # Blenderで開く
    blender --python server/visualization/mesh_to_blender.py -- experiments/session_xxx
"""

import argparse
import numpy as np
from pathlib import Path
import sys


def load_mesh_vertices(session_dir: str) -> np.ndarray:
    """セッションからメッシュ頂点を読み込む"""
    session_path = Path(session_dir)
    mesh_dir = session_path / "mesh"

    # 最新のメッシュを探す
    latest_path = mesh_dir / "latest.npy"
    if latest_path.exists():
        print(f"Loading: {latest_path}")
        return np.load(latest_path)

    # フレームファイルを探す
    frame_files = sorted(mesh_dir.glob("frame_*.npy"))
    if frame_files:
        latest_frame = frame_files[-1]
        print(f"Loading: {latest_frame}")
        return np.load(latest_frame)

    raise FileNotFoundError(f"No mesh data found in {mesh_dir}")


def save_as_ply(vertices: np.ndarray, output_path: str):
    """頂点データをPLYファイルとして保存"""
    num_vertices = len(vertices)

    # 高さに基づいて色を計算
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    y_range = max(y_max - y_min, 0.1)

    with open(output_path, 'w') as f:
        # PLYヘッダー
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # 頂点データ
        for v in vertices:
            # 高さに基づいた色（青→緑のグラデーション）
            t = (v[1] - y_min) / y_range
            r = int(50)
            g = int(80 + t * 175)
            b = int(255 - t * 128)
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {r} {g} {b}\n")

    print(f"Saved PLY: {output_path} ({num_vertices} vertices)")


def export_to_ply(session_dir: str, output_path: str = None):
    """メッシュをPLYファイルにエクスポート"""
    vertices = load_mesh_vertices(session_dir)

    if output_path is None:
        session_path = Path(session_dir)
        output_path = str(session_path / "mesh" / "mesh.ply")

    save_as_ply(vertices, output_path)
    return output_path


def blender_import_mesh(session_dir: str):
    """Blender内でメッシュをインポート（Blenderスクリプトとして実行）"""
    try:
        import bpy
    except ImportError:
        print("Error: This function must be run inside Blender")
        return False

    # PLYファイルを生成
    ply_path = export_to_ply(session_dir)

    # 既存オブジェクトを削除
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # PLYをインポート
    bpy.ops.wm.ply_import(filepath=ply_path)

    # インポートしたオブジェクトを取得
    mesh_obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None

    if mesh_obj:
        # 頂点カラーを表示するマテリアルを設定
        mat = bpy.data.materials.new(name="VertexColorMaterial")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # 既存ノードをクリア
        nodes.clear()

        # 頂点カラーノードを作成
        vertex_color_node = nodes.new(type='ShaderNodeVertexColor')
        vertex_color_node.location = (-300, 0)

        # Principled BSDFノード
        bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf_node.location = (0, 0)

        # 出力ノード
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        output_node.location = (300, 0)

        # ノードを接続
        links.new(vertex_color_node.outputs['Color'], bsdf_node.inputs['Base Color'])
        links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

        # マテリアルをオブジェクトに適用
        if mesh_obj.data.materials:
            mesh_obj.data.materials[0] = mat
        else:
            mesh_obj.data.materials.append(mat)

        # ビューポートシェーディングをマテリアルプレビューに設定
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'MATERIAL'
                        break

    # カメラとライトを追加
    bpy.ops.object.camera_add(location=(5, -5, 5))
    camera = bpy.context.active_object
    camera.rotation_euler = (1.1, 0, 0.8)
    bpy.context.scene.camera = camera

    bpy.ops.object.light_add(type='SUN', location=(10, -10, 10))

    # ビューをフィット
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for region in area.regions:
                if region.type == 'WINDOW':
                    with bpy.context.temp_override(area=area, region=region):
                        bpy.ops.view3d.view_all()
                    break

    print("Mesh imported to Blender with vertex colors!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export mesh to PLY for Blender")
    parser.add_argument("session_dir", help="Session directory path")
    parser.add_argument("--output", "-o", help="Output PLY file path")
    parser.add_argument("--blender", "-b", action="store_true",
                        help="Import directly to Blender (run from within Blender)")

    # Blenderから実行された場合、--の後の引数を処理
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)

    if args.blender:
        blender_import_mesh(args.session_dir)
    else:
        ply_path = export_to_ply(args.session_dir, args.output)
        print(f"\n" + "=" * 50)
        print("Blenderで色付きメッシュを開く方法")
        print("=" * 50)
        print(f"\n【推奨】自動インポート（色設定込み）:")
        print(f"  blender --python server/visualization/mesh_to_blender.py -- {args.session_dir} --blender")
        print(f"\n【手動】PLYをインポート後、色を表示:")
        print(f"  1. File → Import → Stanford (.ply) → {ply_path}")
        print(f"  2. オブジェクトを選択")
        print(f"  3. 右上のビューポートシェーディングを「Material Preview」に変更")
        print(f"  4. マテリアルプロパティで新規マテリアル作成")
        print(f"  5. シェーダーエディタで Vertex Color ノードを Base Color に接続")
        print("=" * 50)


if __name__ == "__main__":
    main()
