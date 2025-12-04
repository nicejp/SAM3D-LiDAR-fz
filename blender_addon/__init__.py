bl_info = {
    "name": "LiDAR-LLM Pipeline",
    "author": "nicejp",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > LLM",
    "description": "LiDAR点群からLLMを使って3Dオブジェクトを生成",
    "category": "3D View",
}

import bpy
from bpy.props import StringProperty, EnumProperty, BoolProperty
from bpy.types import Panel, Operator, PropertyGroup
import subprocess
import json
import os
from pathlib import Path


# プロパティグループ
class LLMPipelineProperties(PropertyGroup):
    """パイプラインの設定を保持するプロパティグループ"""

    session_dir: StringProperty(
        name="Session Directory",
        description="セッションディレクトリのパス",
        default="",
        subtype='DIR_PATH'
    )

    user_prompt: StringProperty(
        name="Prompt",
        description="LLMへの指示",
        default="この点群を3Dオブジェクトにして"
    )

    model_name: EnumProperty(
        name="Model",
        description="使用するLLMモデル",
        items=[
            ('gpt-oss:120b', 'gpt-oss 120B', '高精度モデル'),
            ('qwen3-coder:30b', 'Qwen3 Coder 30B', '軽量モデル'),
            ('codellama:34b', 'CodeLlama 34B', 'コード特化モデル'),
        ],
        default='gpt-oss:120b'
    )

    python_path: StringProperty(
        name="Python Path",
        description="venv内のPythonパス",
        default=os.path.expanduser("~/LiDAR-LLM-MCP/venv/bin/python"),
        subtype='FILE_PATH'
    )

    project_dir: StringProperty(
        name="Project Directory",
        description="LiDAR-LLM-MCPプロジェクトのパス",
        default=os.path.expanduser("~/LiDAR-LLM-MCP"),
        subtype='DIR_PATH'
    )

    status_message: StringProperty(
        name="Status",
        description="現在のステータス",
        default="待機中"
    )


# オペレーター: パイプライン実行
class LIDAR_OT_run_pipeline(Operator):
    """LLMパイプラインを実行"""
    bl_idname = "lidar.run_pipeline"
    bl_label = "Generate 3D"
    bl_description = "点群からLLMを使って3Dオブジェクトを生成"

    def execute(self, context):
        props = context.scene.llm_pipeline

        if not props.session_dir:
            self.report({'ERROR'}, "セッションディレクトリを選択してください")
            return {'CANCELLED'}

        session_dir = bpy.path.abspath(props.session_dir)

        if not os.path.exists(session_dir):
            self.report({'ERROR'}, f"ディレクトリが見つかりません: {session_dir}")
            return {'CANCELLED'}

        props.status_message = "パイプライン実行中..."

        try:
            # Pythonパスとプロジェクトディレクトリを取得
            python_path = bpy.path.abspath(props.python_path)
            project_dir = bpy.path.abspath(props.project_dir)

            # パスの存在確認
            if not os.path.exists(python_path):
                self.report({'ERROR'}, f"Pythonが見つかりません: {python_path}")
                props.status_message = "エラー: Python not found"
                return {'CANCELLED'}

            # パイプラインを実行
            cmd = [
                python_path, "-m", "server.phase1_minimal.pipeline",
                session_dir,
                "--prompt", props.user_prompt,
                "--model", props.model_name
            ]

            # 作業ディレクトリを設定
            work_dir = project_dir

            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=300
            )

            # 出力ファイルのパス
            output_blend = os.path.join(session_dir, "output", "result.blend")

            # 成功判定: returncode==0 または 出力ファイルが存在する
            if result.returncode == 0 or os.path.exists(output_blend):
                if os.path.exists(output_blend):
                    # 現在のシーンにインポート
                    with bpy.data.libraries.load(output_blend) as (data_from, data_to):
                        data_to.objects = data_from.objects

                    # オブジェクトをシーンにリンク
                    for obj in data_to.objects:
                        if obj is not None:
                            context.collection.objects.link(obj)

                    props.status_message = "生成完了！"
                    self.report({'INFO'}, "3Dオブジェクトを生成しました")
                else:
                    props.status_message = "出力ファイルが見つかりません"
                    self.report({'WARNING'}, "出力ファイルが見つかりません")
            else:
                # エラー詳細を表示（Warningをフィルタ）
                error_msg = result.stderr
                # RuntimeWarningを除外
                error_lines = [l for l in error_msg.split('\n') if 'RuntimeWarning' not in l and l.strip()]
                error_msg = '\n'.join(error_lines[-5:])  # 最後の5行

                props.status_message = "エラー発生"
                print(f"Pipeline stdout: {result.stdout}")
                print(f"Pipeline stderr: {result.stderr}")
                self.report({'ERROR'}, f"パイプラインエラー: {error_msg[:300]}")
                return {'CANCELLED'}

        except subprocess.TimeoutExpired:
            props.status_message = "タイムアウト"
            self.report({'ERROR'}, "パイプラインがタイムアウトしました")
            return {'CANCELLED'}
        except Exception as e:
            props.status_message = f"エラー: {str(e)[:50]}"
            self.report({'ERROR'}, f"エラー: {str(e)}")
            return {'CANCELLED'}

        return {'FINISHED'}


# オペレーター: セッション一覧を取得
class LIDAR_OT_list_sessions(Operator):
    """利用可能なセッションを一覧表示"""
    bl_idname = "lidar.list_sessions"
    bl_label = "List Sessions"
    bl_description = "experimentsフォルダ内のセッションを一覧表示"

    def execute(self, context):
        props = context.scene.llm_pipeline
        # experimentsフォルダを探す
        project_dir = bpy.path.abspath(props.project_dir)
        experiments_dir = os.path.join(project_dir, "experiments")

        if os.path.exists(experiments_dir):
            sessions = sorted(Path(experiments_dir).glob("session_*"))
            if sessions:
                msg = "利用可能なセッション:\n"
                for s in sessions[-5:]:  # 最新5件
                    msg += f"  - {s.name}\n"
                self.report({'INFO'}, msg)
            else:
                self.report({'WARNING'}, "セッションが見つかりません")
        else:
            self.report({'WARNING'}, f"フォルダが見つかりません: {experiments_dir}")

        return {'FINISHED'}


# オペレーター: 点群をインポート
class LIDAR_OT_import_pointcloud(Operator):
    """点群PLYファイルをインポート"""
    bl_idname = "lidar.import_pointcloud"
    bl_label = "Import Point Cloud"
    bl_description = "セッションの点群をインポート"

    def execute(self, context):
        props = context.scene.llm_pipeline

        if not props.session_dir:
            self.report({'ERROR'}, "セッションディレクトリを選択してください")
            return {'CANCELLED'}

        session_dir = bpy.path.abspath(props.session_dir)
        ply_path = os.path.join(session_dir, "output", "pointcloud", "merged.ply")

        if os.path.exists(ply_path):
            bpy.ops.wm.ply_import(filepath=ply_path)
            self.report({'INFO'}, f"点群をインポートしました: {ply_path}")
        else:
            self.report({'ERROR'}, f"点群ファイルが見つかりません: {ply_path}")
            return {'CANCELLED'}

        return {'FINISHED'}


# UIパネル
class LIDAR_PT_main_panel(Panel):
    """メインパネル"""
    bl_label = "LiDAR-LLM Pipeline"
    bl_idname = "LIDAR_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'LLM'

    def draw(self, context):
        layout = self.layout
        props = context.scene.llm_pipeline

        # セッション選択
        box = layout.box()
        box.label(text="Session", icon='FILE_FOLDER')
        box.prop(props, "session_dir", text="")
        box.operator("lidar.list_sessions", icon='VIEWZOOM')

        # プロンプト入力
        box = layout.box()
        box.label(text="Prompt", icon='TEXT')
        box.prop(props, "user_prompt", text="")

        # モデル選択
        box = layout.box()
        box.label(text="Model", icon='PREFERENCES')
        box.prop(props, "model_name", text="")

        # 実行ボタン
        layout.separator()
        row = layout.row(align=True)
        row.scale_y = 2.0
        row.operator("lidar.run_pipeline", icon='PLAY')

        # 追加オプション
        layout.separator()
        box = layout.box()
        box.label(text="Tools", icon='TOOL_SETTINGS')
        box.operator("lidar.import_pointcloud", icon='IMPORT')

        # 設定
        layout.separator()
        box = layout.box()
        box.label(text="Settings", icon='PREFERENCES')
        box.prop(props, "project_dir", text="Project")
        box.prop(props, "python_path", text="Python")

        # ステータス表示
        layout.separator()
        box = layout.box()
        box.label(text=f"Status: {props.status_message}", icon='INFO')


# 登録するクラス
classes = [
    LLMPipelineProperties,
    LIDAR_OT_run_pipeline,
    LIDAR_OT_list_sessions,
    LIDAR_OT_import_pointcloud,
    LIDAR_PT_main_panel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.llm_pipeline = bpy.props.PointerProperty(type=LLMPipelineProperties)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.llm_pipeline


if __name__ == "__main__":
    register()
