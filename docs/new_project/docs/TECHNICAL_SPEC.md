# SAM3D-LiDAR-Fusion 技術仕様書

---

## 1. システム概要

### 1.1 目的

SAM 3D（生成AI）で作成した3Dメッシュを、iPad LiDAR（実測データ）で補正・融合し、以下を両立した3Dモデルを生成する：

- **正確な寸法** （LiDAR由来）
- **綺麗なトポロジー** （SAM 3D由来）
- **完全な形状** （裏側含む）

### 1.2 技術的アプローチ

本システムは **Template Fitting** および **Non-rigid Registration** の技術領域に属する。

```
「SAM 3Dで作った理想的な形の風船を、
 LiDARという実測の型枠の中で膨らませて、
 表面をピタッと吸着させる」
```

---

## 2. コンポーネント詳細

### 2.1 SAM 3 (Segment Anything Model 3)

**役割:** 「最強のハサミ」（2D切り抜き）

| 項目 | 仕様 |
|------|------|
| 入力 | RGB画像 |
| 出力 | バイナリマスク（2D） |
| プロンプト | クリック座標 / テキスト |
| 3D能力 | なし |

**処理フロー:**
```python
# SAM 3によるマスク生成
mask = sam3.segment(image, click_point=(512, 384))
# → 白黒のマスク画像（対象物=白, 背景=黒）
```

### 2.2 SAM 3D (Meta 3D Gen)

**役割:** 「3D彫刻家」（立体生成）

| 項目 | 仕様 |
|------|------|
| 入力 | RGBA画像（背景透明） |
| 出力 | 3Dメッシュ（OBJ/GLB） |
| 裏側生成 | AI推測による |
| 寸法精度 | 低（Unitless） |

**処理フロー:**
```python
# SAM 3のマスクを使って背景透明化
rgba_image = apply_mask(original_image, mask)

# SAM 3Dで3Dメッシュ生成
mesh = sam3d.generate(rgba_image)
# → 裏側を含む完全な3Dメッシュ
```

**注意:** SAM 3Dへの入力は「マスク画像」ではなく「背景透明のカラー画像（RGBA）」

### 2.3 点群抽出モジュール

**役割:** LiDAR点群から対象オブジェクトを抽出

**アルゴリズム（逆投影法）:**

3D点群の中から対象を「探す」のではなく、全点群を2D投影してマスクでフィルタリング。

```python
def extract_object_pointcloud(pointcloud, mask, camera_matrix):
    """
    Args:
        pointcloud: 全体の3D点群 (N, 3)
        mask: SAM 3のバイナリマスク (H, W)
        camera_matrix: カメラ内部/外部行列
    Returns:
        対象オブジェクトの点群 (M, 3)
    """
    # 3D点群を2D画像平面に投影
    projected_2d = project_to_image(pointcloud, camera_matrix)

    # マスク内の点のみ抽出
    in_mask = mask[projected_2d[:, 1], projected_2d[:, 0]] > 0

    return pointcloud[in_mask]
```

**課題と対策:**

| 課題 | 説明 | 対策 |
|------|------|------|
| パンチスルー | 手前のゴミ、後ろの壁も選択される | 深度フィルタ（1m〜2mのみ） |
| 位置ズレ | RGB撮影とLiDAR取得のタイミング差 | マスクをDilation（数ピクセル膨張） |

---

## 3. 融合アルゴリズム

### 3.1 Step 1: Coarse Alignment（大まかな位置合わせ）

SAM 3Dのメッシュを、LiDAR点群の座標系に合わせる。

```python
def coarse_alignment(sam3d_mesh, lidar_points):
    # 1. 重心を合わせる
    sam3d_center = np.mean(sam3d_mesh.vertices, axis=0)
    lidar_center = np.mean(lidar_points, axis=0)
    translation = lidar_center - sam3d_center

    # 2. スケールを合わせる（バウンディングボックス比）
    sam3d_size = sam3d_mesh.bounding_box.extent
    lidar_size = np.max(lidar_points, axis=0) - np.min(lidar_points, axis=0)
    scale = np.mean(lidar_size / sam3d_size)

    # 3. PCAで主軸を合わせる（オプション）
    rotation = pca_alignment(sam3d_mesh.vertices, lidar_points)

    return translation, scale, rotation
```

### 3.2 Step 2: Rigid ICP（精密な位置合わせ）

形状を変えずに、位置と回転を微調整。

```python
import open3d as o3d

def rigid_icp(source_mesh, target_points, threshold=0.05):
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_mesh.vertices)

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)

    # ICP実行
    result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    return result.transformation, result.fitness
```

**ICPが失敗する場合:**
- `fitness` スコアが0.5以下 → 初期回転を90度ずらして再実行
- LLMがこの判断を自動化

### 3.3 Step 3: Visibility Check（可視判定）★重要

カメラ位置から見える頂点のみを特定し、裏側を保護する。

```python
def compute_visibility(mesh, camera_position):
    """
    カメラから見える頂点を判定

    Returns:
        visible_mask: bool配列 (頂点数,)
    """
    visible_mask = np.zeros(len(mesh.vertices), dtype=bool)

    for i, vertex in enumerate(mesh.vertices):
        # 頂点からカメラへのレイ
        ray_direction = camera_position - vertex
        ray_direction /= np.linalg.norm(ray_direction)

        # メッシュとの交差判定（自分自身を除く）
        # 遮蔽物がなければ可視
        if not mesh.ray_intersects_any(vertex, ray_direction, exclude_self=True):
            visible_mask[i] = True

    return visible_mask
```

### 3.4 Step 4: Partial Shrinkwrap（部分的吸着）

可視頂点のみをLiDARデータに吸着させ、裏側は維持。

**Blender Python実装:**

```python
import bpy

def partial_shrinkwrap(source_obj, target_obj, visible_vertices):
    """
    Args:
        source_obj: SAM 3Dのメッシュ (Blender Object)
        target_obj: LiDAR点群メッシュ (Blender Object)
        visible_vertices: 可視頂点のインデックスリスト
    """
    # 1. 可視頂点のVertex Groupを作成
    vg = source_obj.vertex_groups.new(name="Visible")
    vg.add(visible_vertices, 1.0, 'REPLACE')

    # 2. Shrinkwrap Modifierを追加
    mod = source_obj.modifiers.new(name="Shrinkwrap", type='SHRINKWRAP')
    mod.target = target_obj
    mod.wrap_method = 'NEAREST_SURFACEPOINT'
    mod.vertex_group = "Visible"  # 可視頂点のみ適用

    # 3. 適用
    bpy.ops.object.modifier_apply(modifier="Shrinkwrap")

    # 4. 境界スムージング（オプション）
    smooth_boundary(source_obj, visible_vertices)
```

---

## 4. LLMオーケストレーション

### 4.1 役割

LLMは3D処理の計算を行わず、「ツールの指揮官」として機能する。

### 4.2 ツール定義

```python
tools = [
    {
        "name": "run_sam3d",
        "description": "SAM 3Dを実行して3Dメッシュを生成",
        "parameters": {"image_path": "str"}
    },
    {
        "name": "run_icp_alignment",
        "description": "ICPで位置合わせを実行",
        "parameters": {
            "source_mesh": "str",
            "target_pcd": "str",
            "init_rotation": "float (optional)"
        }
    },
    {
        "name": "run_blender_fusion",
        "description": "Blenderで融合処理を実行",
        "parameters": {
            "base_mesh": "str",
            "target_pcd": "str",
            "visibility_threshold": "float"
        }
    },
    {
        "name": "check_quality",
        "description": "出力メッシュの品質をチェック",
        "parameters": {"mesh_path": "str"}
    }
]
```

### 4.3 処理フローの例

```
[LLM思考]
1. まずRGB画像からSAM 3Dで3Dメッシュを生成します
   → run_sam3d("input.jpg")
   → Result: "sam_mesh.obj" created

2. LiDAR点群と位置合わせします
   → run_icp_alignment("sam_mesh.obj", "lidar.ply")
   → Result: fitness=0.4 (Low)

3. スコアが低いので、90度回転して再試行します
   → run_icp_alignment("sam_mesh.obj", "lidar.ply", init_rotation=90)
   → Result: fitness=0.92 (High)

4. Blenderで可視領域のみ吸着させます
   → run_blender_fusion("sam_mesh_aligned.obj", "lidar.ply")
   → Result: "final_model.obj" saved

5. 品質チェック
   → check_quality("final_model.obj")
   → Result: OK (頂点数: 5432, 穴なし, 正常トポロジー)
```

### 4.4 エラーハンドリング

| 状況 | LLMの判断 |
|------|----------|
| ICP fitness < 0.5 | 初期回転を変えて最大3回リトライ |
| SAM 3D形状が乖離 | CLIP類似度チェック、ユーザー確認 |
| Blender実行エラー | パラメータ調整、代替手法提案 |
| 無限ループ検知 | 最大リトライ回数で停止、ユーザーに報告 |

---

## 5. データフォーマット

### 5.1 入力データ

```
session_xxx/
├── rgb/
│   ├── frame_000.jpg          # RGB画像 (1920×1440)
│   └── ...
├── depth/
│   ├── frame_000.npy          # 深度マップ (256×192, float32)
│   └── ...
├── camera/
│   ├── frame_000.json         # カメラパラメータ
│   └── ...
└── metadata.json              # セッション情報
```

**camera_xxx.json:**
```json
{
  "intrinsic": {
    "fx": 1000.0, "fy": 1000.0,
    "cx": 960.0, "cy": 720.0
  },
  "extrinsic": {
    "transform": [[...], [...], [...], [...]]  // 4x4行列
  },
  "timestamp": 1701619200.123
}
```

### 5.2 中間データ

```
session_xxx/
├── masks/
│   └── frame_000_mask.png     # SAM 3マスク
├── pointcloud/
│   └── object.ply             # 抽出された点群
├── sam3d/
│   └── generated.obj          # SAM 3D生成メッシュ
└── aligned/
    └── aligned.obj            # ICP後メッシュ
```

### 5.3 出力データ

```
session_xxx/output/
├── final_model.obj            # 最終3Dモデル
├── final_model.mtl            # マテリアル
├── final_model.png            # テクスチャ
├── metadata.json              # 処理ログ
└── quality_report.json        # 品質レポート
```

---

## 6. 性能要件

| 項目 | 目標値 |
|------|--------|
| SAM 3Dメッシュ生成 | < 30秒 |
| ICP位置合わせ | < 5秒 |
| 可視判定 | < 10秒 |
| Shrinkwrap融合 | < 20秒 |
| **総処理時間** | **< 2分** |

---

## 7. 参考文献

1. Meta AI. "Segment Anything Model 3 (SAM 3)." 2024.
2. Meta AI. "Meta 3D Gen." arXiv:2411.19480, 2024.
3. Besl, P. J., & McKay, N. D. "A method for registration of 3-D shapes." IEEE TPAMI, 1992.
4. Blender Foundation. "Shrinkwrap Modifier Documentation." 2024.
