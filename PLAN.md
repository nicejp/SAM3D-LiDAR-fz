# SAM3D-LiDAR-fz 実装計画書

**プロジェクト名:** SAM3D-LiDAR-fz (fusion)
**作成日:** 2025年12月3日
**目標:** SAM 3D（生成） + LiDAR（実測） → LLMがオーケストレーションして融合 → 高精度3Dモデル

---

## 方針

**流用元リポジトリ:**
https://github.com/nicejp/LiDAR-LLM-MCP.git

LiDAR-LLM-MCPプロジェクトのプログラムを極力流用し、新規開発を最小限に抑える。

---

## 実装概要

### 1. RGB画像とLiDAR情報を取得

```
┌─────────────────┐        WiFi (WebSocket)        ┌─────────────────────────┐
│    iPad Pro     │  ────────────────────────────→ │   DGX Spark Server      │
│  (LiDAR搭載)    │   RGB画像 + LiDAR点群          │   + Web GUI             │
└─────────────────┘                                └─────────────────────────┘
```

#### 1-1. ユーザはiPadを使い、RGB画像とLiDAR情報を取得する

**流用コンポーネント:**
- `ipad_app/` - iPadアプリ（Swift）
- ARKitによるRGB (1920×1440) + LiDAR Depth (256×192) 取得

**実装内容:**
- 既存のiPadアプリをそのまま使用
- 必要に応じてUI調整のみ

#### 1-2. NVIDIA DGX Sparkに構築された受信サーバと表示Web GUIで、そのデータを順次受信し、画面表示していく

**流用コンポーネント:**
- `server/data_reception/websocket_server.py` - WebSocketサーバー
- `server/visualization/web_viewer.py` - 点群可視化

**実装内容:**
- WebSocketサーバーでリアルタイムにデータ受信
- Web GUIでRGB画像と3D点群を表示（Three.js等）
- 受信データをセッションディレクトリに保存

**データ形式:**
```
session_xxx/
├── rgb/
│   ├── frame_000.jpg
│   └── ...
├── depth/
│   ├── frame_000.npy
│   └── ...
├── camera/
│   ├── frame_000.json    # カメラ行列（内部/外部）
│   └── ...
└── pointcloud/
    └── accumulated.ply   # 累積点群
```

#### 1-3. ユーザが、Web GUI上に3D point cloudが十分なだけプロットされたと感じたら処理を終了する

**実装内容:**
- Web GUIに「スキャン終了」ボタンを追加
- 点群の累積表示と点数カウンター表示

---

### 2. SAM 3によるRGB画像とLiDARデータ抽出

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│  RGB画像    │ ──→ │   SAM 3     │ ──→ │  2Dマスク       │
│  選択       │     │  セグメント │     │  (白黒画像)     │
└─────────────┘     └─────────────┘     └─────────────────┘
                           │
                           ↓
                    ┌─────────────────┐
                    │  LiDAR点群抽出  │
                    │  (逆投影法)     │
                    └─────────────────┘
```

#### 2-1. ユーザは「1.」で取得されたRGB画像をWeb GUIで一覧し、そのうちの1枚を選択する

**実装内容:**
- Web GUIにサムネイル一覧表示機能
- 画像クリックで選択・拡大表示

#### 2-2. DGX Spark上にあるWeb GUIでは、選択された1枚を拡大し、「SAM 3」を使った領域選択機能をユーザに提供する

**流用コンポーネント:**
- `server/phase2_full/sam3_segmentation.py` - SAM 3セグメンテーション

**実装内容:**
- Web GUI上でクリック座標を取得
- SAM 3にクリック座標を渡してマスク生成
- リアルタイムでマスクプレビュー表示

#### 2-3. システムは、Web GUI上で選択された領域のRGB画像を「SAM 3」を使って抽出する

**処理フロー:**
```python
# SAM 3によるマスク生成
mask = sam3.segment(image, click_point=(x, y))

# マスクを使って背景透明化（RGBA画像生成）
rgba_image = apply_mask(original_image, mask)
# → SAM 3D への入力用
```

**出力:**
- `masks/frame_xxx_mask.png` - バイナリマスク
- `rgba/frame_xxx_rgba.png` - 背景透明のRGBA画像

#### 2-4. システムは、SAM 3を使って選択された領域のRGB画像に相当するLiDARデータを抽出する

**技術詳細: 逆投影法による点群抽出**

3D点群の中から対象を「探す」のではなく、**全点群を2D画像平面に投影し、SAM 3のマスク内に落ちた点だけを抽出**する。

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
    # Step 1: 3D点群を2D画像平面に投影
    # 2D座標 = カメラ内部行列 × カメラ外部行列 × 3D座標
    projected_2d = project_to_image(pointcloud, camera_matrix)

    # Step 2: マスク内の点のみ抽出
    in_mask = mask[projected_2d[:, 1], projected_2d[:, 0]] > 0

    # Step 3: 深度フィルタ（パンチスルー対策）
    # 手前のゴミや後ろの壁を除外
    depth_valid = (pointcloud[:, 2] > min_depth) & (pointcloud[:, 2] < max_depth)

    return pointcloud[in_mask & depth_valid]
```

**課題と対策:**

| 課題 | 説明 | 対策 |
|------|------|------|
| パンチスルー問題 | マスク範囲に手前のゴミや後ろの壁が含まれる | 深度フィルタ（例: 1m〜2mのみ）、DBSCANクラスタリング |
| RGB/LiDARの位置ズレ | 撮影タイミングのズレ | マスクをDilation（数ピクセル膨張） |

**出力:**
- `pointcloud/object.ply` - 抽出されたオブジェクト点群

---

### 3. SAM 3Dオブジェクトの生成とLiDARデータによる補正

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│  RGBA画像   │ ──→ │   SAM 3D    │ ──→ │  3Dメッシュ     │
│  (2-3出力)  │     │  生成       │     │  (裏側含む)     │
└─────────────┘     └─────────────┘     └─────────────────┘
                                                │
                    ┌───────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│                    融合エンジン                         │
│  ┌─────────┐    ┌─────────┐    ┌─────────────────────┐ │
│  │ ICP     │ ─→ │ 可視    │ ─→ │ Shrinkwrap         │ │
│  │ 位置合せ│    │ 判定    │    │ 部分的吸着         │ │
│  └─────────┘    └─────────┘    └─────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                    ↓
            ┌─────────────────┐
            │  高精度3Dモデル │
            │  (補正済み)     │
            └─────────────────┘
```

#### 3-1. ユーザはAIエージェントを呼び出す

**実装内容:**
- Web GUIに「3D生成実行」ボタン
- LLMオーケストレーターが起動し、以下の処理を自動実行

#### 3-2. AIエージェントは、「2-3」で「SAM 3」を使って抽出されたRGB画像から「SAM 3D」を使って、3Dオブジェクトを作る

**新規開発コンポーネント:**
- `server/generation/sam3d_generate.py` - SAM 3D呼び出し

**処理フロー:**
```python
# SAM 3Dで3Dメッシュ生成
# 入力: 背景透明のRGBA画像
# 出力: 裏側を含む完全な3Dメッシュ
mesh = sam3d.generate(rgba_image)
mesh.export("sam3d/generated.obj")
```

**SAM 3Dの特性:**
- 綺麗なトポロジー（構造的に破綻のないメッシュ）
- 裏側もAI推測で生成
- **弱点:** 寸法が不正確（Unitless）

#### 3-3. AIエージェントは、「3-2」で「SAM 3D」によって生成された3DオブジェクトをBlenderに読み込む

**流用コンポーネント:**
- `blender_addon/` - Blenderアドオン骨格

**実装内容:**
- Blenderをバックグラウンド実行（`blender --background --python script.py`）
- SAM 3DメッシュをOBJ形式でインポート

#### 3-4. AIエージェントは、「2-4」で抽出されたLiDARデータを使って、「3-3」で抽出された3Dオブジェクトを補正する

**技術詳細: Template Fitting + Shrinkwrap**

「SAM 3Dで作った理想的な形の風船を、LiDARという実測の型枠の中で膨らませて、表面をピタッと吸着させる」

**Step 1: 大まかな位置合わせ (Coarse Alignment)**
```python
# 重心を合わせる
translation = lidar_center - sam3d_center

# スケールを合わせる（バウンディングボックス比）
scale = np.mean(lidar_size / sam3d_size)

# PCAで主軸を合わせる
rotation = pca_alignment(sam3d_mesh.vertices, lidar_points)
```

**Step 2: 精密な位置合わせ (Rigid ICP)**
```python
import open3d as o3d

# ICP（Iterative Closest Point）で位置・回転を微調整
result = o3d.pipelines.registration.registration_icp(
    source_pcd, target_pcd, threshold=0.05,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)
# fitness スコアで成功/失敗を判定
```

**Step 3: 可視判定 (Visibility Check)**
```python
def compute_visibility(mesh, camera_position):
    """カメラから見える頂点を判定"""
    visible_mask = []
    for vertex in mesh.vertices:
        # レイキャストで遮蔽判定
        if not mesh.ray_intersects_any(vertex, camera_position):
            visible_mask.append(True)  # 見える
        else:
            visible_mask.append(False) # 裏側
    return visible_mask
```

**Step 4: 部分的吸着 (Partial Shrinkwrap)**
```python
# Blender Python API
import bpy

# 可視頂点のVertex Groupを作成
vg = source_obj.vertex_groups.new(name="Visible")
vg.add(visible_vertices, 1.0, 'REPLACE')

# Shrinkwrap Modifierを追加（可視頂点のみ）
mod = source_obj.modifiers.new(name="Shrinkwrap", type='SHRINKWRAP')
mod.target = lidar_mesh
mod.wrap_method = 'NEAREST_SURFACEPOINT'
mod.vertex_group = "Visible"  # ★ここが重要

# 適用
bpy.ops.object.modifier_apply(modifier="Shrinkwrap")
```

**なぜ部分的吸着が必要か:**
- LiDARは「オモテ面」のデータしかない
- 単純にShrinkwrapすると、SAM 3Dの「裏側」メッシュまで無理やり引っ張られて潰れる
- **解決策:** 可視頂点のみLiDARに吸着、裏側はSAM 3Dの形状を維持

#### 3-5. AIエージェントは、「3-4」で補正された3DオブジェクトをBlenderに読み込む

**実装内容:**
- 補正済み3DモデルをBlenderで最終確認
- 境界スムージング（ラプラシアンスムージング）
- 出力フォーマット: OBJ / PLY / USDZ

**出力:**
```
session_xxx/output/
├── final_model.obj      # 最終3Dモデル
├── final_model.mtl      # マテリアル
├── metadata.json        # 処理ログ
└── quality_report.json  # 品質レポート
```

---

## LLMオーケストレーター

AIエージェントは処理パイプライン全体を指揮する「司令塔」として機能する。

### ツール定義

```python
tools = [
    {"name": "run_sam3d", "description": "SAM 3Dで3Dメッシュ生成"},
    {"name": "run_icp_alignment", "description": "ICP位置合わせ"},
    {"name": "run_blender_fusion", "description": "Blender融合処理"},
    {"name": "check_quality", "description": "品質チェック"}
]
```

### エラーハンドリング

| 状況 | LLMの判断 |
|------|----------|
| ICP fitness < 0.5 | 初期回転を変えて最大3回リトライ |
| SAM 3D形状が乖離 | CLIP類似度チェック、ユーザー確認 |
| Blender実行エラー | パラメータ調整、代替手法提案 |

---

## LiDAR-LLM-MCPからの流用一覧

| コンポーネント | 流用元 | 用途 |
|--------------|--------|------|
| iPadアプリ | `ipad_app/` | LiDAR + RGB取得 |
| WebSocketサーバー | `server/data_reception/` | データ受信 |
| SAM 3セグメント | `server/phase2_full/sam3_segmentation.py` | マスク生成 |
| 点群可視化 | `server/visualization/` | デバッグ |
| Blenderアドオン骨格 | `blender_addon/` | GUI |
| Docker環境 | `lidar-llm-mcp:sam3-tested` | SAM 3 + decord済み |

## 新規開発一覧

| コンポーネント | ファイル | 内容 |
|--------------|----------|------|
| SAM 3D統合 | `server/generation/sam3d_generate.py` | SAM 3D呼び出し |
| ICP位置合わせ | `server/fusion/icp_alignment.py` | Open3D ICP |
| 可視判定 | `server/fusion/visibility_check.py` | レイキャスト |
| 部分的吸着 | `server/fusion/shrinkwrap.py` | Blender Shrinkwrap |
| LLMオーケストレーター | `server/orchestrator/agent.py` | エージェント |
| Web GUI強化 | `server/web_gui/` | 画像選択、3D表示 |

---

## 開発フェーズ

### Phase 1: データ取得パイプライン
- [ ] LiDAR-LLM-MCPからコンポーネント移行
- [ ] Web GUI実装（画像一覧、点群表示）
- [ ] WebSocket受信サーバー起動確認

### Phase 2: セグメンテーション
- [ ] SAM 3統合・動作確認
- [ ] Web GUIクリック → マスク生成
- [ ] 点群抽出（逆投影法）実装

### Phase 3: 3D生成・融合
- [ ] SAM 3Dセットアップ
- [ ] ICP位置合わせ実装
- [ ] 可視判定実装
- [ ] Blender Shrinkwrap実装

### Phase 4: LLMオーケストレーション
- [ ] ツール定義
- [ ] エージェントフレームワーク構築
- [ ] エラーハンドリング実装

---

## 技術スタック

| カテゴリ | 技術 |
|---------|------|
| 3D生成 | SAM 3D (Meta), SAM 3 (Meta) |
| 点群処理 | Open3D, NumPy |
| メッシュ処理 | Blender API, PyMeshLab |
| LLMエージェント | LangGraph / AutoGen |
| LLM | Claude 3.5 Sonnet |
| 通信 | WebSocket (asyncio) |
| コンテナ | Docker, NGC PyTorch |
| Web GUI | Three.js, React |

---

## 参考資料

- [SAM 3 GitHub](https://github.com/facebookresearch/sam3)
- [Meta 3D Gen Paper](https://arxiv.org/abs/2411.19480)
- [Open3D ICP Tutorial](http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html)
- [Blender Shrinkwrap Modifier](https://docs.blender.org/manual/en/latest/modeling/modifiers/deform/shrinkwrap.html)
- プロジェクト内技術仕様書: [docs/SAM3D_LIDAR_FUSION.md](docs/SAM3D_LIDAR_FUSION.md)
