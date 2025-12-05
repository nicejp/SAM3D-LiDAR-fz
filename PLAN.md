# SAM3D-LiDAR-fz 実装計画書

**プロジェクト名:** SAM3D-LiDAR-fz (fusion)
**作成日:** 2025年12月3日
**最終更新:** 2025年12月4日
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
- `ipad_app/Sources/ARManager.swift` - ARセッション管理
- `ipad_app/Sources/ContentView.swift` - SwiftUI画面
- ARKitによるRGB (1920×1440) + LiDAR Depth (256×192) + メッシュ頂点取得

**修正が必要:**
現在のiPadアプリはメッシュ頂点のみを送信しており、**RGB画像は送信されていない**。
SAM 3を使用するにはRGB画像が必要なため、iPadアプリを修正する。

**推奨方式: ハイブリッド送信方式**

| データ種別 | 送信頻度 | 用途 |
|-----------|---------|------|
| メッシュ頂点 | 3fps | リアルタイム3D表示 |
| RGB画像 (JPEG) | 1fps | SAM 3用 |
| 深度マップ (Float32) | 1fps | 点群抽出用 |
| カメラパラメータ | 1fps | 座標変換用 |

**実装内容 (ARManager.swift修正):**
```swift
// session(_ session: ARSession, didUpdate frame: ARFrame) 内に追加
// RGB画像を送信（1fpsに制限）
if frameCount % 30 == 0 {  // 30fpsから1fpsに間引き
    if let capturedImage = frame.capturedImage,
       let jpegData = pixelBufferToJPEG(capturedImage) {
        // 既存のメッシュデータに加えてRGBも送信
        packet["rgb"] = jpegData.base64EncodedString()
    }
    if let depthMap = frame.sceneDepth?.depthMap {
        let depthData = depthMapToData(depthMap)
        packet["depth"] = depthData.base64EncodedString()
        packet["depth_width"] = CVPixelBufferGetWidth(depthMap)
        packet["depth_height"] = CVPixelBufferGetHeight(depthMap)
    }
    packet["intrinsics"] = cameraIntrinsicsToDict(frame.camera.intrinsics)
}
```

**送信データフォーマット (JSON):**
```json
{
  "type": "frame_data",
  "frame_id": 100,
  "timestamp": 1234567890.123,
  "vertices": [[x, y, z], ...],
  "vertex_count": 30000,
  "rgb": "<base64 encoded JPEG>",
  "depth": "<base64 encoded Float32 array>",
  "depth_width": 256,
  "depth_height": 192,
  "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 640.0, "cy": 480.0},
  "transform": [/* 4x4 transform matrix */]
}
```

#### 1-2. NVIDIA DGX Sparkに構築された受信サーバと表示Web GUIで、そのデータを順次受信し、画面表示していく

**流用コンポーネント:**
- `server/data_reception/websocket_server.py` - WebSocketサーバー（RGB受信対応済み）
- `server/visualization/web_viewer.py` - Three.jsベースのリアルタイム3Dビューワー

**web_viewer.pyの機能:**
- Three.jsによるブラウザ3D表示
- リアルタイムメッシュ更新（高さベースの色分け）
- カメラ位置マーカーと移動軌跡表示
- 自動ブラウザ起動機能

**起動方法:**
```bash
# Webビューワーを起動（ブラウザが自動で開く）
python -m server.visualization.web_viewer --port 8765

# 表示されるURL:
# - Web Viewer: http://localhost:8080
# - iPad接続先: ws://<サーバーIP>:8765
```

**保存されるデータ形式:**
```
experiments/session_YYYYMMDD_HHMMSS/
├── rgb/
│   ├── frame_000000.jpg      # RGB画像 (1920×1440)
│   ├── frame_000001.jpg
│   └── ...
├── depth/
│   ├── frame_000000.npy      # 深度マップ (256×192, Float32)
│   ├── frame_000001.npy
│   └── ...
├── camera/
│   ├── frame_000000.json     # カメラパラメータ
│   ├── frame_000001.json
│   └── ...
├── mesh/
│   └── accumulated.ply       # 累積メッシュ（オプション）
└── metadata.json             # セッション情報
```

#### 1-3. ユーザが、Web GUI上に3D point cloudが十分なだけプロットされたと感じたら処理を終了する

**実装内容:**
- Web GUIに「スキャン終了」ボタンを追加
- 点群の累積表示と点数カウンター表示
- iPadアプリ側の「停止」ボタンでも終了可能

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

**流用コンポーネント:**
- `server/phase2_full/click_selector.py` - OpenCV GUIでクリック座標を選択

**実装内容:**
- Web GUIまたはOpenCV GUIでサムネイル一覧表示
- 画像クリックで選択・拡大表示

**使い方 (OpenCV版):**
```bash
# 画像でクリック座標を選択
python -m server.phase2_full.click_selector experiments/session_xxx
# → クリック位置の座標 (x, y) を取得
```

#### 2-2. DGX Spark上にあるWeb GUIでは、選択された1枚を拡大し、「SAM 3」を使った領域選択機能をユーザに提供する

**流用コンポーネント:**
- `server/phase2_full/sam3_segmentation.py` - SAM 3セグメンテーション
- `server/phase2_full/sam3_demo.py` - Gradio Webデモ（ブラウザでクリックセグメント）

**SAM 3のセットアップ (Docker内):**
```bash
# lidar-llm-mcp:sam3-tested コンテナを使用
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --network=host \
  -v ~/SAM3D-LiDAR-fz:/workspace \
  -it lidar-llm-mcp:sam3-tested

# コンテナ内でSAM 3をクローン（初回のみ）
cd /workspace
git clone https://github.com/facebookresearch/sam3.git

# PYTHONPATHを設定（毎回必要）
export PYTHONPATH=/workspace:/workspace/sam3:$PYTHONPATH
```

**使い方:**
```bash
# Webデモ（ブラウザでクリック操作）
python -m server.phase2_full.sam3_demo
# → ブラウザでアクセス: http://localhost:7860

# コマンドラインで直接実行
python -m server.phase2_full.sam3_segmentation experiments/session_xxx --click 512,384
```

#### 2-3. システムは、Web GUI上で選択された領域のRGB画像を「SAM 3」を使って抽出する

**処理フロー:**
```python
# SAM 3によるマスク生成
masks, scores, logits = segmentor.segment_image(
    image,
    point_coords=[(x, y)],
    point_labels=[1],  # 1=前景
    multimask_output=True
)

# ベストマスクを選択
best_idx = np.argmax(scores)
best_mask = masks[best_idx]

# マスクを使って背景透明化（RGBA画像生成）
rgba_image = np.dstack([original_image, (best_mask * 255).astype(np.uint8)])
# → SAM 3D への入力用
```

**出力:**
- `output/segmented/masks/mask_000000.png` - バイナリマスク
- `output/segmented/masks/mask_000000.npy` - マスク配列

#### 2-4. システムは、SAM 3を使って選択された領域のRGB画像に相当するLiDARデータを抽出する

**技術詳細: 逆投影法による点群抽出**

3D点群の中から対象を「探す」のではなく、**全点群を2D画像平面に投影し、SAM 3のマスク内に落ちた点だけを抽出**する。

**流用コンポーネント:**
- `server/phase2_full/sam3_segmentation.py` 内の `masked_depth_to_pointcloud()` 関数

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
- `output/segmented/segmented_object.ply` - 抽出されたオブジェクト点群（色付き）

---

### 3. SAM 3Dオブジェクトの生成とLiDARデータによる補正

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│  RGBA画像   │ ──→ │   SAM 3D    │ ──→ │  3Dメッシュ     │
│  (2-3出力)  │     │  Objects    │     │  (裏側含む)     │
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
- `server/generation/sam3d_generate.py` - SAM 3D Objects呼び出し

**処理フロー:**
```python
# SAM 3D Objectsで3Dメッシュ生成
# 入力: RGB画像 + バイナリマスク
# 出力: Gaussian Splat → PLYファイル
from sam_3d_objects import Inference

inference = Inference(config_path, compile=False)
output = inference(image, mask, seed=42)
output["gs"].save_ply("sam3d/generated.ply")
```

**SAM 3D Objectsの特性:**
- 綺麗なトポロジー（構造的に破綻のないメッシュ）
- 裏側もAI推測で生成（オクルージョン対応）
- **弱点:** 寸法が不正確（Unitless）

#### 3-3. AIエージェントは、「3-2」で「SAM 3D」によって生成された3DオブジェクトをBlenderに読み込む

**流用コンポーネント:**
- `blender_addon/` - Blenderアドオン骨格

**実装内容:**
- Blenderをバックグラウンド実行（`blender --background --python script.py`）
- SAM 3DメッシュをPLY/OBJ形式でインポート

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

#### 融合パイプラインの使い方

**推奨: 自動融合（auto_fuse）**

SAM 3D Gaussian SplatとLiDAR点群を1コマンドで融合する。SciPyベースで実装されており、DGX Sparkホスト上で安定動作する。

```bash
# 自動融合を実行
python3 -m server.fusion.auto_fuse \
    --sam3d sam3d_output.ply \
    --lidar lidar_points.ply \
    -o fused_output.ply

# オプション
python3 -m server.fusion.auto_fuse \
    --sam3d sam3d_output.ply \
    --lidar lidar_points.ply \
    --threshold 0.5 \          # スナップ閾値（デフォルト1.0）
    --scale lidar \            # 出力スケール（lidar/sam3d）
    -o fused_output.ply
```

**出力ファイル:**
- `fused_output.ply` - 融合済みGaussian Splat（LiDARスケール）
- `fused_output.points.ply` - 融合後の点群（確認用）

**Open3D版パイプライン（参考）:**
```bash
# 個別ファイルを指定して実行（Open3Dが必要、DGX Sparkでは不安定）
python -m server.fusion.run \
    --sam3d sam3d_output.ply \
    --lidar lidar_points.ply \
    --camera 0 0 2 \
    -o fused_output.ply

# セッションディレクトリから自動検出
python -m server.fusion.run experiments/session_xxx
```

**各モジュールの個別実行:**
```bash
# ICP位置合わせのみ
python -m server.fusion.icp_alignment sam3d.ply lidar.ply -o aligned.ply --visualize

# 可視判定のみ
python -m server.fusion.visibility_check mesh.ply --camera 0 0 2 --visualize

# Shrinkwrapのみ
python -m server.fusion.shrinkwrap mesh.ply lidar.ply -o shrinkwrap_result.ply
```

**Pythonからの使用:**
```python
from server.fusion import ICPAligner, VisibilityChecker, ShrinkwrapProcessor
from server.fusion.run import FusionPipeline
import numpy as np

# 方法1: パイプライン全体を実行
pipeline = FusionPipeline()
result = pipeline.run(
    sam3d_mesh_path="sam3d_output.ply",
    lidar_pcd_path="lidar_points.ply",
    camera_positions=np.array([0, 0, 2]),
    output_path="fused_output.ply"
)

# 方法2: 個別モジュールを使用
aligner = ICPAligner()
icp_result = aligner.align("sam3d.ply", "lidar.ply")

checker = VisibilityChecker()
visible_mask = checker.compute_visibility(mesh, camera_positions)

shrinkwrap = ShrinkwrapProcessor()
shrinkwrap_result = shrinkwrap.process("mesh.ply", "lidar.ply", visible_mask)
```

---

## SAM 3D Objectsのセットアップ

### 入手方法

Meta社が公式にリリースしたSAM 3D Objectsを使用する。

**GitHub:** https://github.com/facebookresearch/sam-3d-objects

### インストール手順

```bash
# 1. リポジトリをクローン
cd /workspace
git clone https://github.com/facebookresearch/sam-3d-objects.git
cd sam-3d-objects

# 2. セットアップ手順に従う
# doc/setup.md を参照

# 3. Hugging Faceからチェックポイントをダウンロード
# 事前にHugging Faceでアクセスリクエストが必要
pip install huggingface_hub
huggingface-cli download facebook/sam-3d-objects --local-dir checkpoints/sam-3d-objects

# 4. 動作確認
python -c "from sam_3d_objects import Inference; print('SAM 3D Objects OK')"
```

### WSL2 (x86_64/Windows) でのセットアップ手順

WSL2 + NVIDIA GPU (例: GeForce RTX 4060 Ti) でのセットアップ手順。

#### 前提条件

- Windows 11 + WSL2 (Ubuntu 22.04/24.04)
- NVIDIA GPU + ドライバー (Windows側にインストール)
- Miniconda/Anaconda

#### Step 1: Conda環境の作成

```bash
# sam3d専用環境を作成
conda create -n sam3d python=3.11 -y
conda activate sam3d
```

#### Step 2: PyTorch + CUDAのインストール

```bash
# PyTorch 2.5.1 + CUDA 12.4 (推奨)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDAが認識されているか確認
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

#### Step 3: SAM 3D Objectsのクローン

```bash
cd ~
git clone https://github.com/facebookresearch/sam-3d-objects.git
cd sam-3d-objects
```

#### Step 4: 依存関係のインストール

```bash
# fvcore, iopath (PyTorch3Dの依存)
pip install "git+https://github.com/facebookresearch/fvcore"
pip install "git+https://github.com/facebookresearch/iopath"

# その他の依存関係
pip install hydra-core omegaconf einops hatch-requirements-txt
pip install gradio pillow numpy scipy
```

#### Step 5: PyTorch3Dのインストール（ソースビルド）

```bash
# x86_64でもプリビルドホイールが対応していない場合、ソースビルドが必要

# 環境変数を設定
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 40シリーズの場合

# ソースからビルド（約5-10分）
pip install "git+https://github.com/facebookresearch/pytorch3d.git@main" --no-build-isolation
```

**TORCH_CUDA_ARCH_LISTの値:**
| GPU | 値 |
|-----|-----|
| RTX 40シリーズ (4060Ti, 4070, 4080, 4090) | 8.9 |
| RTX 30シリーズ (3060, 3070, 3080, 3090) | 8.6 |
| RTX 20シリーズ | 7.5 |
| DGX Spark GB110 | 12.0 |

#### Step 6: SAM 3D Objects本体のインストール

```bash
cd ~/sam-3d-objects

# 開発モードでインストール（依存関係は手動でインストール済みなので--no-deps）
pip install -e . --no-deps
```

#### Step 7: チェックポイントのダウンロード

```bash
cd ~/sam-3d-objects

# Hugging Face CLIをインストール
pip install huggingface_hub

# チェックポイントをダウンロード
# ※事前にHugging Faceでfacebook/sam-3d-objectsへのアクセスリクエストが必要
mkdir -p checkpoints/hf
huggingface-cli download facebook/sam-3d-objects --local-dir checkpoints/hf
```

**注意:** Hugging Faceで https://huggingface.co/facebook/sam-3d-objects にアクセスし、「Request access」をクリックしてアクセス権を取得する必要があります（通常即時承認）。

#### Step 8: 動作確認

```bash
cd ~/sam-3d-objects
conda activate sam3d

# モデル読み込みテスト
python -c "
import sys
sys.path.append('notebook')
from inference import Inference

config_path = 'checkpoints/hf/pipeline.yaml'
print('Loading SAM 3D Objects model...')
inference = Inference(config_path, compile=False)
print('Model loaded successfully!')
"
```

成功すると以下が表示される:
```
Loading SAM 3D Objects model...
Model loaded successfully!
```

#### Step 9: Web UIの起動と3D生成

```bash
# WSL2ターミナルで実行
cd ~/SAM3D-LiDAR-fz
git pull  # 最新版を取得
conda activate sam3d

# Web UIを起動
python server/generation/sam3d_web_ui.py --port 8000
```

**DGX Sparkからアクセス:**
1. WSL2のIPアドレスを確認: `hostname -I | awk '{print $1}'`
2. **Firefox**で `http://<WSL2のIP>:8000` にアクセス（※Chromeではアップロードエラーが発生する場合あり）
3. RGBA画像（背景透明PNG）をアップロード
4. シード値を設定（オプション）
5. 「3D生成」ボタンをクリック
6. 生成完了後、PLYファイルをダウンロード

**生成時間の目安（RTX 4060 Ti）:**
- モデル読み込み: 約30秒（初回のみ）
- 3D生成: 約7-8分

**出力ファイル:**
- 保存先: `~/sam3d_outputs/`
- ファイル名: `sam3d_YYYYMMDD_HHMMSS_seed{シード値}.ply`

### DGX Spark (ARM64/aarch64) での追加手順

DGX SparkはARM64アーキテクチャのため、PyTorch3Dのプリビルドホイールが存在しない。
ソースからビルドする必要がある。

#### PyTorch3Dのインストール（ARM64専用）

```bash
# 1. 依存関係のインストール
pip install "git+https://github.com/facebookresearch/fvcore"
pip install "git+https://github.com/facebookresearch/iopath"

# 2. 環境変数の設定（重要！）
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="12.0"  # DGX Spark GB110用、仮想モード互換

# 3. mainブランチからソースビルド
pip install "git+https://github.com/facebookresearch/pytorch3d.git@main" --no-build-isolation
```

**注意点:**
- `TORCH_CUDA_ARCH_LIST="12.0"` はDGX SparkのCUDA計算能力12.1に対応（仮想モード互換のため12.0を指定）
- `--no-build-isolation` は既存のPyTorchを使用するために必要
- ビルドには約10〜15分かかる
- 古いタグ（v0.7.7など）はPyTorch 2.10と非互換、必ずmainブランチを使用

#### SAM 3D Objects本体のインストール（ARM64）

```bash
cd /workspace/sam-3d-objects

# hatch-requirements-txtプラグインをインストール
pip install hatch-requirements-txt

# 依存関係を個別インストール
pip install hydra-core omegaconf einops

# --no-depsで本体のみインストール
pip install -e . --no-deps
```

### ARM64 (DGX Spark) での制限事項

**注意:** DGX Spark (ARM64/aarch64) ではSAM 3D Objectsの完全な動作は現時点で困難。

**動作しない依存関係:**
- `pytorch3d.renderer`: CUDAリンクエラー（ARM64非対応）
- `open3d`: ARM64ホイールなし
- `kaolin`: ARM64ホイールなし、usd-coreも非対応
- `spconv`: ビルド複雑（pccm/cumm依存）

**代替案:**
1. **Meta Web Demo** (後述) - 検証・プロトタイプ用
2. **x86_64環境** (WSL2, クラウドGPU) - 本格運用用
3. **LiDARのみ** - SAM 3D部分をスキップ

### Meta公式デモサイト

環境構築なしでSAM 3D Objectsを試せる公式デモ:

**URL:** https://www.aidemos.meta.com/segment-anything/editor/convert-image-to-3d

**使い方:**
1. 画像をアップロード
2. 対象物をクリックしてマスクを選択
3. 「Convert to 3D」で3Dモデル生成
4. 結果をダウンロード

**利点:** 無料、環境構築不要、即座に結果確認可能
**制限:** API連携不可、バッチ処理不可、手動操作のみ

### 基本的な使い方

```python
from sam_3d_objects import Inference

# モデルを読み込み
config_path = "checkpoints/sam-3d-objects/pipeline.yaml"
inference = Inference(config_path, compile=False)

# 画像とマスクから3D生成
output = inference(image, mask, seed=42)

# PLYファイルとして保存
output["gs"].save_ply("output.ply")
```

### 入力/出力フォーマット

| 項目 | フォーマット |
|------|------------|
| 入力画像 | RGB画像 (PIL Image または NumPy array) |
| 入力マスク | バイナリマスク (対象物=1, 背景=0) |
| 出力 | Gaussian Splat → PLYファイル |

### Web UI（リモートアクセス用）

WSL2上のSAM 3D Objectsに、DGX Sparkなど他のホストからアクセスするためのWeb UI。

**起動方法（WSL2側）:**
```bash
cd ~/sam-3d-objects
conda activate sam3d

# Web UIを起動
python /path/to/SAM3D-LiDAR-fz/server/generation/sam3d_web_ui.py --port 8000

# または、SAM3D-LiDAR-fzディレクトリから
cd ~/SAM3D-LiDAR-fz
python -m server.generation.sam3d_web_ui --port 8000
```

**アクセス方法（DGX Spark側）:**
```bash
# WSL2のIPアドレスを確認（WSL2側で実行）
hostname -I | awk '{print $1}'

# ブラウザでアクセス
# http://<WSL2のIP>:8000
```

**オプション:**
| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--host` | バインドするホスト | 0.0.0.0 |
| `--port` | ポート番号 | 8000 |
| `--share` | Gradio公開リンクを作成 | False |
| `--sam3d-path` | sam-3d-objectsディレクトリ | ~/sam-3d-objects |
| `--output-dir` | PLY出力ディレクトリ | ~/sam3d_outputs |

**使い方:**
1. RGBA画像（背景透明のPNG）をアップロード
2. シード値を設定（オプション、再現性のため）
3. 「3D生成」ボタンをクリック
4. 生成されたPLYファイルをダウンロード

---

## LLM構成

### 使用するLLM

| LLM | 用途 | 備考 |
|-----|------|------|
| **gpt-oss 120B** | メイン処理 | ollama経由、導入済み |
| **qwen3-coder:30b** | 軽量代替 | 必要に応じてダウンロード |
| **Claude Desktop** | Blender操作 | Blender MCP経由 |

### gpt-oss 120B (導入済み)

```bash
# サーバー起動
sudo systemctl start ollama

# 停止
sudo systemctl stop ollama

# 動作確認
ollama run gpt-oss:120b "こんにちは"
```

### qwen3-coder:30b (未導入)

```bash
# ダウンロード（約18GB）
ollama pull qwen3-coder:30b

# 使用
ollama run qwen3-coder:30b "Blender Pythonで立方体を作成するコードを書いてください"
```

### Blender MCP (Claude Desktop連携)

Claude APIを直接使用せず、**Blender MCPプラグイン**を通じてClaude Desktopと通信する。

**構成:**
```
┌─────────────┐     MCP通信      ┌─────────────────┐
│   Blender   │ ←──────────────→ │  Claude Desktop │
│  (MCP対応)  │                  │                 │
└─────────────┘                  └─────────────────┘
```

**Blender MCPプラグインの確認:**
```
Blender → Edit → Preferences → Add-ons → "Claude" で検索
```

---

## 評価計画

### 使用データセット

**Replica Dataset** (Meta社提供)
- 18個の高精度3D室内シーン再構築データ
- HDRテクスチャ、セマンティックラベル付き
- Ground Truthとして使用

**GitHub:** https://github.com/facebookresearch/Replica-Dataset

### Replica Datasetのセットアップ

```bash
# 1. リポジトリをクローン
git clone https://github.com/facebookresearch/Replica-Dataset.git
cd Replica-Dataset

# 2. データセットをダウンロード
# Linuxの場合
./download.sh

# 3. 配置
# ~/datasets/Replica-Dataset/ に保存

# 4. シンボリックリンク作成
ln -s ~/datasets/Replica-Dataset ~/SAM3D-LiDAR-fz/datasets/Replica
```

### 評価指標

| 指標 | 説明 |
|------|------|
| **Chamfer Distance** | 点群間の平均距離 |
| **Hausdorff Distance** | 点群間の最大距離 |
| **IoU (Intersection over Union)** | メッシュの重なり度合い |
| **寸法精度** | Ground Truthとのサイズ比較 |

### 評価手順

```bash
# 1. Replica Datasetから画像・深度を取得
python scripts/extract_replica_frames.py datasets/Replica/room_0

# 2. SAM 3 + SAM 3D + 融合パイプラインを実行
python -m server.pipeline experiments/replica_room_0

# 3. 評価スクリプトを実行
python scripts/evaluate.py \
  --prediction experiments/replica_room_0/output/final_model.obj \
  --ground_truth datasets/Replica/room_0/mesh.ply
```

---

## LiDAR-LLM-MCPからの流用一覧

### そのまま使えるもの

| コンポーネント | パス | 用途 |
|--------------|------|------|
| WebSocketサーバー | `server/data_reception/websocket_server.py` | データ受信（RGB対応済み） |
| Web 3Dビューワー | `server/visualization/web_viewer.py` | リアルタイム表示 |
| 点群可視化 | `server/visualization/pointcloud_viewer.py` | デバッグ |
| SAM 3セグメント | `server/phase2_full/sam3_segmentation.py` | マスク生成 |
| クリック選択 | `server/phase2_full/click_selector.py` | 座標選択 |
| SAM 3デモ | `server/phase2_full/sam3_demo.py` | Gradio Web UI |
| Blenderアドオン骨格 | `blender_addon/` | GUI |
| Docker環境 | `lidar-llm-mcp:sam3-tested` | SAM 3 + decord済み |

### 修正が必要なもの

| コンポーネント | 変更内容 |
|--------------|----------|
| `ipad_app/Sources/ARManager.swift` | RGB画像・深度マップ送信を追加 |
| `blender_addon/operators.py` | Shrinkwrap処理を追加 |

### 新規開発が必要なもの

| コンポーネント | ファイル | 内容 | 状態 |
|--------------|----------|------|------|
| SAM 3D Web UI | `server/generation/sam3d_web_ui.py` | SAM 3D Objects Gradio UI | ✅ 完了 |
| ICP位置合わせ | `server/fusion/icp_alignment.py` | Open3D ICP | ✅ 完了 |
| 可視判定 | `server/fusion/visibility_check.py` | レイキャスト | ✅ 完了 |
| 部分的吸着 | `server/fusion/shrinkwrap.py` | Open3D/Blender Shrinkwrap | ✅ 完了 |
| 融合パイプライン | `server/fusion/run.py` | 全モジュール統合 | ✅ 完了 |
| **自動融合** | `server/fusion/auto_fuse.py` | SciPyベース自動融合 | ✅ 完了 |
| Gaussian Splat変換 | `server/fusion/gaussian_splat.py` | GS PLY読み書き | ✅ 完了 |
| LLMオーケストレーター | `server/orchestrator/agent.py` | エージェント | 未実装 |

---

## Docker環境

### 使用イメージ

```bash
# ベースイメージ（NVIDIA NGC）
nvcr.io/nvidia/pytorch:25.11-py3

# カスタムイメージ（SAM 3 + decord インストール済み）
lidar-llm-mcp:sam3-tested
```

### コンテナ起動コマンド

```bash
docker run --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network=host \
  -v ~/SAM3D-LiDAR-fz:/workspace \
  -v ~/datasets:/workspace/datasets \
  -it lidar-llm-mcp:sam3-tested

# コンテナ内でPYTHONPATHを設定
export PYTHONPATH=/workspace:/workspace/sam3:$PYTHONPATH
```

### Dockerフラグの説明

| フラグ | 説明 |
|-------|------|
| `--gpus all` | GPU使用 |
| `--ipc=host` | ホストの共有メモリを使用（PyTorchのメモリエラー対策） |
| `--ulimit memlock=-1` | メモリロック制限を解除 |
| `--network=host` | ホストのネットワークを直接使用 |

### 融合パイプライン用のセットアップ

融合パイプライン（ICP位置合わせ、可視判定、Shrinkwrap）は**DGX Sparkのホスト上**で実行する。

**注意:** SAM 3用DockerコンテナはPython 3.12を使用しており、Open3DはPython 3.12に対応していない（3.8〜3.11のみ）。そのため融合パイプラインはDockerコンテナ内ではなくホスト上で実行する。

**処理の使い分け:**

| 処理 | 実行場所 | 理由 |
|-----|---------|------|
| LiDAR点群取得 | DGX Spark (ホスト) | WebSocket受信 |
| SAM 3セグメンテーション | DGX Spark (Docker) | Python 3.12 + SAM 3 |
| SAM 3D 3D生成 | WSL2 | PyTorch3D必要 |
| **融合パイプライン** | **DGX Spark (ホスト)** | Open3D (Python 3.11以下) |

**DGX Sparkホスト上でのセットアップ:**
```bash
# Pythonバージョン確認（3.11以下が必要）
python3 --version

# Open3Dをインストール
pip install open3d

# 動作確認
python3 -c "import open3d; print(f'Open3D {open3d.__version__} OK')"
```

**融合パイプラインの動作確認:**
```bash
cd ~/SAM3D-LiDAR-fz

# ヘルプ表示
python3 -m server.fusion.icp_alignment --help
python3 -m server.fusion.visibility_check --help
python3 -m server.fusion.shrinkwrap --help

# パイプライン全体
python3 -m server.fusion.run --help
```

**データフロー:**
```
DGX Spark                              WSL2
─────────────────────────────────────────────────────
[Docker] SAM 3でRGBA生成
    ↓
    ──────→ RGBA画像転送 ──────→ [WSL2] SAM 3Dで3D生成
                                        ↓
    ←────── PLYファイル転送 ←──────────┘
    ↓
[ホスト] 融合パイプライン実行
    ↓
高精度3Dモデル完成
```

---

## 開発フェーズ

### Phase 1: データ取得パイプライン
- [x] LiDAR-LLM-MCPからコンポーネント移行 ✅ (2025/12/4)
- [x] iPadアプリにRGB/深度送信機能を追加 ✅ (2025/12/4)
- [x] WebSocketサーバー修正（RGB/深度保存対応） ✅ (2025/12/4)
- [x] Web GUI動作確認 ✅ (2025/12/4) - 26フレーム取得成功

### Phase 2: セグメンテーション
- [x] SAM 3統合・動作確認 ✅ (2025/12/4)
- [x] Web GUIクリック → マスク生成 ✅ (2025/12/4) - sam3_demo UI改善
- [x] 点群抽出（逆投影法）実装 ✅ (2025/12/4) - 3D点群出力(PLY)動作確認
- [x] RGBA画像生成 ✅ (2025/12/4) - SAM 3D用背景透明画像出力

### Phase 3: 3D生成・融合
- [x] SAM 3D Objectsセットアップ ✅ (2025/12/5) - WSL2環境で動作確認
- [x] SAM 3D Web UI実装 ✅ (2025/12/5) - リモートアクセス用Gradio UI
- [x] SAM 3D 3D生成テスト ✅ (2025/12/5) - PLYファイル生成・ダウンロード成功
- [x] ICP位置合わせ実装 ✅ (2025/12/5) - `server/fusion/icp_alignment.py`
- [x] 可視判定実装 ✅ (2025/12/5) - `server/fusion/visibility_check.py`
- [x] Blender Shrinkwrap実装 ✅ (2025/12/5) - `server/fusion/shrinkwrap.py`
- [x] 融合パイプライン統合 ✅ (2025/12/5) - `server/fusion/run.py`
- [x] 自動融合プログラム実装 ✅ (2025/12/5) - `server/fusion/auto_fuse.py`

### Phase 3.5: 多視点LiDAR融合（高密度化）

**課題:** 単一フレームのLiDARデータが粗すぎて、SAM 3Dとの融合で品質が低下する

**解決策:** SAM 2のビデオトラッキング機能を使用して、複数フレームの点群を自動統合

#### データ取得方式

**方式A: 自作iPadアプリ（現行）**
- WebSocketでリアルタイム送信
- RGB + 深度 + カメラパラメータ
- 制御しやすいが、同期精度に課題

**方式B: Omniscientアプリ（推奨）**
- App Store: https://apps.apple.com/app/omniscient/id1609646889
- 動画 + LiDAR深度 + カメラポーズを同期取得
- ARKit Recording形式でエクスポート
- 高精度なカメラトラッキング

**Omniscientの出力形式:**
```
recording/
├── video.mp4           # RGB動画
├── depth/              # LiDAR深度フレーム
│   ├── 0000.png        # 16bit深度画像
│   ├── 0001.png
│   └── ...
├── camera_poses.json   # 各フレームのカメラポーズ
└── intrinsics.json     # カメラ内部パラメータ
```

#### 実装計画

- [ ] Omniscientデータ読み込み
  - 動画・深度・カメラポーズのパース
  - `server/multiview/omniscient_loader.py`
- [ ] SAM 2ビデオトラッキング統合
  - 1フレームでクリックセグメント → 他フレームに自動伝播
  - `server/multiview/sam2_tracker.py`
- [ ] 多視点点群統合
  - 各フレームからセグメント点群を抽出
  - カメラポーズで位置合わせ（ICPより高精度）
  - `server/multiview/pointcloud_fusion.py`
- [ ] 統合パイプライン
  - Omniscientデータから自動処理
  - `server/multiview/run.py`

#### 処理フロー

```
┌─────────────────────────────────────────────────────────────┐
│ Omniscientで撮影                                            │
│   動画(MP4) + 深度(PNG) + カメラポーズ(JSON)                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ フレーム0でクリックセグメント                               │
│   SAM 2でオブジェクトを指定                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ SAM 2がフレーム1〜Nにマスク自動伝播                         │
│   ビデオトラッキング機能                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 各フレームの点群を抽出                                      │
│   マスク × 深度 × カメラパラメータ → 3D点群                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ カメラポーズで位置合わせ・統合                              │
│   Omniscientの高精度ポーズを使用（ICP不要）                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 高密度統合点群 → SAM 3Dと融合                               │
└─────────────────────────────────────────────────────────────┘
```

#### 期待効果

| 項目 | 現状（単一フレーム） | 改善後（10フレーム統合） |
|------|---------------------|-------------------------|
| 点群密度 | 4,831点 | 50,000点以上 |
| 遮蔽部分 | 欠損あり | 補完される |
| ノイズ | 単一計測 | 平均化で低減 |
| 位置合わせ精度 | - | Omniscientポーズで高精度 |

### Phase 4: LLMオーケストレーション

**目的:** 融合処理の自動化と品質評価によるリトライ制御

#### 実装計画

- [x] gpt-oss 120bセットアップ ✅
  - ollamaでダウンロード済み
  - `sudo systemctl start ollama` で起動
  - `ollama run gpt-oss:120b` で動作確認済み
- [x] DGX SparkにClaude Desktopセットアップ ✅
  - Claude Desktopインストール済み
  - Blender MCP連携設定完了
- [ ] ツール定義
  - SAM 3セグメンテーション実行
  - SAM 3D 3D生成実行
  - 融合処理実行
  - 品質評価実行
- [ ] 品質評価スクリプト作成
  - Chamfer Distance計算
  - スナップ率評価
  - 2Dレンダリング画像生成
  - `server/evaluation/quality_check.py`
- [ ] エージェントフレームワーク構築
  - LLMがメトリクスを解釈
  - パラメータ調整・リトライ指示
  - `server/orchestrator/agent.py`
- [ ] Blender MCP連携
- [ ] エラーハンドリング実装

#### LLM品質評価・リトライ機構

**処理フロー:**
```
┌─────────────────────────────────────────────────────────────┐
│ 融合実行                                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 品質評価スクリプト（Python）                                │
│   - Chamfer Distance計算                                    │
│   - スナップ率                                              │
│   - 点群密度                                                │
│   - 2Dレンダリング画像生成（front/side/top）                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ LLMが判断                                                   │
│   - 数値メトリクス（JSON）を解釈                            │
│   - レンダリング画像を確認（マルチモーダルLLM）             │
│   - 閾値未達なら → リトライ指示                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ リトライ時にパラメータ調整                                  │
│   - --threshold 値を変更                                    │
│   - SAM 3Dのseed値を変更                                    │
│   - 別フレームを選択                                        │
└─────────────────────────────────────────────────────────────┘
```

**品質レポート形式（LLMへの入力）:**
```json
{
  "fusion_result": {
    "snap_ratio": 0.48,
    "snapped_points": 2345,
    "mean_distance": 0.032,
    "median_distance": 0.028
  },
  "quality_metrics": {
    "chamfer_distance": 0.045,
    "point_density": 4831,
    "coverage_ratio": 0.85
  },
  "render_images": [
    "renders/front.png",
    "renders/side.png",
    "renders/top.png"
  ],
  "parameters_used": {
    "threshold": 1.0,
    "sam3d_seed": 42
  }
}
```

**LLM判断基準:**
| メトリクス | 合格基準 | リトライ時の調整 |
|-----------|---------|-----------------|
| snap_ratio | > 0.3 (30%) | threshold を上げる |
| chamfer_distance | < 0.05 | SAM 3D seedを変更 |
| coverage_ratio | > 0.8 (80%) | 別フレームを選択 |

**リトライ制御:**
- 最大リトライ回数: 3回
- リトライ間隔: パラメータ変更後即時
- 成功条件: 全メトリクスが閾値以内

### Phase 5: 評価

**目的:** 提案手法の有効性を定量的に実証する

#### 評価フレームワーク

Replica Datasetを使用して、3段階の手法を比較評価する。

```
┌────────────────────────────────────────────────────────────────┐
│ Replica Dataset                                                │
│   RGB画像 + 深度/点群 + Ground Truth 3Dメッシュ               │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ (1) SAM 3D単体                                                 │
│     RGB → SAM 3D → 3Dモデル                                   │
│     ※ベースライン（比較対象）                                 │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ (2) SAM 3D + LiDAR融合                                         │
│     RGB → SAM 3D → 融合(auto_fuse) → 精緻化モデル             │
│     ※提案手法の基本効果                                       │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ (3) SAM 3D + LiDAR融合 + LLMオーケストレーション               │
│     RGB → SAM 3D → 融合 → LLM評価・リトライ → 最適化モデル    │
│     ※提案手法の完全版                                         │
└────────────────────────────────────────────────────────────────┘
```

#### 比較評価表（論文用）

| 手法 | Chamfer Distance↓ | IoU↑ | 寸法誤差↓ |
|------|------------------|------|----------|
| (1) SAM 3D単体 | - | - | - |
| (2) + LiDAR融合 | - | - | - |
| (3) + LLM最適化 | - | - | - |
| Ground Truth | 0.00 | 1.00 | 0.0% |

#### 研究の新規性を示すポイント

| 比較 | 示せること |
|------|-----------|
| (1) vs (2) | LiDAR融合による精度向上効果 |
| (2) vs (3) | LLMオーケストレーションによる追加改善 |
| (1) vs (3) | 提案手法全体の効果 |

#### 実装計画

- [ ] Replica Datasetセットアップ
  - データセットダウンロード
  - RGB/深度/Ground Truth抽出スクリプト
  - `scripts/extract_replica_frames.py`
- [ ] 評価スクリプト作成
  - Chamfer Distance計算
  - IoU計算
  - 寸法誤差計算
  - `server/evaluation/metrics.py`
- [ ] 自動評価パイプライン
  - 全シーン一括評価
  - 結果CSV/JSON出力
  - `server/evaluation/run_evaluation.py`
- [ ] 定量評価実施
  - 複数シーンで評価
  - 統計処理（平均、標準偏差）
- [ ] 結果まとめ
  - 比較表作成
  - グラフ生成
  - 論文用図表

#### 評価手順

```bash
# 1. Replica Datasetから画像・深度を取得
python scripts/extract_replica_frames.py datasets/Replica/room_0

# 2. 各手法で3Dモデル生成
# (1) SAM 3D単体
python -m server.generation.sam3d_generate input.png -o sam3d_only.ply

# (2) SAM 3D + LiDAR融合
python -m server.fusion.auto_fuse \
    --sam3d sam3d_only.ply \
    --lidar replica_depth.ply \
    -o fused.ply

# (3) SAM 3D + LiDAR + LLM最適化
python -m server.orchestrator.agent \
    --input input.png \
    --lidar replica_depth.ply \
    -o optimized.ply

# 3. 評価スクリプトを実行
python -m server.evaluation.run_evaluation \
    --predictions sam3d_only.ply fused.ply optimized.ply \
    --ground_truth datasets/Replica/room_0/mesh.ply \
    -o evaluation_results.json
```

---

## 技術スタック

| カテゴリ | 技術 |
|---------|------|
| 3D生成 | SAM 3D Objects (Meta), SAM 3 (Meta) |
| 点群処理 | Open3D, NumPy |
| メッシュ処理 | Blender API (bpy), PyMeshLab |
| LLM (ローカル) | gpt-oss 120B, qwen3-coder:30b (ollama) |
| LLM (リモート) | Claude Desktop (Blender MCP経由) |
| 通信 | WebSocket (asyncio) |
| コンテナ | Docker, NGC PyTorch |
| Web GUI | Three.js, Gradio |
| iOS開発 | Swift, ARKit, RealityKit |

---

## ディレクトリ構成

```
SAM3D-LiDAR-fz/
├── PLAN.md                      # 本ファイル
├── README.md
├── requirements.txt
│
├── docs/
│   ├── SAM3D_LIDAR_FUSION.md    # 技術仕様書
│   └── new_project/
│       ├── SETUP.md             # 環境構築
│       └── TECHNICAL_SPEC.md    # 詳細仕様
│
├── ipad_app/                    # iPadアプリ (Swift) ← 修正必要
│   ├── Sources/
│   │   ├── ARManager.swift      # RGB送信機能追加
│   │   └── ContentView.swift
│   └── DisasterScanner/
│
├── server/                      # サーバー (Python)
│   ├── data_reception/
│   │   └── websocket_server.py  # 流用
│   ├── visualization/
│   │   ├── web_viewer.py        # 流用
│   │   └── pointcloud_viewer.py # 流用
│   ├── phase2_full/
│   │   ├── sam3_segmentation.py # 流用
│   │   ├── sam3_demo.py         # 流用
│   │   └── click_selector.py    # 流用
│   ├── generation/              # 新規
│   │   ├── sam3d_web_ui.py      # Web UI（WSL2用）
│   │   └── sam3d_generate.py    # （予定）
│   ├── fusion/                  # 新規
│   │   ├── icp_alignment.py
│   │   ├── visibility_check.py
│   │   └── shrinkwrap.py
│   └── orchestrator/            # 新規
│       └── agent.py
│
├── blender_addon/               # Blenderアドオン ← 一部修正
│   └── __init__.py
│
├── scripts/                     # 評価スクリプト
│   ├── extract_replica_frames.py
│   └── evaluate.py
│
├── experiments/                 # 実験データ（.gitignore）
│   └── session_xxx/
│
└── datasets/                    # データセット（シンボリックリンク）
    └── Replica -> ~/datasets/Replica-Dataset
```

---

## 参考資料

### 公式リポジトリ
- [SAM 3 GitHub](https://github.com/facebookresearch/sam3)
- [SAM 3D Objects GitHub](https://github.com/facebookresearch/sam-3d-objects)
- [SAM 3D Body GitHub](https://github.com/facebookresearch/sam-3d-body)
- [Replica Dataset GitHub](https://github.com/facebookresearch/Replica-Dataset)

### 技術ドキュメント
- [Open3D ICP Tutorial](http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html)
- [Blender Shrinkwrap Modifier](https://docs.blender.org/manual/en/latest/modeling/modifiers/deform/shrinkwrap.html)

### プロジェクト内ドキュメント
- [技術仕様書](docs/SAM3D_LIDAR_FUSION.md)
- [環境構築ガイド](docs/new_project/docs/SETUP.md)
