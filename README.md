# SAM3D-LiDAR-fz (fusion)

SAM 3D（生成） + LiDAR（実測） → LLMがオーケストレーションして融合 → 高精度3Dモデル を目指します

## 概要

iPad ProのLiDARセンサーで取得した実測データと、SAM 3Dで生成したAI推測3Dモデルを融合し、高精度な3Dモデルを生成するシステムです。

## システム構成

```
┌─────────────────┐        WiFi (WebSocket)        ┌─────────────────────────┐
│    iPad Pro     │  ────────────────────────────→ │   DGX Spark Server      │
│  (LiDAR搭載)    │   RGB画像 + LiDAR点群          │   + Web GUI             │
└─────────────────┘                                └─────────────────────────┘
                                                            │
                                                            ↓
                                                   ┌─────────────────┐
                                                   │  SAM 3 + SAM 3D │
                                                   │  + 融合エンジン │
                                                   └─────────────────┘
                                                            │
                                                            ↓
                                                   ┌─────────────────┐
                                                   │  高精度3Dモデル │
                                                   └─────────────────┘
```

## 環境構築

### 1. Python環境のセットアップ

```bash
cd ~/SAM3D-LiDAR-fz

# Python 3.11で仮想環境を作成
python3.11 -m venv venv
source venv/bin/activate

# 依存関係をインストール
pip install -r server/requirements.txt
```

### 2. Dockerコンテナの起動（SAM 3使用時）

```bash
# SAM 3セットアップ済みコンテナを使用（推奨）
docker run --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network=host \
  -v ~/SAM3D-LiDAR-fz:/workspace \
  -v ~/datasets:/workspace/datasets \
  -it sam3d-lidar:sam3-ready

# PYTHONPATHを設定（毎回必要）
export PYTHONPATH=/workspace:/workspace/sam3:$PYTHONPATH
```

**初回セットアップ（sam3d-lidar:sam3-readyがない場合）:**
```bash
# ベースコンテナを起動
docker run --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network=host \
  -v ~/SAM3D-LiDAR-fz:/workspace \
  --name sam3d-setup \
  -it lidar-llm-mcp:sam3-tested

# コンテナ内でSAM 3をクローン
cd /workspace
git clone https://github.com/facebookresearch/sam3.git
exit

# コンテナを保存
docker commit sam3d-setup sam3d-lidar:sam3-ready
docker rm sam3d-setup
```

### 3. SAM 3D Objects のセットアップ (WSL2/Windows)

WSL2環境（Ubuntu + NVIDIA GPU）でSAM 3D Objectsをセットアップする手順。

```bash
# 1. Conda環境を作成
conda create -n sam3d python=3.11 -y
conda activate sam3d

# 2. PyTorch + CUDAをインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. SAM 3D Objectsをクローン
cd ~
git clone https://github.com/facebookresearch/sam-3d-objects.git
cd sam-3d-objects

# 4. 依存関係をインストール
pip install "git+https://github.com/facebookresearch/fvcore"
pip install "git+https://github.com/facebookresearch/iopath"
pip install hydra-core omegaconf einops hatch-requirements-txt gradio pillow numpy scipy

# 5. PyTorch3Dをソースビルド
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 40シリーズの場合（詳細はPLAN.md参照）
pip install "git+https://github.com/facebookresearch/pytorch3d.git@main" --no-build-isolation

# 6. SAM 3D Objects本体をインストール
pip install -e . --no-deps

# 7. チェックポイントをダウンロード（要Hugging Faceアクセス申請）
pip install huggingface_hub
mkdir -p checkpoints/hf
huggingface-cli download facebook/sam-3d-objects --local-dir checkpoints/hf

# 8. 動作確認
python -c "
import sys; sys.path.append('notebook')
from inference import Inference
inference = Inference('checkpoints/hf/pipeline.yaml', compile=False)
print('Model loaded successfully!')
"
```

詳細な手順は [PLAN.md](PLAN.md) の「WSL2 (x86_64/Windows) でのセットアップ手順」を参照。

**DGX Spark (ARM64) でのPyTorch3D追加インストール:**

SAM 3D Objectsを使用する場合、PyTorch3Dが必要。DGX SparkはARM64アーキテクチャのため、ソースからビルドする。

```bash
# 依存関係
pip install "git+https://github.com/facebookresearch/fvcore"
pip install "git+https://github.com/facebookresearch/iopath"

# 環境変数設定（DGX Spark GB110用）
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="12.0"

# ソースからビルド（約10-15分）
pip install "git+https://github.com/facebookresearch/pytorch3d.git@main" --no-build-isolation
```

### 4. iPadアプリのセットアップ

#### 要件

- iPad Pro 第3世代以降（LiDAR搭載モデル）
- iOS 17.0以上
- Xcode 15.0以上（Mac）

#### Xcodeプロジェクト作成

1. Xcode → "Create New Project"
2. iOS → App を選択
3. 設定:
   - **Product Name:** DisasterScanner
   - **Interface:** SwiftUI
   - **Language:** Swift
4. 保存先: `ipad_app/` ディレクトリを選択

#### ソースファイル追加

`ipad_app/Sources/` フォルダ内のファイルをXcodeプロジェクトに追加:

```
Sources/
├── ContentView.swift   # メインUI（既存ファイルを置き換え）
└── ARManager.swift     # LiDAR + RGB取得・送信
```

**追加方法:**
1. Xcode → File → Add Files to "DisasterScanner"
2. `Sources/` 内のファイルを選択
3. "Copy items if needed" はオフにする

#### Info.plist設定（重要）

Xcode → TARGETS → DisasterScanner → Info で以下を追加:

| Key | Value |
|-----|-------|
| Privacy - Camera Usage Description | LiDARスキャンのためにカメラを使用します |
| Privacy - Local Network Usage Description | DGX Sparkサーバーにデータを送信します |

#### ビルド＆実行

1. iPad Pro を Mac に USB 接続
2. Xcode 上部でデバイスを選択
3. ▶️ ボタンでビルド＆実行

詳細は [ipad_app/README.md](ipad_app/README.md) を参照してください。

## 使い方

### Step 1: WebSocketサーバーの起動

```bash
# 方法1: WebSocketサーバーのみ起動
python -m server.data_reception.websocket_server --port 8765

# 方法2: Web 3Dビューワー付きで起動（推奨）
python -m server.visualization.web_viewer --port 8765
# → ブラウザで http://localhost:8080 にアクセス
```

### Step 2: iPadアプリからスキャン

1. iPadアプリを起動
2. サーバーIPアドレスを入力（DGX SparkのIP）
3. 「スキャン開始」をタップ
4. 対象物をゆっくり周回撮影
5. 「停止」でスキャン終了

### Step 3: SAM 3でセグメンテーション

```bash
# Dockerコンテナ内で実行
export PYTHONPATH=/workspace:/workspace/sam3:$PYTHONPATH

# Gradio Web UIを使用（推奨）
python -m server.phase2_full.sam3_demo
# → ブラウザで http://localhost:7860 にアクセス
```

**sam3_demo Web UI:**
1. セッションフォルダをドロップダウンで選択
2. サムネイルギャラリーから画像をクリック → 自動で読み込み
3. 右側の大きな画像をクリック → セグメント実行
4. 「背景モード」ONで除外領域を指定
5. 「RGBA画像を出力 (SAM 3D用)」で背景透明画像を生成
6. 「3D点群を出力 (PLY)」で点群生成

```bash
# コマンドラインで直接実行する場合
python -m server.phase2_full.sam3_segmentation experiments/session_xxx --click 512,384
```

### Step 4: SAM 3Dで3D生成

SAM 3D ObjectsはWSL2上で動作。Web UI経由でDGX Sparkからアクセス可能。

**WSL2側でWeb UIを起動:**
```bash
cd ~/sam-3d-objects
conda activate sam3d

# Web UIを起動（ポート8000）
python ~/SAM3D-LiDAR-fz/server/generation/sam3d_web_ui.py --port 8000
```

**DGX Spark側からアクセス:**
```bash
# WSL2のIPを確認（WSL2側で hostname -I）
# ブラウザでアクセス: http://<WSL2のIP>:8000
```

**使い方:**
1. Step 3で生成したRGBA画像をアップロード
2. シード値を設定（オプション）
3. 「3D生成」ボタンをクリック
4. 生成されたPLYファイルをダウンロード

### Step 5: 融合処理（未実装）

```bash
# ICP位置合わせ + Shrinkwrap融合
python -m server.fusion.run experiments/session_xxx
```

## ディレクトリ構成

```
SAM3D-LiDAR-fz/
├── PLAN.md                      # 実装計画書
├── README.md                    # 本ファイル
├── server/
│   ├── data_reception/          # データ受信
│   │   └── websocket_server.py
│   ├── visualization/           # 可視化
│   │   ├── web_viewer.py
│   │   └── pointcloud_viewer.py
│   ├── phase2_full/             # SAM 3セグメンテーション
│   │   ├── sam3_segmentation.py
│   │   ├── sam3_demo.py
│   │   └── click_selector.py
│   ├── generation/              # SAM 3D生成
│   │   └── sam3d_web_ui.py      # Web UI（WSL2用）
│   ├── fusion/                  # 融合処理（未実装）
│   └── orchestrator/            # LLMオーケストレーター（未実装）
├── ipad_app/                    # iPadアプリ (Swift)
├── blender_addon/               # Blenderアドオン
├── experiments/                 # 実験データ
└── datasets/                    # 評価用データセット
```

## 保存されるデータ形式

```
experiments/session_YYYYMMDD_HHMMSS/
├── rgb/
│   ├── frame_000000.jpg      # RGB画像 (1920×1440)
│   └── ...
├── depth/
│   ├── frame_000000.npy      # 深度マップ (256×192, Float32)
│   └── ...
├── camera/
│   ├── frame_000000.json     # カメラパラメータ
│   └── ...
└── metadata.json             # セッション情報
```

## 技術スタック

| カテゴリ | 技術 |
|---------|------|
| 3D生成 | SAM 3D Objects (Meta), SAM 3 (Meta) |
| 点群処理 | Open3D, NumPy |
| メッシュ処理 | Blender API (bpy) |
| LLM | gpt-oss 120B, Claude Desktop (Blender MCP経由) |
| 通信 | WebSocket (asyncio) |
| コンテナ | Docker, NGC PyTorch |
| Web GUI | Three.js, Gradio |
| iOS開発 | Swift, ARKit, RealityKit |

## 参考資料

- [実装計画書 (PLAN.md)](PLAN.md)
- [SAM 3 GitHub](https://github.com/facebookresearch/sam3)
- [SAM 3D Objects GitHub](https://github.com/facebookresearch/sam-3d-objects)
- [Replica Dataset GitHub](https://github.com/facebookresearch/Replica-Dataset)

## トラブルシューティング

### iPadアプリ関連

| 問題 | 解決策 |
|------|--------|
| 「LiDAR非対応デバイス」エラー | iPad Pro 第3世代以降が必要。シミュレーターでは動作しません |
| 接続エラー | サーバーIP確認、WebSocketサーバー起動確認、同一ネットワーク接続確認 |
| ビルドエラー（Optional binding） | ARManager.swift: `let capturedImage = frame.capturedImage` に変更 |
| ビルドエラー（Variable never mutated） | `var packet` を `let packet` に変更 |

### サーバー関連

| 問題 | 解決策 |
|------|--------|
| websocketsインポートエラー | `pip install websockets` |
| SAM 3が動作しない | Dockerコンテナ内でPYTHONPATHを設定 |

## ライセンス

TBD
