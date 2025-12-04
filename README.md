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

# 仮想環境を作成
python3 -m venv venv
source venv/bin/activate

# 依存関係をインストール
pip install -r server/requirements.txt
```

### 2. Dockerコンテナの起動（SAM 3使用時）

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

### 3. iPadアプリのセットアップ

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

# 画像でクリック座標を選択
python -m server.phase2_full.click_selector experiments/session_xxx

# SAM 3でセグメント
python -m server.phase2_full.sam3_segmentation experiments/session_xxx --click 512,384

# または、Gradio Web UIを使用
python -m server.phase2_full.sam3_demo
# → ブラウザで http://localhost:7860 にアクセス
```

### Step 4: SAM 3Dで3D生成（未実装）

```bash
# SAM 3D Objectsで3Dメッシュ生成
python -m server.generation.sam3d_generate experiments/session_xxx
```

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
│   ├── generation/              # SAM 3D生成（未実装）
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

## ライセンス

TBD
