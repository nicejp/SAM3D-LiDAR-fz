# LiDAR-LLM-MCP → SAM3D-LiDAR-Fusion 移行ガイド

---

## 1. プロジェクトの変化

### 1.1 方向性の変更

| 項目 | 旧 (LiDAR-LLM-MCP) | 新 (SAM3D-LiDAR-Fusion) |
|------|---------------------|-------------------------|
| **主目的** | LLMによるBlenderコード生成 | SAM 3D + LiDAR融合 |
| **3D生成方法** | LLMがBlenderコードを書く | SAM 3Dが3Dメッシュを生成 |
| **LiDARの役割** | 点群からの形状認識 | 生成メッシュの寸法補正 |
| **LLMの役割** | 3Dコード生成者 | パイプラインオーケストレーター |

### 1.2 技術スタックの変化

```
旧: iPad → LiDAR点群 → LLM → Blenderコード → 3Dモデル

新: iPad → RGB画像 + LiDAR点群
         ↓
    SAM 3 (2Dマスク)
         ↓
    SAM 3D (3Dメッシュ生成)
         ↓
    ICP + Shrinkwrap (LiDAR融合)
         ↓
    LLM (オーケストレーション)
         ↓
    高精度3Dモデル
```

---

## 2. 再利用可能なコンポーネント

### 2.1 そのまま使えるもの

| コンポーネント | パス | 用途 |
|--------------|------|------|
| iPadアプリ | `ipad_app/` | LiDAR + RGB取得 |
| WebSocketサーバー | `server/data_reception/websocket_server.py` | データ受信 |
| 点群可視化 | `server/visualization/web_viewer.py` | デバッグ |
| SAM 3セグメント | `server/phase2_full/sam3_segmentation.py` | マスク生成 |
| Blenderアドオン骨格 | `blender_addon/` | GUI |

### 2.2 修正が必要なもの

| コンポーネント | 変更内容 |
|--------------|----------|
| `phase2_full/pipeline.py` | SAM 3D統合を追加 |
| `blender_addon/operators.py` | Shrinkwrap処理を追加 |

### 2.3 新規開発が必要なもの

| コンポーネント | 内容 |
|--------------|------|
| `server/generation/sam3d_generate.py` | SAM 3D呼び出し |
| `server/fusion/icp_alignment.py` | ICP位置合わせ |
| `server/fusion/visibility_check.py` | 可視判定 |
| `server/fusion/shrinkwrap.py` | 部分的吸着 |
| `server/orchestrator/agent.py` | LLMエージェント |

---

## 3. Docker環境の引き継ぎ

### 3.1 既存イメージ

```bash
# 確認
docker images | grep lidar-llm-mcp

# 利用可能なイメージ
# lidar-llm-mcp:sam3-ready   - SAM 3インストール済み
# lidar-llm-mcp:sam3-tested  - SAM 3 + decord + 動作確認済み
```

### 3.2 イメージのリネーム（オプション）

```bash
# 新プロジェクト名でタグ付け
docker tag lidar-llm-mcp:sam3-tested sam3d-lidar-fusion:base

# 確認
docker images | grep sam3d
```

---

## 4. Ollamaの引き継ぎ

### 4.1 ダウンロード済みモデル

```bash
# 確認
ollama list

# 期待される出力
# NAME                 SIZE
# gpt-oss:120b        60GB
# qwen3-coder:30b     18GB
```

### 4.2 サービス管理

```bash
# 起動
sudo systemctl start ollama

# 停止
sudo systemctl stop ollama

# 状態確認
sudo systemctl status ollama
```

---

## 5. データセットの引き継ぎ

### 5.1 保存場所

```bash
# Replica Dataset
~/datasets/Replica-Dataset/

# 実験データ
~/LiDAR-LLM-MCP/experiments/session_xxx/
```

### 5.2 移行方法

```bash
# シンボリックリンクで共有（推奨）
ln -s ~/datasets ~/SAM3D-LiDAR-Fusion/datasets
ln -s ~/LiDAR-LLM-MCP/experiments ~/SAM3D-LiDAR-Fusion/experiments_old
```

---

## 6. ファイルコピーコマンド

新プロジェクト作成時のコピーコマンド：

```bash
# 新リポジトリ作成
mkdir -p ~/SAM3D-LiDAR-Fusion
cd ~/SAM3D-LiDAR-Fusion
git init

# ドキュメントをコピー
cp ~/LiDAR-LLM-MCP/docs/new_project/README.md .
cp ~/LiDAR-LLM-MCP/docs/new_project/PLAN.md .
mkdir -p docs
cp ~/LiDAR-LLM-MCP/docs/new_project/docs/* docs/

# iPadアプリをコピー
cp -r ~/LiDAR-LLM-MCP/ipad_app .

# サーバーコードをコピー（構造を変更）
mkdir -p server/receiver
cp ~/LiDAR-LLM-MCP/server/data_reception/*.py server/receiver/

mkdir -p server/segmentation
cp ~/LiDAR-LLM-MCP/server/phase2_full/sam3_segmentation.py server/segmentation/

mkdir -p server/utils
cp ~/LiDAR-LLM-MCP/server/visualization/*.py server/utils/

# Blenderアドオンをコピー
cp -r ~/LiDAR-LLM-MCP/blender_addon .

# .gitignore
cp ~/LiDAR-LLM-MCP/.gitignore .

# 初回コミット
git add .
git commit -m "Initial commit: Migrate from LiDAR-LLM-MCP"
```

---

## 7. 新規作成が必要なディレクトリ

```bash
mkdir -p server/generation      # SAM 3D
mkdir -p server/fusion          # ICP + Shrinkwrap
mkdir -p server/orchestrator    # LLMエージェント
mkdir -p tests
mkdir -p experiments
```

---

## 8. requirements.txt

```txt
# Core
numpy>=1.24.0
scipy>=1.10.0
open3d>=0.17.0

# Deep Learning
torch>=2.0.0
torchvision>=0.15.0

# SAM 3 (インストール済みの場合はコメントアウト)
# sam3 @ git+https://github.com/facebookresearch/sam3.git

# LLM
anthropic>=0.18.0
langchain>=0.1.0
langgraph>=0.0.20

# Web
websockets>=11.0
aiohttp>=3.8.0

# Visualization
matplotlib>=3.7.0
gradio>=4.0.0

# Image
Pillow>=10.0.0
opencv-python>=4.8.0

# Mesh
trimesh>=4.0.0
pymeshlab>=2023.12
```

---

## 9. 作成済みドキュメント一覧

| ファイル | 説明 |
|---------|------|
| `README.md` | プロジェクト概要 |
| `PLAN.md` | 開発計画 |
| `docs/TECHNICAL_SPEC.md` | 技術仕様書（Gemini議論のまとめ） |
| `docs/SETUP.md` | 環境構築ガイド |
| `MIGRATION.md` | 本ファイル（移行ガイド） |

---

## 10. 次のアクション

1. **GitHubで新リポジトリ作成**
   - リポジトリ名: `SAM3D-LiDAR-Fusion`
   - 公開/非公開を選択

2. **ローカルで初期化**
   ```bash
   cd ~/SAM3D-LiDAR-Fusion
   git remote add origin https://github.com/nicejp/SAM3D-LiDAR-Fusion.git
   git push -u origin main
   ```

3. **Phase 1開発開始**
   - SAM 3Dのセットアップ
   - 正常系パイプラインの実装
