# SAM3D-LiDAR-Fusion 環境構築ガイド

---

## 1. システム要件

### 1.1 サーバー側（DGX Spark推奨）

| 項目 | 要件 |
|------|------|
| OS | Ubuntu 24.04 (ARM64) |
| GPU | NVIDIA GPU (Blackwell推奨) |
| メモリ | 128GB以上 |
| ストレージ | 500GB以上 |
| CUDA | 12.6+ |
| Python | 3.11+ |

### 1.2 クライアント側（iPad）

| 項目 | 要件 |
|------|------|
| デバイス | iPad Pro 11"/12.9" (第3世代以降, LiDAR搭載) |
| OS | iOS 17.0+ |
| 開発環境 | macOS + Xcode 15+ |

---

## 2. サーバー環境構築

### 2.1 Dockerベースセットアップ（推奨）

#### Step 1: NGCコンテナを取得

```bash
# NVIDIA PyTorchコンテナ（ARM64 + CUDA対応）
docker pull nvcr.io/nvidia/pytorch:25.11-py3
```

#### Step 2: コンテナを起動

```bash
docker run --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/SAM3D-LiDAR-Fusion:/workspace \
  -v ~/datasets:/workspace/datasets \
  -p 8765:8765 \
  -it nvcr.io/nvidia/pytorch:25.11-py3
```

#### Step 3: SAM 3をインストール

```bash
cd /workspace

# SAM 3リポジトリをクローン
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .

# Hugging Face認証（モデルダウンロードに必要）
python -c "from huggingface_hub import login; login()"
# トークン: https://huggingface.co/settings/tokens で生成
```

#### Step 4: decordをビルド（動画処理用）

ARM64 + FFmpeg 6環境ではソースからビルドが必要：

```bash
# 依存パッケージ
apt-get update && apt-get install -y \
  ffmpeg libavcodec-dev libavformat-dev libavutil-dev \
  libswscale-dev libavfilter-dev libavdevice-dev

# libnvcuvidリンク
ln -s /usr/lib/aarch64-linux-gnu/libnvcuvid.so.1 /usr/local/cuda/lib64/stubs/libnvcuvid.so

# decordクローン
cd /tmp
git clone --recursive https://github.com/dmlc/decord
cd decord

# FFmpeg 6パッチ
sed -i '/#include <libavcodec\/avcodec.h>/a #include <libavcodec/bsf.h>' src/video/ffmpeg/ffmpeg_common.h
sed -i 's/AVCodec \*dec/const AVCodec *dec/g' src/video/video_reader.cc
sed -i 's/AVInputFormat \*iformat/const AVInputFormat *iformat/g' src/video/nvcodec/cuda_threaded_decoder.h
sed -i 's/AVInputFormat \*iformat/const AVInputFormat *iformat/g' src/video/nvcodec/cuda_threaded_decoder.cc

# ビルド
mkdir build && cd build
cmake .. -DUSE_CUDA=ON
make -j$(nproc)

# インストール
cd ../python
pip install .
```

#### Step 5: その他の依存関係

```bash
pip install open3d numpy scipy websockets aiohttp
pip install langchain langgraph  # LLMエージェント用
```

#### Step 6: コンテナを保存

```bash
# 別ターミナルで（ホスト側）
docker ps  # CONTAINER_ID を確認
docker commit <CONTAINER_ID> sam3d-lidar-fusion:ready

# 次回からは
docker run --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/SAM3D-LiDAR-Fusion:/workspace \
  -p 8765:8765 \
  -it sam3d-lidar-fusion:ready
```

---

### 2.2 SAM 3D のセットアップ

**注意:** 2024年12月時点で、Meta SAM 3Dの公式リリースは限定的です。

#### オプション A: Meta 3D Gen API（利用可能な場合）

```bash
# Meta AI Developer APIを使用
pip install meta-ai-sdk

# API Key設定
export META_AI_API_KEY="your_key_here"
```

#### オプション B: 代替モデル（TripoSR等）

```bash
# TripoSR（オープンソースの画像→3D変換）
pip install triposr

# 使用例
python -c "
from triposr import TripoSR
model = TripoSR()
mesh = model.generate('input.png')
mesh.export('output.obj')
"
```

---

### 2.3 Ollama（ローカルLLM）セットアップ

```bash
# インストール（ホスト側、Dockerコンテナ外）
curl -fsSL https://ollama.com/install.sh | sh

# サービス起動
sudo systemctl start ollama

# モデルダウンロード
ollama pull gpt-oss:120b      # 大規模（60GB）
ollama pull qwen3-coder:30b   # 軽量代替（18GB）

# 動作確認
ollama run gpt-oss:120b "こんにちは"
```

---

### 2.4 Blender セットアップ

```bash
# Blender 4.0+ をインストール（Ubuntu）
sudo snap install blender --classic

# または公式サイトからダウンロード
wget https://download.blender.org/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz
tar -xf blender-4.0.2-linux-x64.tar.xz
sudo mv blender-4.0.2-linux-x64 /opt/blender

# パスを通す
echo 'export PATH="/opt/blender:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 動作確認
blender --version
```

---

## 3. iPadアプリセットアップ

### 3.1 Xcodeプロジェクト

```bash
# MacBookで実行
cd ~/SAM3D-LiDAR-Fusion/ipad_app

# Xcodeでプロジェクトを開く
open ScannerApp.xcodeproj
```

### 3.2 Info.plist設定

以下の権限を追加：

| Key | Value |
|-----|-------|
| Privacy - Camera Usage Description | LiDARスキャンのためにカメラを使用します |
| Privacy - Local Network Usage Description | サーバーにデータを送信します |

### 3.3 ビルド & 実機テスト

1. iPad Pro（LiDAR搭載）をMacにUSB接続
2. Xcode上部でデバイスを選択
3. ▶️ ボタンでビルド・実行
4. 初回はiPad側で「信頼」を許可

---

## 4. 動作確認

### 4.1 サーバー側

```bash
# コンテナ内でPYTHONPATHを設定
export PYTHONPATH=/workspace:/workspace/sam3:$PYTHONPATH

# SAM 3テスト
python -c "from sam3.model_builder import build_sam3_image_model; print('SAM 3 OK')"

# Open3Dテスト
python -c "import open3d as o3d; print('Open3D OK')"

# WebSocketサーバー起動テスト
python -m server.receiver --port 8765
```

### 4.2 iPad接続テスト

1. サーバーでWebSocketサーバーを起動
2. iPadアプリでサーバーIPを入力（例: `192.168.1.100:8765`）
3. 「接続」をタップ
4. 「録画開始」でスキャン
5. サーバーログでデータ受信を確認

---

## 5. トラブルシューティング

### 5.1 CUDA関連

```bash
# GPU認識確認
nvidia-smi

# PyTorch CUDA確認
python -c "import torch; print(torch.cuda.is_available())"
```

### 5.2 SAM 3モデルダウンロードエラー

```bash
# Hugging Faceキャッシュクリア
rm -rf ~/.cache/huggingface

# 再認証
huggingface-cli login
```

### 5.3 WebSocket接続失敗

```bash
# ファイアウォール確認
sudo ufw status
sudo ufw allow 8765

# ポート使用確認
netstat -tlnp | grep 8765
```

### 5.4 メモリ不足

```bash
# スワップ追加
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## 6. LiDAR-LLM-MCPからの移行

既存のLiDAR-LLM-MCPプロジェクトからの移行：

### 6.1 再利用可能なDockerイメージ

```bash
# 既存イメージ確認
docker images | grep lidar-llm-mcp

# 存在する場合、ベースとして使用可能
# lidar-llm-mcp:sam3-tested にはSAM 3 + decordがインストール済み
docker run --gpus all --ipc=host \
  -v ~/SAM3D-LiDAR-Fusion:/workspace \
  -it lidar-llm-mcp:sam3-tested
```

### 6.2 コピーすべきファイル

```bash
# iPadアプリ
cp -r ~/LiDAR-LLM-MCP/ipad_app ~/SAM3D-LiDAR-Fusion/

# WebSocketサーバー
cp -r ~/LiDAR-LLM-MCP/server/data_reception ~/SAM3D-LiDAR-Fusion/server/receiver

# 可視化ツール
cp -r ~/LiDAR-LLM-MCP/server/visualization ~/SAM3D-LiDAR-Fusion/server/utils/

# Blenderアドオン骨格
cp -r ~/LiDAR-LLM-MCP/blender_addon ~/SAM3D-LiDAR-Fusion/
```

---

## 7. 次のステップ

環境構築が完了したら：

1. [技術仕様書](TECHNICAL_SPEC.md) で詳細を確認
2. [開発計画](../PLAN.md) でタスクを確認
3. Phase 1（技術検証）から開始
