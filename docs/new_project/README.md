# SAM3D-LiDAR-Fusion

<div align="center">

**AI生成3Dモデル + LiDAR実測データ による高精度3D再構築システム**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/)
[![Blender](https://img.shields.io/badge/Blender-4.0+-orange.svg)](https://www.blender.org/)

</div>

---

## 概要

SAM3D-LiDAR-Fusionは、**生成AIによる綺麗な3Dモデル**と**LiDAR実測による正確な寸法**を融合し、高精度な3Dオブジェクトを生成するシステムです。

### コンセプト

```
┌─────────────────────┐     ┌─────────────────────┐
│   SAM 3D (生成AI)   │     │  iPad LiDAR (実測)  │
│                     │     │                     │
│  ・綺麗なトポロジー  │     │  ・正確な寸法       │
│  ・裏側も生成       │     │  ・表面位置が正確   │
│  ・寸法は不正確     │     │  ・裏側データなし   │
└─────────┬───────────┘     └──────────┬──────────┘
          │                            │
          └──────────┬─────────────────┘
                     ↓
          ┌─────────────────────┐
          │  Template Fitting   │
          │  + Shrinkwrap       │
          │                     │
          │  可視領域 → 吸着    │
          │  裏側 → 維持        │
          └─────────┬───────────┘
                    ↓
          ┌─────────────────────┐
          │   高精度3Dモデル    │
          │                     │
          │  ・正確な寸法       │
          │  ・綺麗なメッシュ   │
          │  ・完全な形状       │
          └─────────────────────┘
```

### 主な特徴

- **2系統データ融合**: 生成AI（SAM 3D）+ 実測（LiDAR）のハイブリッド
- **裏側保持**: LiDARで見えない部分はAI生成形状を維持
- **LLMオーケストレーション**: パイプライン全体をLLMが指揮・最適化
- **リアルタイム対応**: iPadスキャンからBlender出力まで自動化

---

## クイックスタート

### 必要環境

- **サーバー**: NVIDIA GPU搭載マシン（DGX Spark推奨）
- **クライアント**: iPad Pro（LiDAR搭載）
- **ソフトウェア**: Python 3.11+, Blender 4.0+, Docker

### インストール

```bash
# リポジトリをクローン
git clone https://github.com/nicejp/SAM3D-LiDAR-Fusion.git
cd SAM3D-LiDAR-Fusion

# Docker環境を起動
docker compose up -d

# 依存関係をインストール
pip install -r requirements.txt
```

### 基本的な使い方

```bash
# Step 1: iPadでスキャン → データ受信
python -m server.receiver --port 8765

# Step 2: SAM 3でセグメンテーション
python -m server.sam3_segment experiments/session_xxx --click 512,384

# Step 3: SAM 3Dで3Dメッシュ生成
python -m server.sam3d_generate experiments/session_xxx

# Step 4: LiDARデータで融合・補正
python -m server.fusion experiments/session_xxx

# 出力: experiments/session_xxx/output/final_model.obj
```

---

## システムアーキテクチャ

```
┌────────────────────────────────────────────────────────────────┐
│                        iPad Pro                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  ARKit: RGB (1920×1440) + LiDAR Depth (256×192)          │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
                           ↓ WiFi (WebSocket)
┌────────────────────────────────────────────────────────────────┐
│                      DGX Spark Server                          │
│                                                                │
│  ┌────────────────┐    ┌────────────────┐                     │
│  │ SAM 3          │    │ Point Cloud    │                     │
│  │ (2Dマスク生成) │───→│ Extraction     │                     │
│  └────────────────┘    └────────────────┘                     │
│         │                      │                               │
│         ↓                      ↓                               │
│  ┌────────────────┐    ┌────────────────┐                     │
│  │ SAM 3D         │    │ LiDAR点群      │                     │
│  │ (3Dメッシュ生成)│    │ (実測データ)   │                     │
│  └────────────────┘    └────────────────┘                     │
│         │                      │                               │
│         └──────────┬───────────┘                               │
│                    ↓                                           │
│         ┌────────────────────────┐                             │
│         │ Fusion Engine          │                             │
│         │ ・ICP位置合わせ        │                             │
│         │ ・可視判定             │                             │
│         │ ・Shrinkwrap吸着       │                             │
│         └────────────────────────┘                             │
│                    ↓                                           │
│         ┌────────────────────────┐                             │
│         │ LLM Orchestrator       │                             │
│         │ ・エラーハンドリング    │                             │
│         │ ・パラメータ最適化      │                             │
│         │ ・Blenderスクリプト生成 │                             │
│         └────────────────────────┘                             │
│                    ↓                                           │
│         ┌────────────────────────┐                             │
│         │ Blender Export         │                             │
│         │ → .obj / .ply / .usdz  │                             │
│         └────────────────────────┘                             │
└────────────────────────────────────────────────────────────────┘
```

---

## 技術的背景

### SAM 3 vs SAM 3D

| 項目 | SAM 3 | SAM 3D |
|------|-------|--------|
| 役割 | 2D切り抜き | 3D生成 |
| 入力 | RGB画像 | RGBA画像 |
| 出力 | バイナリマスク | 3Dメッシュ |
| 裏側 | 不可 | AI推測で生成 |

### 融合アルゴリズム

1. **Coarse Alignment**: 重心・スケール合わせ（PCA）
2. **Rigid ICP**: 位置・回転の精密調整
3. **Visibility Check**: カメラから見える頂点を特定
4. **Shrinkwrap**: 可視頂点のみLiDARに吸着、裏側は維持

---

## ドキュメント

- [技術仕様書](docs/TECHNICAL_SPEC.md) - 詳細な技術解説
- [環境構築ガイド](docs/SETUP.md) - インストール手順
- [開発計画](PLAN.md) - ロードマップ
- [API リファレンス](docs/API.md) - モジュール詳細

---

## ライセンス

MIT License

---

## 引用

```bibtex
@software{sam3d_lidar_fusion,
  title = {SAM3D-LiDAR-Fusion: Hybrid 3D Reconstruction with Generative AI and LiDAR Measurements},
  author = {nicejp},
  year = {2024},
  url = {https://github.com/nicejp/SAM3D-LiDAR-Fusion}
}
```

---

## 謝辞

- [SAM 3](https://github.com/facebookresearch/sam3) - Meta AI
- [Meta 3D Gen / SAM 3D](https://ai.meta.com/research/publications/meta-3d-gen/) - Meta AI
- [Open3D](http://www.open3d.org/) - Intel
- [Blender](https://www.blender.org/) - Blender Foundation
