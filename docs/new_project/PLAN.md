# SAM3D-LiDAR-Fusion 開発計画

**プロジェクト名:** SAM3D-LiDAR-Fusion
**作成日:** 2024年12月3日
**作成者:** nicejp

---

## 1. プロジェクト概要

### 1.1 目的

SAM 3D（生成AI）で作成した3Dモデルを、iPad LiDAR（実測データ）で補正・融合し、高精度な3Dオブジェクトを生成するシステムを構築する。

### 1.2 技術的新規性

- **2系統データ融合**: 生成AI + 実測のハイブリッドアプローチ
- **可視判定による部分吸着**: 裏側を潰さずにオモテ面のみ補正
- **LLMオーケストレーション**: パイプライン全体をLLMが指揮

### 1.3 成功の定義

**Phase 1（技術検証）:**
- SAM 3D → ICP → Blender Shrinkwrap の正常系パイプラインが動作

**Phase 2（融合システム）:**
- 可視判定による部分的吸着が動作
- 裏側の形状が維持されたまま、オモテ面がLiDARに吸着

**Phase 3（LLMエージェント）:**
- LLMがパイプライン全体をオーケストレーション
- エラー時の自動リカバリー

---

## 2. 開発フェーズ

### Phase 1: 技術検証（正常系スクリプト）

**目標:** 各コンポーネントの連携確認

```
タスク:
├── [ ] SAM 3Dセットアップ
│   ├── [ ] Meta SAM 3Dのインストール
│   ├── [ ] 単体テスト（画像→3Dメッシュ）
│   └── [ ] 出力フォーマット確認（OBJ/GLB）
│
├── [ ] 点群抽出パイプライン
│   ├── [ ] SAM 3マスク → 点群フィルタリング
│   ├── [ ] カメラ行列を使った投影計算
│   └── [ ] パンチスルー対策（深度フィルタ）
│
├── [ ] ICP位置合わせ
│   ├── [ ] Open3Dによる実装
│   ├── [ ] Coarse Alignment（PCA）
│   └── [ ] Rigid ICP
│
└── [ ] Blender Shrinkwrap
    ├── [ ] バックグラウンド実行
    ├── [ ] Shrinkwrap Modifier適用
    └── [ ] 出力エクスポート
```

### Phase 2: 融合システム

**目標:** 可視判定による部分的吸着

```
タスク:
├── [ ] 可視判定（Visibility Check）
│   ├── [ ] カメラ位置からの可視頂点計算
│   ├── [ ] Vertex Group生成
│   └── [ ] 裏側頂点のマスキング
│
├── [ ] 部分的Shrinkwrap
│   ├── [ ] 可視頂点のみ吸着
│   ├── [ ] 裏側は変形なし
│   └── [ ] 境界のスムージング
│
└── [ ] トポロジー不一致対策
    ├── [ ] CLIP類似度判定
    ├── [ ] 再生成/スキップ判断
    └── [ ] ユーザー確認フロー
```

### Phase 3: LLMエージェント

**目標:** 自動エラーハンドリングとパラメータ最適化

```
タスク:
├── [ ] ツール定義
│   ├── [ ] run_sam3d()
│   ├── [ ] run_icp_alignment()
│   ├── [ ] run_blender_fusion()
│   └── [ ] check_quality()
│
├── [ ] エージェントフレームワーク
│   ├── [ ] LangGraph / AutoGen選定
│   ├── [ ] プロンプト設計
│   └── [ ] リトライロジック
│
└── [ ] ユーザーインタラクション
    ├── [ ] 曖昧な指示の解釈
    ├── [ ] パラメータ調整UI
    └── [ ] 進捗レポート
```

---

## 3. システム構成

### 3.1 ディレクトリ構成

```
SAM3D-LiDAR-Fusion/
├── README.md
├── PLAN.md
├── requirements.txt
├── docker-compose.yml
│
├── docs/
│   ├── TECHNICAL_SPEC.md      # 技術仕様書
│   ├── SETUP.md               # 環境構築
│   └── API.md                 # APIリファレンス
│
├── server/
│   ├── receiver/              # iPadデータ受信
│   │   └── websocket_server.py
│   │
│   ├── segmentation/          # SAM 3セグメンテーション
│   │   ├── sam3_segment.py
│   │   └── point_extraction.py
│   │
│   ├── generation/            # SAM 3D生成
│   │   └── sam3d_generate.py
│   │
│   ├── fusion/                # 融合エンジン
│   │   ├── icp_alignment.py
│   │   ├── visibility_check.py
│   │   └── shrinkwrap.py
│   │
│   ├── orchestrator/          # LLMオーケストレーター
│   │   ├── agent.py
│   │   └── tools.py
│   │
│   └── utils/
│       ├── camera_utils.py
│       └── mesh_utils.py
│
├── ipad_app/                  # iPadアプリ（Swift）
│   └── (LiDAR-LLM-MCPから引き継ぎ)
│
├── blender_addon/             # Blenderアドオン
│   └── (LiDAR-LLM-MCPから引き継ぎ)
│
├── experiments/               # 実験データ（.gitignore）
│   └── session_xxx/
│
└── tests/
    ├── test_sam3d.py
    ├── test_icp.py
    └── test_fusion.py
```

### 3.2 技術スタック

| カテゴリ | 技術 |
|---------|------|
| **3D生成** | SAM 3D (Meta), SAM 3 (Meta) |
| **点群処理** | Open3D, NumPy |
| **メッシュ処理** | Blender API, PyMeshLab |
| **LLMエージェント** | LangGraph / AutoGen |
| **LLM** | Claude 3.5 Sonnet, gpt-oss 120B |
| **通信** | WebSocket (asyncio) |
| **コンテナ** | Docker, NGC PyTorch |

---

## 4. LiDAR-LLM-MCPからの引き継ぎ

### 4.1 再利用するコンポーネント

| コンポーネント | 元ファイル | 用途 |
|--------------|-----------|------|
| iPadアプリ | `ipad_app/` | LiDAR + RGB取得 |
| WebSocketサーバー | `server/data_reception/` | データ受信 |
| SAM 3セグメント | `server/phase2_full/sam3_segmentation.py` | マスク生成 |
| 点群可視化 | `server/visualization/` | デバッグ |
| Blenderアドオン骨格 | `blender_addon/` | GUI |

### 4.2 新規開発が必要なコンポーネント

| コンポーネント | 内容 |
|--------------|------|
| SAM 3D統合 | Meta SAM 3Dのセットアップと呼び出し |
| 融合エンジン | ICP + 可視判定 + Shrinkwrap |
| LLMオーケストレーター | エージェントフレームワーク |

### 4.3 環境構築情報

**引き継ぐDocker情報:**
- ベースイメージ: `nvcr.io/nvidia/pytorch:25.11-py3`
- 保存済みイメージ: `lidar-llm-mcp:sam3-tested`
- SAM 3: インストール済み
- decord: FFmpeg 6パッチ適用済み

**Ollama (LLM):**
- gpt-oss 120B: ダウンロード済み
- 起動: `sudo systemctl start ollama`

---

## 5. 技術的リスクと対策

### 5.1 SAM 3Dの入手・動作

| リスク | 対策 |
|-------|------|
| SAM 3Dが公開されていない | Meta 3D Genの論文・デモを参考に代替手法を検討 |
| ARM64非対応 | x86_64クラウド環境でのリモート実行 |

### 5.2 融合アルゴリズム

| リスク | 対策 |
|-------|------|
| ICPが局所解に陥る | 複数の初期回転でリトライ、LLMが判断 |
| トポロジー不一致 | CLIP類似度で事前チェック、ユーザー確認 |
| 境界アーティファクト | ラプラシアンスムージング |

### 5.3 LLMエージェント

| リスク | 対策 |
|-------|------|
| 無限ループ | 最大リトライ回数を設定 |
| コスト超過 | ローカルLLM (gpt-oss) をフォールバック |

---

## 6. 評価計画

### 6.1 定量評価

| 指標 | 方法 |
|------|------|
| 寸法精度 | Ground Truth (CADモデル) との比較 |
| 形状精度 | Chamfer Distance, Hausdorff Distance |
| 処理時間 | 各ステップのベンチマーク |

### 6.2 定性評価

- ユーザースタディ（使いやすさ）
- 視覚的品質（レンダリング結果）

---

## 7. 参考資料

- [SAM 3 GitHub](https://github.com/facebookresearch/sam3)
- [Meta 3D Gen Paper](https://arxiv.org/abs/2411.19480)
- [Open3D ICP Tutorial](http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html)
- [Blender Shrinkwrap Modifier](https://docs.blender.org/manual/en/latest/modeling/modifiers/deform/shrinkwrap.html)
