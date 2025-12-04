# LiDAR-LLM Pipeline Blender Add-on

BlenderでLiDAR点群からLLMを使って3Dオブジェクトを生成するアドオンです。

## インストール

### 方法1: zipファイルからインストール

```bash
# アドオンをzipに圧縮
cd ~/LiDAR-LLM-MCP
zip -r lidar_llm_addon.zip blender_addon/

# Blenderでインストール
# 1. Blender起動
# 2. Edit → Preferences → Add-ons
# 3. "Install..." ボタン
# 4. lidar_llm_addon.zip を選択
# 5. "LiDAR-LLM Pipeline" をチェック
```

### 方法2: 直接コピー

```bash
# Blenderのアドオンフォルダにコピー
cp -r ~/LiDAR-LLM-MCP/blender_addon ~/.config/blender/4.0/scripts/addons/lidar_llm_pipeline

# Blenderを再起動して有効化
```

## 使い方

1. Blenderを起動
2. 3D Viewportのサイドバーを開く（Nキー）
3. 「LLM」タブをクリック
4. セッションディレクトリを選択
5. プロンプトを入力（例: "この点群を椅子にして"）
6. 「Generate 3D」をクリック

## UIパネル

```
┌─────────────────────────────────┐
│ LiDAR-LLM Pipeline              │
├─────────────────────────────────┤
│ Session                         │
│ [セッションディレクトリ選択]      │
│ [List Sessions]                 │
├─────────────────────────────────┤
│ Prompt                          │
│ [この点群を3Dオブジェクトにして]  │
├─────────────────────────────────┤
│ Model                           │
│ [gpt-oss 120B ▼]                │
├─────────────────────────────────┤
│ [    Generate 3D    ]           │
├─────────────────────────────────┤
│ Tools                           │
│ [Import Point Cloud]            │
├─────────────────────────────────┤
│ Status: 待機中                   │
└─────────────────────────────────┘
```

## 機能

| 機能 | 説明 |
|------|------|
| Generate 3D | LLMを使って3Dオブジェクトを生成 |
| List Sessions | 利用可能なセッションを一覧表示 |
| Import Point Cloud | 点群PLYファイルをインポート |

## トラブルシューティング

### 「パイプラインエラー」が出る

1. DGX Sparkでollamaが起動しているか確認
   ```bash
   sudo systemctl start ollama
   ```

2. venv環境が有効か確認
   ```bash
   source ~/LiDAR-LLM-MCP/venv/bin/activate
   ```

### 「セッションが見つかりません」が出る

セッションディレクトリのパスが正しいか確認してください。
例: `/home/nicejp/LiDAR-LLM-MCP/experiments/session_20251202_222212`
