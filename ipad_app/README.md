# DisasterScanner - iPad LiDARスキャナーアプリ

iPad Pro の LiDAR センサーを使用して、RGB画像と深度マップを同時取得し、DGX Sparkサーバーに送信するアプリです。

## 要件

- iPad Pro 第3世代以降（LiDAR搭載モデル）
- iOS 17.0以上
- Xcode 15.0以上（Mac）

## セットアップ

### 1. Xcodeプロジェクト作成

1. Xcode → "Create New Project"
2. iOS → App を選択
3. 設定:
   - **Product Name:** DisasterScanner
   - **Interface:** SwiftUI
   - **Language:** Swift
4. **保存先:** このディレクトリ（`ipad_app/`）を選択

### 2. ソースファイル追加

`Sources/` フォルダ内のファイルをXcodeプロジェクトに追加:

```
Sources/
├── ContentView.swift   # メインUI（既存ファイルを置き換え）
└── ARManager.swift     # LiDAR + RGB取得・送信
```

**追加方法:**
- Xcode → File → Add Files to "DisasterScanner"
- `Sources/` 内のファイルを選択
- "Copy items if needed" はオフ

### 3. Info.plist設定

Xcode → TARGETS → DisasterScanner → Info で以下を追加:

| Key | Value |
|-----|-------|
| Privacy - Camera Usage Description | LiDARスキャンのためにカメラを使用します |
| Privacy - Local Network Usage Description | DGX Sparkサーバーにデータを送信します |

### 4. ビルド＆実行

1. iPad Pro を Mac に USB 接続
2. Xcode 上部でデバイスを選択
3. ▶️ ボタンでビルド＆実行

## 使い方

1. アプリを起動
2. サーバーIPアドレスを入力（DGX SparkのIP）
3. 「スキャン開始」をタップ
4. 対象物をゆっくり周回撮影
5. 「停止」でスキャン終了

## データフォーマット

ハイブリッド送信方式で、2種類のパケットを送信します：

### 1. メッシュ頂点パケット（3fps）

リアルタイム3D表示用のメッシュ頂点データ：

```json
{
  "type": "mesh_vertices",
  "frame_id": 1,
  "timestamp": 1234567890.123,
  "vertices": [[x, y, z], ...],
  "vertex_count": 30000,
  "total_vertices": 150000,
  "camera_position": [x, y, z],
  "transform": [/* 4x4 transform matrix */]
}
```

### 2. RGB/深度パケット（1fps）

SAM 3セグメンテーション用のRGB画像と深度マップ：

```json
{
  "type": "frame_data",
  "frame_id": 1,
  "timestamp": 1234567890.123,
  "rgb": "<base64 encoded JPEG>",
  "depth": "<base64 encoded Float32 array>",
  "depth_width": 256,
  "depth_height": 192,
  "intrinsics": {
    "fx": 1000.0,
    "fy": 1000.0,
    "cx": 640.0,
    "cy": 480.0
  },
  "camera_position": [x, y, z],
  "transform": [/* 4x4 transform matrix */]
}
```

### 送信頻度

| データ種別 | 送信頻度 | 用途 |
|-----------|---------|------|
| メッシュ頂点 | 3fps | リアルタイム3D表示 |
| RGB画像 (JPEG) | 1fps | SAM 3用 |
| 深度マップ (Float32) | 1fps | 点群抽出用 |

## トラブルシューティング

### 「LiDAR非対応デバイス」エラー
- iPad Pro 第3世代以降が必要です
- シミュレーターでは動作しません

### 接続エラー
- サーバーIPアドレスを確認
- DGX Spark側でWebSocketサーバーが起動しているか確認
- 同じネットワークに接続しているか確認
