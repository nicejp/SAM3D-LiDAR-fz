#!/usr/bin/env python3
"""
Claude API Interface
ClaudeのマルチモーダルAPIを使用して高品質な3Dモデル生成コードを取得

環境変数:
    ANTHROPIC_API_KEY: Claude APIキー

使い方:
    # 環境変数でAPIキーを設定
    export ANTHROPIC_API_KEY="sk-ant-xxxxx"

    # テスト実行
    python -m server.phase2_full.claude_interface
"""

import os
import json
import base64
from pathlib import Path
from typing import Optional, List
import requests

# Anthropic API設定
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-4-20250514"  # 最新モデル
MAX_TOKENS = 8192


def get_api_key() -> Optional[str]:
    """APIキーを取得"""
    return os.environ.get("ANTHROPIC_API_KEY")


def check_api_available() -> bool:
    """APIが利用可能か確認"""
    return get_api_key() is not None


def encode_image(image_path: str) -> Optional[dict]:
    """
    画像をbase64エンコード

    Args:
        image_path: 画像ファイルパス

    Returns:
        Claude API用の画像オブジェクト、またはNone
    """
    path = Path(image_path)
    if not path.exists():
        return None

    # MIMEタイプを判定
    suffix = path.suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    media_type = mime_types.get(suffix, "image/jpeg")

    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")

    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": data
        }
    }


def generate(
    prompt: str,
    images: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.3
) -> str:
    """
    Claude APIでテキスト生成（マルチモーダル対応）

    Args:
        prompt: ユーザープロンプト
        images: 画像ファイルパスのリスト
        system_prompt: システムプロンプト
        model: モデル名
        max_tokens: 最大トークン数
        temperature: 温度パラメータ

    Returns:
        生成されたテキスト
    """
    api_key = get_api_key()
    if not api_key:
        return "Error: ANTHROPIC_API_KEY environment variable not set"

    # コンテンツを構築
    content = []

    # 画像を追加
    if images:
        for image_path in images:
            image_obj = encode_image(image_path)
            if image_obj:
                content.append(image_obj)

    # テキストを追加
    content.append({
        "type": "text",
        "text": prompt
    })

    # リクエストペイロード
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }

    if system_prompt:
        payload["system"] = system_prompt

    # APIリクエスト
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    try:
        response = requests.post(
            ANTHROPIC_API_URL,
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code == 200:
            data = response.json()
            # レスポンスからテキストを抽出
            text_content = [
                block["text"]
                for block in data.get("content", [])
                if block.get("type") == "text"
            ]
            return "\n".join(text_content)
        else:
            return f"Error: {response.status_code} - {response.text}"

    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"


def generate_blender_code_claude(
    pointcloud_stats: dict,
    user_prompt: str,
    rgb_images: Optional[List[str]] = None,
    model: str = DEFAULT_MODEL
) -> str:
    """
    点群統計情報、RGB画像、ユーザー指示からBlender Pythonコードを生成

    Args:
        pointcloud_stats: 点群の統計情報
        user_prompt: ユーザーからの指示
        rgb_images: RGB画像ファイルパスのリスト（最大5枚）
        model: 使用するモデル

    Returns:
        Blender Pythonコード
    """
    system_prompt = """あなたはBlender Python APIのエキスパートです。
ユーザーから与えられた3D点群の統計情報と、オプションでRGB画像に基づいて、
高品質なBlender Pythonコードを生成してください。

ルール:
1. bpyモジュールを使用してください
2. 既存のオブジェクトを削除してから新しいオブジェクトを作成してください
3. オブジェクトの位置は点群の中心(center)を基準にしてください
4. オブジェクトのサイズは点群の寸法(dimensions)を参考にしてください
5. RGB画像がある場合は、見た目や形状を参考にしてください
6. メッシュのディテール（エッジループ、ベベル等）を適切に追加してください
7. マテリアルとテクスチャも必要に応じて設定してください
8. コードのみを出力し、説明は含めないでください
9. コードは```python と ``` で囲んでください

出力するコードの構造:
1. 既存オブジェクトのクリア
2. メッシュオブジェクトの作成
3. 頂点・面の定義
4. モディファイアの適用（必要に応じて）
5. マテリアルの設定
6. シーン設定（カメラ、ライト）

利用可能な情報:
- center: 点群の中心座標 [x, y, z]
- dimensions: 点群の寸法 [width, height, depth]
- width, height, depth: 各軸の寸法（メートル単位）
- num_points: 点の数
- linearity: 線形性（0-1、1に近いほど棒状）
- planarity: 平面性（0-1、1に近いほど平面状）
- sphericity: 球形性（0-1、1に近いほど球状）
"""

    # プロンプトを構築
    prompt_parts = [
        "以下の情報に基づいて、高品質なBlender Pythonコードを生成してください。",
        "",
        "## 点群の統計情報:",
        "```json",
        json.dumps(pointcloud_stats, indent=2, ensure_ascii=False),
        "```",
        "",
        "## ユーザーの指示:",
        user_prompt,
    ]

    if rgb_images:
        prompt_parts.extend([
            "",
            "## 参考画像:",
            "添付した画像を参考に、形状やディテールを再現してください。",
        ])

    prompt = "\n".join(prompt_parts)

    # 画像は最大5枚まで
    images_to_use = rgb_images[:5] if rgb_images else None

    response = generate(
        prompt=prompt,
        images=images_to_use,
        system_prompt=system_prompt,
        model=model,
        temperature=0.2  # コード生成は低温度で
    )

    # コードブロックを抽出
    code = extract_code_block(response)
    return code


def extract_code_block(text: str) -> str:
    """テキストからPythonコードブロックを抽出"""
    import re

    # ```python ... ``` を探す
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # ``` ... ``` を探す
    pattern = r"```\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    return text.strip()


def analyze_image(
    image_path: str,
    model: str = DEFAULT_MODEL
) -> str:
    """
    画像を分析して何が写っているか説明

    Args:
        image_path: 画像ファイルパス
        model: 使用するモデル

    Returns:
        分析結果
    """
    prompt = """この画像に写っているものを詳しく説明してください。
特に以下の点に注目してください：
1. 主要なオブジェクト
2. 形状の特徴
3. 色やテクスチャ
4. 3Dモデル化する際に重要な要素

日本語で回答してください。"""

    return generate(
        prompt=prompt,
        images=[image_path],
        model=model,
        temperature=0.5
    )


def get_session_images(session_dir: str, max_images: int = 5) -> List[str]:
    """
    セッションディレクトリからRGB画像を取得

    Args:
        session_dir: セッションディレクトリ
        max_images: 最大画像数

    Returns:
        画像パスのリスト
    """
    session_path = Path(session_dir)
    rgb_dir = session_path / "rgb"

    if not rgb_dir.exists():
        return []

    images = sorted(rgb_dir.glob("frame_*.jpg"))

    if not images:
        return []

    # 均等にサンプリング
    if len(images) <= max_images:
        return [str(p) for p in images]

    step = len(images) // max_images
    sampled = [images[i * step] for i in range(max_images)]
    return [str(p) for p in sampled]


if __name__ == "__main__":
    print("Claude API Interface")
    print("=" * 50)
    print(f"API Available: {check_api_available()}")

    if not check_api_available():
        print("\nTo use Claude API:")
        print("  export ANTHROPIC_API_KEY=\"sk-ant-xxxxx\"")
        exit(1)

    # テスト用の点群統計
    test_stats = {
        "num_points": 50000,
        "center": [0.0, 0.5, 1.2],
        "dimensions": [0.45, 0.9, 0.45],
        "width": 0.45,
        "height": 0.9,
        "depth": 0.45,
        "linearity": 0.15,
        "planarity": 0.25,
        "sphericity": 0.6
    }

    print("\nBlenderコード生成テスト:")
    code = generate_blender_code_claude(
        test_stats,
        "この点群から椅子を作成してください。背もたれと4本の脚があるシンプルな椅子です。"
    )
    print(code[:500] + "..." if len(code) > 500 else code)
