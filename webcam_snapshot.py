#!/usr/bin/env python3
"""
webcam_snapshot.py — ウェブカム画像から物語を生成し、HTML ギャラリーとして出力

使い方:
    source venv/bin/activate
    python webcam_snapshot.py --count 10
    python webcam_snapshot.py --count 5 --output gallery.html
    python webcam_snapshot.py --count 20 --interval 2.0
"""

import argparse
import base64
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image


# ────────────────────────────────────────────
# 設定
# ────────────────────────────────────────────
DEFAULT_URL = "http://localhost:8080"
IMG_SIZE = (336, 336)
MAX_TOKENS = 60
TEMPERATURE = 0.8
JPEG_QUALITY = 85
DISPLAY_WIDTH = 336  # HTML表示サイズ

SYSTEM_PROMPT = (
    "あなたは映像詩人です。画像を見て、1行の短い物語を日本語で書いてください。"
    "簡潔に、情緒的に。1文だけ出力し、他には何も書かないでください。"
)


# ────────────────────────────────────────────
# 画像処理
# ────────────────────────────────────────────
def frame_to_base64(frame: np.ndarray) -> str:
    """OpenCV BGR フレーム → 336x336 リサイズ → JPEG base64"""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def frame_to_display_base64(frame: np.ndarray) -> str:
    """表示用: 元解像度のまま JPEG base64"""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ────────────────────────────────────────────
# LLM 呼び出し
# ────────────────────────────────────────────
def request_story(base_url: str, image_b64: str, session: requests.Session) -> str:
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        },
                    },
                    {"type": "text", "text": "この光景から物語を。"},
                ],
            },
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False,
    }
    resp = session.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    story = data["choices"][0]["message"]["content"].strip()
    return story.split("\n")[0].strip()


# ────────────────────────────────────────────
# HTML 生成
# ────────────────────────────────────────────
def generate_html(entries: list, output_path: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cards = ""
    for i, entry in enumerate(entries):
        cards += f"""
    <div class="card">
      <div class="number">#{i + 1}</div>
      <img src="data:image/jpeg;base64,{entry['image_b64']}" alt="snapshot {i + 1}">
      <p class="story">{entry['story']}</p>
      <div class="meta">{entry['time']} | {entry['elapsed']:.2f}s</div>
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Webcam Stories — {timestamp}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: "Hiragino Kaku Gothic ProN", "Noto Sans JP", sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    padding: 2rem;
  }}
  h1 {{
    text-align: center;
    color: #e0c097;
    margin-bottom: 0.5rem;
    font-size: 1.5rem;
  }}
  .subtitle {{
    text-align: center;
    color: #888;
    margin-bottom: 2rem;
    font-size: 0.85rem;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    gap: 1.5rem;
    max-width: 1200px;
    margin: 0 auto;
  }}
  .card {{
    background: #16213e;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    position: relative;
  }}
  .card img {{
    width: 100%;
    display: block;
  }}
  .number {{
    position: absolute;
    top: 8px;
    left: 8px;
    background: rgba(0,0,0,0.6);
    color: #e0c097;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.8rem;
  }}
  .story {{
    padding: 1rem 1.2rem;
    font-size: 1rem;
    line-height: 1.7;
    color: #f0f0f0;
  }}
  .meta {{
    padding: 0 1.2rem 0.8rem;
    font-size: 0.75rem;
    color: #666;
  }}
</style>
</head>
<body>
  <h1>Webcam Stories</h1>
  <div class="subtitle">{timestamp} — {len(entries)} snapshots</div>
  <div class="grid">{cards}
  </div>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")


# ────────────────────────────────────────────
# メイン
# ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ウェブカム → VLM → HTML ギャラリー")
    parser.add_argument("--count", type=int, default=5, help="撮影枚数")
    parser.add_argument("--device", type=int, default=0, help="カメラデバイス番号")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="llama-server URL")
    parser.add_argument("--output", type=str, default=None, help="出力HTMLファイル名")
    parser.add_argument("--interval", type=float, default=0.0, help="撮影間隔（秒）。0で最速")
    args = parser.parse_args()

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"stories_{ts}.html"

    # カメラ初期化
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"ERROR: カメラ {args.device} を開けません。", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # サーバー確認
    session = requests.Session()
    try:
        health = session.get(f"{args.url}/health", timeout=5)
        status = health.json().get("status", "unknown")
        if status != "ok":
            print(f"WARNING: サーバーステータス: {status}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: サーバーに接続できません ({e})", file=sys.stderr)
        sys.exit(1)

    print(f"撮影枚数: {args.count}")
    print(f"出力先:   {args.output}")
    print(f"間隔:     {args.interval}s" if args.interval > 0 else "間隔:     最速")
    print("─" * 50)

    entries = []
    for i in range(args.count):
        # 間隔待ち
        if i > 0 and args.interval > 0:
            time.sleep(args.interval)

        # フレーム取得（数フレーム読み捨ててバッファを最新に）
        for _ in range(3):
            cap.grab()
        ret, frame = cap.read()
        if not ret:
            print(f"  [{i+1}/{args.count}] フレーム取得失敗、スキップ", file=sys.stderr)
            continue

        t_start = time.perf_counter()
        infer_b64 = frame_to_base64(frame)
        display_b64 = frame_to_display_base64(frame)

        try:
            story = request_story(args.url, infer_b64, session)
        except Exception as e:
            story = f"[エラー: {e}]"

        elapsed = time.perf_counter() - t_start
        now = datetime.now().strftime("%H:%M:%S")

        entries.append({
            "image_b64": display_b64,
            "story": story,
            "time": now,
            "elapsed": elapsed,
        })

        print(f"  [{i+1}/{args.count}] {elapsed:.2f}s — {story}")

    cap.release()
    session.close()

    if entries:
        generate_html(entries, args.output)
        print("─" * 50)
        print(f"完了! {len(entries)} 枚の物語を {args.output} に保存しました。")
    else:
        print("画像が取得できませんでした。", file=sys.stderr)


if __name__ == "__main__":
    main()
