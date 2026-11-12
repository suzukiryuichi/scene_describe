#!/usr/bin/env python3
"""
Webcam scene describer.

ウェブカメラで撮影した画像を Qwen VLM を載せた llama-server の
OpenAI 互換 API に送り、周囲の状況を日本語で詳しく説明します。
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import requests
from PIL import Image


DEFAULT_URL = "http://127.0.0.1:8080"
DEFAULT_OUTPUT_DIR = Path("scene_logs")
DEFAULT_HTML_NAME_FORMAT = "%Y%m%d%H%M.html"
DEFAULT_VLM_PROMPT_MODE = "caution"
DEFAULT_INTERVAL = 10.0
DEFAULT_MAX_SIDE = 336
DEFAULT_JPEG_QUALITY = 72
DEFAULT_TIMEOUT = 120.0
DEFAULT_HTML_IMAGE_MAX_SIDE = 480

SYSTEM_PROMPT = """あなたは自動車の車載カメラによるフロント映像から周辺状況を説明する視覚アシスタントです。
画像に写っている事実を根拠に、日本語で具体的かつ実用的に説明してください。

守ること:
- 見えているものと推測を分け、不確かな点は「可能性があります」と書く。
- 人物が写る場合、個人特定や顔認識はせず、姿勢・位置・動作など観察できる範囲だけ述べる。
- 危険、障害物、移動の妨げ、注意すべき変化があれば明示する。
- 看板や店舗名などの文字を読む必要はありません。文字起こしや看板内容の説明はしないでください。
- 道路状況、歩行者、車両、信号、障害物など、前方確認で注意すべき点を優先する。
- 短い感想や物語ではなく、周辺状況の情報整理として出力する。
"""

USER_PROMPT = """この自動車の車載カメラのフロント映像について、周辺状況を詳しく説明してください。
次の観点を含めてください。

1. 全体概要
2. 主要な物体・人物・設備
3. 位置関係と空間の様子
4. 動作・変化していそうなこと
5. 注意点・危険・不確実な点

看板の文字を読む必要はありません。
"""

CAUTION_SYSTEM_PROMPT = """あなたは自動車の車載カメラによるフロント映像から注意点を抽出する視覚アシスタントです。
入力画像が複数ある場合は古い順に並んでいます。画像全体、主要な物体・人物・設備、位置関係、画像間の動作を確認したうえで、出力は注意点だけにしてください。

守ること:
- 人物が写る場合、個人特定や顔認識はしない。
- 見えている事実と推測を混ぜず、不確かな点は「可能性」と書く。
- 看板や店舗名などの文字を読む必要はありません。文字起こしや看板内容の説明はしないでください。
- 道路状況、歩行者、車両、信号、障害物など、前方確認で注意すべき点だけを出力する。
- 注意点がなければ「注意点なし」とだけ出力する。
- 最大3項目、各項目は短い完結文にする。
- 文末まで出力し、途中で切れた文を残さない。"""

CAUTION_USER_PROMPT = """この自動車の車載カメラのフロント映像を確認し、注意点のみを日本語で出力してください。
複数の画像がある場合は、画像間の位置変化から動いている対象を確認してください。
看板の文字を読む必要はありません。
全体概要、物体、位置関係、動作は内部で確認し、説明としては出さないでください。"""


@dataclass
class SceneEntry:
    timestamp: str
    elapsed_sec: float
    image_file: str
    description: str


@dataclass
class FrameSnapshot:
    frame_id: int
    frame: Any
    captured_at: float


def resize_for_inference(frame: Any, max_side: int) -> Image.Image:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        image = image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
    return image


def image_to_base64(image: Image.Image, quality: int) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def save_frame(frame: Any, path: Path, quality: int) -> None:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image.save(path, format="JPEG", quality=quality)


def check_server(base_url: str, session: requests.Session) -> None:
    try:
        resp = session.get(f"{base_url}/health", timeout=5)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"VLM サーバへ接続できません: {base_url}\n"
            f"先に ./02_start_server.sh を実行してください。\n"
            f"詳細: {exc}"
        ) from exc


def describe_scene(
    base_url: str,
    image_b64: str | list[str],
    session: requests.Session,
    max_tokens: int,
    temperature: float,
    timeout: float,
    prompt_mode: str,
    extra_system_prompt: str = "",
    extra_user_prompt: str = "",
    retry_on_length: bool = False,
) -> str:
    if prompt_mode == "caution":
        system_prompt = CAUTION_SYSTEM_PROMPT
        user_prompt = CAUTION_USER_PROMPT
    else:
        system_prompt = SYSTEM_PROMPT
        user_prompt = USER_PROMPT
    if extra_system_prompt:
        system_prompt = f"{system_prompt}\n\n{extra_system_prompt}"
    if extra_user_prompt:
        user_prompt = f"{user_prompt}\n\n{extra_user_prompt}"
    image_b64_list = [image_b64] if isinstance(image_b64, str) else image_b64
    user_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
        }
        for encoded_image in image_b64_list
    ]
    user_content.append({"type": "text", "text": user_prompt})
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_content,
        },
    ]
    attempts = 2 if retry_on_length else 1
    for attempt in range(attempts):
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens if attempt == 0 else max_tokens * 2,
            "temperature": temperature,
            "stream": False,
        }
        resp = session.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        content = choice["message"]["content"].strip()
        if choice.get("finish_reason") != "length":
            return content
    return content


def write_jsonl(path: Path, entry: SceneEntry) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def image_src_for_html(html_path: Path, output_dir: Path, image_file: str) -> str:
    image_path = output_dir / image_file
    if image_path.exists():
        try:
            image = Image.open(image_path)
            image.thumbnail((DEFAULT_HTML_IMAGE_MAX_SIDE, DEFAULT_HTML_IMAGE_MAX_SIDE), Image.LANCZOS)
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=68)
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"
        except OSError:
            pass

    try:
        image_src = os.path.relpath(image_path, html_path.parent)
    except ValueError:
        image_src = str(image_path)
    return html.escape(image_src)


def write_html(
    path: Path,
    output_dir_or_entries: Path | list[SceneEntry],
    entries: list[SceneEntry] | None = None,
) -> None:
    if entries is None:
        output_dir = path.parent
        entries = output_dir_or_entries  # type: ignore[assignment]
    else:
        output_dir = output_dir_or_entries  # type: ignore[assignment]

    cards = []
    for entry in reversed(entries):
        image_src = image_src_for_html(path, output_dir, entry.image_file)
        description = html.escape(entry.description).replace("\n", "<br>")
        cards.append(
            f"""<article class="entry">
  <img src="{image_src}" alt="webcam snapshot">
  <div class="body">
    <div class="meta">{html.escape(entry.timestamp)} / {entry.elapsed_sec:.2f}s</div>
    <p>{description}</p>
  </div>
</article>"""
        )

    document = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Webcam Scene Descriptions</title>
<style>
body {{
  margin: 0;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Noto Sans JP", sans-serif;
  background: #f5f6f8;
  color: #1f2933;
}}
main {{
  max-width: 1180px;
  margin: 0 auto;
  padding: 24px;
}}
h1 {{
  margin: 0 0 18px;
  font-size: 24px;
  font-weight: 700;
}}
.entry {{
  display: grid;
  grid-template-columns: minmax(280px, 42%) 1fr;
  gap: 16px;
  margin-bottom: 18px;
  padding: 14px;
  background: #fff;
  border: 1px solid #d8dee8;
  border-radius: 8px;
}}
.entry img {{
  width: 100%;
  height: auto;
  border-radius: 6px;
  display: block;
}}
.meta {{
  color: #687586;
  font-size: 13px;
  margin-bottom: 10px;
}}
p {{
  margin: 0;
  line-height: 1.75;
}}
@media (max-width: 760px) {{
  main {{ padding: 14px; }}
  .entry {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<main>
<h1>Webcam Scene Descriptions</h1>
{''.join(cards)}
</main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def resolve_html_path(output_dir: Path, html_file: str, html_name_format: str) -> Path:
    if html_file:
        path = Path(html_file)
        return path if path.is_absolute() else output_dir / path
    return output_dir / datetime.now().strftime(html_name_format)


def open_camera(device: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"カメラを開けません: device={device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


class LatestFrameCamera:
    def __init__(self, cap: cv2.VideoCapture, warmup_frames: int, sample_fps: float, temporal_frames: int) -> None:
        self.cap = cap
        self.warmup_frames = max(0, warmup_frames)
        self.sample_interval = 1.0 / max(0.001, sample_fps)
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)
        self._frame: Any | None = None
        self._frame_id = 0
        self._captured_at = 0.0
        self._sampled_frames: deque[FrameSnapshot] = deque(maxlen=max(1, temporal_frames))
        self._last_sampled_at = 0.0
        self._error: Exception | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def latest(self, min_frame_id: int = 0, timeout: float = 5.0) -> FrameSnapshot:
        deadline = time.monotonic() + timeout
        while True:
            if self._error is not None:
                raise RuntimeError("カメラ画像を取得できませんでした") from self._error
            if not self._ready.wait(timeout=max(0.0, min(0.1, deadline - time.monotonic()))):
                if time.monotonic() >= deadline:
                    raise RuntimeError("カメラ画像を取得できませんでした")
                continue
            with self._lock:
                if self._frame is not None and self._frame_id > min_frame_id:
                    return FrameSnapshot(self._frame_id, self._frame.copy(), self._captured_at)
            if time.monotonic() >= deadline:
                with self._lock:
                    if self._frame is not None:
                        return FrameSnapshot(self._frame_id, self._frame.copy(), self._captured_at)
                raise RuntimeError("カメラ画像を取得できませんでした")
            time.sleep(0.01)

    def latest_sequence(self, min_frame_id: int = 0, timeout: float = 5.0) -> list[FrameSnapshot]:
        latest = self.latest(min_frame_id=min_frame_id, timeout=timeout)
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                snapshots = [
                    FrameSnapshot(snapshot.frame_id, snapshot.frame.copy(), snapshot.captured_at)
                    for snapshot in self._sampled_frames
                ]
                sequence_is_ready = len(snapshots) == self._sampled_frames.maxlen
            if sequence_is_ready or time.monotonic() >= deadline:
                return snapshots or [latest]
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))

    def _run(self) -> None:
        try:
            skipped = 0
            while not self._stop.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue
                if skipped < self.warmup_frames:
                    skipped += 1
                    continue
                captured_at = time.perf_counter()
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
                    self._captured_at = captured_at
                    if captured_at - self._last_sampled_at >= self.sample_interval:
                        self._sampled_frames.append(FrameSnapshot(self._frame_id, frame.copy(), captured_at))
                        self._last_sampled_at = captured_at
                    self._ready.set()
        except Exception as exc:
            self._error = exc
            self._ready.set()


class AsyncFrameSaver:
    def __init__(self, quality: int, max_queue_size: int = 4) -> None:
        self.quality = quality
        self._queue: queue.Queue[tuple[Any, Path] | None] = queue.Queue(maxsize=max_queue_size)
        self._thread = threading.Thread(target=self._run, name="frame-save", daemon=True)
        self._dropped = 0
        self._error_lock = threading.Lock()
        self._error: Exception | None = None

    @property
    def dropped(self) -> int:
        return self._dropped

    def start(self) -> None:
        self._thread.start()

    def enqueue(self, frame: Any, path: Path) -> None:
        self._raise_if_failed()
        try:
            self._queue.put_nowait((frame.copy(), path))
        except queue.Full:
            self._dropped += 1

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5.0)

    def wait_until_idle(self) -> None:
        self._queue.join()
        self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        with self._error_lock:
            error = self._error
        if error is not None:
            raise RuntimeError("画像を保存できませんでした") from error

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                frame, path = item
                try:
                    save_frame(frame, path, self.quality)
                except Exception as exc:
                    with self._error_lock:
                        if self._error is None:
                            self._error = exc
            finally:
                self._queue.task_done()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ウェブカメラ画像をQwen VLMで分析し、周辺状況を日本語で説明する"
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="llama-server のURL")
    parser.add_argument("--device", type=int, default=0, help="カメラデバイス番号")
    parser.add_argument("--width", type=int, default=1280, help="カメラ取得幅")
    parser.add_argument("--height", type=int, default=720, help="カメラ取得高さ")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="連続実行時の撮影間隔秒")
    parser.add_argument("--count", type=int, default=1, help="撮影回数。0 で無限")
    parser.add_argument("--continuous", action="store_true", help="無限に実行")
    parser.add_argument(
        "--vlm-prompt-mode",
        choices=("caution", "detailed"),
        default=DEFAULT_VLM_PROMPT_MODE,
        help="vlmで使うプロンプト。caution=注意点のみ、detailed=従来の詳細説明",
    )
    parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE, help="推論用画像の最大辺")
    parser.add_argument("--quality", type=int, default=DEFAULT_JPEG_QUALITY, help="JPEG品質")
    parser.add_argument("--max-tokens", type=int, default=60, help="説明文の最大トークン数")
    parser.add_argument(
        "--retry-on-length",
        action="store_true",
        help="出力が最大トークン数に達した場合、上限を2倍にして再推論する",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="生成温度")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="VLM API タイムアウト秒")
    parser.add_argument("--warmup-frames", type=int, default=1, help="撮影前に捨てるフレーム数")
    parser.add_argument("--sample-fps", type=float, default=5.0, help="動き確認用のフレーム取得FPS")
    parser.add_argument("--temporal-frames", type=int, default=10, help="1回のVLM分析に渡す時系列フレーム数")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="ログ出力先")
    parser.add_argument(
        "--html-file",
        default="",
        help="HTML出力ファイル名。未指定なら起動時刻から生成",
    )
    parser.add_argument(
        "--html-name-format",
        default=DEFAULT_HTML_NAME_FORMAT,
        help="--html-file未指定時のstrftime形式",
    )
    parser.add_argument(
        "--no-index-html",
        action="store_true",
        help="互換用の index.html 更新を行わない",
    )
    parser.add_argument("--history", type=int, default=50, help="今回の起動中にHTMLへ残す件数")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "descriptions.jsonl"
    html_path = resolve_html_path(args.output_dir, args.html_file, args.html_name_format)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    index_html_path = args.output_dir / "index.html"

    session = requests.Session()
    check_server(args.url, session)
    cap = open_camera(args.device, args.width, args.height)
    camera = LatestFrameCamera(cap, args.warmup_frames, args.sample_fps, args.temporal_frames)
    saver = AsyncFrameSaver(args.quality)
    camera.start()
    saver.start()

    target_count = 0 if args.continuous else args.count
    shot_index = 0
    last_frame_id = 0
    session_entries: list[SceneEntry] = []
    print("周辺状況の説明を開始します。終了するには Ctrl+C を押してください。")
    print(f"Temporal sampling: {args.sample_fps:.2f} FPS x {args.temporal_frames} frames")
    print(f"HTML: {html_path}")

    try:
        while target_count == 0 or shot_index < target_count:
            shot_index += 1
            started = time.perf_counter()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            safe_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = images_dir / f"scene_{safe_ts}_{shot_index:04d}.jpg"

            snapshots = camera.latest_sequence(min_frame_id=last_frame_id)
            snapshot = snapshots[-1]
            last_frame_id = snapshot.frame_id
            frame_age = max(0.0, time.perf_counter() - snapshot.captured_at)
            saver.enqueue(snapshot.frame, image_path)
            image_b64 = [
                image_to_base64(resize_for_inference(sample.frame, args.max_side), args.quality)
                for sample in snapshots
            ]
            description = describe_scene(
                args.url,
                image_b64,
                session,
                args.max_tokens,
                args.temperature,
                args.timeout,
                prompt_mode=args.vlm_prompt_mode,
                retry_on_length=args.retry_on_length,
            )

            elapsed = time.perf_counter() - started
            entry = SceneEntry(
                timestamp=timestamp,
                elapsed_sec=elapsed,
                image_file=str(image_path.relative_to(args.output_dir)),
                description=description,
            )
            saver.wait_until_idle()
            write_jsonl(jsonl_path, entry)
            session_entries.append(entry)
            entries = session_entries[-args.history :]
            write_html(html_path, args.output_dir, entries)
            if not args.no_index_html and html_path.resolve() != index_html_path.resolve():
                write_html(index_html_path, args.output_dir, entries)

            print(f"\n[{timestamp}] {elapsed:.2f}s / frame_age={frame_age:.2f}s")
            print(description)
            sys.stdout.flush()

            if target_count == 0 or shot_index < target_count:
                time.sleep(max(0.0, args.interval - elapsed))
    except KeyboardInterrupt:
        print("\n停止しました。")
    finally:
        camera.stop()
        saver.stop()
        cap.release()
        session.close()
        if saver.dropped:
            print(f"保存キュー混雑により {saver.dropped} 件の画像保存をスキップしました。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
