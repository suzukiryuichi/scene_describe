#!/usr/bin/env python3
"""
Video scene describer.

撮影済み動画ファイルから一定間隔でフレームを抜き出し、
webcam_scene_describer.py と同じ VLM 分析で注意点を出力します。
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import requests

from webcam_scene_describer import (
    DEFAULT_HTML_NAME_FORMAT,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_SIDE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TIMEOUT,
    DEFAULT_URL,
    DEFAULT_VLM_PROMPT_MODE,
    SceneEntry,
    check_server,
    describe_scene,
    image_to_base64,
    resize_for_inference,
    resolve_html_path,
    save_frame,
    write_html,
    write_jsonl,
)


VIDEO_SYSTEM_PROMPT_CONTEXT = """前提:
- 入力画像は自動車の車載カメラによるフロント映像を時系列順に並べたものです。
- 複数の画像がある場合は、約10 FPSで連続撮影された画像です。位置の変化を比較し、移動中の対象を優先して確認してください。
- 看板や店舗名などの文字を読む必要はありません。文字起こしや看板内容の説明はしないでください。
- 道路状況、歩行者、車両、信号、障害物など、前方確認で注意すべき点を優先してください。"""

VIDEO_USER_PROMPT_CONTEXT = """画像は自動車の車載カメラのフロント映像で、複数ある場合は古い順に並んでいます。
画像間の位置変化から、歩行者、自転車、車両など動いている対象を確認してください。
看板の文字を読む必要はありません。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="撮影済み動画からフレームを抽出し、Qwen VLMで注意点を日本語で出力する"
    )
    parser.add_argument("--video", type=Path, required=True, help="分析する動画ファイル")
    parser.add_argument("--url", default=DEFAULT_URL, help="llama-server のURL")
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=1.0,
        help="VLM分析を開始する間隔秒",
    )
    parser.add_argument("--sample-fps", type=float, default=5.0, help="動き確認用のフレーム取得FPS")
    parser.add_argument("--temporal-frames", type=int, default=10, help="1回のVLM分析に渡す時系列フレーム数")
    parser.add_argument("--start-sec", type=float, default=0.0, help="分析開始位置秒")
    parser.add_argument("--end-sec", type=float, default=0.0, help="分析終了位置秒。0 で動画末尾まで")
    parser.add_argument("--max-samples", type=int, default=0, help="最大分析回数。0 で制限なし")
    parser.add_argument(
        "--vlm-prompt-mode",
        choices=("caution", "detailed"),
        default=DEFAULT_VLM_PROMPT_MODE,
        help="vlmで使うプロンプト。caution=注意点のみ、detailed=従来の詳細説明",
    )
    parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE, help="VLM推論用画像の最大辺")
    parser.add_argument("--quality", type=int, default=DEFAULT_JPEG_QUALITY, help="JPEG品質")
    parser.add_argument("--max-tokens", type=int, default=120, help="説明文の最大トークン数")
    parser.add_argument(
        "--retry-on-length",
        action="store_true",
        help="出力が最大トークン数に達した場合、上限を2倍にして再推論する",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="生成温度")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="VLM API タイムアウト秒")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="ログ出力先")
    parser.add_argument("--html-file", default="", help="HTML出力ファイル名。未指定なら起動時刻から生成")
    parser.add_argument(
        "--html-name-format",
        default=DEFAULT_HTML_NAME_FORMAT,
        help="--html-file未指定時のstrftime形式",
    )
    parser.add_argument("--no-index-html", action="store_true", help="互換用の index.html 更新を行わない")
    parser.add_argument("--history", type=int, default=50, help="今回の起動中にHTMLへ残す件数")
    return parser.parse_args()


def open_video(path: Path) -> cv2.VideoCapture:
    if not path.exists():
        raise FileNotFoundError(f"動画ファイルが見つかりません: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"動画ファイルを開けません: {path}")
    return cap


def read_sample_frame(
    cap: cv2.VideoCapture,
    current_frame_index: int,
    target_frame_index: int,
) -> tuple[Any | None, int]:
    target_frame_index = max(0, target_frame_index)
    if target_frame_index < current_frame_index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
        current_frame_index = target_frame_index
    while True:
        ok, frame = cap.read()
        if not ok:
            return None, current_frame_index
        frame_index = current_frame_index
        current_frame_index += 1
        if frame_index >= target_frame_index:
            return frame, current_frame_index


def read_temporal_frames(
    cap: cv2.VideoCapture,
    current_frame_index: int,
    position_sec: float,
    video_fps: float,
    sample_fps: float,
    temporal_frames: int,
    end_sec: float,
) -> tuple[list[Any], int]:
    frames = []
    for offset in range(max(1, temporal_frames)):
        sample_position_sec = position_sec + offset / max(0.001, sample_fps)
        if sample_position_sec > end_sec:
            break
        target_frame_index = int(round(sample_position_sec * video_fps))
        frame, current_frame_index = read_sample_frame(cap, current_frame_index, target_frame_index)
        if frame is None:
            break
        frames.append(frame)
    return frames, current_frame_index


def format_video_timestamp(position_sec: float) -> str:
    return str(timedelta(seconds=int(position_sec)))


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

    cap = open_video(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("動画のFPSを取得できませんでした")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if frame_count > 0 else 0.0
    end_sec = args.end_sec if args.end_sec > 0 else duration_sec
    if end_sec <= 0 or args.start_sec >= end_sec:
        raise RuntimeError(f"分析範囲が不正です: start={args.start_sec}, end={args.end_sec}")

    video_stem = args.video.stem
    sample_index = 0
    position_sec = max(0.0, args.start_sec)
    current_frame_index = 0
    session_entries: list[SceneEntry] = []
    print("動画の注意点出力を開始します。")
    print(f"Video: {args.video}")
    print(f"Duration: {duration_sec:.2f}s / FPS: {fps:.2f}")
    print(f"Temporal sampling: {args.sample_fps:.2f} FPS x {args.temporal_frames} frames")
    print(f"HTML: {html_path}")

    try:
        while position_sec <= end_sec:
            if args.max_samples > 0 and sample_index >= args.max_samples:
                break

            sample_index += 1
            started = time.perf_counter()
            frames, current_frame_index = read_temporal_frames(
                cap,
                current_frame_index,
                position_sec,
                fps,
                args.sample_fps,
                args.temporal_frames,
                end_sec,
            )
            if not frames:
                break

            frame = frames[-1]
            now = datetime.now()
            timestamp = f"{now:%Y-%m-%d %H:%M:%S} / video {format_video_timestamp(position_sec)}"
            safe_ts = now.strftime("%Y%m%d_%H%M%S")
            image_path = images_dir / f"{video_stem}_{safe_ts}_{sample_index:04d}.jpg"
            save_frame(frame, image_path, args.quality)
            image_b64 = [
                image_to_base64(resize_for_inference(sample_frame, args.max_side), args.quality)
                for sample_frame in frames
            ]
            description = describe_scene(
                args.url,
                image_b64,
                session,
                args.max_tokens,
                args.temperature,
                args.timeout,
                prompt_mode=args.vlm_prompt_mode,
                extra_system_prompt=VIDEO_SYSTEM_PROMPT_CONTEXT,
                extra_user_prompt=VIDEO_USER_PROMPT_CONTEXT,
                retry_on_length=args.retry_on_length,
            )

            elapsed = time.perf_counter() - started
            entry = SceneEntry(
                timestamp=timestamp,
                elapsed_sec=elapsed,
                image_file=str(image_path.relative_to(args.output_dir)),
                description=description,
            )
            write_jsonl(jsonl_path, entry)
            session_entries.append(entry)
            entries = session_entries[-args.history :]
            write_html(html_path, args.output_dir, entries)
            if not args.no_index_html and html_path.resolve() != index_html_path.resolve():
                write_html(index_html_path, args.output_dir, entries)

            print(f"\n[{timestamp}] {elapsed:.2f}s")
            print(description)
            sys.stdout.flush()
            position_sec += max(0.001, args.frame_interval)
    finally:
        cap.release()
        session.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
