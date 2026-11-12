#!/usr/bin/env python3
"""
Stopped-vehicle dual video scene describer.

撮影済みのフロントカメラ動画とバックカメラ動画から同じ時刻のフレームを抜き出し、
停車中の自車が発進・停車継続・乗降・周囲確認を行う際の注意点を VLM 分析して出力します。
"""

from __future__ import annotations

import argparse
import base64
import difflib
import hashlib
import html
import io
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2
import requests
from PIL import Image

from webcam_scene_describer import (
    DEFAULT_HTML_NAME_FORMAT,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_URL,
    DEFAULT_VLM_PROMPT_MODE,
    SceneEntry,
    check_server,
    image_to_base64,
    image_src_for_html,
    resize_for_inference,
    resolve_html_path,
    save_frame,
    write_jsonl,
)


DUAL_VIDEO_SYSTEM_PROMPT_CONTEXT = """前提:
- 入力画像は自動車の車載カメラ映像を横に結合したものを時系列順に並べたものです。
- 自車は停車中です。走行中の進路維持ではなく、停車中の周囲確認、発進可否、停車継続、乗降・ドア開閉時の安全確認に関係する注意点を優先してください。
- 複数画像がある場合は、各画像の変化を比較し、静止物ではなく、自車周辺で位置が変化している対象を最優先で確認してください。
- 特に、接近、離脱、横切り、自車付近を通過する歩行者、自転車、車両を優先してください。
- 結合画像は2つのカメラ領域で構成されています。左側が「FRONT VIEW ONLY」、右側が「REAR VIEW ONLY」です。
- 左側の「FRONT VIEW ONLY」は、フロント動画の上半分だけを切り出した前方道路映像です。
- 右側の「REAR VIEW ONLY」は、バックカメラの後方道路映像です。
- 左側の危険は必ず direction を「前方」とし、右側の危険は必ず direction を「後方」としてください。前方と後方を取り違えないでください。
- 乗客、運転者、車内の座席・荷物・内装などの情報は不要です。車内や乗員の様子は説明しないでください。
- 看板や店舗名などの文字を読む必要はありません。文字起こしや看板内容の説明はしないでください。

危険判定:
- 危険箇所は重要度の高い順に最大3項目まで挙げ、道路全体の状況説明や通常の風景説明は出力しないでください。
- 重要度は、自車との距離、自車への接近速度、発進・停車継続・ドア開閉への影響で判断してください。
- 優先順位は、1. 自車直近を通過・接近する歩行者、2. 自車直近を通過・接近する自転車、3. 自車直近を通過・接近する車両、4. 発進経路を直接妨げる対象、5. 停車継続やドア開閉範囲へ進入する対象、6. その他です。
- ドア開閉や乗降に関する注意は、画像から自車側方または後方近くの歩行者・自転車・車両などが確認できる場合だけ出力してください。
- 静止している車両だけでは危険として出力しないでください。発進経路を直接塞いでいる、自車直近ですれ違いに大きく影響する、ドア開閉や乗降に直接影響する、または画像間で相対位置が変化し自車へ接近している場合だけ危険として扱ってください。
- 歩道、店舗、交差点、駐車車両などが見えるだけで、写っていない歩行者や車両の飛び出しを危険として追加しないでください。
- 路肩や歩道上のコーン、消火栓、看板、植栽などは、自車の発進経路、停車位置、ドア開閉範囲、または車両直近を直接妨げる場合だけ危険として扱ってください。
- 車両の種類、色、駐車中/停車中/走行中の状態を区別する必要はありません。色名は書かず、すべて「車両」と書いてください。
- 画像や補足情報から根拠を持って挙げられる危険箇所が3項目未満の場合は、架空の危険を追加せず、確認できた項目だけを出力してください。危険を3件埋める必要はありません。
- 画像から見えていない歩行者、横断車両、ドア開き、飛び出しなどを「可能性」として追加しないでください。
- 「〜しそう」「〜かもしれない」「〜が予想される」「〜の可能性がある」などの推測は禁止です。画像に確認できない対象を理由に注意点を生成してはいけません。

出力形式:
- direction には「前方」「左後方」「右後方」「後方」「補足情報」のいずれかだけを入れてください。
- 通常は「前方」または「後方」だけを使い、「左後方」「右後方」は使わないでください。
- message には、停車中の自車に対する危険の根拠と注意点を「〜ため、発進時の〜に注意」「〜ため、停車継続中の〜に注意」「〜ため、ドア開閉時の〜に注意」のような短い完結文で書いてください。
- message には方向やカメラ領域名を含めないでください。方向は direction にだけ書いてください。
- 画像から停車中の安全確認に関する注意点が見当たらず、補足情報による注意喚起も不要な場合は {"risks":[]} だけを返してください。
- 位置情報、天気情報、時刻情報などの補足情報が入力されている場合は、画像から分かる注意点に加えて、運転上必要な場合だけ出力に反映してください。
- 補足情報は画像で見えている事実とは区別して扱い、通学路、学校付近、住宅街、朝夕の通勤・通学時間帯、夜間、雨、降雨予報、積雪、強風、視界不良など安全上意味がある場合に限って短く注意喚起してください。
- 補足情報による注意点は、direction を「補足情報」としてください。
- 補足情報から断定できないことは断定せず、「付近」「可能性」「この後」など不確実性が分かる表現にしてください。
- 出力は次のJSON形式だけにしてください。JSON以外のMarkdown、コードブロック、説明文、前置き、後書きは一切出力しないでください。

出力例:
{"risks":[{"direction":"前方","message":"車両直近を自転車が通過しているため、発進時の接触に注意"},{"direction":"後方","message":"車両が接近しているため、停車継続中の後方確認に注意"}]}"""

DUAL_VIDEO_USER_PROMPT_CONTEXT = """画像は同時刻の車載映像を結合したもので、複数ある場合は古い順に並んでいます。
画面配置は、左側が前方道路映像、右側が後方道路映像です。
自車は停車中です。画像間の位置変化から、歩行者、自転車、車両など自車の近くを通過・接近・離脱する対象を確認してください。
前方と後方の両方を確認し、停車中の周囲確認、発進可否、停車継続、乗降・ドア開閉時の安全確認に関する注意点のみを日本語で出力してください。
乗客、運転者、座席、荷物、内装など車内や乗員に関する情報は出力しないでください。
看板の文字を読む必要はありません。
注意点がある場合は、危険箇所を重要度の高い順に3項目まで挙げてください。
画像に明確に写っていない危険を推測で追加しないでください。
歩道や交差点が見えるだけで、写っていない歩行者や車両の飛び出しを追加しないでください。
車両の種類、色、駐停車状態を区別せず、色名は出力せずにすべて「車両」と表現してください。
出力文には「左側」「右側」「前方設置カメラ」「後方設置カメラ」などの画面領域名を書かず、direction の値で方向を示してください。
指定されたJSON形式だけを出力してください。"""

ALLOWED_DIRECTIONS = {"前方", "左後方", "右後方", "後方", "補足情報"}
IMAGE_SPECULATION_TERMS = ("可能性", "予想", "かもしれない", "恐れ", "おそれ")
VEHICLE_COLOR_PATTERN = re.compile(
    r"(?:"
    r"白(?:い|色(?:い|の)?)?|黒(?:い|色(?:い|の)?)?|赤(?:い|色(?:い|の)?)?|青(?:い|色(?:い|の)?)?|"
    r"黄(?:色(?:い|の)?|色)?|緑(?:色(?:い|の)?|色)?|茶(?:色(?:い|の)?|色)?|灰(?:色(?:い|の)?|色)?|"
    r"オレンジ(?:色の?)?|ピンク(?:色の?)?|紫(?:色の?)?|グレー(?:の)?|シルバー(?:の)?|銀(?:色(?:の)?)?"
    r")"
    r"(?:\s*の?\s*)"
    r"(?:車両|車|自動車|乗用車|トラック|バス|タクシー|バン|ワゴン|SUV)",
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DUAL_VIDEO_OUTPUT_DIR = SCRIPT_DIR / "scene_logs"
DEFAULT_DUAL_VIDEO_MAX_SIDE = 768
DEFAULT_API_CONNECT_TIMEOUT = 10.0
DEFAULT_API_READ_TIMEOUT = 300.0
DEFAULT_API_MAX_RETRIES = 3
DEFAULT_CHUNK_FRAMES = 6
DEFAULT_CHUNK_OVERLAP = 2
SCENE_ID_PROFILE = "video_scene_describer3:stopped-v2"
FALLBACK_STEPS = (
    {"name": "normal", "max_side_scale": 1.0, "quality_delta": 0, "max_tokens_scale": 1.0, "frame_step": 1},
    {"name": "reduced_image", "max_side_scale": 0.85, "quality_delta": -8, "max_tokens_scale": 1.0, "frame_step": 1},
    {"name": "reduced_frames", "max_side_scale": 0.75, "quality_delta": -12, "max_tokens_scale": 0.8, "frame_step": 2},
)


class RetryableApiError(RuntimeError):
    pass


class ApiRequestFailed(RuntimeError):
    pass


@dataclass(frozen=True)
class ChunkAnalysisResult:
    chunk_index: int
    chunk_id: str
    description: str
    error: str = ""
    elapsed_sec: float = 0.0
    skipped: bool = False
    frame_range: str = ""
    frame_count: int = 0
    video_time_range: str = ""
    payload_bytes: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str = ""
    retry_count: int = 0
    fallback: str = ""
    status: str = ""


@dataclass(frozen=True)
class ApiJsonResult:
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str = ""
    retry_count: int = 0
    payload_bytes: int = 0


@dataclass(frozen=True)
class FallbackConfig:
    index: int
    name: str
    max_side: int
    quality: int
    frame_step: int
    max_tokens: int


@dataclass
class DualVideoSceneEntry(SceneEntry):
    sampling_fps: float = 0.0
    temporal_frames: int = 0
    sampling_duration_sec: float = 0.0
    images_sent: int = 0


class SceneImageEncoder:
    def __init__(self, frames: list[Any], image_format: str) -> None:
        self.frames = frames
        self.image_format = image_format
        self.cache: dict[tuple[int, str, str, int, int], str] = {}
        self.lock = threading.Lock()

    def encode(
        self,
        frame_indices: list[int],
        *,
        fallback_name_value: str,
        max_side: int,
        quality: int,
        scene_id: str,
        video_label: str,
    ) -> list[str]:
        started = time.perf_counter()
        encoded: list[str] = []
        hits = 0
        misses = 0
        for frame_index in frame_indices:
            key = (frame_index, fallback_name_value, self.image_format, max_side, quality)
            with self.lock:
                cached = self.cache.get(key)
            if cached is not None:
                hits += 1
                encoded.append(cached)
                continue
            frame = self.frames[frame_index]
            resized = resize_for_inference(frame, max_side)
            if self.image_format == "jpeg":
                value = image_to_base64(resized, quality)
            else:
                value = encode_image_payload(resized, self.image_format, quality)
            with self.lock:
                self.cache[key] = value
            misses += 1
            encoded.append(value)
        elapsed = time.perf_counter() - started
        log(
            "[encode] "
            f"video={video_label or '-'} scene_id={scene_id or '-'} frames={len(frame_indices)} "
            f"cache_hits={hits} cache_misses={misses} elapsed_sec={elapsed:.2f} "
            f"fallback={fallback_name_value} image_format={self.image_format} max_side={max_side} quality={quality}"
        )
        return encoded


JSONL_WRITE_LOCK = threading.Lock()
LOG_LEVELS = {"quiet": 0, "normal": 1, "debug": 2}
current_log_level = LOG_LEVELS["normal"]


def set_log_level(level: str) -> None:
    global current_log_level
    current_log_level = LOG_LEVELS[level]


def should_log(level: str) -> bool:
    return current_log_level >= LOG_LEVELS[level]


def log(message: str, level: str = "debug", *, stream: Any = sys.stderr) -> None:
    if should_log(level):
        print(message, file=stream, flush=True)


def sampling_duration_sec(sample_fps: float, temporal_frames: int) -> float:
    return max(0, temporal_frames - 1) / max(0.001, sample_fps)


def write_dual_video_html(path: Path, output_dir: Path, entries: list[DualVideoSceneEntry]) -> None:
    cards = []
    for entry in reversed(entries):
        image_src = image_src_for_html(path, output_dir, entry.image_file)
        description = html.escape(entry.description).replace("\n", "<br>")
        sampling_meta = (
            f" / sampling_fps={entry.sampling_fps:.2f}"
            f" / temporal_frames={entry.temporal_frames}"
            f" / sampling_duration_sec={entry.sampling_duration_sec:.2f}"
            f" / images_sent={entry.images_sent}"
        )
        cards.append(
            f"""<article class="entry">
  <img src="{image_src}" alt="dual video snapshot">
  <div class="body">
    <div class="meta">{html.escape(entry.timestamp)} / {entry.elapsed_sec:.2f}s{sampling_meta}</div>
    <p>{description}</p>
  </div>
</article>"""
        )

    document = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stopped Vehicle Dual Video Scene Descriptions</title>
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
<h1>Stopped Vehicle Dual Video Scene Descriptions</h1>
{''.join(cards)}
</main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def empty_scene_timing() -> dict[str, Any]:
    return {
        "scene_status": "executed",
        "chunk_count": 0,
        "chunk_elapsed_sec_list": [],
        "chunk_metrics": [],
        "chunk_max_elapsed_sec": 0.0,
        "chunk_min_elapsed_sec": 0.0,
        "chunk_total_elapsed_sec": 0.0,
        "merge_elapsed_sec": 0.0,
        "merge_status": "skipped",
        "merge_reason": "scene_incomplete",
        "failed_chunk_count": 0,
        "resumed_chunk_count": 0,
    }


def chunk_metric_from_result(result: ChunkAnalysisResult) -> dict[str, Any]:
    return {
        "chunk_id": result.chunk_id,
        "frame_range": result.frame_range,
        "frame_count": result.frame_count,
        "video_time_range": result.video_time_range,
        "payload_bytes": result.payload_bytes,
        "elapsed_sec": float(result.elapsed_sec or 0.0),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "finish_reason": result.finish_reason,
        "retry_count": result.retry_count,
        "fallback": result.fallback,
        "status": result.status or ("resumed" if result.skipped else "failed" if result.error else "executed"),
    }


def update_chunk_timing(timing: dict[str, Any], chunk_results: list[ChunkAnalysisResult]) -> None:
    elapsed_list = [float(result.elapsed_sec or 0.0) for result in chunk_results]
    resumed_count = len([result for result in chunk_results if result.skipped])
    executed_count = len([result for result in chunk_results if not result.skipped])
    timing.update(
        {
            "chunk_count": len(chunk_results),
            "chunk_elapsed_sec_list": elapsed_list,
            "chunk_metrics": [chunk_metric_from_result(result) for result in chunk_results],
            "chunk_max_elapsed_sec": max(elapsed_list) if elapsed_list else 0.0,
            "chunk_min_elapsed_sec": min(elapsed_list) if elapsed_list else 0.0,
            "chunk_total_elapsed_sec": sum(elapsed_list),
            "failed_chunk_count": len([result for result in chunk_results if result.error]),
            "resumed_chunk_count": resumed_count,
        }
    )
    if chunk_results and resumed_count == len(chunk_results):
        timing["scene_status"] = "chunk_resumed"
    elif resumed_count > 0 and executed_count > 0:
        timing["scene_status"] = "partial_resume"
    elif chunk_results:
        timing["scene_status"] = "executed"


def format_timing_detail(video_timestamp: str, timing: dict[str, Any]) -> str:
    lines = [f"[timing] video={video_timestamp}", f"scene_status={timing.get('scene_status', 'executed')}"]
    chunk_metrics = timing.get("chunk_metrics") or []
    if chunk_metrics:
        for index, metric in enumerate(chunk_metrics, start=1):
            lines.extend(
                [
                    "",
                    f"chunk{index}",
                    f"frame_range={metric.get('frame_range', '')}",
                    f"video_time={metric.get('video_time_range', '')}",
                    f"elapsed={float(metric.get('elapsed_sec', 0.0) or 0.0):.1f}s",
                    f"status={metric.get('status', '')}",
                    f"payload_bytes={int(metric.get('payload_bytes', 0) or 0)}",
                    f"retry_count={int(metric.get('retry_count', 0) or 0)}",
                    f"fallback={metric.get('fallback', '')}",
                ]
            )
    else:
        for index, elapsed in enumerate(timing.get("chunk_elapsed_sec_list", []), start=1):
            lines.extend(["", f"chunk{index}", f"elapsed={float(elapsed or 0.0):.1f}s"])
    lines.extend(
        [
            "",
            "merge",
            f"status={timing.get('merge_status', 'skipped')}",
            f"reason={timing.get('merge_reason', '')}",
            f"elapsed={float(timing.get('merge_elapsed_sec', 0.0) or 0.0):.1f}s",
        ]
    )
    return "\n".join(lines)


def index_html_will_update(html_path: Path, index_html_path: Path, no_index_html: bool) -> bool:
    return not no_index_html and html_path.resolve() != index_html_path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="フロント/バックの撮影済み動画からフレームを抽出し、停車中の前後注意点をQwen VLMで日本語出力する"
    )
    parser.add_argument("--front-video", type=Path, required=True, help="分析するフロントカメラ動画ファイル")
    parser.add_argument("--back-video", type=Path, required=True, help="分析するバックカメラ動画ファイル")
    parser.add_argument(
        "--location-info",
        default="",
        help="任意の位置情報や道路環境情報。例: 通学路付近、学校付近、住宅街など",
    )
    parser.add_argument(
        "--weather-info",
        default="",
        help="任意の天気情報や予報。例: この後雨、路面凍結の可能性、強風など",
    )
    parser.add_argument(
        "--time-info",
        default="",
        help="任意の時刻情報。例: 朝の通学時間帯、夕方、夜間、17:30頃など",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="llama-server のURL")
    parser.add_argument("--start-sec", type=float, default=0.0, help="分析開始位置秒")
    parser.add_argument(
        "--end-sec",
        type=float,
        default=0.0,
        help="分析終了位置秒。0 で短い方の動画末尾まで",
    )
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=1.0,
        help="VLM分析を開始する間隔秒",
    )
    parser.add_argument("--sample-fps", type=float, default=5.0, help="動き確認用のフレーム取得FPS")
    parser.add_argument("--temporal-frames", type=int, default=5, help="1回のVLM分析に渡す時系列フレーム数")
    parser.add_argument("--max-samples", type=int, default=0, help="最大分析回数。0 で制限なし")
    parser.add_argument(
        "--vlm-prompt-mode",
        choices=("caution", "detailed"),
        default=DEFAULT_VLM_PROMPT_MODE,
        help="互換用。video_scene_describer3.py は停車時注意点JSON出力の専用プロンプトを使う",
    )
    parser.add_argument("--max-side", type=int, default=DEFAULT_DUAL_VIDEO_MAX_SIDE, help="VLM推論用結合画像の最大辺")
    parser.add_argument("--quality", type=int, default=DEFAULT_JPEG_QUALITY, help="JPEG品質")
    parser.add_argument(
        "--image-format",
        choices=("jpeg", "png"),
        default="jpeg",
        help="VLM APIへ送る画像形式。デフォルトは従来通りjpeg",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        help="VLM APIへ送るJPEG品質。未指定なら --quality を使う",
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=None,
        help="VLM APIへ送る画像の最大辺。未指定なら --max-side を使う",
    )
    parser.add_argument("--max-tokens", type=int, default=240, help="説明文の最大トークン数")
    parser.add_argument(
        "--retry-on-length",
        action="store_true",
        help="出力が最大トークン数に達した場合、上限を2倍にして再推論する",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="生成温度")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="互換用。指定した場合は read timeout として扱う秒数",
    )
    parser.add_argument("--connect-timeout", type=float, default=DEFAULT_API_CONNECT_TIMEOUT, help="VLM API 接続タイムアウト秒")
    parser.add_argument("--read-timeout", type=float, default=DEFAULT_API_READ_TIMEOUT, help="VLM API 読み取りタイムアウト秒")
    parser.add_argument("--api-max-retries", type=int, default=DEFAULT_API_MAX_RETRIES, help="VLM API 一時失敗時の最大試行回数")
    parser.add_argument("--max-workers", type=int, default=2, help="chunk単位API呼び出しの最大並列数")
    parser.add_argument("--chunk-frames", type=int, default=DEFAULT_CHUNK_FRAMES, help="1チャンクに含める最大フレーム数。0で分割しない")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="チャンク間で重ねるフレーム数")
    parser.add_argument(
        "--chunk-merge-mode",
        choices=("local", "vlm"),
        default="local",
        help="チャンク結果の統合方法。local=追加推論なしで重複排除、vlm=従来通りVLMで統合",
    )
    parser.add_argument("--no-resume", action="store_true", help="中間結果による完了済みシーンのスキップを無効化")
    parser.add_argument(
        "--show-timing-detail",
        action="store_true",
        help="chunkごとの処理時間とmerge時間をnormalログに表示する",
    )
    parser.add_argument(
        "--log-level",
        choices=("quiet", "normal", "debug"),
        default="normal",
        help="画面ログの詳細度。quiet=エラーのみ、normal=シーン単位、debug=詳細",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DUAL_VIDEO_OUTPUT_DIR,
        help="ログ出力先",
    )
    parser.add_argument("--html-file", default="", help="HTML出力ファイル名。未指定なら起動時刻から生成")
    parser.add_argument(
        "--html-name-format",
        default=DEFAULT_HTML_NAME_FORMAT,
        help="--html-file未指定時のstrftime形式",
    )
    parser.add_argument("--no-index-html", action="store_true", help="互換用の index.html 更新を行わない")
    parser.add_argument("--history", type=int, default=50, help="今回の起動中にHTMLへ残す件数")
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> None:
    args.frame_interval = max(0.001, args.frame_interval)
    args.sample_fps = max(0.001, args.sample_fps)
    args.temporal_frames = max(1, args.temporal_frames)
    args.max_workers = max(1, args.max_workers)
    args.chunk_overlap = max(0, args.chunk_overlap)


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


def video_duration_sec(cap: cv2.VideoCapture, label: str) -> tuple[float, int, float]:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError(f"{label}動画のFPSを取得できませんでした")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if frame_count > 0 else 0.0
    return fps, frame_count, duration_sec


def format_video_timestamp(position_sec: float) -> str:
    return str(timedelta(seconds=int(position_sec)))


def resize_to_height(frame: Any, target_height: int) -> Any:
    height, width = frame.shape[:2]
    if height == target_height:
        return frame
    scale = target_height / height
    return cv2.resize(frame, (max(1, int(width * scale)), target_height), interpolation=cv2.INTER_AREA)


def label_frame(frame: Any, label: str) -> Any:
    output = frame.copy()
    text_width, text_height = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    label_width = min(output.shape[1], text_width + 20)
    label_height = text_height + 18
    cv2.rectangle(output, (0, 0), (label_width, label_height), (0, 0, 0), thickness=-1)
    cv2.putText(
        output,
        label,
        (10, label_height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def crop_front_forward_view(front_frame: Any) -> Any:
    height = front_frame.shape[0]
    return front_frame[: max(1, height // 2), :]


def combine_front_back_frames(front_frame: Any, back_frame: Any) -> Any:
    front_forward = crop_front_forward_view(front_frame)
    target_height = min(front_forward.shape[0], back_frame.shape[0])
    front_resized = resize_to_height(front_forward, target_height)
    back_resized = resize_to_height(back_frame, target_height)
    front_labeled = label_frame(front_resized, "FRONT VIEW ONLY")
    back_labeled = label_frame(back_resized, "REAR VIEW ONLY")
    return cv2.hconcat([front_labeled, back_labeled])


def front_region_percent(front_frame: Any, back_frame: Any) -> int:
    front_forward = crop_front_forward_view(front_frame)
    target_height = min(front_forward.shape[0], back_frame.shape[0])
    front_width = round(front_forward.shape[1] * target_height / front_forward.shape[0])
    back_width = round(back_frame.shape[1] * target_height / back_frame.shape[0])
    return round(100 * front_width / (front_width + back_width))


def extract_json_object(content: str) -> str:
    text = content.strip()
    if text == "注意点なし":
        return '{"risks":[]}'
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    if start < 0:
        raise ValueError("JSONオブジェクトが見つかりません")

    in_string = False
    escape = False
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("JSONオブジェクトが閉じられていません")


def format_dual_video_description(content: str) -> str:
    try:
        data = json.loads(extract_json_object(content))
    except json.JSONDecodeError as exc:
        raise ValueError("JSONとして解釈できません") from exc
    risks = data.get("risks") if isinstance(data, dict) else None
    if not isinstance(risks, list):
        raise ValueError("risks が配列ではありません")
    if len(risks) > 3:
        raise ValueError("危険箇所が3項目を超えています")
    if not risks:
        return "注意点なし"

    messages = []
    for index, risk in enumerate(risks, start=1):
        if not isinstance(risk, dict):
            raise ValueError(f"risks[{index - 1}] がオブジェクトではありません")
        direction = risk.get("direction")
        message = risk.get("message")
        if direction not in ALLOWED_DIRECTIONS:
            raise ValueError(f"risks[{index - 1}].direction が不正です")
        if not isinstance(message, str) or not message.strip():
            raise ValueError(f"risks[{index - 1}].message が空です")
        if direction != "補足情報" and any(term in message for term in IMAGE_SPECULATION_TERMS):
            continue
        messages.append(f"【{direction}】{remove_vehicle_color_mentions(message.strip())}")
    return "\n".join(messages) if messages else "注意点なし"


def remove_vehicle_color_mentions(message: str) -> str:
    return VEHICLE_COLOR_PATTERN.sub("車両", message)


def build_supplemental_user_prompt(location_info: str, weather_info: str, time_info: str) -> str:
    lines = []
    if location_info.strip():
        lines.append(f"- 位置情報: {location_info.strip()}")
    if weather_info.strip():
        lines.append(f"- 天気情報: {weather_info.strip()}")
    if time_info.strip():
        lines.append(f"- 時刻情報: {time_info.strip()}")
    if not lines:
        return ""
    return (
        "補足情報が入力されています。画像の内容を主根拠にしつつ、運転上必要な場合だけ次の情報を注意点に反映してください。\n"
        + "\n".join(lines)
    )


def api_timeout(connect_timeout: float, read_timeout: float) -> tuple[float, float]:
    return (max(0.001, connect_timeout), max(0.001, read_timeout))


def estimate_payload_size_bytes(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    with JSONL_WRITE_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_completed_scene_results(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("type") == "scene_complete" and data.get("scene_id") and data.get("description"):
                completed[str(data["scene_id"])] = data
    return completed


def load_completed_chunk_results(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("type") == "chunk_complete" and data.get("scene_id") and data.get("chunk_id"):
                completed[(str(data["scene_id"]), str(data["chunk_id"]))] = data
    return completed


def video_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def analysis_profile(args: argparse.Namespace, max_side: int, quality: int) -> str:
    """Return a stable fingerprint of every setting that can affect a scene result."""
    profile = {
        "profile": SCENE_ID_PROFILE,
        "system_prompt": DUAL_VIDEO_SYSTEM_PROMPT_CONTEXT,
        "user_prompt": DUAL_VIDEO_USER_PROMPT_CONTEXT,
        "sample_fps": args.sample_fps,
        "temporal_frames": args.temporal_frames,
        "frame_interval": args.frame_interval,
        "end_sec": args.end_sec,
        "max_side": max_side,
        "quality": quality,
        "image_format": args.image_format,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "retry_on_length": args.retry_on_length,
        "chunk_frames": args.chunk_frames,
        "chunk_overlap": args.chunk_overlap,
        "chunk_merge_mode": args.chunk_merge_mode,
        "location_info": args.location_info.strip(),
        "weather_info": args.weather_info.strip(),
        "time_info": args.time_info.strip(),
    }
    encoded = json.dumps(profile, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def scene_id_for(
    front_video: Path,
    back_video: Path,
    sample_index: int,
    position_sec: float,
    profile_hash: str,
) -> str:
    key = {
        "profile_hash": profile_hash,
        "front_video": video_identity(front_video),
        "back_video": video_identity(back_video),
        "sample_index": sample_index,
        "position_sec": round(position_sec, 6),
    }
    serialized = json.dumps(key, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]


def is_retryable_status(status_code: int) -> bool:
    return 500 <= status_code <= 599


def sleep_for_retry(attempt_index: int) -> None:
    time.sleep(min(2.0 * (2**attempt_index), 12.0))


def chunk_ranges(total: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    if chunk_size <= 0 or total <= chunk_size:
        return [(0, total)]
    overlap = max(0, min(overlap, chunk_size - 1))
    ranges = []
    start = 0
    while start < total:
        end = min(total, start + chunk_size)
        ranges.append((start, end))
        if end >= total:
            break
        start = max(start + 1, end - overlap)
    return ranges


def fallback_config(max_side: int, quality: int, max_tokens: int, fallback_index: int) -> FallbackConfig:
    step = FALLBACK_STEPS[min(fallback_index, len(FALLBACK_STEPS) - 1)]
    return FallbackConfig(
        index=fallback_index,
        name=str(step["name"]),
        max_side=max(128, int(max_side * float(step["max_side_scale"]))),
        quality=max(40, min(95, quality + int(step["quality_delta"]))),
        frame_step=max(1, int(step["frame_step"])),
        max_tokens=max(80, int(max_tokens * float(step["max_tokens_scale"]))),
    )


def encode_image_payload(image: Image.Image, image_format: str, jpeg_quality: int) -> str:
    buf = io.BytesIO()
    normalized_format = image_format.lower()
    if normalized_format == "png":
        image.save(buf, format="PNG")
    else:
        image.save(buf, format="JPEG", quality=jpeg_quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def image_mime_type(image_format: str) -> str:
    return "image/png" if image_format.lower() == "png" else "image/jpeg"


def build_image_content(image_b64: list[str], image_format: str) -> list[dict[str, Any]]:
    mime_type = image_mime_type(image_format)
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"},
        }
        for encoded_image in image_b64
    ]


def select_frame_indices(start: int, end: int, frame_step: int) -> list[int]:
    indices = list(range(start, end, max(1, frame_step)))
    if start < end and (not indices or indices[-1] != end - 1):
        indices.append(end - 1)
    return indices


def frame_range_label(start: int, end: int) -> str:
    return f"{start + 1}-{end}" if end > start else ""


def video_time_range_label(scene_position_sec: float, sample_fps: float, start: int, end: int) -> str:
    if end <= start:
        return ""
    start_sec = scene_position_sec + start / max(0.001, sample_fps)
    end_sec = scene_position_sec + (end - 1) / max(0.001, sample_fps)
    return f"{start_sec:.1f}-{end_sec:.1f}s"


def format_description_with_raw_context(raw_content: str) -> str:
    try:
        return format_dual_video_description(raw_content)
    except ValueError as exc:
        snippet = raw_content.replace("\n", "\\n")[:500]
        raise ValueError(f"{exc}; raw={snippet!r}") from exc


def parse_formatted_risks(description: str) -> list[tuple[str, str]]:
    risks: list[tuple[str, str]] = []
    if description.strip() == "注意点なし":
        return risks
    for line in description.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^【([^】]+)】(.+)$", line)
        if not match:
            continue
        direction = match.group(1).strip()
        message = match.group(2).strip()
        if direction in ALLOWED_DIRECTIONS and message:
            risks.append((direction, remove_vehicle_color_mentions(message)))
    return risks


def risk_object_type(message: str) -> str:
    for object_type, terms in (
        ("pedestrian", ("歩行者", "人 ", "人が", "人を")),
        ("bicycle", ("自転車",)),
        ("vehicle", ("車両", "自動車", "乗用車", "トラック", "バス", "タクシー")),
    ):
        if any(term in message for term in terms):
            return object_type
    return "other"


def local_risk_priority(direction: str, message: str) -> tuple[int, int, int]:
    """Rank risks using the same explicit priority policy as the VLM prompt."""
    if direction == "補足情報":
        return (6, 0, 0)
    object_rank = {"pedestrian": 0, "bicycle": 1, "vehicle": 2, "other": 3}[risk_object_type(message)]
    is_near = any(term in message for term in ("直近", "すぐ近く", "付近", "近く"))
    is_moving = any(term in message for term in ("通過", "接近", "横切", "進入"))
    blocks_departure = any(term in message for term in ("発進経路", "進路を塞", "発進を妨"))
    affects_door_or_stop = any(term in message for term in ("ドア開閉", "乗降", "停車継続"))
    if is_near and is_moving and object_rank < 3:
        return (object_rank, 0, 0)
    if blocks_departure:
        return (3, object_rank, 0)
    if affects_door_or_stop:
        return (4, object_rank, 0)
    return (5, object_rank, 0 if is_moving else 1)


def normalized_risk_text(message: str) -> str:
    return re.sub(r"[\s、。・,.;；：:（）()]", "", message)


def risks_are_duplicates(left: tuple[str, str], right: tuple[str, str]) -> bool:
    if left[0] != right[0] or risk_object_type(left[1]) != risk_object_type(right[1]):
        return False
    left_text = normalized_risk_text(left[1])
    right_text = normalized_risk_text(right[1])
    if left_text == right_text:
        return True
    if min(len(left_text), len(right_text)) < 12:
        return False
    return difflib.SequenceMatcher(None, left_text, right_text).ratio() >= 0.78


def merge_chunk_descriptions_locally(chunk_results: list[ChunkAnalysisResult]) -> str:
    candidates: list[tuple[tuple[int, int, int], int, int, str, str]] = []
    for result in chunk_results:
        if result.error:
            continue
        for item_index, (direction, message) in enumerate(parse_formatted_risks(result.description)):
            candidates.append(
                (local_risk_priority(direction, message), result.chunk_index, item_index, direction, message)
            )

    # Sort all chunks together before limiting the output. Chunk/item order is only
    # a tie-breaker because each chunk is already requested in importance order.
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    merged: list[tuple[str, str]] = []
    for _, _, _, direction, message in candidates:
        candidate = (direction, message)
        if any(risks_are_duplicates(candidate, existing) for existing in merged):
            continue
        merged.append(candidate)
        if len(merged) >= 3:
            break
    if not merged:
        return "注意点なし"
    return "\n".join(f"【{direction}】{message}" for direction, message in merged)


def analyze_dual_video_chunk(
    *,
    base_url: str,
    chunk_image_b64: list[str],
    image_format: str,
    system_prompt: str,
    user_prompt: str,
    chunk_index: int,
    chunk_count: int,
    start: int,
    end: int,
    max_tokens: int,
    temperature: float,
    timeout: tuple[float, float],
    retry_on_length: bool,
    api_max_retries: int,
    scene_id: str,
    video_label: str,
    request_context: str,
    fallback_name_value: str,
    intermediate_path: Path | None,
    running_state: dict[str, Any],
    scene_position_sec: float,
    sample_fps: float,
) -> ChunkAnalysisResult:
    chunk_id = f"{chunk_index:02d}-{start + 1:02d}_{end:02d}"
    image_content = build_image_content(chunk_image_b64, image_format)
    context_prompt = (
        f"これは scene_id={scene_id or '-'} のチャンク {chunk_index}/{chunk_count} です。"
        f"対象フレームはシーン内の {start + 1}〜{end} 枚目です。"
        "前後チャンクと重複するフレームがあるため、同じ危険を重複して増やさず、"
        "このチャンクで画像から確認できる注意点だけをJSONで出力してください。"
    )
    chunk_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                *image_content,
                {"type": "text", "text": "\n\n".join(part for part in (user_prompt, context_prompt) if part)},
            ],
        },
    ]
    payload_size = estimate_payload_size_bytes(
        {
            "messages": chunk_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
    )
    lock: threading.Lock = running_state["lock"]
    with lock:
        running_state["count"] += 1
        running_chunks = running_state["count"]
    started = time.perf_counter()
    log(
        "[chunk:start] "
        f"video={video_label or '-'} scene_id={scene_id or '-'} chunk_id={chunk_id} "
        f"chunk={chunk_index}/{chunk_count} running_chunks={running_chunks} payload_bytes={payload_size} "
        f"fallback={fallback_name_value}"
    )
    try:
        status = "failed"
        with requests.Session() as chunk_session:
            api_result = request_dual_video_json(
                base_url,
                chunk_messages,
                chunk_session,
                max_tokens,
                temperature,
                timeout,
                retry_on_length,
                max_retries=api_max_retries,
                scene_id=scene_id,
                chunk_id=chunk_id,
                video_label=video_label,
                request_context=f"{request_context}:chunk",
                fallback_name_value=fallback_name_value,
            )
        chunk_description = format_description_with_raw_context(api_result.content)
        elapsed = time.perf_counter() - started
        status = "success"
        metric_fields = {
            "frame_range": frame_range_label(start, end),
            "frame_count": end - start,
            "video_time_range": video_time_range_label(scene_position_sec, sample_fps, start, end),
            "payload_bytes": payload_size,
            "prompt_tokens": api_result.prompt_tokens,
            "completion_tokens": api_result.completion_tokens,
            "total_tokens": api_result.total_tokens,
            "finish_reason": api_result.finish_reason,
            "retry_count": api_result.retry_count,
            "fallback": fallback_name_value,
            "status": "executed",
        }
        if intermediate_path is not None:
            append_jsonl(
                intermediate_path,
                {
                    "type": "chunk_complete",
                    "scene_id": scene_id,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "video": video_label,
                    "description": chunk_description,
                    "elapsed_sec": elapsed,
                    "fallback": fallback_name_value,
                    "payload_bytes": payload_size,
                    **metric_fields,
                    "finished_at": utc_now_iso(),
                },
            )
        return ChunkAnalysisResult(chunk_index, chunk_id, chunk_description, elapsed_sec=elapsed, **metric_fields)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        error = f"{type(exc).__name__}: {exc}"
        metric_fields = {
            "frame_range": frame_range_label(start, end),
            "frame_count": end - start,
            "video_time_range": video_time_range_label(scene_position_sec, sample_fps, start, end),
            "payload_bytes": payload_size,
            "fallback": fallback_name_value,
            "status": "failed",
        }
        if intermediate_path is not None:
            append_jsonl(
                intermediate_path,
                {
                    "type": "chunk_failed",
                    "scene_id": scene_id,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "video": video_label,
                    "error": error,
                    "elapsed_sec": elapsed,
                    "fallback": fallback_name_value,
                    "payload_bytes": payload_size,
                    **metric_fields,
                    "finished_at": utc_now_iso(),
                },
            )
        return ChunkAnalysisResult(chunk_index, chunk_id, "解析失敗", error=error, elapsed_sec=elapsed, **metric_fields)
    finally:
        with lock:
            running_state["count"] -= 1
            running_chunks = running_state["count"]
        elapsed = time.perf_counter() - started
        log(
            "[chunk:end] "
            f"video={video_label or '-'} scene_id={scene_id or '-'} chunk_id={chunk_id} "
            f"chunk={chunk_index}/{chunk_count} running_chunks={running_chunks} elapsed_sec={elapsed:.2f} "
            f"status={status} fallback={fallback_name_value}"
        )


def describe_dual_video_scene(
    base_url: str,
    image_b64: list[str] | None,
    session: requests.Session,
    max_tokens: int,
    temperature: float,
    timeout: tuple[float, float],
    prompt_mode: str,
    supplemental_user_prompt: str,
    retry_on_length: bool,
    front_percent: int,
    *,
    scene_id: str = "",
    video_label: str = "",
    request_context: str = "scene",
    api_max_retries: int = DEFAULT_API_MAX_RETRIES,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    chunk_merge_mode: str = "local",
    intermediate_path: Path | None = None,
    fallback_name_value: str = "normal",
    image_format: str = "jpeg",
    max_workers: int = 2,
    completed_chunks: dict[tuple[str, str], dict[str, Any]] | None = None,
    image_encoder: SceneImageEncoder | None = None,
    max_side: int = DEFAULT_DUAL_VIDEO_MAX_SIDE,
    quality: int = DEFAULT_JPEG_QUALITY,
    timing: dict[str, Any] | None = None,
    scene_position_sec: float = 0.0,
    sample_fps: float = 1.0,
) -> str:
    _ = prompt_mode  # Kept for CLI compatibility; this path must use JSON-only prompts.
    system_prompt = (
        f"{DUAL_VIDEO_SYSTEM_PROMPT_CONTEXT}\n"
        f"- 今回の結合画像では、左側の FRONT VIEW ONLY 領域は画像幅の約{front_percent}%です。"
    )
    user_prompt = "\n\n".join(
        part
        for part in (DUAL_VIDEO_USER_PROMPT_CONTEXT, supplemental_user_prompt)
        if part
    )
    total_frames = len(image_encoder.frames) if image_encoder is not None else len(image_b64 or [])
    ranges = chunk_ranges(total_frames, chunk_frames, chunk_overlap)
    if len(ranges) > 1:
        worker_count = min(max(1, max_workers), len(ranges))
        log(
            f"[progress] analyzing chunks {len(ranges)} chunks max_workers={worker_count}",
            level="normal",
        )
        log(
            "[chunks:parallel_start] "
            f"video={video_label or '-'} scene_id={scene_id or '-'} chunks={len(ranges)} max_workers={worker_count} "
            f"fallback=chunk-level"
        )
        chunk_results: dict[int, ChunkAnalysisResult] = {}
        running_state: dict[str, Any] = {"count": 0, "lock": threading.Lock()}
        for chunk_index, (start, end) in enumerate(ranges, start=1):
            chunk_id = f"{chunk_index:02d}-{start + 1:02d}_{end:02d}"
            completed_chunk = (completed_chunks or {}).get((scene_id, chunk_id))
            if completed_chunk and completed_chunk.get("description"):
                description = str(completed_chunk["description"])
                chunk_results[chunk_index] = ChunkAnalysisResult(
                    chunk_index,
                    chunk_id,
                    description,
                    elapsed_sec=float(completed_chunk.get("elapsed_sec", 0.0) or 0.0),
                    skipped=True,
                    frame_range=str(completed_chunk.get("frame_range") or frame_range_label(start, end)),
                    frame_count=int(completed_chunk.get("frame_count", end - start) or 0),
                    video_time_range=str(
                        completed_chunk.get("video_time_range")
                        or video_time_range_label(scene_position_sec, sample_fps, start, end)
                    ),
                    payload_bytes=int(completed_chunk.get("payload_bytes", 0) or 0),
                    prompt_tokens=completed_chunk.get("prompt_tokens"),
                    completion_tokens=completed_chunk.get("completion_tokens"),
                    total_tokens=completed_chunk.get("total_tokens"),
                    finish_reason=str(completed_chunk.get("finish_reason", "") or ""),
                    retry_count=int(completed_chunk.get("retry_count", 0) or 0),
                    fallback=str(completed_chunk.get("fallback", "") or ""),
                    status="resumed",
                )
                log(
                    "[chunk:resume] "
                    f"video={video_label or '-'} scene_id={scene_id or '-'} chunk_id={chunk_id} "
                    f"chunk={chunk_index}/{len(ranges)} reason=chunk_complete"
                )

        submitted_total = 0
        for fallback_index in range(len(FALLBACK_STEPS)):
            pending = [
                (chunk_index, start, end)
                for chunk_index, (start, end) in enumerate(ranges, start=1)
                if chunk_index not in chunk_results or chunk_results[chunk_index].error
            ]
            if not pending:
                break
            fallback = fallback_config(max_side, quality, max_tokens, fallback_index)
            log(
                "[fallback] "
                f"video={video_label or '-'} scene_id={scene_id or '-'} scope=chunk "
                f"fallback={fallback.name} pending_chunks={len(pending)}/{len(ranges)} "
                f"max_image_side={fallback.max_side} jpeg_quality={fallback.quality} frame_step={fallback.frame_step}",
                level="normal" if fallback_index > 0 else "debug",
            )
            futures = {}
            with ThreadPoolExecutor(max_workers=min(worker_count, len(pending))) as executor:
                for chunk_index, start, end in pending:
                    chunk_id = f"{chunk_index:02d}-{start + 1:02d}_{end:02d}"
                    frame_indices = select_frame_indices(start, end, fallback.frame_step)
                    if image_encoder is not None:
                        chunk_image_b64 = image_encoder.encode(
                            frame_indices,
                            fallback_name_value=fallback.name,
                            max_side=fallback.max_side,
                            quality=fallback.quality,
                            scene_id=scene_id,
                            video_label=video_label,
                        )
                    else:
                        chunk_image_b64 = (image_b64 or [])[start:end]
                    log(
                        "[chunk:submitted] "
                        f"video={video_label or '-'} scene_id={scene_id or '-'} chunk_id={chunk_id} "
                        f"chunk={chunk_index}/{len(ranges)} fallback={fallback.name}"
                    )
                    future = executor.submit(
                        analyze_dual_video_chunk,
                        base_url=base_url,
                        chunk_image_b64=chunk_image_b64,
                        image_format=image_format,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        chunk_index=chunk_index,
                        chunk_count=len(ranges),
                        start=start,
                        end=end,
                        max_tokens=fallback.max_tokens,
                        temperature=temperature,
                        timeout=timeout,
                        retry_on_length=retry_on_length,
                        api_max_retries=api_max_retries,
                        scene_id=scene_id,
                        video_label=video_label,
                        request_context=request_context,
                        fallback_name_value=fallback.name,
                        intermediate_path=intermediate_path,
                        running_state=running_state,
                        scene_position_sec=scene_position_sec,
                        sample_fps=sample_fps,
                    )
                    futures[future] = (chunk_index, chunk_id)
                    submitted_total += 1

                for future in as_completed(futures):
                    chunk_index, chunk_id = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        result = ChunkAnalysisResult(chunk_index, chunk_id, "解析失敗", error=error)
                    chunk_results[chunk_index] = result
                    if completed_chunks is not None and not result.error:
                        completed_chunks[(scene_id, result.chunk_id)] = {
                            "scene_id": scene_id,
                            "chunk_id": result.chunk_id,
                            "description": result.description,
                            "elapsed_sec": result.elapsed_sec,
                            **chunk_metric_from_result(result),
                        }

        ordered_chunk_results = [chunk_results[index] for index in sorted(chunk_results)]
        failed_chunks = [result for result in ordered_chunk_results if result.error]
        if timing is not None:
            update_chunk_timing(timing, ordered_chunk_results)
        if failed_chunks:
            log(
                "[chunks:partial_failure] "
                f"video={video_label or '-'} scene_id={scene_id or '-'} failed_chunks={len(failed_chunks)}/{len(ranges)} "
                f"fallback={fallback_name_value}"
            )
            if len(failed_chunks) == len(ranges):
                raise ApiRequestFailed(
                    f"すべてのチャンク解析に失敗しました: {len(failed_chunks)}/{len(ranges)}"
                )
        chunk_descriptions = []
        for result in ordered_chunk_results:
            suffix = f"（解析失敗: {result.error}）" if result.error else ""
            chunk_descriptions.append(f"チャンク{result.chunk_index}: {result.description}{suffix}")
        log(
            "[chunks:parallel_end] "
            f"video={video_label or '-'} scene_id={scene_id or '-'} chunks={len(ranges)} "
            f"submitted={submitted_total} resumed={len([r for r in ordered_chunk_results if r.skipped])} "
            f"failed={len(failed_chunks)} fallback=chunk-level"
        )
        if chunk_merge_mode == "local":
            merge_started = time.perf_counter()
            merged_description = merge_chunk_descriptions_locally(ordered_chunk_results)
            elapsed = time.perf_counter() - merge_started
            if timing is not None:
                timing["merge_elapsed_sec"] = elapsed
                timing["merge_status"] = "executed"
                timing["merge_reason"] = "local_merge"
            if intermediate_path is not None:
                append_jsonl(
                    intermediate_path,
                    {
                        "type": "merge_complete",
                        "scene_id": scene_id,
                        "video": video_label,
                        "chunk_count": len(ranges),
                        "failed_chunk_count": len(failed_chunks),
                        "fallback": "local",
                        "elapsed_sec": elapsed,
                        "merge_status": "executed",
                        "merge_reason": "local_merge",
                        "payload_bytes": 0,
                        "finished_at": utc_now_iso(),
                    },
                )
            log(
                "[merge:end] "
                f"video={video_label or '-'} scene_id={scene_id or '-'} elapsed_sec={elapsed:.2f} "
                "status=success fallback=local"
            )
            return merged_description

        merge_prompt = (
            "次のチャンク別注意点を統合し、動画シーン全体として重要度の高い停車時の注意点を最大3項目にまとめてください。\n"
            "重複は統合し、画像に明確に写っていない危険を追加しないでください。\n"
            "解析失敗のチャンクは、失敗した事実だけを踏まえ、未確認の危険を推測で追加しないでください。\n"
            "direction と message を持つ指定JSON形式だけで出力してください。\n\n"
            + "\n".join(chunk_descriptions)
        )
        merge_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": merge_prompt}]},
        ]
        merge_payload_size = estimate_payload_size_bytes(
            {"messages": merge_messages, "max_tokens": max_tokens, "temperature": temperature, "stream": False}
        )
        merge_started = time.perf_counter()
        log("[progress] merging chunk results", level="normal")
        if intermediate_path is not None:
            append_jsonl(
                intermediate_path,
                {
                    "type": "merge_start",
                    "scene_id": scene_id,
                    "video": video_label,
                    "chunk_count": len(ranges),
                    "failed_chunk_count": len(failed_chunks),
                    "fallback": "chunk-level",
                    "payload_bytes": merge_payload_size,
                    "started_at": utc_now_iso(),
                },
            )
        log(
            "[merge:start] "
            f"video={video_label or '-'} scene_id={scene_id or '-'} chunks={len(ranges)} "
            f"failed_chunks={len(failed_chunks)} payload_bytes={merge_payload_size} fallback=chunk-level"
        )
        try:
            api_result = request_dual_video_json(
                base_url,
                merge_messages,
                session,
                max_tokens,
                temperature,
                timeout,
                retry_on_length,
                max_retries=api_max_retries,
                scene_id=scene_id,
                chunk_id="merge",
                video_label=video_label,
                request_context=f"{request_context}:merge",
                fallback_name_value="chunk-level",
            )
            merged_description = format_description_with_raw_context(api_result.content)
            elapsed = time.perf_counter() - merge_started
            if timing is not None:
                timing["merge_elapsed_sec"] = elapsed
                timing["merge_status"] = "executed"
                timing["merge_reason"] = "api_called" if api_result.retry_count == 0 else "api_retry"
            if intermediate_path is not None:
                append_jsonl(
                    intermediate_path,
                    {
                        "type": "merge_complete",
                        "scene_id": scene_id,
                        "video": video_label,
                        "chunk_count": len(ranges),
                        "failed_chunk_count": len(failed_chunks),
                        "fallback": "chunk-level",
                        "elapsed_sec": elapsed,
                        "merge_status": "executed",
                        "merge_reason": "api_called" if api_result.retry_count == 0 else "api_retry",
                        "payload_bytes": merge_payload_size,
                        "prompt_tokens": api_result.prompt_tokens,
                        "completion_tokens": api_result.completion_tokens,
                        "total_tokens": api_result.total_tokens,
                        "finish_reason": api_result.finish_reason,
                        "retry_count": api_result.retry_count,
                        "finished_at": utc_now_iso(),
                    },
                )
            log(
                "[merge:end] "
                f"video={video_label or '-'} scene_id={scene_id or '-'} elapsed_sec={elapsed:.2f} "
                f"status=success fallback=chunk-level"
            )
            return merged_description
        except Exception as exc:
            elapsed = time.perf_counter() - merge_started
            if timing is not None:
                timing["merge_elapsed_sec"] = elapsed
                timing["merge_status"] = "failed"
                timing["merge_reason"] = "api_failed"
            error = f"{type(exc).__name__}: {exc}"
            if intermediate_path is not None:
                append_jsonl(
                    intermediate_path,
                    {
                        "type": "merge_failed",
                        "scene_id": scene_id,
                        "video": video_label,
                        "chunk_count": len(ranges),
                        "failed_chunk_count": len(failed_chunks),
                        "fallback": "chunk-level",
                        "elapsed_sec": elapsed,
                        "merge_status": "failed",
                        "merge_reason": "api_failed",
                        "payload_bytes": merge_payload_size,
                        "error": error,
                        "finished_at": utc_now_iso(),
                    },
                )
            log(
                "[merge:end] "
                f"video={video_label or '-'} scene_id={scene_id or '-'} elapsed_sec={elapsed:.2f} "
                f"status=failed fallback=chunk-level error={error}"
            )
            raise

    image_content = build_image_content(image_b64 or [], image_format)
    format_retry_prompt = ""
    last_error = ""
    last_raw_description = ""
    for _ in range(2):
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    *image_content,
                    {"type": "text", "text": "\n\n".join(part for part in (user_prompt, format_retry_prompt) if part)},
                ],
            },
        ]
        api_result = request_dual_video_json(
            base_url,
            messages,
            session,
            max_tokens,
            temperature,
            timeout,
            retry_on_length,
            max_retries=api_max_retries,
            scene_id=scene_id,
            chunk_id="full",
            video_label=video_label,
            request_context=request_context,
            fallback_name_value=fallback_name_value,
        )
        raw_description = api_result.content
        last_raw_description = raw_description
        try:
            return format_dual_video_description(raw_description)
        except ValueError as exc:
            last_error = str(exc)
            format_retry_prompt = (
                f"直前の出力形式または内容が不正でした（{exc}）。指定されたJSON形式だけで、"
                "画像に明確に写る危険だけを出力し、画像根拠のない可能性や予想を書かずに出力し直してください。"
            )
    snippet = last_raw_description.replace("\n", "\\n")[:500]
    raise RuntimeError(f"VLM出力を指定JSON形式として解釈できませんでした: {last_error}; raw={snippet!r}")


def request_dual_video_json(
    base_url: str,
    messages: list[dict[str, Any]],
    session: requests.Session,
    max_tokens: int,
    temperature: float,
    timeout: tuple[float, float],
    retry_on_length: bool,
    *,
    max_retries: int = DEFAULT_API_MAX_RETRIES,
    scene_id: str = "",
    chunk_id: str = "",
    video_label: str = "",
    request_context: str = "",
    fallback_name_value: str = "normal",
) -> ApiJsonResult:
    length_attempts = 2 if retry_on_length else 1
    content = ""
    total_retry_count = 0
    last_payload_size = 0
    last_result = ApiJsonResult(content="")
    for length_attempt in range(length_attempts):
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens if length_attempt == 0 else max_tokens * 2,
            "temperature": temperature,
            "stream": False,
        }
        payload_size = estimate_payload_size_bytes(payload)
        last_payload_size = payload_size
        last_exc: Exception | None = None
        request_started_at = utc_now_iso()
        for retry_index in range(max(1, max_retries)):
            start = time.perf_counter()
            log(
                "[api:start] "
                f"started_at={request_started_at} video={video_label or '-'} "
                f"scene_id={scene_id or '-'} chunk_id={chunk_id or '-'} request={request_context or '-'} "
                f"payload_bytes={payload_size} timeout={timeout} retry={retry_index}/{max_retries - 1} "
                f"fallback={fallback_name_value}"
            )
            try:
                resp = session.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout)
                if is_retryable_status(resp.status_code):
                    raise RetryableApiError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                resp.raise_for_status()
                elapsed = time.perf_counter() - start
                log(
                    "[api:end] "
                    f"ended_at={utc_now_iso()} video={video_label or '-'} scene_id={scene_id or '-'} "
                    f"chunk_id={chunk_id or '-'} request={request_context or '-'} elapsed_sec={elapsed:.2f} "
                    f"payload_bytes={payload_size} timeout={timeout} retry={retry_index} fallback={fallback_name_value}"
                )
                break
            except (requests.ReadTimeout, requests.ConnectionError, RetryableApiError) as exc:
                last_exc = exc
                total_retry_count += 1
                elapsed = time.perf_counter() - start
                log(
                    "[api:retryable_error] "
                    f"ended_at={utc_now_iso()} video={video_label or '-'} scene_id={scene_id or '-'} "
                    f"chunk_id={chunk_id or '-'} request={request_context or '-'} elapsed_sec={elapsed:.2f} "
                    f"payload_bytes={payload_size} timeout={timeout} retry={retry_index}/{max_retries - 1} "
                    f"fallback={fallback_name_value} error={type(exc).__name__}: {exc}"
                )
                if retry_index + 1 >= max(1, max_retries):
                    raise ApiRequestFailed(
                        f"VLM API request failed after {max_retries} attempts "
                        f"(video={video_label or '-'}, scene_id={scene_id or '-'}, "
                        f"chunk_id={chunk_id or '-'}, request={request_context or '-'}, "
                        f"payload_bytes={payload_size}, timeout={timeout}, fallback={fallback_name_value}): {exc}"
                    ) from exc
                sleep_for_retry(retry_index)
        else:
            raise ApiRequestFailed(
                f"VLM API request failed without response "
                f"(video={video_label or '-'}, scene_id={scene_id or '-'}, chunk_id={chunk_id or '-'})"
            ) from last_exc
        try:
            data = resp.json()
            choice = data["choices"][0]
            content = choice["message"]["content"].strip()
            usage = data.get("usage") if isinstance(data, dict) else {}
            if not isinstance(usage, dict):
                usage = {}
            last_result = ApiJsonResult(
                content=content,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                finish_reason=str(choice.get("finish_reason") or ""),
                retry_count=total_retry_count,
                payload_bytes=payload_size,
            )
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raw_snippet = getattr(resp, "text", "").replace("\n", "\\n")[:500]
            log(
                "[api:bad_json] "
                f"video={video_label or '-'} scene_id={scene_id or '-'} chunk_id={chunk_id or '-'} "
                f"request={request_context or '-'} fallback={fallback_name_value} "
                f"error={type(exc).__name__}: {exc} raw={raw_snippet!r}"
            )
            raise ValueError(f"APIレスポンスJSONを解釈できません: {type(exc).__name__}: {exc}; raw={raw_snippet!r}") from exc
        if choice.get("finish_reason") != "length":
            return last_result
    if last_result.content:
        return last_result
    return ApiJsonResult(content=content, retry_count=total_retry_count, payload_bytes=last_payload_size)


def main() -> int:
    args = parse_args()
    normalize_args(args)
    set_log_level(args.log_level)
    if args.timeout is not None:
        args.read_timeout = args.timeout
    current_sampling_duration_sec = sampling_duration_sec(args.sample_fps, args.temporal_frames)
    effective_max_side = args.max_image_side if args.max_image_side is not None else args.max_side
    effective_jpeg_quality = args.jpeg_quality if args.jpeg_quality is not None else args.quality
    effective_max_side = max(128, effective_max_side)
    effective_jpeg_quality = max(1, min(95, effective_jpeg_quality))
    profile_hash = analysis_profile(args, effective_max_side, effective_jpeg_quality)
    timeout_value = api_timeout(args.connect_timeout, args.read_timeout)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "descriptions.jsonl"
    intermediate_path = args.output_dir / "dual_video_stopped_scene_results.jsonl"
    completed_scene_results = {} if args.no_resume else load_completed_scene_results(intermediate_path)
    completed_chunk_results = {} if args.no_resume else load_completed_chunk_results(intermediate_path)
    html_path = resolve_html_path(args.output_dir, args.html_file, args.html_name_format)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    index_html_path = args.output_dir / "index.html"
    log(f"HTML: {html_path}", level="normal", stream=sys.stdout)
    if index_html_will_update(html_path, index_html_path, args.no_index_html):
        log(f"Index HTML: {index_html_path}", level="normal", stream=sys.stdout)

    session = requests.Session()
    check_server(args.url, session)

    front_cap = open_video(args.front_video)
    back_cap = open_video(args.back_video)
    front_fps, front_frame_count, front_duration_sec = video_duration_sec(front_cap, "フロント")
    back_fps, back_frame_count, back_duration_sec = video_duration_sec(back_cap, "バック")
    duration_sec = min(front_duration_sec, back_duration_sec)
    end_sec = args.end_sec if args.end_sec > 0 else duration_sec
    end_sec = min(end_sec, duration_sec)
    if end_sec <= 0 or args.start_sec >= end_sec:
        raise RuntimeError(f"分析範囲が不正です: start={args.start_sec}, end={args.end_sec}")

    front_stem = args.front_video.stem
    back_stem = args.back_video.stem
    sample_index = 0
    position_sec = max(0.0, args.start_sec)
    current_front_frame_index = 0
    current_back_frame_index = 0
    session_entries: list[DualVideoSceneEntry] = []
    pending_entry: DualVideoSceneEntry | None = None
    pending_jsonl_written = False
    log("停車時向けの前後動画注意点出力を開始します。", level="normal", stream=sys.stdout)
    log(f"Front video: {args.front_video}", level="normal", stream=sys.stdout)
    log(f"Back video: {args.back_video}", level="normal", stream=sys.stdout)
    log(
        f"Front duration: {front_duration_sec:.2f}s / FPS: {front_fps:.2f} / Frames: {front_frame_count}",
        level="normal",
        stream=sys.stdout,
    )
    log(
        f"Back duration: {back_duration_sec:.2f}s / FPS: {back_fps:.2f} / Frames: {back_frame_count}",
        level="normal",
        stream=sys.stdout,
    )
    log(f"Frame interval: {args.frame_interval:.2f}s", level="normal", stream=sys.stdout)
    log(f"Temporal sampling: {args.sample_fps:.2f} FPS x {args.temporal_frames} frames", level="normal", stream=sys.stdout)
    log(f"Sampling duration:\n{current_sampling_duration_sec:.2f} sec", level="normal", stream=sys.stdout)
    log(f"Analysis end: {end_sec:.2f}s", level="normal", stream=sys.stdout)
    log(
        f"API timeout: connect={timeout_value[0]:.1f}s read={timeout_value[1]:.1f}s / retries={args.api_max_retries}",
        level="normal",
        stream=sys.stdout,
    )
    log(f"Chunk frames: {args.chunk_frames} / overlap={args.chunk_overlap}", level="normal", stream=sys.stdout)
    log(f"Chunk max_workers: {args.max_workers}", level="normal", stream=sys.stdout)
    log(f"Chunk merge mode: {args.chunk_merge_mode}", level="normal", stream=sys.stdout)
    log(
        f"Image payload: format={args.image_format} max_image_side={effective_max_side} "
        f"jpeg_quality={effective_jpeg_quality}",
        level="normal",
        stream=sys.stdout,
    )
    if completed_scene_results:
        log(
            f"Resume: {len(completed_scene_results)} completed scenes loaded from {intermediate_path}",
            level="normal",
            stream=sys.stdout,
        )
    if completed_chunk_results:
        log(
            f"Resume: {len(completed_chunk_results)} completed chunks loaded from {intermediate_path}",
            level="normal",
            stream=sys.stdout,
        )
    supplemental_user_prompt = build_supplemental_user_prompt(args.location_info, args.weather_info, args.time_info)
    if args.location_info:
        log(f"Location info: {args.location_info}", level="normal", stream=sys.stdout)
    if args.weather_info:
        log(f"Weather info: {args.weather_info}", level="normal", stream=sys.stdout)
    if args.time_info:
        log(f"Time info: {args.time_info}", level="normal", stream=sys.stdout)

    started = time.perf_counter()
    interrupted = False
    try:
        while position_sec <= end_sec:
            if args.max_samples > 0 and sample_index >= args.max_samples:
                break

            sample_index += 1
            current_scene_id = scene_id_for(
                args.front_video,
                args.back_video,
                sample_index,
                position_sec,
                profile_hash,
            )
            if current_scene_id in completed_scene_results:
                completed = completed_scene_results[current_scene_id]
                scene_timing = empty_scene_timing()
                scene_timing.update(
                    {
                        "scene_status": "scene_resumed",
                        "chunk_count": int(completed.get("chunk_count", 0) or 0),
                        "chunk_elapsed_sec_list": completed.get("chunk_elapsed_sec_list", []) or [],
                        "chunk_metrics": completed.get("chunk_metrics", []) or [],
                        "chunk_max_elapsed_sec": float(completed.get("chunk_max_elapsed_sec", 0.0) or 0.0),
                        "chunk_min_elapsed_sec": float(completed.get("chunk_min_elapsed_sec", 0.0) or 0.0),
                        "chunk_total_elapsed_sec": float(completed.get("chunk_total_elapsed_sec", 0.0) or 0.0),
                        "merge_elapsed_sec": 0.0,
                        "merge_status": "resumed",
                        "merge_reason": "scene_complete_resumed",
                        "failed_chunk_count": int(completed.get("failed_chunk_count", 0) or 0),
                        "resumed_chunk_count": int(completed.get("chunk_count", 0) or 0),
                    }
                )
                entry = DualVideoSceneEntry(
                    timestamp=str(completed.get("timestamp", f"video {format_video_timestamp(position_sec)}")),
                    elapsed_sec=float(completed.get("elapsed_sec", 0.0)),
                    image_file=str(completed.get("image_file", "")),
                    description=str(completed["description"]),
                    sampling_fps=float(completed.get("sampling_fps", args.sample_fps) or 0.0),
                    temporal_frames=int(completed.get("temporal_frames", args.temporal_frames) or 0),
                    sampling_duration_sec=float(
                        completed.get("sampling_duration_sec", current_sampling_duration_sec) or 0.0
                    ),
                    images_sent=int(completed.get("images_sent", completed.get("temporal_frames", 0)) or 0),
                )
                session_entries.append(entry)
                entries = session_entries[-args.history :]
                write_dual_video_html(html_path, args.output_dir, entries)
                if index_html_will_update(html_path, index_html_path, args.no_index_html):
                    write_dual_video_html(index_html_path, args.output_dir, entries)
                log(
                    f"[resume] skip video={format_video_timestamp(position_sec)}",
                    level="normal",
                )
                if args.show_timing_detail:
                    log(
                        format_timing_detail(format_video_timestamp(position_sec), scene_timing),
                        level="normal",
                    )
                log(
                    f"[resume:debug] skip completed scene_id={current_scene_id} video={format_video_timestamp(position_sec)}",
                    level="debug",
                )
                position_sec += args.frame_interval
                continue

            started = time.perf_counter()
            front_frames, current_front_frame_index = read_temporal_frames(
                front_cap,
                current_front_frame_index,
                position_sec,
                front_fps,
                args.sample_fps,
                args.temporal_frames,
                end_sec,
            )
            back_frames, current_back_frame_index = read_temporal_frames(
                back_cap,
                current_back_frame_index,
                position_sec,
                back_fps,
                args.sample_fps,
                args.temporal_frames,
                end_sec,
            )
            temporal_frame_count = min(len(front_frames), len(back_frames))
            if temporal_frame_count == 0:
                break

            combined_frames = [
                combine_front_back_frames(front_frame, back_frame)
                for front_frame, back_frame in zip(front_frames[:temporal_frame_count], back_frames[:temporal_frame_count])
            ]
            combined_frame = combined_frames[-1]
            now = datetime.now()
            timestamp = f"{now:%Y-%m-%d %H:%M:%S} / video {format_video_timestamp(position_sec)}"
            safe_ts = now.strftime("%Y%m%d_%H%M%S")
            image_path = images_dir / f"{front_stem}_{back_stem}_{safe_ts}_{sample_index:04d}.jpg"
            save_frame(combined_frame, image_path, args.quality)
            pending_entry = DualVideoSceneEntry(
                timestamp=timestamp,
                elapsed_sec=0.0,
                image_file=str(image_path.relative_to(args.output_dir)),
                description="中断されたため、説明は生成されていません。",
                sampling_fps=args.sample_fps,
                temporal_frames=args.temporal_frames,
                sampling_duration_sec=current_sampling_duration_sec,
                images_sent=temporal_frame_count,
            )
            pending_jsonl_written = False
            video_label = f"{args.front_video.name}+{args.back_video.name}@{format_video_timestamp(position_sec)}"
            log(
                "[scene:start] "
                f"video={format_video_timestamp(position_sec)} scene={sample_index} "
                f"temporal_frames={temporal_frame_count}",
                level="normal",
            )
            log(
                "[scene:start:debug] "
                f"video={video_label} scene_id={current_scene_id} sample_index={sample_index} "
                f"position_sec={position_sec:.3f} temporal_frame_count={temporal_frame_count}",
                level="debug",
            )
            scene_timing = empty_scene_timing()
            try:
                description = ""
                image_encoder = SceneImageEncoder(combined_frames, args.image_format)
                scene_ranges = chunk_ranges(len(combined_frames), args.chunk_frames, args.chunk_overlap)
                if len(scene_ranges) > 1:
                    description = describe_dual_video_scene(
                        args.url,
                        None,
                        session,
                        args.max_tokens,
                        args.temperature,
                        timeout_value,
                        prompt_mode=args.vlm_prompt_mode,
                        supplemental_user_prompt=supplemental_user_prompt,
                        retry_on_length=args.retry_on_length,
                        front_percent=front_region_percent(front_frames[-1], back_frames[-1]),
                        scene_id=current_scene_id,
                        video_label=video_label,
                        request_context="scene",
                        api_max_retries=args.api_max_retries,
                        chunk_frames=args.chunk_frames,
                        chunk_overlap=args.chunk_overlap,
                        chunk_merge_mode=args.chunk_merge_mode,
                        intermediate_path=intermediate_path,
                        fallback_name_value="chunk-level",
                        image_format=args.image_format,
                        max_workers=args.max_workers,
                        completed_chunks=completed_chunk_results,
                        image_encoder=image_encoder,
                        max_side=effective_max_side,
                        quality=effective_jpeg_quality,
                        timing=scene_timing,
                        scene_position_sec=position_sec,
                        sample_fps=args.sample_fps,
                    )
                else:
                    for fallback_index in range(len(FALLBACK_STEPS)):
                        fallback = fallback_config(
                            effective_max_side,
                            effective_jpeg_quality,
                            args.max_tokens,
                            fallback_index,
                        )
                        if fallback_index > 0:
                            log(
                                "[fallback] "
                                f"video={video_label} scene_id={current_scene_id} scope=scene "
                                f"fallback={fallback.name} reason=consecutive_api_failures "
                                f"lightweight_image_payload=true image_format={args.image_format} "
                                f"max_image_side={fallback.max_side} jpeg_quality={fallback.quality} "
                                f"frame_step={fallback.frame_step}",
                                level="normal",
                            )
                        frame_indices = select_frame_indices(0, len(combined_frames), fallback.frame_step)
                        image_b64 = image_encoder.encode(
                            frame_indices,
                            fallback_name_value=fallback.name,
                            max_side=fallback.max_side,
                            quality=fallback.quality,
                            scene_id=current_scene_id,
                            video_label=video_label,
                        )
                        try:
                            description = describe_dual_video_scene(
                                args.url,
                                image_b64,
                                session,
                                fallback.max_tokens,
                                args.temperature,
                                timeout_value,
                                prompt_mode=args.vlm_prompt_mode,
                                supplemental_user_prompt=supplemental_user_prompt,
                                retry_on_length=args.retry_on_length,
                                front_percent=front_region_percent(front_frames[-1], back_frames[-1]),
                                scene_id=current_scene_id,
                                video_label=video_label,
                                request_context="scene",
                                api_max_retries=args.api_max_retries,
                                chunk_frames=args.chunk_frames,
                                chunk_overlap=args.chunk_overlap,
                                chunk_merge_mode=args.chunk_merge_mode,
                                intermediate_path=intermediate_path,
                                fallback_name_value=fallback.name,
                                image_format=args.image_format,
                                max_workers=args.max_workers,
                                completed_chunks=completed_chunk_results,
                                timing=scene_timing,
                                scene_position_sec=position_sec,
                                sample_fps=args.sample_fps,
                            )
                            break
                        except ApiRequestFailed:
                            if fallback_index + 1 >= len(FALLBACK_STEPS):
                                raise
                            continue
            except Exception as exc:
                pending_entry.elapsed_sec = time.perf_counter() - started
                pending_entry.description = f"推論失敗: {type(exc).__name__}: {exc}"
                write_jsonl(jsonl_path, pending_entry)
                pending_jsonl_written = True
                append_jsonl(
                    intermediate_path,
                    {
                        "type": "scene_failed",
                        "scene_id": current_scene_id,
                        "video": video_label,
                        "timestamp": timestamp,
                        "elapsed_sec": pending_entry.elapsed_sec,
                        "image_file": pending_entry.image_file,
                        "sampling_fps": args.sample_fps,
                        "temporal_frames": args.temporal_frames,
                        "sampling_duration_sec": current_sampling_duration_sec,
                        "images_sent": temporal_frame_count,
                        "error": f"{type(exc).__name__}: {exc}",
                        "finished_at": utc_now_iso(),
                    },
                )
                log(
                    "[scene:failed] "
                    f"video={format_video_timestamp(position_sec)} elapsed={pending_entry.elapsed_sec:.1f}s "
                    f"error={type(exc).__name__}: {exc}",
                    level="quiet",
                )
                log(
                    "[scene:failed:debug] "
                    f"video={video_label} scene_id={current_scene_id} elapsed_sec={pending_entry.elapsed_sec:.2f} "
                    f"error={type(exc).__name__}: {exc}",
                    level="debug",
                )
                position_sec += args.frame_interval
                continue

            elapsed = time.perf_counter() - started
            entry = DualVideoSceneEntry(
                timestamp=timestamp,
                elapsed_sec=elapsed,
                image_file=str(image_path.relative_to(args.output_dir)),
                description=description,
                sampling_fps=args.sample_fps,
                temporal_frames=args.temporal_frames,
                sampling_duration_sec=current_sampling_duration_sec,
                images_sent=temporal_frame_count,
            )
            pending_entry = entry
            pending_jsonl_written = False
            write_jsonl(jsonl_path, entry)
            pending_jsonl_written = True
            append_jsonl(
                intermediate_path,
                {
                    "type": "scene_complete",
                    "scene_id": current_scene_id,
                    "video": video_label,
                    "timestamp": timestamp,
                    "elapsed_sec": elapsed,
                    "image_file": entry.image_file,
                    "description": description,
                    "sampling_fps": args.sample_fps,
                    "temporal_frames": args.temporal_frames,
                    "sampling_duration_sec": current_sampling_duration_sec,
                    "images_sent": temporal_frame_count,
                    "scene_status": scene_timing["scene_status"],
                    "chunk_count": scene_timing["chunk_count"],
                    "chunk_elapsed_sec_list": scene_timing["chunk_elapsed_sec_list"],
                    "chunk_metrics": scene_timing["chunk_metrics"],
                    "chunk_max_elapsed_sec": scene_timing["chunk_max_elapsed_sec"],
                    "chunk_min_elapsed_sec": scene_timing["chunk_min_elapsed_sec"],
                    "chunk_total_elapsed_sec": scene_timing["chunk_total_elapsed_sec"],
                    "merge_elapsed_sec": scene_timing["merge_elapsed_sec"],
                    "merge_status": scene_timing["merge_status"],
                    "merge_reason": scene_timing["merge_reason"],
                    "failed_chunk_count": scene_timing["failed_chunk_count"],
                    "resumed_chunk_count": scene_timing["resumed_chunk_count"],
                    "finished_at": utc_now_iso(),
                },
            )
            if args.show_timing_detail:
                log(
                    format_timing_detail(format_video_timestamp(position_sec), scene_timing),
                    level="normal",
                )
            log(
                "[scene:end] "
                f"video={format_video_timestamp(position_sec)} elapsed={elapsed:.1f}s status=success "
                f"images_sent={temporal_frame_count} "
                f"chunk_max={scene_timing['chunk_max_elapsed_sec']:.1f}s "
                f"chunk_total={scene_timing['chunk_total_elapsed_sec']:.1f}s "
                f"merge={scene_timing['merge_elapsed_sec']:.1f}s",
                level="normal",
            )
            log(
                "[scene:end:debug] "
                f"video={video_label} scene_id={current_scene_id} elapsed_sec={elapsed:.2f} status=success",
                level="debug",
            )
            session_entries.append(entry)
            entries = session_entries[-args.history :]
            write_dual_video_html(html_path, args.output_dir, entries)
            if index_html_will_update(html_path, index_html_path, args.no_index_html):
                write_dual_video_html(index_html_path, args.output_dir, entries)
            pending_entry = None
            pending_jsonl_written = False

            log(f"\n[{timestamp}] {elapsed:.2f}s", level="normal", stream=sys.stdout)
            log(description, level="normal", stream=sys.stdout)
            position_sec += args.frame_interval
    except KeyboardInterrupt:
        interrupted = True
        log("\n停止しました。HTMLを保存します。", level="normal", stream=sys.stdout)
    finally:
        if pending_entry is not None:
            if pending_entry.description.startswith("中断されたため"):
                pending_entry.elapsed_sec = time.perf_counter() - started
            if not pending_jsonl_written:
                write_jsonl(jsonl_path, pending_entry)
            if pending_entry not in session_entries:
                session_entries.append(pending_entry)
        entries = session_entries[-args.history :]
        write_dual_video_html(html_path, args.output_dir, entries)
        if index_html_will_update(html_path, index_html_path, args.no_index_html):
            write_dual_video_html(index_html_path, args.output_dir, entries)
        if interrupted:
            log(f"HTML: {html_path}", level="quiet", stream=sys.stdout)
            if index_html_will_update(html_path, index_html_path, args.no_index_html):
                log(f"Index HTML: {index_html_path}", level="quiet", stream=sys.stdout)
        else:
            log(f"HTML: {html_path}", level="quiet", stream=sys.stdout)
        front_cap.release()
        back_cap.release()
        session.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
