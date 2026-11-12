#!/usr/bin/env python3
"""
Stopped-vehicle dual video scene describer.

撮影済みのフロントカメラ動画とバックカメラ動画から同じ時刻のフレームを抜き出し、
停車中の通行状況と防犯上の観察を蓄積し、開始時と最新の障害物を比較します。
車両に戻った際のフィードバックを最大3件、 final_report.json / .txt / .html に保存します。
通行の観測時点数は台数・人数ではなく、連続画像に写る対象の重複を含みます。
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


DUAL_VIDEO_SYSTEM_PROMPT_CONTEXT = """停車中の車載映像を時系列で観測してください。
左の FRONT VIEW ONLY は前方、右の REAR VIEW ONLY は後方です。方向を取り違えないでください。
前方映像は上半分の切り出しです。映らない直近路面、側方、ドア開閉範囲、正確な距離は推測禁止です。
車内・乗員、看板の文字、車両の色は説明不要です。犯罪意図や人物属性を推測しないでください。
observations は内部観測で、通行、停止、滞留、接触などを件数制限なく記録します。
object_type は pedestrian/bicycle/vehicle/other/unknown のいずれかだけです。
object_label は対象の具体的な名称（例: 自転車、コーン）、position は映像内で区別できる位置です。
人物は pedestrian、自転車は bicycle、自動車・バイク・トラック等は vehicle としてください。
時刻は与えられた画像の動画開始からの秒数です。不明な値は null または unknown とします。
間隔が大きい画像間の移動や、別カメラ・別時間の対象の同一性を断定しないでください。
temporal_status は最終観測時点の approaching/passing/departed/stationary/unknown です。
単に見えなくなった対象を departed と断定せず、根拠がなければ unknown としてください。
latest_visible は今回の最後の画像で対象を確認できる場合だけ true にします。
evidence_frames は実際に対象を確認した画像の秒数。first_seen/last_seen はその範囲です。
notifications は原則1～2件、最大3件、不要なら空配列です。観測event_idを参照してください。
通知優先順: 最新画像で明瞭な車両直近の接近・通過、発進経路の妨害、停車継続・乗降への直接影響、
明確な車体への接触・のぞき込み、その他復帰時に有用な変化。
過去の通常通行は通知不要。人物の存在や視線だけを防犯通知の根拠にしないでください。
notification_reason は near_movement/blocked_path/boarding_or_stop/contact/peering/change/none。
根拠のない影響を補わず、通常の存在・通行は none。position は映像で確認した位置だけ、
evidence は観測事実、confirmation_action は必要な確認行動（不要なら空文字）です。
通知文に「〜ため、〜に注意」の定型は不要です。推測や冗長な注意喚起は禁止です。
JSONは最後まで閉じてください。空白や改行は最小限にし、観測件数を減らして帳尻を合わせないでください。
通知文章は後段で生成するため、notifications は event_id の参照だけにし、文章や観測項目を重複出力しないでください。
JSONのみを出力してください。例:
{"latest_frame_sec":18.0,"observations":[{"event_id":"event_001","object_type":"bicycle",
"direction":"後方","action":"passing","temporal_status":"departed","first_seen":12.0,
"last_seen":15.0,"evidence_frames":[12.0,15.0],"latest_visible":false,
"position":"","evidence":"後方を通過し離脱","notification_reason":"none","confirmation_action":""}],
"notifications":[]}
"""

DUAL_VIDEO_USER_PROMPT_CONTEXT = """同時刻の前後映像を古い順に提示します。
左が前方、右が後方です。観測は制限なく保存し、最終画像での状態と過去の行動を区別してください。
不明瞭な動きや同一性は unknown。指定JSONのみを返してください。"""

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
DEFAULT_DUAL_VIDEO_OUTPUT_DIR = SCRIPT_DIR / "scene_logs_v4"
DEFAULT_DUAL_VIDEO_MAX_SIDE = 768
DEFAULT_API_CONNECT_TIMEOUT = 10.0
DEFAULT_API_READ_TIMEOUT = 300.0
DEFAULT_API_MAX_RETRIES = 3
DEFAULT_CHUNK_FRAMES = 6
DEFAULT_CHUNK_OVERLAP = 2
SCENE_ID_PROFILE = "video_scene_describer4:return-feedback-v5-frame-identity"
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
    captured_frame_id: str | None = None
    captured_frame_sec: float | None = None


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
        description = html.escape(display_description(entry.description)).replace("\n", "<br>")
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
        description="フロント/バックの撮影済み動画からフレームを抽出し、停車中の事故・防犯上の注意点を分析し、動画全体の上位3件を出力する"
    )
    parser.add_argument("--notification-style", choices=NOTIFICATION_STYLES, default="action_oriented")
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
        help="VLM分析を開始する間隔秒（既定: 1秒ごとに10 FPSの画像を10枚取得）",
    )
    parser.add_argument("--sample-fps", type=float, default=10.0, help="動き確認用のフレーム取得FPS（既定: 0.1秒間隔）")
    parser.add_argument("--temporal-frames", type=int, default=10, help="1回のVLM分析に渡す時系列フレーム数")
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
    parser.add_argument("--max-tokens", type=int, default=2048, help="構造化JSONの初回最大トークン数（再試行では増量）")
    parser.add_argument(
        "--retry-on-length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="出力上限到達・JSON構文不正時に上限を増やして再推論する（既定: 有効）",
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
    frame_metadata: list[dict[str, Any]] | None = None,
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
        if frame_metadata is not None:
            frame_metadata.append({"frame_index": current_frame_index - 1,
                                   "sec": (current_frame_index - 1) / video_fps})
    return frames, current_frame_index


def combined_frame_id(front_index: int, back_index: int) -> str:
    return f"front:{front_index}/rear:{back_index}"


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


NOTIFICATION_STYLES = ("object_only", "situation", "action_oriented")


def build_timed_image_content(images: list[str], image_format: str,
                              times: list[float]) -> list[dict[str, Any]]:
    if len(images) != len(times):
        raise ValueError("画像と時刻の数が一致しません")
    content = []
    for image, second in zip(images, times):
        content.append({"type": "text", "text": f"動画開始から {second!r} 秒"})
        content.extend(build_image_content([image], image_format))
    return content


def validate_observations(data: dict[str, Any]) -> dict[str, Any]:
    import math

    def second(value: Any) -> bool:
        return value is None or (type(value) in (int, float) and math.isfinite(value) and value >= 0)

    latest = data.get("latest_frame_sec")
    if not second(latest):
        raise ValueError("latest_frame_sec が不正です")
    events = data.get("observations")
    notices = data.get("notifications", [])
    if not isinstance(events, list) or not isinstance(notices, list) or len(notices) > 3:
        raise ValueError("observations/notifications が不正です")
    ids = set()
    normalized = []
    for item in events:
        if not isinstance(item, dict):
            raise ValueError("観測がオブジェクトではありません")
        event = dict(item)
        identifier = event.get("event_id")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("event_id が空または重複しています")
        ids.add(identifier)
        if event.get("direction") not in ("前方", "後方"):
            raise ValueError("観測の方向が不正です")
        object_value = event.get("object_type")
        aliases = {"car": "vehicle", "automobile": "vehicle", "motorcycle": "vehicle",
                   "motorbike": "vehicle", "truck": "vehicle", "bus": "vehicle",
                   "person": "pedestrian", "people": "pedestrian", "cyclist": "bicycle",
                   "bike": "bicycle", "歩行者": "pedestrian", "人物": "pedestrian",
                   "車両": "vehicle", "自転車": "bicycle"}
        if isinstance(object_value, str):
            object_value = object_value.strip().lower()
            event["object_type"] = aliases.get(object_value, object_value)
        if event.get("object_type") not in ("pedestrian", "bicycle", "vehicle", "other", "unknown"):
            raise ValueError("object_type が不正です")
        states = ("approaching", "passing", "departed", "stationary", "unknown")
        if event.get("temporal_status") not in states:
            raise ValueError("temporal_status が不正です")
        if not isinstance(event.get("action"), str) or not event["action"]:
            raise ValueError("action が不正です")
        first, last = event.get("first_seen"), event.get("last_seen")
        frames = event.get("evidence_frames")
        if not second(first) or not second(last) or not isinstance(frames, list) or any(
                t is None or not second(t) for t in frames):
            raise ValueError("観測時刻が不正です")
        if first is not None and last is not None and first > last:
            raise ValueError("観測時刻が逆転しています")
        if frames and (first != min(frames) or last != max(frames)):
            raise ValueError("観測時刻と根拠フレームが一致しません")
        if latest is not None and any(t > latest for t in frames):
            raise ValueError("最新時刻より後の根拠フレームです")
        if not isinstance(event.get("latest_visible", False), bool):
            raise ValueError("latest_visible が不正です")
        event.setdefault("latest_visible", False)
        for key in ("position", "evidence", "confirmation_action", "object_label"):
            event.setdefault(key, "")
            if not isinstance(event[key], str):
                raise ValueError(f"{key} が文字列ではありません")
        event.setdefault("notification_reason", "none")
        if event["notification_reason"] not in ("near_movement", "blocked_path", "boarding_or_stop",
                                                 "contact", "peering", "change", "none"):
            raise ValueError("notification_reason が不正です")
        normalized.append(event)
    for notice in notices:
        if not isinstance(notice, dict) or notice.get("event_id") not in ids:
            raise ValueError("通知のイベント参照が不正です")
    result = dict(observations=normalized, notifications=notices, latest_frame_sec=latest)
    for key in ("latest_frame_id", "frame_timeline"):
        if key in data:
            result[key] = data[key]
    return result


def display_description(description: str) -> str:
    if not description.lstrip().startswith("{"):
        return description
    data = json.loads(description)
    return "\n".join(f"【{e['direction']}】{e['evidence']}" for e in data["observations"]) or "観測なし"


def validate_frame_evidence(description: str, times: list[float],
                            frame_timeline: list[dict[str, Any]] | None = None) -> str:
    if not description.lstrip().startswith("{"):
        return description
    data = json.loads(description)
    if frame_timeline is not None and [f["sec"] for f in frame_timeline] != times:
        raise ValueError("フレーム情報と送信時刻が一致しません")
    # Identity metadata comes only from the decoder, never from model output.
    data.pop("latest_frame_id", None)
    data.pop("frame_timeline", None)
    if frame_timeline:
        data["frame_timeline"] = frame_timeline
        data["latest_frame_id"] = frame_timeline[-1]["frame_id"]
    # The caller owns the frame timeline; model metadata cannot advance it.
    data["latest_frame_sec"] = max(times) if times else None
    for event in data["observations"]:
        event.pop("evidence_frame_ids", None)
        event.pop("observation_source", None)
        for second in event["evidence_frames"]:
            if not any(abs(second - actual) < 0.001 for actual in times):
                raise ValueError("送信していないフレームが根拠に指定されています")
        event["evidence_frames"] = [min(times, key=lambda t: abs(t-second))
                                    for second in event["evidence_frames"]]
        if event["evidence_frames"]:
            event["first_seen"] = min(event["evidence_frames"])
            event["last_seen"] = max(event["evidence_frames"])
        if frame_timeline is not None:
            event["evidence_frame_ids"] = list(dict.fromkeys(
                f["frame_id"] for f in frame_timeline if f["sec"] in event["evidence_frames"]))
    return json.dumps(validate_observations(data), ensure_ascii=False)


def notification_variants(event: dict[str, Any], current: bool) -> dict[str, str]:
    label = event.get("object_label") or {"pedestrian": "歩行者", "bicycle": "自転車",
        "vehicle": "車両", "other": "対象", "unknown": "対象"}[event["object_type"]]
    subject = f"{event['direction']}の{event['position']}{label}"
    reason = event["notification_reason"]
    if reason in ("contact", "peering"):
        behavior = "車体に触れる" if reason == "contact" else "車内をのぞき込む"
        situation = f"離席中、{subject}が{behavior}様子が確認されました。"
    elif current:
        state = {"approaching": "接近しています", "passing": "通過しています",
                 "stationary": "停止・滞留しています", "unknown": "確認されています"}.get(
                     event["temporal_status"], "確認されています")
        situation = f"{subject}が{state}。"
    else:
        situation = f"離席中、{subject}が確認されました。"
    action = event["confirmation_action"].strip()
    return dict(object_only=f"{subject}。", situation=situation,
                action_oriented=situation + action)


def format_dual_video_description(content: str) -> str:
    try:
        data = json.loads(extract_json_object(content))
    except json.JSONDecodeError as exc:
        raise ValueError("JSONとして解釈できません") from exc
    if isinstance(data, dict) and "observations" in data:
        return json.dumps(validate_observations(data), ensure_ascii=False)
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


def is_security_observation(message: str) -> bool:
    persistent_gaze = (
        any(term in message for term in ("同一人物", "同じ人物"))
        and any(term in message for term in ("継続", "見続け", "繰り返し"))
        and any(term in message for term in ("見て", "見つめ", "注視", "視線"))
    )
    return persistent_gaze or any(term in message for term in (
        "のぞき込", "覗き込", "ドアに触", "車体に触", "繰り返しの往復",
    ))


def local_risk_priority(direction: str, message: str) -> tuple[int, int, int]:
    """Rank risks using the same explicit priority policy as the VLM prompt."""
    if direction == "補足情報":
        return (7, 0, 0)
    object_rank = {"pedestrian": 0, "bicycle": 1, "vehicle": 2, "other": 3}[risk_object_type(message)]
    is_near = any(term in message for term in ("直近", "すぐ近く", "付近", "近く"))
    is_moving = any(term in message for term in ("通過", "接近", "横切", "進入"))
    blocks_departure = any(term in message for term in ("発進経路", "進路を塞", "発進を妨"))
    affects_door_or_stop = any(term in message for term in ("ドア開閉", "乗降", "停車継続"))
    if is_near and is_moving and object_rank < 3:
        return (object_rank, 0, 0)
    if is_security_observation(message):
        return (3, 0, 0)
    if blocks_departure:
        return (4, object_rank, 0)
    if affects_door_or_stop:
        return (5, object_rank, 0)
    return (6, object_rank, 0 if is_moving else 1)


def normalized_risk_text(message: str) -> str:
    return re.sub(r"[\s、。・,.;；：:（）()]", "", message)


def risks_are_duplicates(left: tuple[str, str], right: tuple[str, str]) -> bool:
    if left[0] != right[0] or risk_object_type(left[1]) != risk_object_type(right[1]):
        return False
    if is_security_observation(left[1]) != is_security_observation(right[1]):
        return False
    left_text = normalized_risk_text(left[1])
    right_text = normalized_risk_text(right[1])
    if left_text == right_text:
        return True
    if min(len(left_text), len(right_text)) < 12:
        return False
    return difflib.SequenceMatcher(None, left_text, right_text).ratio() >= 0.78


def merge_chunk_descriptions_locally(chunk_results: list[ChunkAnalysisResult]) -> str:
    structured = [json.loads(r.description) for r in chunk_results
                  if not r.error and r.description.lstrip().startswith('{')]
    if structured:
        observations = []
        latest = None
        timeline = {}
        latest_id = None
        for index, data in enumerate(structured):
            if data.get("latest_frame_sec") is not None:
                if latest is None or data["latest_frame_sec"] >= latest:
                    latest = data["latest_frame_sec"]
                    latest_id = data.get("latest_frame_id")
            for frame in data.get("frame_timeline", []):
                timeline[frame["frame_id"]] = frame
            for event in data["observations"]:
                observations.append({**event, "event_id": f"chunk_{index}_{event['event_id']}",
                                     "observation_source": f"chunk_{index}"})
        return json.dumps({"observations": observations, "notifications": [],
                           "latest_frame_sec": latest, "latest_frame_id": latest_id,
                           "frame_timeline": sorted(timeline.values(), key=lambda f: f["sec"])}, ensure_ascii=False)
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
    frame_times: list[float] | None = None,
    frame_timeline: list[dict[str, Any]] | None = None,
) -> ChunkAnalysisResult:
    chunk_id = f"{chunk_index:02d}-{start + 1:02d}_{end:02d}"
    image_content = build_timed_image_content(chunk_image_b64, image_format, frame_times or [])
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
            api_result, chunk_description = request_observation_json(
                base_url,
                chunk_messages,
                chunk_session,
                max_tokens,
                temperature,
                timeout,
                retry_on_length,
                frame_times=frame_times or [],
                frame_timeline=frame_timeline,
                max_retries=api_max_retries,
                scene_id=scene_id,
                chunk_id=chunk_id,
                video_label=video_label,
                request_context=f"{request_context}:chunk",
                fallback_name_value=fallback_name_value,
            )
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
    frame_times: list[float] | None = None,
    frame_timeline: list[dict[str, Any]] | None = None,
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
    times = frame_times or [scene_position_sec + i / sample_fps for i in range(total_frames)]
    if len(times) != total_frames or (frame_timeline is not None and len(frame_timeline) != total_frames):
        raise ValueError("画像とフレーム情報の数が一致しません")
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
                        frame_times=[times[i] for i in (frame_indices if image_encoder is not None else range(start, end))],
                        frame_timeline=([frame_timeline[i] for i in (frame_indices if image_encoder is not None else range(start, end))]
                                        if frame_timeline is not None else None),
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
        if chunk_merge_mode == "local" or any(r.description.lstrip().startswith("{") for r in ordered_chunk_results):
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
            "人物の存在や視線だけで防犯通知を生成しないでください。別チャンクの人物を根拠なく同一人物と結びつけないでください。\n"
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

    image_content = build_timed_image_content(image_b64 or [], image_format,
        times)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [*image_content, {"type": "text", "text": user_prompt}]},
    ]
    _, description = request_observation_json(
        base_url, messages, session, max_tokens, temperature, timeout, retry_on_length,
        frame_times=times, frame_timeline=frame_timeline,
        max_retries=api_max_retries, scene_id=scene_id, chunk_id="full", video_label=video_label,
        request_context=request_context, fallback_name_value=fallback_name_value)
    return description


def request_observation_json(
    base_url: str, messages: list[dict[str, Any]], session: requests.Session,
    max_tokens: int, temperature: float, timeout: tuple[float, float], retry_on_length: bool,
    *, frame_times: list[float], frame_timeline: list[dict[str, Any]] | None = None, **kwargs: Any,
) -> tuple[ApiJsonResult, str]:
    retry_messages = list(messages)
    budget = max_tokens
    for attempt in range(3):
        result = request_dual_video_json(base_url, retry_messages, session, budget,
                                        temperature, timeout, retry_on_length, **kwargs)
        try:
            description = validate_frame_evidence(format_description_with_raw_context(result.content), frame_times,
                                                 frame_timeline)
            return result, description
        except ValueError as exc:
            if attempt == 2:
                raise ValueError(f"観測JSONの検証に3回失敗しました: {exc}") from exc
            # Give the model the actual failed answer, plus concrete schema guidance.
            retry_messages = [*messages, {"role": "assistant", "content": result.content},
                {"role": "user", "content":
                 f"検証エラー: {str(exc).split('; raw=')[0]}。全観測を保持して完全なJSONを再生成してください。"
                 "object_type は pedestrian/bicycle/vehicle/other/unknown。car・motorcycle は vehicle、"
                 "person は pedestrian。notifications は event_id 参照のみです。"
                 "時刻は提示された画像だけを根拠にし、不明な状態は unknown としてください。"}]
            budget = max(budget, result.completion_tokens or 0) * 2
    raise AssertionError("unreachable")


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
    length_attempts = 3 if retry_on_length else 1
    content = ""
    total_retry_count = 0
    last_payload_size = 0
    last_result = ApiJsonResult(content="")
    for length_attempt in range(length_attempts):
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens * (2 ** length_attempt),
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
        malformed = False
        try:
            json.loads(extract_json_object(content))
        except (ValueError, TypeError):
            malformed = True
        if last_result.finish_reason != "length" and not malformed:
            return last_result
        if length_attempt + 1 < length_attempts:
            log(f"[api:json_retry] reason={'length' if last_result.finish_reason == 'length' else 'invalid_json'} "
                f"max_tokens={payload['max_tokens']} next_max_tokens={payload['max_tokens'] * 2}", level="normal")
            total_retry_count += 1
        elif retry_on_length:
            raise ValueError(f"JSON出力が不完全です（finish_reason={last_result.finish_reason}, "
                             f"max_tokens={payload['max_tokens']}、{length_attempts}回試行）")
    if last_result.content:
        return last_result
    return ApiJsonResult(content=content, retry_count=total_retry_count, payload_bytes=last_payload_size)


ENVIRONMENT_COMPARISON_PROMPT = """停車車両に戻ったドライバーのため、2枚の結合画像を比較してください。
1枚目は観測開始時、2枚目は最新の観測時です。各画像の左は前方、右は後方です。
映像で明確に発進・後退経路を妨げる、または停車継続・乗降に直接影響する静止障害物だけを確認します。
前方は上半分の切り出しです。映らない側方、ドア範囲、直近路面、正確な距離を推測しないでください。
道路脇に見えるだけの遠い物体、通過中の人物や車両は障害物に含めません。
最新画像でも残っている障害物を、開始時から存在する物も含めて挙げてください。
開始時になかったと断定できるのは、対応する場所が開始画像でも明瞭に見え、空いていた場合だけです。
遮蔽、画角の違い、低解像度の場合は initial_state を unknown にしてください。
latest_visible は最新画像で障害物が明瞭に見える場合だけ true にします。
position は映像内で区別できる位置を簡潔に記録してください。不明なら空文字です。
障害物が消えた場合は出力しないでください。ドライバーが実際に忘れたとは断定しないでください。
出力はJSONのみ:
{"obstacles":[{"direction":"後方","object":"コーン","initial_state":"absent",
"latest_visible":true,"position":"中央","evidence":"開始時は空いていた車両直近に最新画像ではコーンがある"}]}
initial_state は present / absent / unknown のみ。確認できない場合は {"obstacles":[]}。
"""


def compare_environment(args: argparse.Namespace, session: requests.Session,
                        first_sec: float, latest_sec: float,
                        latest_frame_id: str | None = None) -> dict[str, Any]:
    frames = []
    timeline = []
    latest_indices = re.fullmatch(r"front:(\d+)/rear:(\d+)", latest_frame_id or "")
    for image_index, second in enumerate((first_sec, latest_sec)):
        captures = [open_video(args.front_video), open_video(args.back_video)]
        try:
            images = []
            indices = []
            seconds = []
            for camera_index, cap in enumerate(captures):
                fps = cap.get(cv2.CAP_PROP_FPS)
                target = (int(latest_indices.group(camera_index + 1))
                          if image_index == 1 and latest_indices else round(second * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"環境比較フレームを取得できません: {second:.2f}s")
                images.append(frame)
                actual_index = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1
                indices.append(actual_index)
                seconds.append(actual_index / fps)
            timeline.append(dict(frame_id=combined_frame_id(*indices), sec=second,
                                 front_sec=seconds[0], rear_sec=seconds[1]))
            combined = combine_front_back_frames(*images)
            image = resize_for_inference(combined, args.max_image_side or args.max_side)
            frames.append(encode_image_payload(image, args.image_format, args.jpeg_quality or args.quality))
        finally:
            for cap in captures:
                cap.release()
    content = [{"type": "text", "text": f"開始={first_sec:.2f}秒、最新={latest_sec:.2f}秒。"}]
    content.extend(build_image_content(frames, args.image_format))
    result = request_dual_video_json(args.url, [
        {"role": "system", "content": ENVIRONMENT_COMPARISON_PROMPT},
        {"role": "user", "content": content}], session, max(600, args.max_tokens),
        args.temperature, api_timeout(args.connect_timeout, args.read_timeout),
        args.retry_on_length, max_retries=args.api_max_retries, request_context="environment_comparison")
    data = json.loads(extract_json_object(result.content))
    obstacles = data.get("obstacles")
    if not isinstance(obstacles, list):
        raise ValueError("obstacles が配列ではありません")
    validated = []
    for item in obstacles:
        if not isinstance(item, dict) or item.get("direction") not in ("前方", "後方"):
            raise ValueError("障害物の方向が不正です")
        if item.get("initial_state") not in ("present", "absent", "unknown"):
            raise ValueError("障害物の開始状態が不正です")
        if not isinstance(item.get("latest_visible"), bool):
            raise ValueError("latest_visible が真偽値ではありません")
        if not all(isinstance(item.get(k), str) and item[k].strip() for k in ("object", "evidence")):
            raise ValueError("障害物の根拠または対象が空です")
        item.setdefault("position", "")
        if not isinstance(item["position"], str):
            raise ValueError("障害物の位置が文字列ではありません")
        if item["latest_visible"]:
            validated.append(item)
    return {"status": "completed", "first_sec": timeline[0]["sec"], "latest_sec": timeline[-1]["sec"],
            "latest_frame_id": timeline[-1]["frame_id"], "frame_timeline": timeline,
            "obstacles": validated}


def same_notification_subject(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Require shared image evidence and matching descriptions; never infer a track."""
    if left["direction"] != right["direction"]:
        return False
    left_ids, right_ids = left.get("evidence_frame_ids", []), right.get("evidence_frame_ids", [])
    if left_ids or right_ids:
        shared = bool(set(left_ids) & set(right_ids))
    else:
        shared = any(abs(a - b) < 1e-6 for a in left["evidence_frames"] for b in right["evidence_frames"])
    if not shared:
        return False
    normalize = lambda value: re.sub(r"[\s。、,.]", "", value)
    if left.get("observation_source") == "environment" or right.get("observation_source") == "environment":
        # Cross-pass matching needs a concrete object name AND a distinguishing position.
        return (left["notification_reason"] == right["notification_reason"] == "blocked_path"
                and left["temporal_status"] == right["temporal_status"] == "stationary"
                and bool(normalize(left.get("object_label", "")))
                and normalize(left.get("object_label", "")) == normalize(right.get("object_label", ""))
                and bool(normalize(left["position"]))
                and normalize(left["position"]) == normalize(right["position"]))
    return (left["object_type"] == right["object_type"]
            and left["notification_reason"] == right["notification_reason"]
            and left["temporal_status"] == right["temporal_status"]
            and normalize(left["action"]) == normalize(right["action"])
            and normalize(left["position"]) == normalize(right["position"])
            and bool(normalize(left["evidence"]))
            and normalize(left["evidence"]) == normalize(right["evidence"]))


def group_notification_candidates(candidates: list[tuple[int, dict[str, Any], bool]]) -> list[list[Any]]:
    groups: list[list[Any]] = []
    for candidate in candidates:
        event = candidate[1]
        for group in groups:
            if not all(event["observation_source"] != member[1]["observation_source"]
                       and same_notification_subject(event, member[1]) for member in group):
                continue
            # Ambiguous matches (e.g. two indistinguishable pedestrians in one chunk) stay separate.
            if any(sum(other[1]["observation_source"] == member[1]["observation_source"]
                       and same_notification_subject(event, other[1]) for other in candidates) != 1
                   or sum(other[1]["observation_source"] == event["observation_source"]
                          and same_notification_subject(member[1], other[1]) for other in candidates) != 1
                   for member in group):
                continue
            group.append(candidate)
            break
        else:
            groups.append([candidate])
    return groups


def build_final_report(entries: list[DualVideoSceneEntry], *, interrupted: bool = False,
                       environment: dict[str, Any] | None = None,
                       notification_style: str = "action_oriented") -> dict[str, Any]:
    if notification_style not in NOTIFICATION_STYLES:
        raise ValueError("通知表現が不正です")
    observations = []
    latest = None
    latest_id = None
    analyzed_latest = None
    analyzed_ids: set[str] = set()
    failed = 0
    for index, entry in enumerate(entries):
        if entry.captured_frame_sec is not None and (latest is None or entry.captured_frame_sec >= latest):
            latest, latest_id = entry.captured_frame_sec, entry.captured_frame_id
        if entry.description.startswith(("推論失敗:", "中断されたため")):
            failed += 1
            continue
        if entry.description.lstrip().startswith("{"):
            data = validate_observations(json.loads(entry.description))
            frame = data["latest_frame_sec"]
            if frame is not None:
                analyzed_latest = max(analyzed_latest if analyzed_latest is not None else 0, frame)
                if latest is None or frame > latest:
                    latest, latest_id = frame, data.get("latest_frame_id")
            analyzed_ids.update(f["frame_id"] for f in data.get("frame_timeline", []))
            for event in data["observations"]:
                observations.append({**event, "event_id": f"scene_{index}_{event['event_id']}",
                    "observation_source": f"scene_{index}/{event.get('observation_source', 'full')}"})
        else:
            # Old text logs lack temporal evidence. Preserve them without inventing state.
            for number, (direction, message) in enumerate(parse_formatted_risks(entry.description)):
                observations.append(dict(event_id=f"legacy_{index}_{number}", object_type="unknown",
                    direction=direction, action="unknown", temporal_status="unknown", first_seen=None,
                    last_seen=None, evidence_frames=[], latest_visible=False, position="",
                    evidence=message, notification_reason="none", confirmation_action=""))
    environment = environment or {"status": "not_checked", "obstacles": []}
    latest_analyzed = (latest_id in analyzed_ids if latest_id is not None else
                       latest is not None and analyzed_latest is not None and abs(latest - analyzed_latest) < 1e-6)
    if entries and entries[-1].description.startswith(("推論失敗:", "中断されたため")) and entries[-1].captured_frame_sec is None:
        latest_analyzed = False
    for index, item in enumerate(environment.get("obstacles", []) if environment.get("status") == "completed" else []):
        if item.get("latest_visible") is not True:
            continue
        sec = environment.get("latest_sec")
        observations.append(dict(event_id=f"environment_{index}", object_type="other",
            object_label=item["object"], direction=item["direction"], action="stationary",
            temporal_status="stationary", first_seen=sec, last_seen=sec,
            evidence_frames=[sec] if sec is not None else [], latest_visible=True,
            position=item.get("position", ""), evidence=item["evidence"], notification_reason="blocked_path",
            evidence_frame_ids=[environment["latest_frame_id"]] if environment.get("latest_frame_id") else [],
            observation_source="environment",
            confirmation_action="発進前に対象の位置をご確認ください。"))
    priorities = {reason: i for i, reason in enumerate(
        ("near_movement", "blocked_path", "boarding_or_stop", "contact", "peering", "change"))}
    candidates = []
    for event in observations:
        reason = event["notification_reason"]
        is_environment = event.get("observation_source") == "environment"
        reference_sec = environment.get("latest_sec") if is_environment else latest
        reference_id = environment.get("latest_frame_id") if is_environment else latest_id
        frame_matches = (reference_id in event.get("evidence_frame_ids", []) if reference_id is not None else
                         reference_sec is not None and any(abs(t - reference_sec) < 1e-6 for t in event["evidence_frames"]))
        current = ((is_environment or latest_analyzed) and frame_matches and event["latest_visible"] is True
                   and event["temporal_status"] != "departed")
        historical = reason in ("contact", "peering") and bool(event["evidence_frames"])
        if reason not in priorities or not (current or historical):
            continue
        if reason == "near_movement" and event["temporal_status"] not in ("approaching", "passing"):
            continue
        candidates.append((priorities[reason], event, current))
    candidates.sort(key=lambda x: (x[0], -(x[1]["last_seen"] or 0)))
    groups = group_notification_candidates(candidates)
    notifications = []
    selected = groups[:2]
    if len(groups) > 2 and groups[2][0][0] <= 2:
        selected.append(groups[2])
    for group in selected:
        _, event, current = group[0]
        variants = notification_variants(event, current)
        notifications.append(dict(rank=len(notifications)+1, event_id=event["event_id"],
            direction=event["direction"], message=variants[notification_style], variants=variants,
            related_event_ids=[member[1]["event_id"] for member in group],
            observed_at_sec=event["last_seen"], evidence_frame_ids=event.get("evidence_frame_ids", []),
            status="observed"))
    return dict(scope="sampled_video", interrupted=interrupted,
        analyzed_scene_count=len(entries)-failed, failed_scene_count=failed, environment=environment,
        latest_frame_sec=latest, latest_frame_id=latest_id, latest_analyzed_frame_sec=analyzed_latest,
        latest_frame_analysis_status="completed" if latest_analyzed else "unavailable",
        environment_latest_frame_sec=environment.get("latest_sec"),
        observations=observations, notifications=notifications,
        notification_style=notification_style, risks=notifications)


def write_final_report(output_dir: Path, entries: list[DualVideoSceneEntry], *, interrupted: bool = False,
                       environment: dict[str, Any] | None = None,
                       notification_style: str = "action_oriented") -> str:
    report = build_final_report(entries, interrupted=interrupted, environment=environment,
                                notification_style=notification_style)
    (output_dir / "observations.json").write_text(json.dumps(report["observations"], ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "final_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["車両に戻った際のフィードバック（最大3件）",
             f"解析済み: {report['analyzed_scene_count']}シーン / 失敗・未完了: {report['failed_scene_count']}シーン"]
    if interrupted:
        lines.append("中断までの暫定結果です。")
    if report["latest_frame_analysis_status"] != "completed":
        lines.append("最新の通行状況は確認できていません。")
    if report["environment"]["status"] != "completed":
        lines.append("開始時と最新の障害物比較は未完了です。車両周囲を直接確認してください。")
    if not report["notifications"]:
        lines.append("通知すべき情報はありません。")
    for item in report["notifications"]:
        direction = f"【{item['direction']}】" if item["direction"] else ""
        lines.append(f"{item['rank']}. {direction}{item['message']}")
        if item.get("observation_window_count"):
            lines.append(f"   通行を確認した観測時点: {item['observation_window_count']}（台数・人数ではありません）")
    description = "\n".join(lines)
    (output_dir / "final_report.txt").write_text(description + "\n", encoding="utf-8")
    (output_dir / "final_report.html").write_text(
        '<!doctype html><html lang="ja"><meta charset="utf-8"><title>車両に戻った際のフィードバック</title>'
        '<h1>車両に戻った際のフィードバック</h1><pre style="white-space:pre-wrap">'
        + html.escape(description) + '</pre></html>', encoding="utf-8")
    return description


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
                    captured_frame_id=completed.get("captured_frame_id"),
                    captured_frame_sec=completed.get("captured_frame_sec"),
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
            front_metadata: list[dict[str, Any]] = []
            back_metadata: list[dict[str, Any]] = []
            front_frames, current_front_frame_index = read_temporal_frames(
                front_cap,
                current_front_frame_index,
                position_sec,
                front_fps,
                args.sample_fps,
                args.temporal_frames,
                end_sec,
                front_metadata,
            )
            back_frames, current_back_frame_index = read_temporal_frames(
                back_cap,
                current_back_frame_index,
                position_sec,
                back_fps,
                args.sample_fps,
                args.temporal_frames,
                end_sec,
                back_metadata,
            )
            temporal_frame_count = min(len(front_frames), len(back_frames))
            if temporal_frame_count == 0:
                break

            frame_timeline = [dict(
                frame_id=combined_frame_id(front["frame_index"], back["frame_index"]),
                sec=position_sec + offset / args.sample_fps, front_sec=front["sec"], rear_sec=back["sec"])
                for offset, (front, back) in enumerate(zip(front_metadata[:temporal_frame_count],
                                                          back_metadata[:temporal_frame_count]))]

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
                captured_frame_id=frame_timeline[-1]["frame_id"],
                captured_frame_sec=frame_timeline[-1]["sec"],
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
                        frame_times=[f["sec"] for f in frame_timeline],
                        frame_timeline=frame_timeline,
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
                                frame_times=[frame_timeline[i]["sec"] for i in frame_indices],
                                frame_timeline=[frame_timeline[i] for i in frame_indices],
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
                session_entries.append(pending_entry)
                pending_entry = None
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
                captured_frame_id=frame_timeline[-1]["frame_id"],
                captured_frame_sec=frame_timeline[-1]["sec"],
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
                    "captured_frame_id": entry.captured_frame_id,
                    "captured_frame_sec": entry.captured_frame_sec,
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
            log(display_description(description), level="normal", stream=sys.stdout)
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
        environment = {"status": "not_checked", "obstacles": []}
        if not interrupted and session_entries:
            try:
                latest_entry = session_entries[-1]
                latest_sec = (latest_entry.captured_frame_sec if latest_entry.captured_frame_sec is not None else
                              max(args.start_sec, position_sec - args.frame_interval))
                log("観測開始時と最新映像の障害物を比較しています。", level="normal")
                environment = compare_environment(args, session, max(0.0, args.start_sec), latest_sec,
                                                  latest_entry.captured_frame_id)
            except KeyboardInterrupt:
                interrupted = True
                environment = {"status": "interrupted", "obstacles": []}
            except Exception as exc:
                environment = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "obstacles": []}
                log(f"環境比較に失敗: {exc}", level="quiet")
        final_description = write_final_report(args.output_dir, session_entries,
                                               interrupted=interrupted, environment=environment,
                                               notification_style=args.notification_style)
        log(final_description, level="quiet", stream=sys.stdout)
        log(f"Final report: {args.output_dir / 'final_report.html'}", level="quiet", stream=sys.stdout)
        front_cap.release()
        back_cap.release()
        session.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
