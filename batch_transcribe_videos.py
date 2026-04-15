"""
Batch-transcribe video files with Gemini.

- Waits until each uploaded file is ACTIVE before calling the model.
- Retries transient API errors (429, 5xx, etc.) via tenacity.
- Default two-thread pipeline: one thread prepares the next file while the other transcribes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai.types import File, FileState
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_INPUT_DIR = Path("data")

_RETRIABLE_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})

_SENTINEL = object()


@dataclass(frozen=True)
class VideoInfo:
    duration_sec: float
    bitrate_bps: int


def _collect_videos(input_dir: Path, extensions: frozenset[str]) -> list[Path]:
    files: list[Path] = []
    for ext in extensions:
        files.extend(input_dir.rglob(f"*{ext}"))
    return sorted({p.resolve() for p in files if p.is_file()})


def _build_prompt(language: str | None) -> str:
    lang_clause = (
        f"The audio is in {language}. "
        if language
        else "Detect the spoken language automatically. "
    )
    return (
        "Transcribe this video accurately. "
        f"{lang_clause}"
        "Provide ONLY the plain text transcription. "
        "Write it with speaker level information with proper punctuation. "
        "Do NOT include timestamps, headers, titles, or extra formatting. "
        "Output the raw transcribed text as a conversation with speaker names (no timestamps)"
    )


def _is_retriable_api_error(exc: BaseException) -> bool:
    return isinstance(exc, genai_errors.APIError) and exc.code in _RETRIABLE_HTTP_CODES


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return (p.stdout or "").strip()


def probe_video_info(path: Path, ffprobe: str) -> VideoInfo:
    out = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    data = json.loads(out)
    fmt = (data.get("format") or {}) if isinstance(data, dict) else {}

    duration = float(fmt.get("duration") or 0.0)
    if duration <= 0:
        raise ValueError(f"Could not determine duration for {path}")

    bit_rate = fmt.get("bit_rate")
    if bit_rate:
        bitrate_bps = int(bit_rate)
    else:
        size_bytes = path.stat().st_size
        bitrate_bps = int((size_bytes * 8) / duration)

    return VideoInfo(duration_sec=duration, bitrate_bps=bitrate_bps)


def split_video_by_max_bytes(
    input_path: Path,
    output_dir: Path,
    *,
    ffmpeg: str,
    ffprobe: str,
    max_bytes: int,
    safety_ratio: float = 0.90,
    min_segment_sec: int = 10,
    overwrite: bool = False,
) -> list[Path]:
    """
    If input is <= max_bytes, returns [input_path].
    Else, uses ffmpeg segment muxer (-c copy) to create output_dir/<stem>__partNNN.<ext>.

    Note: segment boundaries are keyframe-aligned; sizes are approximate. The safety_ratio
    helps keep most parts under max_bytes.
    """
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    size_bytes = input_path.stat().st_size
    if size_bytes <= max_bytes:
        return [input_path]

    info = probe_video_info(input_path, ffprobe)
    target_bytes = int(max_bytes * safety_ratio)
    seg_sec = int((target_bytes * 8) / max(1, info.bitrate_bps))
    seg_sec = max(min_segment_sec, seg_sec)

    if seg_sec >= int(math.ceil(info.duration_sec)):
        return [input_path]

    out_pattern = output_dir / f"{input_path.stem}__part%03d{input_path.suffix}"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-c",
        "copy",
        "-f",
        "segment",
        "-segment_time",
        str(seg_sec),
        "-reset_timestamps",
        "1",
        str(out_pattern),
    ]
    cmd.insert(1, "-y" if overwrite else "-n")
    subprocess.run(cmd, check=True)

    parts = sorted(output_dir.glob(f"{input_path.stem}__part*{input_path.suffix}"))
    if not parts:
        raise RuntimeError(f"ffmpeg produced no segments for {input_path}")
    return parts


def wait_until_file_active(
    client: genai.Client,
    file_name: str,
    *,
    timeout_sec: float,
    poll_sec: float,
) -> File:
    """Poll files.get until the file is ACTIVE or FAILED."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            meta = client.files.get(name=file_name)
        except genai_errors.APIError as e:
            if _is_retriable_api_error(e):
                time.sleep(poll_sec)
                continue
            raise
        state = meta.state
        if state == FileState.ACTIVE:
            return meta
        if state == FileState.FAILED:
            detail = getattr(meta, "error", None)
            raise RuntimeError(f"File processing failed for {file_name}: {detail!r}")
        time.sleep(poll_sec)
    raise TimeoutError(f"File {file_name} did not become ACTIVE within {timeout_sec}s")


@dataclass
class PreparedVideo:
    index: int
    total: int
    path: Path
    file: File


def _print_job_header(video_path: Path, index: int, total: int, input_dir: Path) -> None:
    rel = video_path.relative_to(input_dir) if video_path.is_relative_to(input_dir) else video_path.name
    print(f"\n{'=' * 60}\n[{index}/{total}] {rel}\n{'=' * 60}")


def transcribe_videos() -> None:
    parser = argparse.ArgumentParser(description="Transcribe videos in a folder using Gemini.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_DEFAULT_INPUT_DIR,
        help=f"Directory to scan for video files (default: {_DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for .txt transcripts (default: next to each video)",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="Gemini model id (default: gemini-2.5-pro)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Spoken language hint, e.g. English (optional)",
    )
    parser.add_argument(
        "--extensions",
        default=".mp4,.mkv,.mov,.webm,.avi,.m4v,.wav",
        help="Comma-separated extensions to include (default: common video types)",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=0,
        help="If >0, split any video larger than this many bytes using ffmpeg before upload",
    )
    parser.add_argument(
        "--max-mb",
        type=float,
        default=0.0,
        help="If >0, split any video larger than this many MB using ffmpeg before upload",
    )
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=None,
        help="Directory to write generated size-based chunks (default: <input-dir>/.chunks)",
    )
    parser.add_argument(
        "--keep-chunks",
        action="store_true",
        help="Keep generated chunk files on disk (default: delete after each chunk is transcribed)",
    )
    parser.add_argument(
        "--ffmpeg",
        default=None,
        help="Path to ffmpeg binary (default: use ffmpeg on PATH)",
    )
    parser.add_argument(
        "--ffprobe",
        default=None,
        help="Path to ffprobe binary (default: use ffprobe on PATH)",
    )
    parser.add_argument(
        "--active-timeout",
        type=float,
        default=900.0,
        help="Max seconds to wait for each uploaded file to become ACTIVE (default: 900)",
    )
    parser.add_argument(
        "--active-poll",
        type=float,
        default=2.0,
        help="Seconds between ACTIVE state polls (default: 2)",
    )
    parser.add_argument(
        "--api-retries",
        type=int,
        default=5,
        help="Max attempts for retriable API errors on upload/generate (default: 5)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Disable pipeline: upload, wait, transcribe one file at a time",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.strip() == "" or api_key == "your_gemini_api_key_here":
        print("Set GEMINI_API_KEY in the environment or a .env file.", file=sys.stderr)
        sys.exit(1)

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    exts = frozenset(
        e.strip().lower() if e.strip().lower().startswith(".") else f".{e.strip().lower()}"
        for e in args.extensions.split(",")
        if e.strip()
    )
    videos = _collect_videos(input_dir, exts)
    if not videos:
        print(f"No video files found under {input_dir} with extensions {sorted(exts)}")
        return

    # Optional: expand oversized videos into smaller chunks on disk before upload.
    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    ffprobe = args.ffprobe or shutil.which("ffprobe")
    generated_chunks: set[Path] = set()

    max_bytes = int(args.max_bytes) if args.max_bytes and args.max_bytes > 0 else 0
    if max_bytes <= 0 and args.max_mb and args.max_mb > 0:
        max_bytes = int(args.max_mb * 1024 * 1024)

    if max_bytes > 0:
        if not ffmpeg or not ffprobe:
            print(
                "Size-based chunking requires ffmpeg and ffprobe on PATH (or pass --ffmpeg/--ffprobe).",
                file=sys.stderr,
            )
            sys.exit(2)

        chunk_dir = (args.chunk_dir.resolve() if args.chunk_dir else (input_dir / ".chunks").resolve())
        chunk_dir.mkdir(parents=True, exist_ok=True)

        expanded: list[Path] = []
        for v in videos:
            try:
                parts = split_video_by_max_bytes(
                    v,
                    chunk_dir,
                    ffmpeg=str(ffmpeg),
                    ffprobe=str(ffprobe),
                    max_bytes=max_bytes,
                    overwrite=False,
                )
                expanded.extend(parts)
                for p in parts:
                    if p.resolve() != v.resolve():
                        generated_chunks.add(p.resolve())
            except Exception as e:
                print(f"Chunking failed for {v}: {e}", file=sys.stderr)
        videos = expanded

        if not videos:
            print("No videos available after chunking.", file=sys.stderr)
            return

    prompt = _build_prompt(args.language)
    output_dir = args.output_dir.resolve() if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    api_attempts = max(1, args.api_retries)
    retry_api = retry(
        stop=stop_after_attempt(api_attempts),
        retry=retry_if_exception(_is_retriable_api_error),
        wait=wait_exponential_jitter(initial=1, max=60),
        reraise=True,
    )

    def make_generate(client: genai.Client) -> Callable[[File], str]:
        @retry_api
        def generate_transcript(ready: File) -> str:
            response = client.models.generate_content(
                model=args.model,
                contents=[ready, prompt],
            )
            text = (response.text or "").strip()
            if not text:
                raise RuntimeError("Empty transcription response")
            return text

        return generate_transcript

    def out_path_for(video_path: Path) -> Path:
        if output_dir:
            return output_dir / f"{video_path.stem}.txt"
        return video_path.with_suffix(".txt")

    def maybe_delete_chunk(path: Path) -> None:
        if args.keep_chunks:
            return
        try:
            rp = path.resolve()
        except Exception:
            return
        if rp in generated_chunks:
            try:
                rp.unlink(missing_ok=True)
            except Exception:
                pass

    if args.sequential:
        client = genai.Client(api_key=api_key)
        generate_transcript = make_generate(client)

        @retry_api
        def upload_file(path: Path) -> File:
            return client.files.upload(file=str(path))

        for i, video_path in enumerate(videos, 1):
            _print_job_header(video_path, i, len(videos), input_dir)
            out_path = out_path_for(video_path)
            uploaded: File | None = None
            try:
                start = time.perf_counter()
                uploaded = upload_file(video_path)
                ready = wait_until_file_active(
                    client,
                    uploaded.name,
                    timeout_sec=args.active_timeout,
                    poll_sec=args.active_poll,
                )
                transcript = generate_transcript(ready)
                elapsed = time.perf_counter() - start
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(transcript, encoding="utf-8")
                print(f"  OK in {elapsed:.1f}s -> {out_path}")
            except Exception as e:
                print(f"  FAILED: {e}", file=sys.stderr)
            finally:
                if uploaded is not None:
                    try:
                        client.files.delete(name=uploaded.name)
                    except Exception:
                        pass
                maybe_delete_chunk(video_path)
        return

    prepare_client = genai.Client(api_key=api_key)
    transcribe_client = genai.Client(api_key=api_key)
    generate_transcript = make_generate(transcribe_client)
    prepared: queue.Queue[PreparedVideo | object] = queue.Queue(maxsize=1)

    @retry_api
    def upload_file(path: Path) -> File:
        return prepare_client.files.upload(file=str(path))

    def producer() -> None:
        try:
            for i, video_path in enumerate(videos, 1):
                uploaded: File | None = None
                try:
                    uploaded = upload_file(video_path)
                    ready = wait_until_file_active(
                        prepare_client,
                        uploaded.name,
                        timeout_sec=args.active_timeout,
                        poll_sec=args.active_poll,
                    )
                    prepared.put(PreparedVideo(i, len(videos), video_path, ready))
                except Exception as e:
                    _print_job_header(video_path, i, len(videos), input_dir)
                    print(f"  PREP FAILED: {e}", file=sys.stderr)
                    if uploaded is not None:
                        try:
                            prepare_client.files.delete(name=uploaded.name)
                        except Exception:
                            pass
                    maybe_delete_chunk(video_path)
        finally:
            prepared.put(_SENTINEL)

    threading.Thread(target=producer, name="video-prepare", daemon=False).start()

    while True:
        item = prepared.get()
        if item is _SENTINEL:
            break
        assert isinstance(item, PreparedVideo)
        _print_job_header(item.path, item.index, item.total, input_dir)
        out_path = out_path_for(item.path)
        try:
            start = time.perf_counter()
            transcript = generate_transcript(item.file)
            elapsed = time.perf_counter() - start
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(transcript, encoding="utf-8")
            print(f"  OK in {elapsed:.1f}s -> {out_path}")
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
        finally:
            try:
                transcribe_client.files.delete(name=item.file.name)
            except Exception:
                pass
            maybe_delete_chunk(item.path)


if __name__ == "__main__":
    transcribe_videos()
