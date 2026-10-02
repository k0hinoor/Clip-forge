"""Safe FFmpeg / ffprobe execution (TRD §9, §20).

* Commands are argument arrays — never ``shell=True``.
* Every argument is validated (strings only, no NUL bytes).
* Every run has a timeout and is killed on cancellation.
* A process-wide semaphore bounds concurrent FFmpeg processes (TRD §32).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from clipforge.core.errors import AppError, ErrorCode
from clipforge.core.logging import get_logger

log = get_logger(__name__)

_ffmpeg_semaphore = threading.BoundedSemaphore(2)
_sem_lock = threading.Lock()


def configure_ffmpeg_concurrency(limit: int) -> None:
    global _ffmpeg_semaphore
    with _sem_lock:
        _ffmpeg_semaphore = threading.BoundedSemaphore(max(1, int(limit)))


class FFmpegError(Exception):
    def __init__(self, message: str, returncode: int | None = None, stderr_tail: str = "") -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr_tail = stderr_tail


def resolve_binary(name_or_path: str) -> str:
    """Resolve an executable from config; refuse anything that is not a real file."""
    candidate = shutil.which(name_or_path) if os.sep not in name_or_path else name_or_path
    if not candidate or not os.path.isfile(candidate) or not os.access(candidate, os.X_OK):
        raise AppError(ErrorCode.MODEL_UNAVAILABLE, "FFmpeg is not installed or not executable.",
                       retryable=False, internal=f"binary not found: {name_or_path}")
    return candidate


def _validate_args(args: Sequence[Any]) -> list[str]:
    out: list[str] = []
    for a in args:
        if isinstance(a, bool) or not isinstance(a, (str, int, float, Path)):
            raise ValueError(f"Invalid ffmpeg argument type: {type(a)!r}")
        s = str(a)
        if "\x00" in s or "\n" in s or "\r" in s:
            raise ValueError("Invalid character in ffmpeg argument")
        out.append(s)
    return out


def run_ffmpeg(
    args: Sequence[Any],
    *,
    ffmpeg_path: str = "ffmpeg",
    timeout: float = 3600,
    cancel_check: Callable[[], bool] | None = None,
    on_progress: Callable[[float], None] | None = None,
    duration: float | None = None,
    error_code: ErrorCode = ErrorCode.RENDER_FAILED,
) -> None:
    """Run ffmpeg with ``args`` (excluding the binary). Raises AppError on failure."""
    binary = resolve_binary(ffmpeg_path)
    cmd = [binary, "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-progress", "pipe:1",
           "-nostats", *_validate_args(args)]
    with _ffmpeg_semaphore:
        _run(cmd, timeout=timeout, cancel_check=cancel_check, on_progress=on_progress,
             duration=duration, error_code=error_code)


def _run(cmd: list[str], *, timeout: float, cancel_check, on_progress, duration, error_code) -> None:
    started = time.monotonic()
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, stdin=subprocess.DEVNULL,
                                shell=False, close_fds=True)
        last_progress = [0.0]

        def _reader() -> None:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.decode("utf-8", "replace").strip()
                if on_progress and duration and line.startswith("out_time_ms="):
                    try:
                        secs = int(line.split("=", 1)[1]) / 1_000_000
                    except ValueError:
                        continue
                    frac = min(max(secs / duration, 0.0), 1.0)
                    if frac - last_progress[0] >= 0.02:
                        last_progress[0] = frac
                        try:
                            on_progress(frac)
                        except Exception:  # progress reporting must never kill a render
                            log.debug("progress callback failed", exc_info=True)

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()
        try:
            while True:
                try:
                    rc = proc.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if cancel_check and cancel_check():
                    _kill(proc)
                    raise AppError(ErrorCode.JOB_CANCELLED)
                if time.monotonic() - started > timeout:
                    _kill(proc)
                    raise AppError(error_code, internal=f"ffmpeg timeout after {timeout}s")
        except BaseException:
            _kill(proc)
            raise
        finally:
            reader.join(timeout=2)
        if rc != 0:
            err.seek(0)
            tail = err.read()[-4000:].decode("utf-8", "replace")
            log.warning("ffmpeg failed", extra={"returncode": rc, "stderr_tail": tail[-1500:]})
            raise AppError(error_code, internal=f"ffmpeg exited {rc}: {tail[-1500:]}")


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


# --------------------------------------------------------------------- probe
@dataclass
class StreamInfo:
    index: int
    codec_type: str
    codec_name: str | None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    channels: int | None = None
    sample_rate: int | None = None
    duration: float | None = None
    rotation: int = 0


@dataclass
class MediaInfo:
    format_name: str
    duration: float
    size_bytes: int
    bit_rate: int | None
    streams: list[StreamInfo] = field(default_factory=list)

    @property
    def video(self) -> StreamInfo | None:
        return next((s for s in self.streams if s.codec_type == "video"), None)

    @property
    def audio(self) -> StreamInfo | None:
        return next((s for s in self.streams if s.codec_type == "audio"), None)

    @property
    def display_size(self) -> tuple[int, int]:
        v = self.video
        if not v or not v.width or not v.height:
            return (0, 0)
        if abs(v.rotation) in (90, 270):
            return (v.height, v.width)
        return (v.width, v.height)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["display_width"], d["display_height"] = self.display_size
        return d


def _parse_rate(rate: str | None) -> float | None:
    if not rate or rate in ("0/0", "0"):
        return None
    try:
        if "/" in rate:
            n, d = rate.split("/", 1)
            return round(float(n) / float(d), 3) if float(d) else None
        return float(rate)
    except (ValueError, ZeroDivisionError):
        return None


def _float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def probe(path: Path, *, ffprobe_path: str = "ffprobe", timeout: float = 30) -> MediaInfo:
    """Run ffprobe and return structured metadata. Raises MEDIA_CORRUPTED on failure."""
    binary = resolve_binary(ffprobe_path)
    cmd = [binary, "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
           *_validate_args([str(path)])]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=timeout, shell=False, check=False,
                             stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as exc:
        raise AppError(ErrorCode.MEDIA_CORRUPTED, internal="ffprobe timeout") from exc
    if res.returncode != 0:
        raise AppError(ErrorCode.MEDIA_CORRUPTED,
                       internal=f"ffprobe rc={res.returncode}: {res.stderr[-1000:].decode('utf-8', 'replace')}")
    try:
        data = json.loads(res.stdout or b"{}")
    except json.JSONDecodeError as exc:
        raise AppError(ErrorCode.MEDIA_CORRUPTED, internal="ffprobe returned invalid json") from exc
    fmt = data.get("format") or {}
    streams: list[StreamInfo] = []
    for s in data.get("streams") or []:
        rotation = 0
        tags = s.get("tags") or {}
        if "rotate" in tags:
            try:
                rotation = int(float(tags["rotate"]))
            except ValueError:
                rotation = 0
        for sd in s.get("side_data_list") or []:
            if "rotation" in sd:
                try:
                    rotation = int(float(sd["rotation"]))
                except (TypeError, ValueError):
                    pass
        streams.append(StreamInfo(
            index=int(s.get("index", len(streams))),
            codec_type=str(s.get("codec_type") or "unknown"),
            codec_name=s.get("codec_name"),
            width=s.get("width"),
            height=s.get("height"),
            fps=_parse_rate(s.get("avg_frame_rate")) or _parse_rate(s.get("r_frame_rate")),
            channels=s.get("channels"),
            sample_rate=int(s["sample_rate"]) if s.get("sample_rate") else None,
            duration=_float(s.get("duration")),
            rotation=rotation,
        ))
    duration = _float(fmt.get("duration")) or max((s.duration or 0.0 for s in streams), default=0.0)
    return MediaInfo(
        format_name=str(fmt.get("format_name") or "unknown"),
        duration=float(duration or 0.0),
        size_bytes=int(fmt.get("size") or (path.stat().st_size if Path(path).exists() else 0)),
        bit_rate=int(fmt["bit_rate"]) if str(fmt.get("bit_rate", "")).isdigit() else None,
        streams=streams,
    )


def ffmpeg_version(ffmpeg_path: str = "ffmpeg") -> str | None:
    try:
        binary = resolve_binary(ffmpeg_path)
        res = subprocess.run([binary, "-version"], capture_output=True, timeout=10, check=False)
        return res.stdout.decode("utf-8", "replace").splitlines()[0] if res.returncode == 0 else None
    except Exception:
        return None


def ffmpeg_has_filter(name: str, ffmpeg_path: str = "ffmpeg") -> bool:
    try:
        binary = resolve_binary(ffmpeg_path)
        res = subprocess.run([binary, "-hide_banner", "-filters"], capture_output=True, timeout=10, check=False)
        return any(line.split()[1:2] == [name] for line in res.stdout.decode("utf-8", "replace").splitlines()
                   if len(line.split()) > 1)
    except Exception:
        return False
