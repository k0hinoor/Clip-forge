"""FFmpeg / ffprobe primitives.

Only low-level media operations live here: probing, audio extraction, silence
detection, frame sampling and encoding-argument construction. Clip composition
lives in :mod:`clipforge.media.compose`, capture in :mod:`clipforge.media.download`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ..config import AppSettings, get_settings
from ..errors import ClipForgeError, ErrorCode
from ..logging_setup import get_logger
from ..system import FfmpegInfo, ffmpeg_info
from .runner import probe_command, run_command

log = get_logger("clipforge.render")


@dataclass
class MediaInfo:
    path: str
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    video_codec: str = ""
    audio_codec: str = ""
    audio_channels: int = 0
    audio_sample_rate: int = 0
    bitrate: int = 0
    rotation: int = 0
    container: str = ""
    filesize: int = 0
    raw: dict[str, Any] | None = None

    @property
    def is_portrait(self) -> bool:
        return self.height > self.width

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "duration": round(self.duration, 3),
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "has_video": self.has_video,
            "has_audio": self.has_audio,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "audio_channels": self.audio_channels,
            "audio_sample_rate": self.audio_sample_rate,
            "bitrate": self.bitrate,
            "rotation": self.rotation,
            "container": self.container,
            "filesize": self.filesize,
        }


# --------------------------------------------------------------------------- #
# Binary discovery
# --------------------------------------------------------------------------- #


def require_ffmpeg() -> FfmpegInfo:
    info = ffmpeg_info()
    if not info.available:
        raise ClipForgeError(
            code=ErrorCode.FFMPEG_NOT_FOUND,
            message="ffmpeg is not available, so CLIPFORGE cannot process video.",
            hint=(
                "Install ffmpeg and set its path in Settings -> Video. On Windows: "
                "winget install Gyan.FFmpeg. Alternatively run "
                'pip install "clipforge[bundled-ffmpeg]" and click Re-detect.'
            ),
            status_code=503,
        )
    return info


def ffmpeg_bin() -> str:
    return require_ffmpeg().ffmpeg


def ffprobe_bin() -> str:
    return require_ffmpeg().ffprobe


def ffmpeg_major_version() -> int:
    version = require_ffmpeg().version
    match = re.match(r"(\d+)", version or "")
    return int(match.group(1)) if match else 0


# --------------------------------------------------------------------------- #
# Probing
# --------------------------------------------------------------------------- #


def probe_media(path: str | Path) -> MediaInfo:
    """Read stream metadata using ffprobe, falling back to parsing ffmpeg output."""
    file_path = Path(path)
    if not file_path.exists():
        raise ClipForgeError(
            code=ErrorCode.NOT_FOUND,
            message=f"Media file missing: {file_path.name}",
            hint="The file may have been deleted or moved. Re-run the analysis for this project.",
            status_code=404,
        )

    info = MediaInfo(path=str(file_path), filesize=file_path.stat().st_size)
    require_ffmpeg()

    ffprobe = ffprobe_bin()
    if ffprobe:
        result = probe_command(
            [
                ffprobe, "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", str(file_path),
            ],
            timeout=60,
            label="ffprobe",
        )
        if result.ok and result.stdout.strip():
            try:
                return _parse_ffprobe(json.loads(result.stdout), info)
            except json.JSONDecodeError:
                log.warning("ffprobe returned unparseable JSON for %s", file_path.name)

    return _probe_with_ffmpeg(file_path, info)


def _parse_ffprobe(payload: dict[str, Any], info: MediaInfo) -> MediaInfo:
    fmt = payload.get("format") or {}
    info.duration = float(fmt.get("duration") or 0.0)
    info.bitrate = int(float(fmt.get("bit_rate") or 0))
    info.container = (fmt.get("format_name") or "").split(",")[0]
    for stream in payload.get("streams") or []:
        kind = stream.get("codec_type")
        if kind == "video" and not info.has_video:
            # Skip cover art / thumbnails pretending to be video streams.
            if stream.get("disposition", {}).get("attached_pic"):
                continue
            info.has_video = True
            info.video_codec = stream.get("codec_name", "")
            info.width = int(stream.get("width") or 0)
            info.height = int(stream.get("height") or 0)
            info.fps = _parse_fraction(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/0")
            rotation = 0
            for side in stream.get("side_data_list") or []:
                if "rotation" in side:
                    try:
                        rotation = int(float(side["rotation"])) % 360
                    except (TypeError, ValueError):
                        rotation = 0
            if rotation in (90, 270):
                info.width, info.height = info.height, info.width
            info.rotation = rotation
            if not info.duration and stream.get("duration"):
                info.duration = float(stream["duration"])
        elif kind == "audio" and not info.has_audio:
            info.has_audio = True
            info.audio_codec = stream.get("codec_name", "")
            info.audio_channels = int(stream.get("channels") or 0)
            info.audio_sample_rate = int(stream.get("sample_rate") or 0)
    return info


def _parse_fraction(value: str) -> float:
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            denominator_f = float(denominator)
            return float(numerator) / denominator_f if denominator_f else 0.0
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _probe_with_ffmpeg(file_path: Path, info: MediaInfo) -> MediaInfo:
    """Fallback for installs without ffprobe: parse ``ffmpeg -i`` output."""
    result = probe_command([ffmpeg_bin(), "-hide_banner", "-i", str(file_path)], timeout=60, label="ffmpeg-probe")
    text = (result.stderr or "") + (result.stdout or "")

    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", text)
    if match:
        hours, minutes, seconds = match.groups()
        info.duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    match = re.search(r"bitrate:\s*(\d+)\s*kb/s", text)
    if match:
        info.bitrate = int(match.group(1)) * 1000
    match = re.search(r"Video:\s*([a-zA-Z0-9_]+).*?(\d{2,5})x(\d{2,5})", text)
    if match:
        info.has_video = True
        info.video_codec = match.group(1)
        info.width = int(match.group(2))
        info.height = int(match.group(3))
    match = re.search(r"(\d+(?:\.\d+)?)\s*fps", text)
    if match:
        info.fps = float(match.group(1))
    match = re.search(r"Audio:\s*([a-zA-Z0-9_]+).*?(\d+)\s*Hz,\s*([a-z0-9.() ]+)", text)
    if match:
        info.has_audio = True
        info.audio_codec = match.group(1)
        info.audio_sample_rate = int(match.group(2))
        channel_text = match.group(3)
        info.audio_channels = 1 if "mono" in channel_text else 2
    if not info.has_video and not info.has_audio:
        raise ClipForgeError(
            code=ErrorCode.UNSUPPORTED_FORMAT,
            message=f"Could not read any streams from {file_path.name}.",
            hint="The file may be corrupt or use an unsupported container. Re-download the source.",
            detail=text[-2000:],
            status_code=422,
        )
    return info


# --------------------------------------------------------------------------- #
# Progress plumbing
# --------------------------------------------------------------------------- #


class FfmpegProgress:
    """Parses ``-progress pipe:1`` output into a 0..1 fraction."""

    def __init__(self, total_seconds: float, callback: Callable[[float], None] | None = None) -> None:
        self.total = max(total_seconds, 0.001)
        self.callback = callback
        self.last = 0.0

    def __call__(self, line: str) -> None:
        match = re.match(r"out_time_ms=(\d+)", line) or re.match(r"out_time_us=(\d+)", line)
        if not match:
            return
        seconds = int(match.group(1)) / 1_000_000
        fraction = max(0.0, min(1.0, seconds / self.total))
        if self.callback and (fraction - self.last > 0.005 or fraction >= 1.0):
            self.last = fraction
            self.callback(fraction)


def progress_args() -> list[str]:
    return ["-progress", "pipe:1", "-nostats"]


# --------------------------------------------------------------------------- #
# Encoder selection
# --------------------------------------------------------------------------- #


def video_encoder_args(settings: AppSettings | None = None, *, fps: int | None = None) -> list[str]:
    """Return ``-c:v ...`` arguments honouring the hardware-acceleration setting."""
    settings = settings or get_settings()
    info = require_ffmpeg()
    accel = settings.hw_accel
    if accel == "auto":
        if "h264_nvenc" in info.encoders:
            accel = "nvenc"
        elif "h264_qsv" in info.encoders:
            accel = "qsv"
        elif "h264_amf" in info.encoders and False:  # AMF quality is inconsistent; opt-in only
            accel = "amf"
        else:
            accel = "none"

    encoder = info.encoder_for(accel)
    args: list[str] = ["-c:v", encoder]
    keyint = str((fps or settings.output_fps) * 2)

    if encoder == "h264_nvenc":
        args += ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "21", "-b:v", "0", "-pix_fmt", "yuv420p"]
    elif encoder == "h264_qsv":
        args += ["-preset", "medium", "-global_quality", "22", "-look_ahead", "1", "-pix_fmt", "nv12"]
    elif encoder == "h264_amf":
        args += ["-quality", "balanced", "-rc", "cqp", "-qp_i", "21", "-qp_p", "23", "-pix_fmt", "yuv420p"]
    elif encoder == "h264_videotoolbox":
        args += ["-b:v", "10M", "-pix_fmt", "yuv420p", "-profile:v", "high"]
    elif encoder == "libx264":
        args += ["-preset", settings.render_preset, "-crf", str(settings.crf), "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1"]
    else:
        args += ["-pix_fmt", "yuv420p"]

    if settings.video_bitrate_kbps:
        args += ["-b:v", f"{settings.video_bitrate_kbps}k", "-maxrate", f"{int(settings.video_bitrate_kbps * 1.5)}k", "-bufsize", f"{settings.video_bitrate_kbps * 3}k"]

    args += ["-g", keyint, "-keyint_min", str(max(1, (fps or settings.output_fps))), "-sc_threshold", "0"]
    return args


def audio_encoder_args(settings: AppSettings | None = None) -> list[str]:
    settings = settings or get_settings()
    return ["-c:a", "aac", "-b:a", f"{settings.audio_bitrate_kbps}k", "-ar", "48000", "-ac", "2"]


def output_mux_args() -> list[str]:
    return ["-movflags", "+faststart", "-movflags", "+use_metadata_tags"]


def fps_args(fps: int) -> list[str]:
    args = ["-r", str(fps)]
    if ffmpeg_major_version() >= 5:
        args += ["-fps_mode", "cfr"]
    return args


# --------------------------------------------------------------------------- #
# Audio extraction
# --------------------------------------------------------------------------- #

AUDIO_FILTERS = {
    "loudnorm": "loudnorm=I=-16:TP=-1.5:LRA=11",
    "dynaudnorm": "dynaudnorm=f=200:g=15",
    "denoise": "afftdn=nf=-25",
    "highpass": "highpass=f=70",
    "voice": "equalizer=f=250:t=q:w=1:g=-2,equalizer=f=3500:t=q:w=2:g=3",
}


def extract_audio(
    source: str | Path,
    destination: str | Path,
    *,
    sample_rate: int = 16000,
    mono: bool = True,
    filters: Sequence[str] = (),
    start: float | None = None,
    duration: float | None = None,
    cancel_key: str = "",
) -> Path:
    """Extract a speech-optimised WAV (default 16 kHz mono) for transcription."""
    out = Path(destination)
    out.parent.mkdir(parents=True, exist_ok=True)
    args: list[str] = [ffmpeg_bin(), "-hide_banner", "-y", "-nostdin"]
    if start is not None:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", str(source)]
    if duration is not None:
        args += ["-t", f"{duration:.3f}"]
    if filters:
        args += ["-af", ",".join(filters)]
    args += ["-vn", "-sn", "-dn", "-ac", "1" if mono else "2", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(out)]
    run_command(args, label="extract-audio", cancel_key=cancel_key, timeout=3600)
    return out


def has_audio_stream(path: str | Path) -> bool:
    try:
        return probe_media(path).has_audio
    except ClipForgeError:
        return False


# --------------------------------------------------------------------------- #
# Silence detection & loudness
# --------------------------------------------------------------------------- #


def detect_silence(
    path: str | Path,
    *,
    noise_db: float = -32.0,
    min_duration: float = 0.35,
    start: float | None = None,
    end: float | None = None,
) -> list[tuple[float, float]]:
    """Return ``(start, end)`` silence intervals using ffmpeg's silencedetect."""
    args: list[str] = [ffmpeg_bin(), "-hide_banner", "-nostdin"]
    if start:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", str(path)]
    if end is not None:
        duration = end - (start or 0.0)
        args += ["-t", f"{max(duration, 0.1):.3f}"]
    args += ["-vn", "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}", "-f", "null", "-"]

    result = probe_command(args, timeout=1800, label="silencedetect")
    text = (result.stderr or "") + (result.stdout or "")
    offset = start or 0.0
    intervals: list[tuple[float, float]] = []
    pending: float | None = None
    for line in text.splitlines():
        match = re.search(r"silence_start:\s*(-?\d+(?:\.\d+)?)", line)
        if match:
            pending = float(match.group(1)) + offset
            continue
        match = re.search(r"silence_end:\s*(-?\d+(?:\.\d+)?)", line)
        if match and pending is not None:
            intervals.append((max(pending, offset), float(match.group(1)) + offset))
            pending = None
    if pending is not None:
        intervals.append((pending, (end or 0.0) or pending))
    return intervals


def measure_loudness(path: str | Path, *, start: float = 0.0, duration: float | None = None) -> dict[str, float]:
    """Run the loudnorm analysis pass and return measured I/TP/LRA values."""
    args: list[str] = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-ss", f"{max(start, 0):.3f}", "-i", str(path)]
    if duration:
        args += ["-t", f"{duration:.3f}"]
    args += ["-vn", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"]
    result = probe_command(args, timeout=900, label="loudnorm-measure")
    text = (result.stderr or "") + (result.stdout or "")
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    out: dict[str, float] = {}
    for key, value in payload.items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def loudnorm_filter(settings: AppSettings, measured: dict[str, float] | None = None) -> str:
    target_i = settings.target_lufs
    target_tp = settings.true_peak_db
    if measured and measured.get("input_i", -70) > -70:
        return (
            f"loudnorm=I={target_i}:TP={target_tp}:LRA=11:"
            f"measured_I={measured.get('input_i')}:measured_TP={measured.get('input_tp')}:"
            f"measured_LRA={measured.get('input_lra')}:measured_thresh={measured.get('input_thresh')}:"
            f"offset={measured.get('target_offset', 0)}:linear=true"
        )
    return f"loudnorm=I={target_i}:TP={target_tp}:LRA=11"


def audio_filter_chain(settings: AppSettings, *, measured: dict[str, float] | None = None, source: str = "voice") -> str:
    """Voice-first audio chain: cleanup, optional denoise, gain, loudness normalisation."""
    parts: list[str] = ["highpass=f=70", "lowpass=f=16000"]
    if settings.noise_reduction:
        parts.append(AUDIO_FILTERS["denoise"])
    if settings.voice_boost:
        parts.append(AUDIO_FILTERS["voice"])
    if settings.voice_gain_db:
        parts.append(f"volume={settings.voice_gain_db}dB")
    parts.append("acompressor=threshold=-18dB:ratio=3:attack=8:release=180:makeup=2")
    if settings.normalize_loudness:
        parts.append(loudnorm_filter(settings, measured))
    if source == "music" or source == "gameplay":
        volume = settings.music_volume if source == "music" else settings.gameplay_volume
        return f"volume={max(0.0, min(1.5, volume)):.3f}"
    return ",".join(parts)


# --------------------------------------------------------------------------- #
# Frames, thumbnails, segments
# --------------------------------------------------------------------------- #


def grab_frames(
    source: str | Path,
    times: Sequence[float],
    destination: Path,
    *,
    width: int = 320,
    quality: int = 3,
) -> list[tuple[float, Path]]:
    """Grab single frames at the given timestamps (used for face/motion analysis)."""
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[tuple[float, Path]] = []
    for timestamp in times:
        target = destination / f"frame_{int(timestamp * 1000):010d}.jpg"
        if target.exists():
            outputs.append((timestamp, target))
            continue
        args = [
            ffmpeg_bin(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
            "-ss", f"{max(timestamp, 0):.3f}", "-i", str(source),
            "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", str(quality), str(target),
        ]
        result = probe_command(args, timeout=120, label="grab-frame")
        if result.ok and target.exists() and target.stat().st_size > 0:
            outputs.append((timestamp, target))
    return outputs


def make_thumbnail(source: str | Path, destination: str | Path, *, time: float = 0.0, width: int = 720) -> Path | None:
    out = Path(destination)
    out.parent.mkdir(parents=True, exist_ok=True)
    args = [
        ffmpeg_bin(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-ss", f"{max(time, 0):.3f}", "-i", str(source),
        "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3", str(out),
    ]
    result = probe_command(args, timeout=120, label="thumbnail")
    return out if result.ok and out.exists() and out.stat().st_size > 0 else None


def extract_segment(
    source: str | Path,
    destination: str | Path,
    start: float,
    end: float,
    *,
    reencode: bool = False,
    cancel_key: str = "",
) -> Path:
    """Cut ``[start, end)`` out of the source.

    Stream-copies by default (near-instant); re-encodes only when the caller
    genuinely needs frame accuracy.
    """
    out = Path(destination)
    out.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.05, end - start)
    args: list[str] = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-y"]
    args += ["-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}"]
    if reencode:
        args += video_encoder_args() + audio_encoder_args() + fps_args(get_settings().output_fps)
    else:
        args += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    args += ["-map_metadata", "-1", str(out)]
    run_command(args, label="cut-segment", cancel_key=cancel_key, timeout=1800)
    return out


def still_image_or_video(path: str | Path) -> bool:
    """True when the asset is a video (has a video stream) rather than an image."""
    try:
        return probe_media(path).has_video
    except ClipForgeError:
        return False


def normalize_background(
    source: str | Path,
    destination: str | Path,
    *,
    width: int,
    height: int,
    fps: int,
    loop: bool = True,
    cancel_key: str = "",
) -> Path:
    """Scale/crop an asset to fill the target frame, optionally looping it."""
    out = Path(destination)
    out.parent.mkdir(parents=True, exist_ok=True)
    args = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-y"]
    info = probe_media(source)
    if info.duration and info.duration < 0.2:
        loop = False
    args += ["-stream_loop", "-1"] if loop else []
    args += ["-i", str(source)]
    args += ["-an", "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},fps={fps}"]
    args += video_encoder_args(None, fps=fps) + fps_args(fps) + ["-t", "0.1"]
    args += ["-map_metadata", "-1", str(out)]
    run_command(args, label="normalize-background", cancel_key=cancel_key, timeout=1800)
    return out


__all__ = [
    "FfmpegProgress",
    "MediaInfo",
    "audio_encoder_args",
    "audio_filter_chain",
    "detect_silence",
    "extract_audio",
    "extract_segment",
    "ffmpeg_bin",
    "ffmpeg_major_version",
    "ffprobe_bin",
    "fps_args",
    "grab_frames",
    "has_audio_stream",
    "loudnorm_filter",
    "make_thumbnail",
    "measure_loudness",
    "output_mux_args",
    "probe_media",
    "progress_args",
    "require_ffmpeg",
    "video_encoder_args",
]
