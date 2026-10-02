"""FFmpeg command builders (TRD §20, §21).

All arguments are generated from validated internal parameters — never raw
user input. Codecs, presets and filters are whitelisted; numbers are checked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from clipforge.media.framing import piecewise_expr
from clipforge.security.files import assert_safe_media_path

VIDEO_CODECS = {"h264": "libx264", "libx264": "libx264"}
AUDIO_CODECS = {"aac": "aac"}
X264_PRESETS = {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"}


def _num(value: Any, lo: float, hi: float, name: str) -> float:
    v = float(value)
    if not (lo <= v <= hi):
        raise ValueError(f"{name}={v} outside [{lo}, {hi}]")
    return v


def _int(value: Any, lo: int, hi: int, name: str) -> int:
    v = int(value)
    if not (lo <= v <= hi) or v != float(value):
        raise ValueError(f"{name}={value} outside [{lo}, {hi}]")
    return v


def video_filter_chain(plan: dict[str, Any], *, fps: float | None) -> tuple[str, str]:
    """Return (filter_complex_prefix, output_label) for framing."""
    out_w = _int(plan["out_w"], 16, 4096, "out_w")
    out_h = _int(plan["out_h"], 16, 4096, "out_h")
    if out_w % 2 or out_h % 2:
        raise ValueError("Output dimensions must be even")
    fps_filter = f",fps={_num(fps, 1, 120, 'fps'):.3f}" if fps else ""
    if plan["mode"] == "fit":
        graph = (
            f"[0:v]split=2[cfbg][cffg];"
            f"[cfbg]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h},"
            f"boxblur=20:2[cfbgb];"
            f"[cffg]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease[cffgs];"
            f"[cfbgb][cffgs]overlay=(W-w)/2:(H-h)/2,setsar=1{fps_filter},format=yuv420p[cfv]"
        )
        return graph, "cfv"
    src_w = _int(plan["src_w"], 16, 16384, "src_w")
    src_h = _int(plan["src_h"], 16, 16384, "src_h")
    crop_w = _int(plan["crop_w"], 2, src_w, "crop_w")
    crop_h = _int(plan["crop_h"], 2, src_h, "crop_h")
    keyframes = plan.get("keyframes") or [{"t": 0, "x": (src_w - crop_w) // 2, "y": (src_h - crop_h) // 2}]
    for k in keyframes:
        _num(k["t"], 0, 1e6, "t")
        _num(k["x"], 0, src_w - crop_w, "x")
        _num(k["y"], 0, src_h - crop_h, "y")
    x_expr = piecewise_expr(keyframes, "x")
    y_expr = piecewise_expr(keyframes, "y")
    graph = (
        f"[0:v]crop=w={crop_w}:h={crop_h}:x={x_expr}:y={y_expr},"
        f"scale={out_w}:{out_h}:flags=lanczos,setsar=1{fps_filter},format=yuv420p[cfv]"
    )
    return graph, "cfv"


def build_render_args(
    *,
    source: Path,
    output: Path,
    start: float,
    duration: float,
    plan: dict[str, Any],
    profile: dict[str, Any],
    has_audio: bool,
    subtitles: Path | None = None,
    fonts_dir: Path | None = None,
) -> list[str]:
    src = assert_safe_media_path(source)
    out = assert_safe_media_path(output)
    start = _num(start, 0, 1e6, "start")
    duration = _num(duration, 0.5, 600, "duration")
    vcodec = VIDEO_CODECS.get(str(profile.get("video_codec", "h264")))
    acodec = AUDIO_CODECS.get(str(profile.get("audio_codec", "aac")))
    if not vcodec or not acodec:
        raise ValueError("Codec not allowed")
    preset = str(profile.get("preset", "veryfast"))
    if preset not in X264_PRESETS:
        raise ValueError("x264 preset not allowed")
    crf = _int(profile.get("crf", 20), 0, 51, "crf")
    abr = _int(profile.get("audio_bitrate_kbps", 160), 32, 512, "audio_bitrate_kbps")
    fps = profile.get("fps")

    graph, label = video_filter_chain(plan, fps=fps)
    if subtitles is not None:
        sub = assert_safe_media_path(subtitles)
        sub_filter = f"subtitles=filename={sub}"
        if fonts_dir is not None:
            sub_filter += f":fontsdir={assert_safe_media_path(fonts_dir)}"
        graph += f";[{label}]{sub_filter}[cfsub]"
        label = "cfsub"

    args: list[str] = ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(src)]
    if not has_audio:
        args += ["-f", "lavfi", "-t", f"{duration:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
    args += ["-filter_complex", graph, "-map", f"[{label}]", "-map", "0:a:0" if has_audio else "1:a:0"]
    args += [
        "-c:v", vcodec, "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-profile:v", "high", "-movflags", "+faststart",
        "-c:a", acodec, "-b:a", f"{abr}k", "-ar", "48000", "-ac", "2",
        "-t", f"{duration:.3f}", "-max_muxing_queue_size", "1024",
        str(out),
    ]
    return args


def build_audio_extract_args(source: Path, output: Path) -> list[str]:
    return ["-i", str(assert_safe_media_path(source)), "-vn", "-map", "0:a:0", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(assert_safe_media_path(output))]


def build_audio_levels_args(audio: Path, levels_file: Path) -> list[str]:
    lf = assert_safe_media_path(levels_file)
    return ["-i", str(assert_safe_media_path(audio)), "-af",
            f"asetnsamples=n=16000,astats=metadata=1:reset=1,"
            f"ametadata=print:key=lavfi.astats.Overall.RMS_level:file={lf}",
            "-f", "null", "-"]


def build_thumbnail_args(video: Path, output: Path, at: float, width: int = 540) -> list[str]:
    at = _num(at, 0, 1e6, "at")
    width = _int(width, 16, 2048, "width")
    return ["-ss", f"{at:.3f}", "-i", str(assert_safe_media_path(video)), "-frames:v", "1",
            "-vf", f"scale={width}:-2", "-q:v", "3", str(assert_safe_media_path(output))]


def build_decode_check_args(video: Path, at: float) -> list[str]:
    at = _num(at, 0, 1e6, "at")
    return ["-ss", f"{at:.3f}", "-i", str(assert_safe_media_path(video)), "-frames:v", "1", "-f", "null", "-"]
