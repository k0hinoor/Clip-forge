"""Clip composition: one ffmpeg filter graph per clip.

Responsibilities:

* cut the clip out of the source (a single trim, or many trims + concat when
  silence removal shortened it),
* build the layout: split screen, podcast only, B-roll, gameplay background,
  cinematic, blurred background,
* smart-reframe the horizontal source into 9:16 / 1:1 / 16:9 using the crop plan,
* apply restrained punch-ins on emotional peaks,
* burn in the caption track (word-level, timed),
* mix voice + gameplay + music with ducking and loudness normalisation,
* encode with the best available encoder (NVENC / QSV / AMF / libx264).

Everything is driven by real analysis output, and the full plan is persisted on
the clip so a render can always be explained - and repeated.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..config import AppSettings, get_settings
from ..errors import ClipForgeError, ErrorCode
from ..logging_setup import get_logger
from .captions import CaptionPlan, ass_escape_filter_path, write_ass, write_srt
from ..system import ffmpeg_info
from .ffmpeg import (
    FfmpegProgress,
    MediaInfo,
    audio_encoder_args,
    audio_filter_chain,
    ffmpeg_bin,
    fps_args,
    loudnorm_filter,
    measure_loudness,
    probe_media,
    video_encoder_args,
)
from .framing import CropPlan, fit_crop_to_panel
from .runner import run_command
from .timeline import Timeline

log = get_logger("clipforge.render")

MAX_ZOOM = 0.09          # punch-ins never exceed ~9%: "do not over-edit"
ZOOM_MIN_GAP = 5.5       # minimum seconds between punch-ins
SPLIT_LAYOUTS = {"split", "broll", "gameplay"}


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #


@dataclass
class ZoomPoint:
    time: float           # output-relative seconds
    strength: float = 1.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"time": round(self.time, 3), "strength": round(self.strength, 3), "reason": self.reason}


@dataclass
class RenderSpec:
    """Everything needed to render one clip."""

    source_path: Path
    output_path: Path
    timeline: Timeline
    layout: str = "split"
    split_ratio: int = 65
    width: int = 1080
    height: int = 1920
    fps: int = 30
    crop_plan: CropPlan | None = None
    captions: CaptionPlan | None = None
    gameplay_path: Path | None = None
    broll_path: Path | None = None
    music_path: Path | None = None
    zoom_points: list[ZoomPoint] = field(default_factory=list)
    language: str = "en"
    title: str = ""
    subtitle_path: Path | None = None
    srt_path: Path | None = None
    settings: AppSettings | None = None
    source_fps: float = 0.0      # measured input frame rate; 0 = unknown
    quality: str = "final"       # final | preview
    cancel_key: str = ""
    notes: list[str] = field(default_factory=list)

    def resolved_settings(self) -> AppSettings:
        return self.settings or get_settings()

    @property
    def duration(self) -> float:
        return max(self.timeline.output_duration, 0.5)


# --------------------------------------------------------------------------- #
# Input bookkeeping
# --------------------------------------------------------------------------- #


class InputList:
    """Keeps ``-i`` indices honest, which is where filter graphs usually break."""

    def __init__(self) -> None:
        self.args: list[str] = []
        self.paths: list[Path] = []

    def add(self, path: Path, *, extra: Sequence[str] = ()) -> int:
        index = len(self.paths)
        self.args += [*extra, "-i", str(path)]
        self.paths.append(path)
        return index

    def index_of(self, path: Path | None) -> int | None:
        if path is None:
            return None
        for index, existing in enumerate(self.paths):
            if existing == path:
                return index
        return None


# --------------------------------------------------------------------------- #
# Expressions
# --------------------------------------------------------------------------- #


def smooth_ramp(start: float, end: float, *, variable: str = "t") -> str:
    """C1-smooth 0 → 1 ramp over ``[start, end]`` as an ffmpeg expression.

    ffmpeg's expression evaluator has no ``smoothstep``, so the hermite curve
    ``u²(3 − 2u)`` with ``u = clip((t − start) / (end − start), 0, 1)`` is written out
    using only the primitives every build ships with (``clip``, ``min``/``max`` and
    arithmetic). Keeping it inline means one expression, no shared state between the
    width and height of a ``crop`` filter.
    """
    span = max(end - start, 0.001)
    unit = f"clip(({variable}-{start:.3f})/{span:.3f},0,1)"
    return f"({unit}*{unit}*(3-2*{unit}))"


def zoom_expression(
    points: Sequence[ZoomPoint],
    *,
    base: float = 1.0,
    duration: float = 0.0,
    variable: str = "in_time",
) -> str:
    """Smooth, bounded punch-in zoom as an ffmpeg expression.

    ``variable`` is the clock the enclosing filter exposes: ``zoompan`` has no
    ``t`` (its expressions run on ``in_time``/``out_time``/``time``), and the crop
    filter's ``w``/``h`` are evaluated once at init, so punch-ins are always drawn by
    ``zoompan`` - never by a dynamic crop. Returns ``""`` when there is nothing to do.
    """
    usable: list[ZoomPoint] = []
    for point in sorted(points, key=lambda item: item.time):
        if point.time < 0.3 or (duration and point.time > duration - 0.4):
            continue
        if usable and point.time - usable[-1].time < ZOOM_MIN_GAP:
            continue
        usable.append(point)
    if not usable:
        return ""

    terms: list[str] = []
    for point in usable:
        amplitude = MAX_ZOOM * max(0.3, min(point.strength, 1.5))
        rise = 0.45
        hold = 1.10
        fall = 0.70
        start = point.time
        terms.append(
            f"{amplitude:.4f}*("
            f"if(between({variable},{start - rise:.3f},{start:.3f}),{smooth_ramp(start - rise, start, variable=variable)},0)"
            f"+if(between({variable},{start:.3f},{start + hold:.3f}),1,0)"
            f"+if(between({variable},{start + hold:.3f},{start + hold + fall:.3f}),"
            f"1-{smooth_ramp(start + hold, start + hold + fall, variable=variable)},0))"
        )
    return f"{base:.4f}+min(0.14,{'+'.join(terms)})"


def zoom_filter(expression: str, *, panel_width: int, panel_height: int, fps: int) -> str:
    """Punch-in step for a panel: ``zoompan`` at ``d=1`` (one output frame per input frame)."""
    frame_rate = max(1, int(round(fps)))
    return (
        f"zoompan=z='{expression}'"
        f":x='iw/2-(iw/zoom)/2':y='ih/2-(ih/zoom)/2'"
        f":d=1:s={panel_width}x{panel_height}:fps={frame_rate}"
    )


def podcast_layer(
    plan: CropPlan | None,
    *,
    panel_width: int,
    panel_height: int,
    zoom: str = "",
    fps: int = 30,
    zoomable: bool = True,
) -> str:
    """Filter chain for the podcast panel: crop → cover-scale → punch-in.

    The scale always preserves the aspect ratio (cover, then trim the rounding
    overflow), so a crop window that does not exactly match the panel can never
    stretch the speaker.
    """
    parts: list[str] = []
    if plan is not None and (plan.crop_width < plan.source_width or plan.crop_height < plan.source_height):
        parts.append(
            f"crop=w={plan.crop_width}:h={plan.crop_height}"
            f":x='{plan.crop_x_expression()}':y='{plan.crop_y_expression()}'"
        )
    parts.append(f"scale={panel_width}:{panel_height}:force_original_aspect_ratio=increase:flags=lanczos")
    parts.append(f"crop={panel_width}:{panel_height}")
    if zoomable and zoom:
        parts.append(zoom_filter(zoom, panel_width=panel_width, panel_height=panel_height, fps=fps))
    parts.append("setsar=1")
    return ",".join(parts)


def background_layer(
    plan: CropPlan | None,
    *,
    width: int,
    height: int,
    zoom: str = "",
    fps: int = 30,
    zoomable: bool = False,
) -> str:
    """Blurred / cinematic background: fill the frame, letterbox nothing."""
    parts: list[str] = []
    if plan is not None and plan.crop_width < plan.source_width and not zoomable:
        parts.append(f"crop=w={plan.crop_width}:h={plan.crop_height}:x='{plan.crop_x_expression()}':y='{plan.crop_y_expression()}'")
    if zoomable and zoom:
        parts.append(zoom_filter(zoom, panel_width=width, panel_height=height, fps=fps))
    parts.append(f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos")
    parts.append(f"crop={width}:{height}")
    parts.append("setsar=1")
    return ",".join(parts)


def asset_layer(
    *,
    input_index: int,
    panel_width: int,
    panel_height: int,
    fps: int,
    duration: float,
    pace: str,
    has_video: bool,
) -> str:
    """Loop / retime / scale gameplay or B-roll into its panel."""
    if not has_video:
        return f"color=c=0x0B0B0F:s={panel_width}x{panel_height}:r={fps}[asset]"
    speed = {"calm": 0.92, "medium": 1.0, "high": 1.12}.get(pace, 1.0)
    filters = []
    if abs(speed - 1.0) > 0.001:
        filters.append(f"setpts=PTS/{speed:.3f}")
    filters += [
        f"scale={panel_width}:{panel_height}:force_original_aspect_ratio=increase:flags=bicubic",
        f"crop={panel_width}:{panel_height}",
        f"fps={fps}",
        f"trim=duration={duration + 0.75:.3f}",
        "setpts=PTS-STARTPTS",
        "eq=saturation=1.06:contrast=1.02",
        "setsar=1",
    ]
    return f"[{input_index}:v]" + ",".join(filters) + "[asset]"


def trim_concat(source_label: str, segments: Sequence[Any], *, kind: str, offset: float = 0.0) -> str:
    """``trim``/``atrim`` + ``concat`` for a multi-segment clip.

    ``offset`` is the source timestamp the input was seeked to (``-ss``). Input
    seeking resets stream timestamps to zero, so every trim is expressed relative
    to that offset - otherwise the cuts silently land in the wrong place.
    """
    parts: list[str] = []
    labels: list[str] = []
    for index, segment in enumerate(segments):
        label = f"{kind}{index}"
        labels.append(label)
        start = max(0.0, float(segment.src_start) - offset)
        end = max(start, float(segment.src_end) - offset)
        if kind == "v":
            parts.append(
                f"[{source_label}]trim=start={start:.3f}:end={end:.3f},"
                f"setpts=PTS-STARTPTS[{label}]"
            )
        else:
            parts.append(
                f"[{source_label}]atrim=start={start:.3f}:end={end:.3f},"
                f"asetpts=PTS-STARTPTS[{label}]"
            )
    output = f"{kind}cat"
    if len(labels) == 1:
        parts[-1] = parts[-1].replace(f"[{labels[0]}]", f"[{output}]")
    else:
        # concat must be told which streams it is stitching: v=1:a=0 for video,
        # v=0:a=1 for audio (getting this wrong links audio into a video pad).
        streams = "v=1:a=0" if kind == "v" else "v=0:a=1"
        parts.append(f"{''.join(f'[{label}]' for label in labels)}concat=n={len(labels)}:{streams}[{output}]")
    return ";".join(parts), output


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #


def build_filter_graph(
    spec: RenderSpec,
    *,
    audio_measured: dict[str, float] | None = None,
) -> tuple[str, str, str, list[str]]:
    """Return ``(filter_complex, video_label, audio_label, input_args)``."""
    settings = spec.resolved_settings()
    timeline = spec.timeline
    duration = spec.duration
    width, height = spec.width, spec.height
    layout = spec.layout
    speed = timeline.speed or 1.0
    multi_segment = timeline.is_trimmed and len(timeline.segments) > 1

    inputs = InputList()
    graph: list[str] = []

    source_start = timeline.segments[0].src_start if timeline.segments else timeline.source_start
    source_end = timeline.segments[-1].src_end if timeline.segments else timeline.source_end
    inputs.add(
        spec.source_path,
        extra=["-ss", f"{source_start:.3f}", "-t", f"{max(source_end - source_start, 0.5):.3f}"],
    )

    # ------------------------------------------------------------------ video
    if multi_segment:
        graph.append(trim_concat("0:v", timeline.segments, kind="v", offset=source_start)[0])
        video_label = "[vcat]"
    else:
        video_label = "[0:v]"
    if abs(speed - 1.0) > 0.001:
        graph.append(f"{video_label}setpts=PTS/{speed:.4f}[vsrc]")
        video_label = "[vsrc]"
    # Normalise the cadence *before* the layout chain: zoompan and the fps filter
    # disagree about timestamps when chained back to back (ffmpeg floods the output
    # queue), and a constant frame rate upstream keeps every later stage honest.
    graph.append(f"{video_label}fps={spec.fps}[vstd]")
    video_label = "[vstd]"

    zoom = zoom_expression(spec.zoom_points, duration=duration)
    # zoompan stamps one output frame per input frame on its own `fps` clock, so
    # it must run at the cadence of its input - the stream normalised to
    # ``spec.fps`` just above. Any other rate would drift away from the audio.
    zoom_fps = int(spec.fps)
    ratio = max(30, min(int(spec.split_ratio), 80))

    if layout in SPLIT_LAYOUTS:
        podcast_height = int(round(height * ratio / 100.0 / 2) * 2)
        asset_height = height - podcast_height
    else:
        podcast_height = height
        asset_height = 0
    # Stored crop plans may be cut for another panel (full frame vs split, or an
    # older aspect ratio); re-cut the window for the panel actually drawn here.
    asset_layout = layout in SPLIT_LAYOUTS and (spec.broll_path if layout == "broll" else spec.gameplay_path) is not None
    crop_plan = fit_crop_to_panel(spec.crop_plan, width, podcast_height if asset_layout else height)

    if layout == "blur":
        # Two chains read the same frames, so the pad has to be split explicitly:
        # ffmpeg treats a label consumed twice as a stream specifier and refuses it.
        graph.append(f"{video_label}split=2[vbg][vfg]")
        graph.append(
            f"[vbg]scale={width}:{height}:force_original_aspect_ratio=increase:flags=bicubic,"
            f"crop={width}:{height},boxblur=luma_radius=40:luma_power=2:chroma_radius=16:chroma_power=2,"
            f"eq=brightness=-0.06:saturation=1.12[bg]"
        )
        graph.append(
            f"[vfg]scale={int(width * 0.95)}:{int(height * 0.95)}:force_original_aspect_ratio=decrease:flags=lanczos[fg]"
        )
        graph.append("[bg][fg]overlay=(W-w)/2:(H-h)/2[vframed]")
        video_out = "[vframed]"
    elif layout == "cinematic":
        graph.append(
            f"{video_label}{background_layer(crop_plan, width=width, height=height, zoom=zoom, fps=zoom_fps)}[vframed]"
        )
        video_out = "[vframed]"
    else:
        asset_path = spec.broll_path if layout == "broll" else spec.gameplay_path
        if layout not in SPLIT_LAYOUTS:
            # Podcast only / blurred: one panel fills the frame, assets are ignored.
            if asset_path is not None:
                spec.notes.append(f"Layout '{layout}' renders the speaker full-frame; the selected asset was not used.")
            graph.append(
                f"{video_label}{podcast_layer(crop_plan, panel_width=width, panel_height=height, zoom=zoom, fps=zoom_fps)}[vframed]"
            )
            video_out = "[vframed]"
        elif asset_path is None:
            spec.notes.append(
                f"No {'B-roll' if layout == 'broll' else 'gameplay'} asset was available, "
                "so the clip was rendered as a full-frame speaker cut."
            )
            graph.append(
                f"{video_label}{podcast_layer(crop_plan, panel_width=width, panel_height=height, zoom=zoom, fps=zoom_fps)}[vframed]"
            )
            video_out = "[vframed]"
        elif layout == "gameplay":
            asset_index = inputs.add(asset_path, extra=["-stream_loop", "-1"])
            asset_info = _safe_probe(asset_path)
            graph.append(
                f"{video_label}{podcast_layer(crop_plan, panel_width=width, panel_height=podcast_height, zoom=zoom, fps=zoom_fps)}[pod]"
            )
            graph.append(
                asset_layer(
                    input_index=asset_index, panel_width=width, panel_height=height,
                    fps=spec.fps, duration=duration, pace=settings.gameplay_pace,
                    has_video=asset_info.has_video,
                )
            )
            graph.append(f"[pod]scale={int(width * 0.8)}:{int(podcast_height * 0.8)}[podsmall]")
            graph.append("[asset][podsmall]overlay=(W-w)/2:(H-h)/2[vframed]")
            video_out = "[vframed]"
        else:
            asset_index = inputs.add(asset_path, extra=["-stream_loop", "-1"])
            asset_info = _safe_probe(asset_path)
            graph.append(
                f"{video_label}{podcast_layer(crop_plan, panel_width=width, panel_height=podcast_height, zoom=zoom, fps=zoom_fps)}[pod]"
            )
            graph.append(
                asset_layer(
                    input_index=asset_index, panel_width=width, panel_height=asset_height,
                    fps=spec.fps, duration=duration, pace=settings.gameplay_pace,
                    has_video=asset_info.has_video,
                )
            )
            graph.append("[pod][asset]vstack=inputs=2[vframed]")
            video_out = "[vframed]"

    # --------------------------------------------------------------- captions
    if spec.captions and spec.captions.lines and spec.subtitle_path and settings.captions_enabled:
        if ffmpeg_info().has_libass:
            subtitle_file = ass_escape_filter_path(spec.subtitle_path)
            graph.append(f"{video_out}ass='{subtitle_file}':shaping=complex[vcaptioned]")
            video_out = "[vcaptioned]"
        else:
            # Degrade instead of failing: the clip still renders and the SRT is
            # still exported, and the UI says exactly why there are no captions.
            spec.notes.append(
                "Captions were not burned in: this ffmpeg build has no libass. "
                "The .srt file is still exported next to the clip."
            )
            log.warning("ffmpeg has no libass; captions will not be burned in")

    graph.append(f"{video_out}format=yuv420p[vout]")

    # ------------------------------------------------------------------ audio
    if multi_segment:
        graph.append(trim_concat("0:a", timeline.segments, kind="a", offset=source_start)[0])
        audio_label = "[acat]"
    else:
        audio_label = "[0:a]"

    voice_chain = audio_filter_chain(settings, measured=audio_measured, source="voice")
    if abs(speed - 1.0) > 0.001:
        voice_chain = f"atempo={speed:.4f}," + voice_chain
    duck_music = spec.music_path is not None and settings.music_enabled and settings.ducking
    if duck_music:
        # The voice feeds both the mix and the ducking side-chain; a filter pad
        # can only be consumed once, so it is split explicitly.
        graph.append(f"{audio_label}{voice_chain},asplit=2[voice][voicekey]")
    else:
        graph.append(f"{audio_label}{voice_chain}[voice]")

    mix_labels = ["[voice]"]
    if asset_path := (spec.gameplay_path if layout in {"split", "gameplay"} else spec.broll_path if layout == "broll" else None):
        asset_index = inputs.index_of(asset_path)
        if asset_index is not None and _safe_probe(asset_path).has_audio and settings.gameplay_volume > 0:
            graph.append(
                f"[{asset_index}:a]atrim=duration={duration + 0.75:.3f},asetpts=PTS-STARTPTS,"
                f"volume={settings.gameplay_volume:.3f}[assetaudio]"
            )
            mix_labels.append("[assetaudio]")

    if spec.music_path is not None and settings.music_enabled:
        music_index = inputs.add(spec.music_path, extra=["-stream_loop", "-1"])
        graph.append(
            f"[{music_index}:a]atrim=duration={duration + 0.75:.3f},asetpts=PTS-STARTPTS,"
            f"volume={settings.music_volume:.3f}[mus]"
        )
        if duck_music:
            graph.append(f"[mus][voicekey]sidechaincompress=threshold=0.035:ratio={ducking_ratio(settings.ducking_db):.1f}:attack=12:release=320[musicducked]")
            mix_labels.append("[musicducked]")
        else:
            mix_labels.append("[mus]")

    if len(mix_labels) > 1:
        graph.append(f"{''.join(mix_labels)}amix=inputs={len(mix_labels)}:duration=first:dropout_transition=0:normalize=0[mixed]")
        final_audio = "[mixed]"
    else:
        final_audio = "[voice]"

    # Loudness normalisation already happened in the voice chain; the tail only
    # guarantees the mixed signal stays inside the ceiling and matches the export
    # sample rate. (Re-applying loudnorm here would flatten the mix twice.)
    tail: list[str] = []
    if settings.normalize_loudness and "loudnorm" not in voice_chain:
        tail.append(loudnorm_filter(settings, audio_measured))
    tail += ["alimiter=limit=0.97", "aresample=48000"]
    graph.append(f"{final_audio}{','.join(tail)}[aout]")

    return ";".join(graph), "vout", "aout", inputs.args


def ducking_ratio(ducking_db: float) -> float:
    """Side-chain compression ratio for the requested music dip (dB, negative).

    Speech normalised to the loudness target sits roughly 10-12 dB above the
    compressor threshold, so ``1 + |dB| / 1.5`` lands the music close to the
    requested reduction while someone is talking (-12 dB -> ratio 9).
    """
    return max(2.0, min(20.0, 1.0 + abs(float(ducking_db)) / 1.5))


def _safe_probe(path: Path | None) -> MediaInfo:
    if path is None:
        return MediaInfo(path="")
    try:
        return probe_media(path)
    except ClipForgeError:
        return MediaInfo(path=str(path))


# --------------------------------------------------------------------------- #
# Command + rendering
# --------------------------------------------------------------------------- #


def build_ffmpeg_command(spec: RenderSpec, *, audio_measured: dict[str, float] | None = None) -> list[str]:
    settings = spec.resolved_settings()
    filter_complex, video_label, audio_label, input_args = build_filter_graph(spec, audio_measured=audio_measured)

    command: list[str] = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", "-loglevel", "warning"]
    command += input_args
    command += ["-filter_complex", filter_complex, "-map", f"[{video_label}]", "-map", f"[{audio_label}]"]

    if spec.quality == "preview":
        command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-pix_fmt", "yuv420p", "-tune", "fastdecode"]
    else:
        command += video_encoder_args(settings, fps=spec.fps)
    command += audio_encoder_args(settings)
    command += fps_args(spec.fps)
    command += ["-movflags", "+faststart", "-map_metadata", "-1", "-shortest", str(spec.output_path)]
    return command


def render(spec: RenderSpec, *, progress: Callable[[float, str], None] | None = None) -> dict[str, Any]:
    """Render one clip and return render statistics."""
    settings = spec.resolved_settings()
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)

    if spec.captions and spec.captions.lines:
        if spec.subtitle_path is None:
            spec.subtitle_path = spec.output_path.with_suffix(".ass")
        write_ass(spec.captions, spec.subtitle_path, width=spec.width, height=spec.height)
        # The .srt lives next to the .ass so one folder holds the whole clip.
        spec.srt_path = write_srt(spec.captions, spec.subtitle_path.with_suffix(".srt"))

    measured: dict[str, float] = {}
    if settings.normalize_loudness and spec.quality != "preview":
        try:
            start = spec.timeline.segments[0].src_start if spec.timeline.segments else spec.timeline.source_start
            measured = measure_loudness(spec.source_path, start=start, duration=min(spec.timeline.source_duration, 180.0))
        except Exception as exc:  # noqa: BLE001 - measurement is an optimisation, not a requirement
            log.debug("loudness measurement skipped: %s", exc)

    command = build_ffmpeg_command(spec, audio_measured=measured or None)
    tracker = FfmpegProgress(spec.duration, (lambda fraction: progress(fraction, "encoding")) if progress else None)

    if progress:
        progress(0.0, "starting the encoder")
    result = run_command(
        command,
        label=f"render:{spec.output_path.name}",
        cancel_key=spec.cancel_key,
        progress=tracker,
        timeout=None,
    )

    info = _safe_probe(spec.output_path)
    if not spec.output_path.exists() or info.duration <= 0:
        raise ClipForgeError(
            code=ErrorCode.RENDER_FAILED,
            message="The clip was rendered but the output file is empty.",
            hint="Check logs/render.log - the encoder likely ran out of resources.",
            status_code=500,
        )
    if progress:
        progress(1.0, "encode complete")

    return {
        "output": str(spec.output_path),
        "duration": round(info.duration, 3),
        "width": info.width,
        "height": info.height,
        "fps": round(info.fps, 2),
        "size_bytes": spec.output_path.stat().st_size,
        "encode_seconds": round(result.duration_seconds, 2),
        "speed_ratio": round(info.duration / max(result.duration_seconds, 0.05), 3),
        "loudness": measured,
        "notes": spec.notes,
    }


def command_preview(spec: RenderSpec) -> str:
    """Readable form of the ffmpeg command (shown in the clip editor / diagnostics)."""
    return shlex.join(build_ffmpeg_command(spec))


def estimate_render_seconds(duration: float, width: int, height: int, *, hw_accel: bool) -> float:
    """Rough throughput estimate used for queue ETAs."""
    pixels = max(width * height, 1)
    base = pixels / (1920 * 1080)
    return duration * (0.35 if hw_accel else 1.35) * base


def graph_summary(spec: RenderSpec) -> list[str]:
    """Human summary of what a render will do (used in the clip panel)."""
    settings = spec.resolved_settings()
    lines: list[str] = []
    lines.append(f"Layout: {spec.layout} ({spec.split_ratio}/{100 - spec.split_ratio})" if spec.layout in SPLIT_LAYOUTS else f"Layout: {spec.layout}")
    lines.append(f"Output: {spec.width}x{spec.height} @ {spec.fps}fps")
    lines.append(f"Duration: {spec.duration:.1f}s (source range {spec.timeline.source_duration:.1f}s)")
    if spec.timeline.is_trimmed:
        lines.append(f"Silence removal: {len(spec.timeline.segments)} keeps, {spec.timeline.removed_seconds:.1f}s removed")
    if abs(spec.timeline.speed - 1.0) > 0.001:
        lines.append(f"Speed: x{spec.timeline.speed:.2f}")
    if spec.crop_plan is not None:
        lines.append(f"Reframing: {spec.crop_plan.mode} crop {spec.crop_plan.crop_width}x{spec.crop_plan.crop_height}")
    if spec.captions is not None:
        lines.append(f"Captions: {len(spec.captions.lines)} lines, {spec.captions.theme.preset} preset, {spec.captions.theme.animation}")
    if spec.gameplay_path is not None:
        lines.append(f"Gameplay: {spec.gameplay_path.name} at {int(settings.gameplay_volume * 100)}% volume")
    if spec.music_path is not None:
        lines.append(f"Music: {spec.music_path.name} at {int(settings.music_volume * 100)}% volume")
    if spec.zoom_points:
        lines.append(f"Punch-ins: {len(spec.zoom_points)} dynamic zooms")
    lines += spec.notes
    return lines


__all__ = [
    "MAX_ZOOM",
    "InputList",
    "RenderSpec",
    "ZoomPoint",
    "asset_layer",
    "build_ffmpeg_command",
    "build_filter_graph",
    "command_preview",
    "ducking_ratio",
    "estimate_render_seconds",
    "graph_summary",
    "podcast_layer",
    "render",
    "trim_concat",
    "zoom_expression",
    "smooth_ramp",
]
