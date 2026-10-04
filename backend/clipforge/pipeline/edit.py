"""Automatic editing: turning a scored moment into a concrete render plan.

For every clip this module decides - from real data, never guesswork:

* **pace** - where the silences are (ffmpeg silencedetect) and which of them to
  remove, producing an explicit :class:`~clipforge.media.timeline.Timeline` that
  every later timestamp is mapped through;
* **framing** - where the speaker is (frame sampling + face detection) and how
  the 16:9 source should be cropped to 9:16, with smooth keyframes;
* **captions** - word-level lines broken at semantic boundaries, with emphasis
  words taken from the transcript and the zoom points;
* **punch-ins** - a small number of dynamic zooms on the emotional peaks;
* **composition** - the layout, the gameplay/B-roll asset and the music bed.

The same plan can be rebuilt from the database at render time, which is what
makes rendering, re-rendering and tweaking cheap and reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..ai.features import CandidateFeatures
from ..ai.language import LanguageProfile
from ..ai.segment import Sentence
from ..ai.transcribe import Word
from ..ai.vision import analyse_frames, sample_times, summarise, vision_available
from ..config import AppSettings, CaptionTheme, get_settings
from ..constants import category_label
from ..logging_setup import get_logger
from ..media import assets as assets_mod
from ..media.captions import CaptionPlan, plan_captions
from ..media.ffmpeg import detect_silence, grab_frames, probe_media
from ..media.framing import CropPlan, plan_for_layout
from ..media.timeline import Timeline, build_timeline
from .context import ProjectPaths

log = get_logger("clipforge.worker")

MAX_VISION_FRAMES = 20
MAX_ZOOM_POINTS = 3


@dataclass
class ClipPlan:
    timeline: Timeline
    crop_plan: CropPlan | None
    captions: CaptionPlan | None
    layout: str
    split_ratio: int
    zoom_points: list[dict[str, Any]]
    gameplay: dict[str, Any] | None
    broll: dict[str, Any] | None
    music: dict[str, Any] | None
    language: str
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeline": self.timeline.to_dict(),
            "crop": self.crop_plan.to_dict() if self.crop_plan else None,
            "captions": self.captions.to_dict() if self.captions else None,
            "layout": {
                "layout": self.layout,
                "split_ratio": self.split_ratio,
                "gameplay": self.gameplay,
                "broll": self.broll,
                "music": self.music,
                "notes": self.notes,
            },
            "zoom_points": self.zoom_points,
            "language": self.language,
        }


# --------------------------------------------------------------------------- #
# Plan creation (during analysis)
# --------------------------------------------------------------------------- #


def build_clip_plan(
    *,
    project_id: str,
    index: int,
    candidate: Any,
    words: Sequence[Word],
    settings: AppSettings,
    language: LanguageProfile,
    paths: ProjectPaths,
    source_path: Path | None = None,
    features: CandidateFeatures | None = None,
    sentences: Sequence[Sentence] = (),
    recent_asset_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Build and persist the edit plan for one clip."""
    notes: list[str] = []
    features = features or candidate.features

    # ------------------------------------------------------------- pacing
    timeline = build_timeline(candidate.start, candidate.end, [], settings=settings)
    if source_path is not None and settings.remove_silence:
        try:
            silences = detect_silence(
                source_path,
                noise_db=settings.silence_threshold_db,
                min_duration=settings.silence_min_duration,
                start=candidate.start,
                end=candidate.end,
            )
            timeline = build_timeline(
                candidate.start,
                candidate.end,
                silences,
                settings=settings,
                min_output_seconds=settings.min_clip_seconds,
                max_output_seconds=settings.max_clip_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - pacing is best-effort
            log.warning("silence detection failed for %s: %s", project_id, exc)
            notes.append("Silence detection failed - the clip keeps its original pacing.")
    notes.extend(timeline.notes)

    caption_words = _remap_words(words, timeline)
    zoom_points = plan_zoom_points(sentences, features, timeline)

    # ------------------------------------------------------------ captions
    captions = None
    if settings.captions_enabled:
        captions = plan_captions(
            caption_words,
            theme=settings.caption,
            language=language.describe() if language else settings.language_hint or "en",
            emphasis_words=_emphasis_words(features),
            zoom_times=[point["time"] for point in zoom_points],
        )
        notes.extend(captions.notes)

    # ------------------------------------------------------------- framing
    crop_plan: CropPlan | None = None
    output_width, output_height = settings.aspect_dims()
    if source_path is not None:
        media = probe_media(source_path)
        analyses = []
        if settings.smart_reframe and vision_available():
            try:
                times = sample_times(candidate.start, candidate.end, interval=2.6, max_samples=MAX_VISION_FRAMES)
                frames = grab_frames(source_path, times, paths.analysis / "frames", width=360)
                analyses = analyse_frames(frames)
                if analyses:
                    notes.append(f"Visual analysis: {summarise(analyses).get('shot_type', 'unknown')} shot.")
            except Exception as exc:  # noqa: BLE001 - vision is optional
                log.debug("visual analysis failed: %s", exc)
                notes.append("Visual analysis unavailable - framing used motion/subject heuristics.")
        layout_hint = settings.layout
        crop_plan = plan_for_layout(
            layout_hint,
            media.width,
            media.height,
            output_width,
            output_height,
            analyses,
            duration=timeline.output_duration,
            smart=settings.smart_reframe,
            tracking=settings.speaker_tracking,
        )
        notes.extend(crop_plan.notes)

    # ------------------------------------------------------- composition
    layout = settings.layout
    clip_key = f"{project_id}:{index}:{candidate.start:.1f}"
    gameplay = broll = music = None

    if layout in {"split", "gameplay"}:
        choice = assets_mod.pick_gameplay(
            clip_key=clip_key,
            category=candidate.category,
            duration=timeline.output_duration,
            settings=settings,
            recent_ids=recent_asset_ids,
        )
        if choice is None:
            layout = "blur" if (crop_plan and crop_plan.source_width > crop_plan.source_height) else "podcast"
            notes.append(f"No gameplay assets yet - switched layout to '{layout}'. Import footage in Assets to get the split screen.")
        else:
            gameplay = choice.to_dict()
            notes.append(f"Gameplay: {choice.name} ({choice.reason}).")
    elif layout == "broll":
        choice = assets_mod.pick_broll(clip_key=clip_key, settings=settings, recent_ids=recent_asset_ids)
        if choice is None:
            layout = "blur"
            notes.append("No B-roll assets available - switched to the blurred-background layout.")
        else:
            broll = choice.to_dict()
            notes.append(f"B-roll: {choice.name} ({choice.reason}).")

    if crop_plan is not None and crop_plan.mode == "fit_blur" and layout == "split":
        notes.append("Frame is too wide for a 9:16 crop; the podcast panel pans between speakers.")

    choice = assets_mod.pick_music(
        clip_key=clip_key,
        settings=settings,
        recent_ids=recent_asset_ids,
        category=candidate.category,
    )
    if choice is not None:
        music = choice.to_dict()
        notes.append(f"Music: {choice.name} at {int(getattr(settings, 'music_volume', 0.12) * 100)}% (ducked under the voice).")

    plan = ClipPlan(
        timeline=timeline,
        crop_plan=crop_plan,
        captions=captions,
        layout=layout,
        split_ratio=settings.split_ratio,
        zoom_points=zoom_points,
        gameplay=gameplay,
        broll=broll,
        music=music,
        language=language.describe() if language else "en",
        notes=notes,
    )
    _write_plan_files(paths, index, plan, caption_words)
    return plan.to_dict()


def _write_plan_files(paths: ProjectPaths, index: int, plan: ClipPlan, caption_words: Sequence[Word]) -> None:
    folder = paths.clip_dir(index)
    (folder / "plan.json").write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (folder / "words.json").write_text(
        json.dumps([word.to_dict() for word in caption_words], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    if plan.captions is not None:
        from ..media.captions import write_srt

        try:
            write_srt(plan.captions, folder / "captions.srt")
        except OSError as exc:
            log.debug("could not write SRT preview: %s", exc)


def _remap_words(words: Sequence[Word], timeline: Timeline) -> list[Word]:
    """Copy words and move their timings onto the rendered timeline."""
    copies = [
        Word(word=word.word, start=word.start, end=word.end, confidence=word.confidence, speaker=word.speaker)
        for word in words
    ]
    if not timeline.is_trimmed and abs(timeline.speed - 1.0) < 0.001:
        base = timeline.source_start
        for word in copies:
            word.start -= base
            word.end -= base
        return copies
    return timeline.map_words(copies)


def _emphasis_words(features: CandidateFeatures) -> list[str]:
    words = features.meta.get("emphasis_words") or []
    if isinstance(words, str):
        words = [word.strip() for word in words.split(",") if word.strip()]
    return [str(word) for word in words][:8]


def plan_zoom_points(
    sentences: Sequence[Sentence],
    features: CandidateFeatures,
    timeline: Timeline,
) -> list[dict[str, Any]]:
    """Punch-ins on the strongest emotional / emphasis beats (never over-edited)."""
    if not sentences:
        return []
    candidates: list[tuple[float, float, str]] = []
    for sentence in sentences:
        emotion = sentence.features.get("emotion", 0.0)
        question = sentence.features.get("question", 0.0)
        if emotion >= 0.35 or question >= 0.5:
            strength = min(1.0, 0.5 + emotion)
            reason = "emotional peak" if emotion >= 0.35 else "question drop"
            candidates.append((sentence.start, strength, reason))

    points: list[dict[str, Any]] = []
    last_time = -999.0
    for start, strength, reason in sorted(candidates, key=lambda item: -item[1]):
        output_time = timeline.source_to_output(start)
        if output_time is None or output_time - last_time < 5.5:
            continue
        points.append({"time": round(output_time, 3), "strength": round(strength, 3), "reason": reason})
        last_time = output_time
        if len(points) >= MAX_ZOOM_POINTS:
            break
    return sorted(points, key=lambda item: item["time"])


# --------------------------------------------------------------------------- #
# Plan reconstruction (at render time)
# --------------------------------------------------------------------------- #


def load_clip_words(clip: Any) -> list[Word]:
    try:
        payload = json.loads(clip.words_json or "[]")
    except json.JSONDecodeError:
        payload = []
    return [
        Word(
            word=item.get("word", ""),
            start=float(item.get("start", 0.0)),
            end=float(item.get("end", 0.0)),
            confidence=float(item.get("confidence", 0.0)),
            speaker=item.get("speaker", ""),
        )
        for item in payload
        if item.get("word")
    ]


def load_timeline(clip: Any, settings: AppSettings) -> Timeline:
    from ..media.timeline import Segment

    try:
        payload = json.loads(clip.trim_json or "{}")
    except json.JSONDecodeError:
        payload = {}
    if payload.get("segments"):
        timeline = Timeline(
            segments=[
                Segment(
                    src_start=float(segment["src_start"]),
                    src_end=float(segment["src_end"]),
                    out_start=float(segment["out_start"]),
                    out_end=float(segment["out_end"]),
                )
                for segment in payload["segments"]
            ],
            speed=float(payload.get("speed", 1.0)),
            removed=[(float(pair[0]), float(pair[1])) for pair in payload.get("removed") or [] if len(pair) == 2],
            source_start=float(payload["segments"][0]["src_start"]),
            source_end=float(payload["segments"][-1]["src_end"]),
            source_duration=float(payload["segments"][-1]["src_end"]) - float(payload["segments"][0]["src_start"]),
            notes=list(payload.get("notes") or []),
        )
        return timeline
    return build_timeline(clip.start, clip.end, [], settings=settings)


# Layout keys the clip editor stores per clip and that must be applied to the
# render settings, otherwise the UI toggle would be decorative.
CLIP_SETTING_KEYS = (
    "captions_enabled",
    "remove_silence",
    "auto_zoom",
    "gameplay_enabled",
    "music_enabled",
    "aspect_ratio",
)


def clip_settings(settings: AppSettings, layout: dict[str, Any]) -> AppSettings:
    """Project settings with this clip's own overrides applied.

    The clip editor writes ``captions_enabled``, ``aspect_ratio``, ``remove_silence``
    and the caption theme into the clip's plan. Rendering has to read them back or
    the editor's controls silently do nothing.
    """
    patch: dict[str, Any] = {
        key: layout[key] for key in CLIP_SETTING_KEYS if layout.get(key) is not None
    }
    theme = layout.get("caption")
    if isinstance(theme, dict) and theme:
        try:
            patch["caption"] = CaptionTheme.model_validate({**settings.caption.model_dump(), **theme}).model_dump()
        except Exception:  # noqa: BLE001 - a stale theme must not break a render
            log.warning("ignoring an unreadable caption theme stored on the clip")
    if not patch:
        return settings
    try:
        return AppSettings.model_validate({**settings.model_dump(), **patch})
    except Exception as exc:  # noqa: BLE001
        log.warning("ignoring clip overrides (%s)", exc)
        return settings


def rebuild_captions(clip: Any, settings: AppSettings, language: str, zoom_times: Sequence[float]) -> CaptionPlan | None:
    if not settings.captions_enabled:
        return None
    words = load_clip_words(clip)
    timeline = load_timeline(clip, settings)
    caption_words = _remap_words(words, timeline)
    return plan_captions(
        caption_words,
        theme=settings.caption,
        language=language,
        zoom_times=zoom_times,
    )


def rebuild_layout(clip: Any, settings: AppSettings) -> dict[str, Any]:
    """The clip's stored plan (``ClipPlan.to_dict()`` shape)."""
    try:
        return json.loads(clip.layout_json or "{}")
    except json.JSONDecodeError:
        return {}


def layout_payload(plan: dict[str, Any]) -> dict[str, Any]:
    """Layout settings inside a stored plan.

    ``ClipPlan.to_dict()`` nests them under ``layout`` (layout, split_ratio,
    gameplay, broll, music, notes); plans written by earlier versions kept them at
    the top level, so both shapes are accepted here rather than in every caller.
    """
    inner = plan.get("layout")
    return inner if isinstance(inner, dict) else plan


def set_layout_payload(plan: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Write layout settings back into a plan, preserving the stored shape."""
    inner = plan.get("layout")
    if isinstance(inner, dict):
        plan["layout"] = payload
    else:
        plan.update(payload)
    return plan


def resolve_asset(entry: dict[str, Any] | None) -> Path | None:
    if not entry:
        return None
    path = entry.get("path")
    if path and Path(path).exists():
        return Path(path)
    asset_id = entry.get("id")
    if asset_id:
        return assets_mod.resolve_asset_path(str(asset_id))
    return None


def clip_plan_summary(clip: Any, settings: AppSettings) -> dict[str, Any]:
    """Plan overview for the clip detail panel (no heavy work)."""
    plan = rebuild_layout(clip, settings)
    layout = layout_payload(plan)
    timeline = load_timeline(clip, settings)
    return {
        "layout": layout.get("layout", settings.layout),
        "split_ratio": layout.get("split_ratio", settings.split_ratio),
        "gameplay": layout.get("gameplay"),
        "broll": layout.get("broll"),
        "music": layout.get("music"),
        "notes": layout.get("notes", []),
        "timeline": timeline.to_dict(),
        "zoom_points": layout.get("zoom_points", []),
        "captions_enabled": settings.captions_enabled,
        "estimated_render_seconds": round(
            timeline.output_duration * (0.4 if settings.hw_accel != "none" else 1.3), 1
        ),
    }


__all__ = [
    "ClipPlan",
    "build_clip_plan",
    "clip_plan_summary",
    "load_clip_words",
    "load_timeline",
    "plan_zoom_points",
    "clip_settings",
    "rebuild_captions",
    "rebuild_layout",
    "layout_payload",
    "set_layout_payload",
    "resolve_asset",
]
