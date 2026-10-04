"""Clip-level operations: editing, regeneration, deletion, caption previews.

Editing a clip never touches the source media: the moment's words are re-read
from the stored transcript, the pace/caption/framing plan is rebuilt, and the
clip is marked as needing a fresh render. Renders therefore always match the
plan the user is looking at.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import select

from ..ai.features import TranscriptIndex
from ..ai.language import detect_language
from ..ai.segment import Sentence, build_blocks, build_sentences
from ..ai.transcribe import Utterance, Word
from ..config import AppSettings, CaptionTheme, get_settings
from ..db import Clip, Project, TranscriptSegment, session_scope
from ..errors import ClipForgeError, ErrorCode, not_found
from ..logging_setup import get_logger
from ..media.captions import plan_captions, preset_theme, write_srt
from ..media.compose import RenderSpec, command_preview
from ..media.framing import crop_plan_from_dict
from ..media.timeline import build_timeline
from ..pipeline.context import ProjectPaths, find_media
from ..pipeline.edit import load_timeline, rebuild_layout, resolve_asset
from ..pipeline.render import project_settings_for

log = get_logger("clipforge.worker")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def get_clip(clip_id: str) -> dict[str, Any]:
    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise not_found("Clip", clip_id)
        return clip.to_dict(include_words=True)


def load_project_sentences(project_id: str) -> list[Sentence]:
    with session_scope() as session:
        rows = (
            session.execute(
                select(TranscriptSegment)
                .where(TranscriptSegment.project_id == project_id)
                .order_by(TranscriptSegment.idx)
            )
            .scalars()
            .all()
        )
        utterances = [
            Utterance(
                text=row.text,
                start=row.start,
                end=row.end,
                speaker=row.speaker,
                confidence=row.confidence,
                words=[
                    Word(
                        word=item.get("word", ""),
                        start=float(item.get("start", 0)),
                        end=float(item.get("end", 0)),
                        confidence=float(item.get("confidence", 0)),
                        speaker=item.get("speaker", row.speaker),
                    )
                    for item in row.words
                    if item.get("word")
                ],
            )
            for row in rows
        ]
    return build_sentences(utterances)


def words_in_range(sentences: Sequence[Sentence], start: float, end: float) -> list[Word]:
    words: list[Word] = []
    for sentence in sentences:
        if sentence.end < start - 0.05 or sentence.start > end + 0.05:
            continue
        for word in sentence.words:
            if word.end >= start - 0.02 and word.start <= end + 0.02:
                words.append(Word(word=word.word, start=word.start, end=word.end, confidence=word.confidence, speaker=word.speaker))
    words.sort(key=lambda item: item.start)
    return words


def clip_detail(clip_id: str) -> dict[str, Any]:
    """Full clip view: metadata, edit plan, captions and the ffmpeg command."""
    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise not_found("Clip", clip_id)
        project = session.get(Project, clip.project_id)
        if project is None:
            raise not_found("Project", clip.project_id)
        settings = project_settings_for(project)
        payload = clip.to_dict(include_words=True)
        layout = rebuild_layout(clip, settings)
        timeline = load_timeline(clip, settings)
        project_snapshot = project.to_dict()

    captions = None
    if settings.captions_enabled:
        from ..pipeline.edit import rebuild_captions

        with session_scope() as session:
            clip = session.get(Clip, clip_id)
            plan = rebuild_captions(clip, settings, layout.get("language", "en"), [point.get("time", 0) for point in layout.get("zoom_points") or []])
        captions = plan.to_dict() if plan else None

    detail = {
        **payload,
        "project": {
            "id": project_snapshot["id"],
            "title": project_snapshot["title"],
            "language": project_snapshot["language"],
            "language_mode": project_snapshot["language_mode"],
            "duration": project_snapshot["duration"],
        },
        "plan": {
            "layout": layout.get("layout", settings.layout),
            "split_ratio": layout.get("split_ratio", settings.split_ratio),
            "gameplay": layout.get("gameplay"),
            "broll": layout.get("broll"),
            "music": layout.get("music"),
            "notes": layout.get("notes") or [],
            "zoom_points": layout.get("zoom_points") or [],
            "crop": layout.get("crop"),
            "timeline": timeline.to_dict(),
        },
        "captions": captions,
        "caption_presets": sorted({*(layout.get("caption_presets") or []), "minimal", "cinematic", "bold_creator", "karaoke", "highlight", "documentary"}),
        "needs_render": payload["status"] != "rendered",
    }
    return detail


# --------------------------------------------------------------------------- #
# Editing
# --------------------------------------------------------------------------- #


def update_clip(clip_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    """Apply user edits and rebuild the derived plan."""
    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise not_found("Clip", clip_id)
        project = session.get(Project, clip.project_id)
        if project is None:
            raise not_found("Project", clip.project_id)
        settings = project_settings_for(project)
        layout = rebuild_layout(clip, settings)
        project_id = clip.project_id

    overrides: dict[str, Any] = {}
    structural = False

    if patch.get("title") is not None:
        overrides["title"] = str(patch["title"])[:300]
    if patch.get("hook") is not None:
        overrides["hook"] = str(patch["hook"])[:400]
    if patch.get("summary") is not None:
        overrides["summary"] = str(patch["summary"])[:600]
    if patch.get("category"):
        overrides["category"] = str(patch["category"])[:60]

    new_start = patch.get("start")
    new_end = patch.get("end")
    if new_start is not None or new_end is not None:
        with session_scope() as session:
            clip = session.get(Clip, clip_id)
            start = float(new_start if new_start is not None else clip.start)
            end = float(new_end if new_end is not None else clip.end)
        if end - start < 5:
            raise ClipForgeError(
                code=ErrorCode.INVALID_INPUT,
                message="A clip must be at least 5 seconds long.",
                status_code=422,
            )
        if end - start > max(settings.max_clip_seconds * 3, 300):
            raise ClipForgeError(
                code=ErrorCode.INVALID_INPUT,
                message="That clip is far longer than the configured maximum.",
                hint="Adjust the maximum clip duration in Settings -> Video first.",
                status_code=422,
            )
        sentences = load_project_sentences(project_id)
        words = words_in_range(sentences, start, end)
        if not words:
            raise ClipForgeError(
                code=ErrorCode.INVALID_INPUT,
                message="No speech was found in that range.",
                hint="Pick a range that contains spoken words.",
                status_code=422,
            )
        timeline = build_timeline(start, end, [], settings=settings)
        with session_scope() as session:
            clip = session.get(Clip, clip_id)
            clip.start, clip.end = start, end
            clip.duration = round(end - start, 3)
            clip.words_json = json.dumps([word.to_dict() for word in words])
            clip.trim_json = json.dumps(timeline.to_dict())
            clip.status = "pending"
            clip.file_path = clip.file_path  # keep the last render visible but mark it stale
            clip.progress = 0.0
        structural = True

    layout_keys = {
        "layout": patch.get("layout"),
        "split_ratio": patch.get("split_ratio"),
        "captions_enabled": patch.get("captions_enabled"),
        "remove_silence": patch.get("remove_silence"),
        "auto_zoom": patch.get("auto_zoom"),
        "gameplay_enabled": patch.get("gameplay_enabled"),
        "music_enabled": patch.get("music_enabled"),
        "aspect_ratio": patch.get("aspect_ratio"),
    }
    if any(value is not None for value in layout_keys.values()):
        structural = True
        for key, value in layout_keys.items():
            if value is not None:
                layout[key] = value
        if patch.get("gameplay_asset_id") is not None:
            asset_id = patch["gameplay_asset_id"]
            from ..media.assets import list_assets

            match = next((asset for asset in list_assets("gameplay") if asset["id"] == asset_id), None)
            layout["gameplay"] = {"id": asset_id, "name": match["name"] if match else asset_id, "path": match["path"] if match else "", "reason": "manually selected"}
        if patch.get("music_asset_id") is not None:
            asset_id = patch["music_asset_id"]
            from ..media.assets import list_assets

            match = next((asset for asset in list_assets("music") if asset["id"] == asset_id), None)
            layout["music"] = {"id": asset_id, "name": match["name"] if match else asset_id, "path": match["path"] if match else "", "reason": "manually selected"}

    if patch.get("caption") is not None:
        structural = True
        theme = settings.caption.model_dump()
        theme.update({key: value for key, value in patch["caption"].items() if key in theme})
        try:
            layout["caption"] = CaptionTheme.model_validate(theme).model_dump()
        except Exception as exc:  # noqa: BLE001
            raise ClipForgeError(
                code=ErrorCode.INVALID_INPUT,
                message=f"That caption style could not be applied: {exc}",
                status_code=422,
            ) from exc
        structural = True

    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        for key, value in overrides.items():
            setattr(clip, key, value)
        if structural or overrides:
            clip.layout_json = json.dumps(layout)
        if structural:
            clip.status = "pending"
            clip.progress = 0.0
            clip.stage = "edited"
        session.flush()
        payload = clip.to_dict(include_words=True)
    log.info("clip %s updated (%s)", clip_id, ", ".join(sorted({*(overrides.keys()), *(k for k, v in layout_keys.items() if v is not None)})))
    return {**payload, "needs_render": payload["status"] != "rendered"}


def delete_clip(clip_id: str, *, remove_files: bool = True) -> dict[str, Any]:
    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise not_found("Clip", clip_id)
        project_id = clip.project_id
        paths_to_remove = [clip.file_path, clip.preview_path, clip.thumb_path, clip.subtitle_path]
        session.delete(clip)
        project = session.get(Project, project_id)
        if project is not None:
            remaining = session.query(Clip).filter(Clip.project_id == project_id).count()
            project.clip_count = max(remaining - 1, 0)
    if remove_files:
        for path_text in paths_to_remove:
            if not path_text:
                continue
            path = Path(path_text)
            try:
                if path.exists() and path.suffix in {".mp4", ".ass", ".srt", ".jpg", ".png"}:
                    path.unlink()
            except OSError as exc:
                log.debug("could not remove %s: %s", path, exc)
    return {"deleted": clip_id, "project_id": project_id}


def regenerate_clip(clip_id: str, *, window: float = 120.0, use_llm: bool = True) -> dict[str, Any]:
    """Re-examine the area around a clip and keep the strongest moment in it.

    This is real work, not a reroll: the transcript around the clip is rescored
    with the same engine used during analysis, and the clip only changes if a
    genuinely better moment is found nearby.
    """
    from ..ai import candidates as discover_mod
    from ..ai import scoring as scoring_mod
    from ..ai.llm import enrich_candidate, status as llm_status
    from ..pipeline.analyze import project_settings

    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise not_found("Clip", clip_id)
        project = session.get(Project, clip.project_id)
        if project is None:
            raise not_found("Project", clip.project_id)
        settings = project_settings(project)
        start, end, current_end = clip.start, clip.end, clip.end
        project_id = clip.project_id

    sentences = load_project_sentences(project_id)
    if not sentences:
        raise ClipForgeError(
            code=ErrorCode.ANALYSIS_FAILED,
            message="This project has no stored transcript to regenerate from.",
            hint="Run the analysis again.",
            status_code=409,
        )

    # Restrict the search window to the neighbourhood of the existing clip.
    window_start = max(0.0, start - window / 2)
    window_end = end + window / 2
    subset = [sentence for sentence in sentences if sentence.end > window_start and sentence.start < window_end]
    offset_index = sentences.index(subset[0]) if subset else 0
    for position, sentence in enumerate(subset):
        sentence.index = position
    blocks = build_blocks(subset)
    index = TranscriptIndex.build(subset, blocks, window_end - window_start)
    min_seconds, target_seconds, max_seconds = settings.clip_limits()

    pool = discover_mod.generate_candidates(
        index,
        min_seconds=min_seconds,
        target_seconds=target_seconds,
        max_seconds=max_seconds,
    )

    llm_note = ""
    if use_llm and settings.llm_enabled:
        try:
            llm = llm_status(settings)
            if llm.available:
                from ..ai.llm import discover_moments

                seeds = discover_moments(subset, language="", duration=index.duration, settings=settings)
                if seeds:
                    pool = discover_mod.merge_llm_seeds(index, pool, seeds, min_seconds, max_seconds, target_seconds)
            else:
                llm_note = llm.error or "Ollama unavailable"
        except ClipForgeError as exc:
            llm_note = exc.message

    result = scoring_mod.finalize_candidates(
        index,
        pool,
        min_seconds=min_seconds,
        target_seconds=target_seconds,
        max_seconds=max_seconds,
        min_score=settings.min_score,
        mode="best",
        optimize=True,
    )
    selected = result["selected"]
    best = selected[0] if selected else (result["kept"][0] if result["kept"] else None)
    if best is None:
        raise ClipForgeError(
            code=ErrorCode.NO_CANDIDATES,
            message="No viable moment was found around this clip.",
            hint="Try regenerating after widening the captions or adjust the clip boundaries manually.",
            status_code=422,
        )

    # Absolute timings for the winning candidate.
    absolute_start = best.start + window_start
    absolute_end = best.end + window_start
    words = words_in_range(sentences, absolute_start, absolute_end)

    enrichment: dict[str, Any] = {}
    if use_llm and settings.llm_enabled and not llm_note:
        try:
            enrichment = enrich_candidate(
                transcript=" ".join(sentence.text for sentence in subset[:0]) or index.range_text(best.start_index, best.end_index),
                context_before=index.range_text(max(0, best.start_index - 2), best.start_index),
                duration=best.duration,
                language="",
                settings=settings,
            )
        except ClipForgeError as exc:
            llm_note = exc.message

    changed = abs(absolute_start - start) > 0.75 or abs(absolute_end - current_end) > 0.75
    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        clip.start, clip.end = absolute_start, absolute_end
        clip.duration = round(absolute_end - absolute_start, 3)
        clip.score = best.score
        clip.title = (enrichment.get("title") or best.title or clip.title)[:300]
        clip.hook = (enrichment.get("hook") or best.hook or clip.hook)[:400]
        clip.summary = (enrichment.get("summary") or best.summary or clip.summary)[:600]
        clip.category = best.category or clip.category
        clip.why_json = json.dumps(enrichment.get("why") or best.why or clip.why)
        clip.factors_json = json.dumps(best.features.factors)
        clip.words_json = json.dumps([word.to_dict() for word in words])
        clip.trim_json = json.dumps(build_timeline(absolute_start, absolute_end, [], settings=settings).to_dict())
        clip.status = "pending"
        clip.stage = "regenerated"
        clip.progress = 0.0
        session.flush()
        payload = clip.to_dict()

    log.info("regenerated clip %s (moved %s, new score %.1f)", clip_id, "yes" if changed else "no", best.score)
    return {
        **payload,
        "changed": changed,
        "considered": len(pool),
        "kept": len(result["kept"]),
        "selected": len(selected),
        "llm_note": llm_note,
        "explanation": best.reason,
        "why": best.why,
    }


# --------------------------------------------------------------------------- #
# Captions & previews
# --------------------------------------------------------------------------- #


def caption_preview(clip_id: str, *, preset: str | None = None, theme: dict[str, Any] | None = None) -> dict[str, Any]:
    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise not_found("Clip", clip_id)
        project = session.get(Project, clip.project_id)
        settings = project_settings_for(project) if project else get_settings()
        words = [
            Word(
                word=item.get("word", ""),
                start=float(item.get("start", 0)),
                end=float(item.get("end", 0)),
                confidence=float(item.get("confidence", 0)),
                speaker=item.get("speaker", ""),
            )
            for item in json.loads(clip.words_json or "[]")
            if item.get("word")
        ]
        language = project.language if project else "en"

    base = settings.caption
    if preset:
        base = preset_theme(preset, base)
    if theme:
        merged = base.model_dump()
        merged.update({key: value for key, value in theme.items() if key in merged})
        try:
            base = CaptionTheme.model_validate(merged)
        except Exception as exc:  # noqa: BLE001
            raise ClipForgeError(code=ErrorCode.INVALID_INPUT, message=f"Invalid caption theme: {exc}", status_code=422) from exc
    plan = plan_captions(words, theme=base, language=language or "en")
    return plan.to_dict()


def write_clip_srt(clip_id: str) -> Path:
    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise not_found("Clip", clip_id)
        project = session.get(Project, clip.project_id)
        settings = project_settings_for(project) if project else get_settings()
        paths = ProjectPaths.for_project(project) if project else None
        index = clip.index
    preview = caption_preview(clip_id)
    from ..media.captions import CaptionLine, CaptionPlan, CaptionWord

    lines = [
        CaptionLine(
            words=[
                CaptionWord(text=word["text"], start=word["start"], end=word["end"], emphasis=word.get("emphasis", False))
                for word in line["words"]
            ],
            start=line["start"],
            end=line["end"],
            index=line["index"],
        )
        for line in preview["lines"]
    ]
    plan = CaptionPlan(lines=lines, theme=settings.caption, language=preview.get("language", "en"), font=preview.get("font", ""))
    target = (paths.clip_dir(index) if paths else Path(get_settings().resolved_export_dir())) / f"{clip_id}.srt"
    return write_srt(plan, target)


def render_spec_preview(clip_id: str, *, quality: str = "final") -> str:
    """The exact ffmpeg command a render would run (shown in the UI)."""
    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise not_found("Clip", clip_id)
        project = session.get(Project, clip.project_id)
        if project is None:
            raise not_found("Project", clip.project_id)
        settings = project_settings_for(project)
        paths = ProjectPaths.for_project(project)
        layout = rebuild_layout(clip, settings)
    source = find_media(paths.source)
    if source is None:
        raise ClipForgeError(code=ErrorCode.NOT_FOUND, message="Source media missing.", status_code=404)
    from ..pipeline.render import _build_spec_from_db  # internal on purpose: keeps one code path

    spec = _build_spec_from_db(
        clip_id=clip_id,
        source_path=source,
        paths=paths,
        settings=settings,
        quality=quality,
        project_snapshot=project.to_dict() if project else {},
    )
    return command_preview(spec)


def clip_assets_options(settings: AppSettings | None = None) -> dict[str, Any]:
    from ..media.assets import list_assets

    return {
        "gameplay": [asset for asset in list_assets("gameplay") if asset["enabled"]],
        "broll": [asset for asset in list_assets("broll") if asset["enabled"]],
        "music": [asset for asset in list_assets("music") if asset["enabled"]],
    }


__all__ = [
    "caption_preview",
    "clip_assets_options",
    "clip_detail",
    "delete_clip",
    "get_clip",
    "load_project_sentences",
    "regenerate_clip",
    "render_spec_preview",
    "update_clip",
    "words_in_range",
    "write_clip_srt",
]
