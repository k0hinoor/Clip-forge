"""Clip-level operations: editing, regeneration, deletion, caption previews.

Editing a clip never touches the source media: the moment's words are re-read
from the stored transcript, the pace/caption/framing plan is rebuilt, and the
clip is marked as needing a fresh render. Renders therefore always match the
plan the user is looking at.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import func, select

from ..ai.features import TranscriptIndex
from ..ai.segment import Sentence, build_blocks, build_sentences
from ..ai.transcribe import Utterance, Word
from ..config import AppSettings, CaptionTheme, get_settings, merge_settings_patch
from ..constants import CAPTION_PRESETS
from ..db import Clip, Project, TranscriptSegment, session_scope
from ..errors import ClipForgeError, ErrorCode, not_found
from ..logging_setup import get_logger
from ..media.captions import plan_captions, preset_theme, write_srt
from ..media.compose import command_preview
from ..pipeline.context import ProjectPaths, find_media
from ..pipeline.edit import (
    clip_settings,
    ensure_layout_asset,
    layout_payload,
    load_timeline,
    plan_timeline,
    rebuild_layout,
    set_layout_payload,
)
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
        project = session.get(Project, project_id)
        preserve_cues = bool(project and project.transcript_timing == "cue")
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
    return build_sentences(utterances, preserve_cue_boundaries=preserve_cues)


def _stored_layout(clip_id: str) -> dict[str, Any]:
    """The clip's stored plan layout block (used to show the effective style)."""
    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            return {}
        return layout_payload(rebuild_layout(clip, get_settings()))


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
        plan = rebuild_layout(clip, settings)
        layout = layout_payload(plan)
        timeline = load_timeline(clip, settings)
        project_snapshot = project.to_dict()

    settings = clip_settings(settings, layout)  # honour the clip's own caption choices
    captions = None
    if settings.captions_enabled:
        from ..pipeline.edit import rebuild_captions

        with session_scope() as session:
            clip = session.get(Clip, clip_id)
            caption_plan = rebuild_captions(clip, settings, plan.get("language", "en"), [point.get("time", 0) for point in plan.get("zoom_points") or []])
        # The full theme travels with the plan so the editor can show which
        # style is actually active (and re-render with exactly that style).
        captions = {**caption_plan.to_dict(), "theme": settings.caption.model_dump()} if caption_plan else None

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
            "zoom_points": plan.get("zoom_points") or layout.get("zoom_points") or [],
            "crop": plan.get("crop"),
            "timeline": timeline.to_dict(),
        },
        "captions": captions,
        "captions_enabled": bool(settings.captions_enabled),
        "caption_presets": sorted(CAPTION_PRESETS),
        "settings": {
            "aspect_ratio": settings.aspect_ratio,
            "remove_silence": settings.remove_silence,
            "auto_zoom": settings.auto_zoom,
            "gameplay_enabled": settings.gameplay_enabled,
            "music_enabled": settings.music_enabled,
        },
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
        plan = rebuild_layout(clip, settings)
        layout = layout_payload(plan)
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
        _retime_clip(clip_id, project_id, new_start, new_end, clip_settings(settings, layout), plan)
        structural = True

    layout_keys = {key: patch.get(key) for key in LAYOUT_PATCH_KEYS}
    for key, value in layout_keys.items():
        if value is not None:
            layout[key] = value
            structural = True

    if layout_keys["remove_silence"] is not None and new_start is None and new_end is None:
        # Pacing is baked into the stored timeline, so recompute it.
        _retime_clip(clip_id, project_id, None, None, clip_settings(settings, layout), plan)

    for kind in ("gameplay", "broll", "music"):
        asset_id = patch.get(f"{kind}_asset_id")
        if asset_id is None:
            continue
        layout[kind] = _asset_entry(kind, str(asset_id)) if asset_id else None
        structural = True

    asset_change = any(patch.get(f"{kind}_asset_id") is not None for kind in ("gameplay", "broll", "music"))
    if asset_change or layout_keys.get("layout") or layout_keys.get("gameplay_enabled") or layout_keys.get("music_enabled"):
        note = ensure_layout_asset(
            layout,
            clip_settings(settings, layout),
            clip_key=f"{project_id}:{clip_id}",
            category=str(patch.get("category") or ""),
        )
        if note:
            layout["notes"] = [*(layout.get("notes") or []), note][-12:]

    if patch.get("caption_preset"):
        preset_name = str(patch["caption_preset"])
        if preset_name not in CAPTION_PRESETS:
            raise ClipForgeError(
                code=ErrorCode.INVALID_INPUT,
                message=f"Unknown caption style '{preset_name}'.",
                hint="Choose one of: " + ", ".join(sorted(CAPTION_PRESETS)),
                status_code=422,
            )

    if patch.get("caption_preset") or patch.get("caption") is not None:
        # Same rules as Settings: a preset applies its look, explicit keys win.
        current = {**settings.caption.model_dump(), **(layout.get("caption") or {})}
        caption_patch = dict(patch.get("caption") or {})
        if patch.get("caption_preset"):
            caption_patch.setdefault("preset", str(patch["caption_preset"]))
        merged = merge_settings_patch({"caption": current}, {"caption": caption_patch})["caption"]
        try:
            layout["caption"] = CaptionTheme.model_validate({key: value for key, value in merged.items() if key in CaptionTheme.model_fields}).model_dump()
        except ValueError as exc:
            raise ClipForgeError(code=ErrorCode.INVALID_INPUT, message=f"That caption style could not be applied: {exc}", status_code=422) from exc
        structural = True

    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        for key, value in overrides.items():
            setattr(clip, key, value)
        if structural or overrides:
            clip.layout_json = json.dumps(set_layout_payload(plan, layout))
        if structural:
            clip.status = "pending"
            clip.progress = 0.0
            clip.stage = "edited"
        session.flush()
        payload = clip.to_dict(include_words=True)
    log.info("clip %s updated (%s)", clip_id, ", ".join(sorted(patch)))
    return {**payload, "needs_render": payload["status"] != "rendered"}


# Plan keys the editor can set directly (see pipeline.edit.CLIP_SETTING_KEYS).
LAYOUT_PATCH_KEYS = ("layout", "split_ratio", "captions_enabled", "remove_silence", "auto_zoom", "gameplay_enabled", "music_enabled", "aspect_ratio")


def _asset_entry(kind: str, asset_id: str) -> dict[str, Any]:
    from ..media.assets import list_assets

    match = next((asset for asset in list_assets(kind) if asset["id"] == asset_id), None)
    if match is None:
        raise ClipForgeError(
            code=ErrorCode.NOT_FOUND,
            message=f"That {'B-roll' if kind == 'broll' else kind} asset does not exist.",
            hint="Pick one from the list, or import it in Assets first.",
            status_code=404,
        )
    return {"id": asset_id, "name": match["name"], "category": match["category"], "path": match["path"], "reason": "chosen in the editor"}


def _retime_clip(clip_id: str, project_id: str, new_start: float | None, new_end: float | None, settings: AppSettings, plan: dict[str, Any]) -> None:
    """Move a clip's boundaries: words, pacing (silence removal) and punch-ins follow."""
    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        start = float(new_start if new_start is not None else clip.start)
        end = float(new_end if new_end is not None else clip.end)
        project = session.get(Project, project_id)
        source = find_media(ProjectPaths.for_project(project).source) if project else None
        project_duration = float(project.duration or 0.0) if project else 0.0
    if project_duration and end > project_duration + 0.5:
        raise ClipForgeError(code=ErrorCode.INVALID_INPUT, message="The clip cannot end after the video does.", status_code=422)
    if end - start < 5:
        raise ClipForgeError(code=ErrorCode.INVALID_INPUT, message="A clip must be at least 5 seconds long.", status_code=422)
    if end - start > max(settings.max_clip_seconds * 3, 300):
        raise ClipForgeError(
            code=ErrorCode.INVALID_INPUT,
            message="That clip is far longer than the configured maximum.",
            hint="Raise the maximum clip length in Settings -> Clips first.",
            status_code=422,
        )
    project_sentences = load_project_sentences(project_id)
    words = words_in_range(project_sentences, start, end)
    selected_sentences = [sentence for sentence in project_sentences if sentence.end >= start and sentence.start <= end]
    if not words and not selected_sentences:
        raise ClipForgeError(
            code=ErrorCode.INVALID_INPUT,
            message="No transcript text was found in that range.",
            hint="Pick a range that overlaps a transcript cue or spoken words.",
            status_code=422,
        )
    timeline = plan_timeline(source, start, end, settings)
    # Punch-ins are output-relative: keep those that still land inside the clip.
    plan["zoom_points"] = [
        point for point in plan.get("zoom_points") or [] if 0.3 <= float(point.get("time", 0)) <= timeline.output_duration - 0.4
    ]
    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        clip.start, clip.end = start, end
        clip.duration = round(timeline.output_duration, 3)
        clip.words_json = json.dumps([word.to_dict() for word in words])
        clip.segments_json = json.dumps(
            {
                "sentences": [sentence.to_dict() for sentence in selected_sentences],
                "sentence_start": selected_sentences[0].index if selected_sentences else 0,
                "sentence_end": selected_sentences[-1].index + 1 if selected_sentences else 0,
            },
            ensure_ascii=False,
        )
        clip.trim_json = json.dumps(timeline.to_dict())


def duplicate_clip(clip_id: str) -> dict[str, Any]:
    """Copy a clip (plan, captions, layout) as a new, unrendered clip."""
    with session_scope() as session:
        original = session.get(Clip, clip_id)
        if original is None:
            raise not_found("Clip", clip_id)
        # A fresh index: the clip folder (plan, captions) is keyed by it.
        next_index = (session.execute(select(func.max(Clip.index)).where(Clip.project_id == original.project_id)).scalar_one() or 0) + 1
        clone = Clip(
            project_id=original.project_id,
            candidate_id=original.candidate_id,
            index=next_index,
            title=f"{original.title} (copy)"[:300],
            hook=original.hook,
            summary=original.summary,
            category=original.category,
            score=original.score,
            why_json=original.why_json,
            factors_json=original.factors_json,
            start=original.start,
            end=original.end,
            duration=original.duration,
            words_json=original.words_json,
            segments_json=original.segments_json,
            trim_json=original.trim_json,
            layout_json=original.layout_json,
            status="pending",
            stage="pending",
        )
        session.add(clone)
        session.flush()
        clone_id = clone.id
        project = session.get(Project, original.project_id)
        if project is not None:
            project.clip_count = session.execute(select(func.count(Clip.id)).where(Clip.project_id == original.project_id)).scalar_one()
    return clip_detail(clone_id)


def delete_clip(clip_id: str, *, remove_files: bool = True) -> dict[str, Any]:
    from ..jobs import queue as job_queue
    from ..jobs.manager import manager

    for job in job_queue.active_jobs(clip_id=clip_id):
        manager().cancel(job["id"])
    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise not_found("Clip", clip_id)
        project_id = clip.project_id
        paths_to_remove = [clip.file_path, clip.preview_path, clip.thumb_path, clip.subtitle_path]
        if clip.subtitle_path:  # the exported .srt sits next to the burned-in .ass
            paths_to_remove.append(str(Path(clip.subtitle_path).with_suffix(".srt")))
        session.delete(clip)
        session.flush()
        project = session.get(Project, project_id)
        if project is not None:
            project.clip_count = session.execute(select(func.count(Clip.id)).where(Clip.project_id == project_id)).scalar_one()
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

    # Sentences keep their source timestamps, so candidates are already absolute.
    absolute_start = best.start
    absolute_end = best.end
    words = words_in_range(sentences, absolute_start, absolute_end)
    selected_sentences = [sentence for sentence in sentences if sentence.end >= absolute_start and sentence.start <= absolute_end]

    enrichment: dict[str, Any] = {}
    if use_llm and settings.llm_enabled and not llm_note:
        try:
            enrichment = enrich_candidate(
                transcript=index.range_text(best.start_index, best.end_index),
                context_before=index.range_text(max(0, best.start_index - 2), best.start_index),
                duration=best.duration,
                language="",
                settings=settings,
            )
        except ClipForgeError as exc:
            llm_note = exc.message

    changed = abs(absolute_start - start) > 0.75 or abs(absolute_end - current_end) > 0.75
    with session_scope() as session:
        project = session.get(Project, project_id)
        source = find_media(ProjectPaths.for_project(project).source) if project else None
    timeline = plan_timeline(source, absolute_start, absolute_end, settings)
    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        clip.start, clip.end = absolute_start, absolute_end
        clip.score = best.score
        clip.title = (enrichment.get("title") or best.title or clip.title)[:300]
        clip.hook = (enrichment.get("hook") or best.hook or clip.hook)[:400]
        clip.summary = (enrichment.get("summary") or best.summary or clip.summary)[:600]
        clip.category = best.category or clip.category
        clip.why_json = json.dumps(enrichment.get("why") or best.why or clip.why)
        clip.factors_json = json.dumps(best.features.factors)
        clip.words_json = json.dumps([word.to_dict() for word in words])
        clip.segments_json = json.dumps(
            {"sentences": [sentence.to_dict() for sentence in selected_sentences]}, ensure_ascii=False
        )
        clip.trim_json = json.dumps(timeline.to_dict())
        clip.duration = round(timeline.output_duration, 3)
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


SAMPLE_CAPTION_TEXT = (
    "This is how your captions will look on every clip you render, "
    "with the key words highlighted exactly when they are spoken."
)


def _sample_words() -> list[Word]:
    words: list[Word] = []
    cursor = 0.2
    for token in SAMPLE_CAPTION_TEXT.split():
        length = 0.18 + 0.035 * len(token)
        words.append(Word(word=token, start=round(cursor, 3), end=round(cursor + length, 3), confidence=0.95))
        cursor += length + 0.06
    return words


def caption_preview(clip_id: str = "", *, preset: str | None = None, theme: dict[str, Any] | None = None) -> dict[str, Any]:
    """Caption lines for a clip (or sample text when ``clip_id`` is empty) in a given style."""
    if preset and preset not in CAPTION_PRESETS:
        raise ClipForgeError(
            code=ErrorCode.INVALID_INPUT,
            message=f"Unknown caption style '{preset}'.",
            hint="Choose one of: " + ", ".join(sorted(CAPTION_PRESETS)),
            status_code=422,
        )
    cues: list[dict[str, Any]] = []
    if clip_id:
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
            if words:  # clip words are source-timed; previews start at zero
                offset = words[0].start
                for word in words:
                    word.start, word.end = word.start - offset, word.end - offset
            language = project.language if project else "en"
            if not words:
                try:
                    stored_segments = json.loads(clip.segments_json or "{}")
                except json.JSONDecodeError:
                    stored_segments = {}
                cue_rows = stored_segments.get("sentences", []) if isinstance(stored_segments, dict) else []
                from ..pipeline.edit import _remap_cues

                cues = _remap_cues(cue_rows, load_timeline(clip, settings))
    else:
        settings = get_settings()
        words = _sample_words()
        language = "en"

    base = settings.caption
    if preset:
        base = preset_theme(preset, base)
    elif clip_id:
        # No explicit preset: fall back to the style stored on this clip.
        stored = _stored_layout(clip_id).get("caption")
        if isinstance(stored, dict) and stored:
            base = CaptionTheme.model_validate({**base.model_dump(), **stored})
    if theme:
        merged = base.model_dump()
        merged.update({key: value for key, value in theme.items() if key in merged})
        try:
            base = CaptionTheme.model_validate(merged)
        except ValueError as exc:
            raise ClipForgeError(code=ErrorCode.INVALID_INPUT, message=f"Invalid caption theme: {exc}", status_code=422) from exc
    plan = plan_captions(words, theme=base, language=language or "en", cues=cues if cues else None)
    return {**plan.to_dict(), "theme": base.model_dump(), "captions_enabled": settings.captions_enabled}


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
            text_override="" if line.get("has_word_timings", True) else line.get("text", ""),
        )
        for line in preview["lines"]
    ]
    plan = CaptionPlan(
        lines=lines,
        theme=preview.get("theme") or settings.caption,
        language=preview.get("language", "en"),
        font=preview.get("font", ""),
    )
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
        snapshot = project.to_dict()
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
        project_snapshot=snapshot,
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
    "duplicate_clip",
    "get_clip",
    "load_project_sentences",
    "regenerate_clip",
    "render_spec_preview",
    "update_clip",
    "words_in_range",
    "write_clip_srt",
]
