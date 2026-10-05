"""Project lifecycle: creation, listing, status, transcripts and cleanup."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from ..ai.language import detect_language, language_name
from ..config import AppSettings, merge_settings_patch
from ..constants import STAGES, stage_public_list
from ..db import Candidate, Clip, Job, Project, TranscriptSegment, session_scope, slugify
from ..errors import not_found
from ..logging_setup import get_logger
from ..media.download import classify_url, parse_youtube_url
from ..pipeline.context import ProjectPaths, find_media
from . import events

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Creation
# --------------------------------------------------------------------------- #


def create_project(
    *,
    url: str = "",
    title: str = "",
    source_type: str = "youtube",
    options: dict[str, Any] | None = None,
    source_path: Path | None = None,
    move_source: bool = False,
) -> dict[str, Any]:
    """Create a project row (no analysis yet).

    ``source_path`` (uploads and local imports) is copied into the project -
    or moved, with ``move_source``, when it is a temporary upload file.
    """
    if source_type == "youtube":
        video_id = parse_youtube_url(url)
    elif source_type == "url":
        # Any other link: a direct .mp4 or a site yt-dlp can extract from.
        video_id = classify_url(url).source_id
    else:
        video_id = ""

    if title:
        name = title
    elif source_type == "youtube":
        name = "YouTube video"
    elif source_type == "url":
        name = "Linked video"
    else:
        name = source_path.stem if source_path else "Uploaded video"
    with session_scope() as session:
        project = Project(
            title=name[:400],
            slug=slugify(name),
            source_url=url,
            source_type=source_type,
            source_id=video_id,
            status="draft",
            stage="draft",
            status_message="ready to analyse",
            settings_json=json.dumps(options or {}),
        )
        session.add(project)
        session.flush()
        project_id = project.id
        snapshot = project.to_dict()

    paths = ProjectPaths.for_snapshot(snapshot).ensure()
    # Store the layout now: the title changes once the real metadata arrives and
    # the folder must not move out from under the files already inside it.
    with session_scope() as session:
        project = session.get(Project, project_id)
        if project is not None:
            project.paths_json = json.dumps(paths.to_dict())

    if source_path is not None and source_type == "upload":
        from ..media.download import safe_upload_path

        target = safe_upload_path(paths.source, source_path.name)
        if source_path.resolve() != target.resolve():
            if move_source:
                shutil.move(str(source_path), str(target))
            else:
                shutil.copy2(source_path, target)

    log.info("created project %s (%s)", project_id, name[:60])
    events.publish("project.created", {"project": snapshot}, project_id=project_id)
    return project_detail(project_id)


def update_project(project_id: str, **fields: Any) -> dict[str, Any]:
    with session_scope() as session:
        project = session.get(Project, project_id)
        if project is None:
            raise not_found("Project", project_id)
        for key, value in fields.items():
            if key == "settings" and isinstance(value, dict):
                project.settings_json = json.dumps(value)
            else:
                setattr(project, key, value)
        session.flush()
        snapshot = project.to_dict()
    events.publish("project.updated", {"project": snapshot}, project_id=project_id)
    return snapshot


def merge_project_options(project_id: str, options: dict[str, Any]) -> dict[str, Any]:
    with session_scope() as session:
        project = session.get(Project, project_id)
        if project is None:
            raise not_found("Project", project_id)
        patch = {key: value for key, value in options.items() if value is not None and key in AppSettings.model_fields}
        current = merge_settings_patch(project.settings, patch)
        project.settings_json = json.dumps(current)
        session.flush()
        return current


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def list_projects(*, search: str = "", limit: int = 100, offset: int = 0) -> dict[str, Any]:
    with session_scope() as session:
        query = select(Project)
        if search:
            query = query.where(Project.title.ilike(f"%{search.strip()}%"))
        total = session.execute(select(func.count()).select_from(query.subquery())).scalar_one()
        projects = session.execute(query.order_by(Project.created_at.desc()).offset(offset).limit(limit)).scalars().all()
        ids = [project.id for project in projects]

        clip_counts: dict[str, dict[str, int]] = {}
        active: dict[str, dict[str, Any]] = {}
        if ids:
            rows = session.execute(
                select(Clip.project_id, Clip.status, func.count(Clip.id)).where(Clip.project_id.in_(ids)).group_by(Clip.project_id, Clip.status)
            ).all()
            for project_id, status, count in rows:
                clip_counts.setdefault(project_id, {})[status] = count
            jobs = session.execute(
                select(Job).where(Job.project_id.in_(ids), Job.status.in_(("queued", "running"))).order_by(Job.created_at.asc())
            ).scalars().all()
            for job in jobs:
                current = active.get(job.project_id)
                if current is None or (job.status == "running" and current["status"] != "running"):
                    active[job.project_id] = job.to_dict(with_log=False)

        items = []
        for project in projects:
            counts = clip_counts.get(project.id, {})
            rendered = counts.get("rendered", 0)
            items.append(
                {
                    **project.to_dict(),
                    "rendered_clips": rendered,
                    "pending_clips": sum(counts.values()) - rendered,
                    "active_job": active.get(project.id),
                }
            )
    return {"projects": items, "total": int(total)}


def project_detail(project_id: str) -> dict[str, Any]:
    with session_scope() as session:
        project = session.get(Project, project_id)
        if project is None:
            raise not_found("Project", project_id)
        payload = project.to_dict()
        clips = session.execute(select(Clip).where(Clip.project_id == project_id).order_by(Clip.index)).scalars().all()
        candidates = (
            session.execute(select(Candidate).where(Candidate.project_id == project_id).order_by(Candidate.idx)).scalars().all()
        )
        jobs = (
            session.execute(select(Job).where(Job.project_id == project_id).order_by(Job.created_at.desc()).limit(20))
            .scalars()
            .all()
        )
    paths = ProjectPaths.for_snapshot(payload)
    media_path = find_media(paths.source)
    return {
        **payload,
        "language_name": language_name(payload.get("language", "")),
        "paths": paths.to_dict(),
        "source_file": str(media_path) if media_path else "",
        "has_source": media_path is not None,
        "clips": [clip.to_dict() for clip in clips],
        "candidate_count_stored": len(candidates),
        "jobs": [job.to_dict(with_log=False) for job in jobs],
        "stages": stage_public_list(),
    }


def project_status(project_id: str) -> dict[str, Any]:
    """Progress payload polled by the analysis screen (also pushed over SSE)."""
    with session_scope() as session:
        project = session.get(Project, project_id)
        if project is None:
            raise not_found("Project", project_id)
        job = (
            session.execute(
                select(Job)
                .where(Job.project_id == project_id, Job.kind == "analyze")
                .order_by(Job.created_at.desc())
                .limit(1)
            )
            .scalars()
            .first()
        )
        clip_rows = session.execute(select(Clip).where(Clip.project_id == project_id)).scalars().all()
        candidate_rows = session.execute(select(Candidate).where(Candidate.project_id == project_id)).scalars().all()
        payload = project.to_dict()
        job_payload = job.to_dict() if job else None

    stage_states = _stage_states(payload, job_payload)
    analysis_stats = (payload.get("stats") or {}).get("analysis", {})
    diagnostics = analysis_stats.get("diagnostics", {}) if isinstance(analysis_stats, dict) else {}
    candidate_statuses: dict[str, int] = {}
    for candidate in candidate_rows:
        candidate_statuses[candidate.status] = candidate_statuses.get(candidate.status, 0) + 1
    rendered = sum(1 for clip in clip_rows if clip.status == "rendered")
    diagnostics = {
        **(diagnostics if isinstance(diagnostics, dict) else {}),
        "queued": sum(1 for clip in clip_rows if clip.status == "queued"),
        "rendered": rendered,
        "rendering": sum(1 for clip in clip_rows if clip.status == "rendering"),
    }
    has_source = find_media(ProjectPaths.for_snapshot(payload).source) is not None
    return {
        "project": {
            "id": payload["id"],
            "title": payload["title"],
            "channel": payload["channel"],
            "source_type": payload["source_type"],
            "source_url": payload["source_url"],
            "has_source": has_source,
            "word_count": payload["word_count"],
            "segment_count": payload["segment_count"],
            "transcript_source": payload["transcript_source"],
            "transcript_preference": payload["transcript_preference"],
            "transcript_filename": payload["transcript_filename"],
            "transcript_timing": payload["transcript_timing"],
            "status": payload["status"],
            "stage": payload["stage"],
            "progress": payload["progress"],
            "status_message": payload["status_message"],
            "duration": payload["duration"],
            "thumbnail_url": payload["thumbnail_url"],
            "language": payload["language"],
            "language_name": language_name(payload.get("language", "")),
            "language_mode": payload["language_mode"],
            "language_secondary": payload["language_secondary"],
            "speakers": payload["speakers"],
            "clip_count": payload["clip_count"],
            "candidate_count": payload["candidate_count"],
            "error": payload["error"],
            "stats": payload["stats"],
        },
        "job": job_payload,
        "stages": stage_states,
        "candidates": {
            "discovered": int(diagnostics.get("discovered", payload["candidate_count"]) or 0),
            "stored": len(candidate_rows),
            "scored": int(diagnostics.get("scored", 0) or 0),
            "scoring_errors": int(diagnostics.get("scoring_errors", 0) or 0),
            "threshold_pass": int(diagnostics.get("threshold_pass", 0) or 0),
            "context_rejections": int(diagnostics.get("context_rejections", 0) or 0),
            "overlap_rejections": int(diagnostics.get("overlap_rejections", 0) or 0),
            "final_accepted": int(diagnostics.get("final_accepted", payload["clip_count"]) or 0),
            "average_score": float(diagnostics.get("average_score", 0.0) or 0.0),
            "highest_score": float(diagnostics.get("highest_score", 0.0) or 0.0),
            "threshold": float(diagnostics.get("threshold", 0.0) or 0.0),
            "queued": int(diagnostics.get("queued", 0) or 0),
            "rendered": int(diagnostics.get("rendered", 0) or 0),
            "top_rejection_reason": diagnostics.get("top_rejection_reason", ""),
            "statuses": candidate_statuses,
        },
        "diagnostics": diagnostics,
        "clips": {
            "total": len(clip_rows),
            "rendered": rendered,
            "failed": sum(1 for clip in clip_rows if clip.status == "failed"),
            "pending": sum(1 for clip in clip_rows if clip.status == "pending"),
            "queued": sum(1 for clip in clip_rows if clip.status == "queued"),
            "rendering": sum(1 for clip in clip_rows if clip.status == "rendering"),
        },
    }


def _stage_states(project: dict[str, Any], job: dict[str, Any] | None) -> list[dict[str, Any]]:
    current_stage = (job or {}).get("stage") or project.get("stage") or ""
    order = [stage.key for stage in STAGES]
    current_index = order.index(current_stage) if current_stage in order else -1
    status = project.get("status", "draft")
    output: list[dict[str, Any]] = []
    for index, stage in enumerate(STAGES):
        if status == "ready":
            state = "done"
        elif status in {"failed", "cancelled"} and current_index == index:
            state = "failed"
        elif current_index > index:
            state = "done"
        elif current_index == index:
            state = "running"
        elif current_index == -1 and status in {"draft", "queued"}:
            state = "pending"
        else:
            state = "pending"
        output.append(
            {
                "key": stage.key,
                "label": stage.label,
                "detail": stage.detail,
                "weight": stage.weight,
                "state": state,
                "progress": round(float(project.get("progress", 0.0)), 4) if state == "running" else (1.0 if state == "done" else 0.0),
            }
        )
    return output


def project_transcript(
    project_id: str,
    *,
    with_words: bool = True,
    start: float | None = None,
    end: float | None = None,
    limit: int = 5000,
) -> dict[str, Any]:
    with session_scope() as session:
        project = session.get(Project, project_id)
        if project is None:
            raise not_found("Project", project_id)
        query = select(TranscriptSegment).where(TranscriptSegment.project_id == project_id).order_by(TranscriptSegment.idx)
        if start is not None:
            query = query.where(TranscriptSegment.end >= start)
        if end is not None:
            query = query.where(TranscriptSegment.start <= end)
        rows = session.execute(query.limit(limit)).scalars().all()
        payload = project.to_dict()
        segments = [row.to_dict(with_words=with_words) for row in rows]

    return {
        "project_id": project_id,
        "language": payload["language"],
        "language_name": language_name(payload.get("language", "")),
        "language_mode": payload["language_mode"],
        "language_secondary": payload["language_secondary"],
        "engine": (payload.get("stats", {}).get("transcript") or {}).get("engine", ""),
        "model": (payload.get("stats", {}).get("transcript") or {}).get("model", ""),
        "source": payload["transcript_source"],
        "preference": payload["transcript_preference"],
        "filename": payload["transcript_filename"],
        "timing_granularity": payload["transcript_timing"],
        "word_count": payload["word_count"],
        "segment_count": payload["segment_count"],
        "speakers": payload["speakers"],
        "segments": segments,
        "count": len(segments),
    }


def attach_transcript(project_id: str, *, filename: str, suffix: str, cues: list[dict[str, Any]]) -> dict[str, Any]:
    """Persist an uploaded cue transcript and invalidate only transcript-derived data.

    Source metadata, the downloaded/uploaded video, and extracted audio remain
    intact so replacing captions does not cause another download or extraction.
    """
    if not cues:
        raise ValueError("attach_transcript requires at least one parsed cue")
    text = " ".join(str(cue.get("text") or "") for cue in cues)
    language = detect_language(text)
    source = f"uploaded_{suffix.lstrip('.').lower()}"
    word_count = sum(len(str(cue.get("text") or "").split()) for cue in cues)

    with session_scope() as session:
        project = session.get(Project, project_id)
        if project is None:
            raise not_found("Project", project_id)
        session.query(TranscriptSegment).filter(TranscriptSegment.project_id == project_id).delete()
        session.query(Candidate).filter(Candidate.project_id == project_id).delete()
        session.query(Clip).filter(Clip.project_id == project_id).delete()
        for index, cue in enumerate(cues):
            session.add(
                TranscriptSegment(
                    project_id=project_id,
                    idx=index,
                    start=float(cue["start"]),
                    end=float(cue["end"]),
                    text=str(cue["text"]),
                    speaker="SPEAKER_01",
                    language=language.primary,
                    confidence=1.0,
                    words_json="[]",
                )
            )
        project.transcript_source = source
        project.transcript_preference = source
        project.transcript_filename = filename[:260]
        project.transcript_timing = "cue"
        project.segment_count = len(cues)
        project.word_count = word_count
        project.language = language.primary
        project.language_secondary = language.secondary
        project.language_mode = language.mode
        project.language_confidence = language.confidence
        project.candidate_count = 0
        project.clip_count = 0
        project.status = "draft"
        project.stage = "transcript"
        project.progress = 0.0
        project.status_message = f"{len(cues)} transcript cues ready; run analysis"
        project.error_code = ""
        project.error_message = ""
        project.error_hint = ""
        project.analyzed_at = None
        try:
            stats = json.loads(project.stats_json or "{}")
        except (TypeError, ValueError):
            stats = {}
        stats.pop("analysis", None)
        stats["transcript"] = {
            "engine": "uploaded transcript",
            "model": filename,
            "source": source,
            "filename": filename,
            "language": language.primary,
            "timing_granularity": "cue",
            "word_timing_count": 0,
            "segment_count": len(cues),
        }
        stats.pop("speakers", None)
        project.stats_json = json.dumps(stats)
        snapshot = project.to_dict()

    paths = ProjectPaths.for_snapshot(snapshot)
    for name in ("asr_cache.json", "transcript.json", "words.jsonl", "engine.json", "language.json"):
        target = paths.transcript / name
        if name == "language.json":
            target = paths.analysis / name
        try:
            target.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("could not invalidate transcript cache %s: %s", target, exc)
    if paths.analysis.exists():
        for target in paths.analysis.glob("*.json"):
            try:
                target.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("could not invalidate transcript analysis %s: %s", target, exc)

    updated = project_detail(project_id)
    events.publish("project.updated", {"project": updated}, project_id=project_id)
    events.publish(
        "transcript.updated",
        {"project_id": project_id, "source": source, "filename": filename, "segment_count": len(cues)},
        project_id=project_id,
    )
    return updated


def project_candidates(project_id: str, *, status: str = "", limit: int = 300) -> dict[str, Any]:
    with session_scope() as session:
        project = session.get(Project, project_id)
        if project is None:
            raise not_found("Project", project_id)
        query = select(Candidate).where(Candidate.project_id == project_id)
        if status:
            query = query.where(Candidate.status == status)
        rows = session.execute(query.order_by(Candidate.score.desc()).limit(limit)).scalars().all()
        counts: dict[str, int] = {}
        for row in session.execute(select(Candidate).where(Candidate.project_id == project_id)).scalars().all():
            counts[row.status] = counts.get(row.status, 0) + 1
    return {
        "candidates": [row.to_dict() for row in rows],
        "counts": counts,
        "analysis": (project.to_dict().get("stats") or {}).get("analysis", {}),
    }


def project_clips(
    project_id: str,
    *,
    sort: str = "score",
    category: str = "",
    min_score: float = 0.0,
    status: str = "",
) -> dict[str, Any]:
    with session_scope() as session:
        project = session.get(Project, project_id)
        if project is None:
            raise not_found("Project", project_id)
        rows = session.execute(select(Clip).where(Clip.project_id == project_id).order_by(Clip.index)).scalars().all()
        clip_payloads = [row.to_dict() for row in rows]
        project_payload = project.to_dict()

    if category:
        clip_payloads = [clip for clip in clip_payloads if clip["category"] == category]
    if min_score:
        clip_payloads = [clip for clip in clip_payloads if clip["score"] >= min_score]
    if status:
        clip_payloads = [clip for clip in clip_payloads if clip["status"] == status]

    sorters = {
        "score": lambda clip: -clip["score"],
        "duration": lambda clip: -clip["duration"],
        "duration_asc": lambda clip: clip["duration"],
        "category": lambda clip: (clip["category"], -clip["score"]),
        "chronological": lambda clip: clip["start"],
        "title": lambda clip: clip["title"].lower(),
    }
    clip_payloads.sort(key=sorters.get(sort, sorters["score"]))

    categories: dict[str, int] = {}
    for clip in clip_payloads:
        categories[clip["category"]] = categories.get(clip["category"], 0) + 1

    return {
        "project": {
            "id": project_payload["id"],
            "title": project_payload["title"],
            "duration": project_payload["duration"],
            "language": project_payload["language"],
            "language_name": language_name(project_payload.get("language", "")),
            "language_mode": project_payload["language_mode"],
            "speakers": project_payload["speakers"],
            "clip_count": len(clip_payloads),
            "analysis": (project_payload.get("stats") or {}).get("analysis", {}),
            "settings": project_payload["settings"],
        },
        "clips": clip_payloads,
        "categories": dict(sorted(categories.items(), key=lambda item: -item[1])),
        "count": len(clip_payloads),
        "sort": sort,
    }


# --------------------------------------------------------------------------- #
# Deletion / maintenance
# --------------------------------------------------------------------------- #


def delete_project(project_id: str, *, remove_files: bool = True) -> dict[str, Any]:
    with session_scope() as session:
        if session.get(Project, project_id) is None:
            raise not_found("Project", project_id)
    # Stop the project's work first: a running analysis or render would keep
    # writing into the folder that is about to disappear.
    from ..jobs.manager import manager

    manager().cancel_project(project_id)
    with session_scope() as session:
        project = session.get(Project, project_id)
        if project is None:
            raise not_found("Project", project_id)
        snapshot = project.to_dict()
        session.query(Clip).filter(Clip.project_id == project_id).delete()
        session.query(Candidate).filter(Candidate.project_id == project_id).delete()
        session.query(TranscriptSegment).filter(TranscriptSegment.project_id == project_id).delete()
        session.query(Job).filter(Job.project_id == project_id).delete()
        session.delete(project)

    removed = False
    if remove_files:
        paths = ProjectPaths.for_snapshot(snapshot)
        if paths.root.exists():
            try:
                shutil.rmtree(paths.root)
                removed = True
            except OSError as exc:
                log.warning("could not remove project folder %s: %s", paths.root, exc)
    events.publish("project.deleted", {"project_id": project_id}, project_id=project_id)
    log.info("deleted project %s (files removed: %s)", project_id, removed)
    return {"deleted": project_id, "files_removed": removed}


def project_disk_usage(project_id: str) -> dict[str, Any]:
    with session_scope() as session:
        project = session.get(Project, project_id)
        if project is None:
            raise not_found("Project", project_id)
        snapshot = project.to_dict()
    paths = ProjectPaths.for_snapshot(snapshot)
    total = 0
    breakdown: dict[str, int] = {}
    if paths.root.exists():
        for path in paths.root.rglob("*"):
            if path.is_file():
                size = path.stat().st_size
                total += size
                breakdown[path.parent.name] = breakdown.get(path.parent.name, 0) + size
    return {"project_id": project_id, "root": str(paths.root), "total_bytes": total, "breakdown": breakdown}


__all__ = [
    "create_project",
    "delete_project",
    "list_projects",
    "merge_project_options",
    "project_candidates",
    "project_clips",
    "project_detail",
    "project_disk_usage",
    "project_status",
    "project_transcript",
    "update_project",
]
