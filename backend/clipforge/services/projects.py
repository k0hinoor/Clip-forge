"""Project lifecycle: creation, listing, status, transcripts and cleanup."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

from sqlalchemy import select

from ..ai.language import language_name
from ..config import AppSettings, get_settings
from ..constants import STAGES, stage_public_list
from ..db import Candidate, Clip, Job, Project, TranscriptSegment, session_scope, slugify, utcnow
from ..errors import ClipForgeError, ErrorCode, not_found
from ..logging_setup import get_logger
from ..media.download import parse_youtube_url
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
) -> dict[str, Any]:
    """Create a project row (no analysis yet)."""
    if source_type == "youtube":
        video_id = parse_youtube_url(url)
    else:
        video_id = ""

    name = title or ("YouTube video" if source_type == "youtube" else (source_path.stem if source_path else "Uploaded video"))
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
    if source_path is not None and source_type != "youtube":
        from ..media.download import safe_upload_path

        target = safe_upload_path(paths.source, source_path.name)
        if source_path.resolve() != target.resolve():
            shutil.copy2(source_path, target)
        with session_scope() as session:
            project = session.get(Project, project_id)
            if project is not None:
                project.slug = slugify(name)
                project.paths_json = json.dumps(paths.to_dict())

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
        current = project.settings
        current.update({key: value for key, value in options.items() if value is not None})
        project.settings_json = json.dumps(current)
        session.flush()
        return current


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def list_projects(*, search: str = "", limit: int = 100, offset: int = 0) -> dict[str, Any]:
    with session_scope() as session:
        query = select(Project).order_by(Project.created_at.desc())
        if search:
            like = f"%{search.lower()}%"
            query = query.where(Project.title.ilike(like))
        projects = session.execute(query.offset(offset).limit(limit)).scalars().all()
        items = []
        for project in projects:
            clashed = session.execute(select(Clip).where(Clip.project_id == project.id)).scalars().all()
            updated = sum(1 for clip in clashed if clip.status == "rendered")
            active = session.execute(
                select(Job).where(Job.project_id == project.id, Job.status.in_(("queued", "running"))).limit(1)
            ).scalars().first()
            items.append(
                {
                    **project.to_dict(),
                    "rendered_clips": updated,
                    "pending_clips": len(clashed) - updated,
                    "active_job": active.to_dict(with_log=False) if active else None,
                }
            )
        total = session.execute(select(Project)).scalars().all()
    return {"projects": items, "total": len(total)}


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
        payload = project.to_dict()
        job_payload = job.to_dict() if job else None

    stage_states = _stage_states(payload, job_payload)
    rendered = sum(1 for clip in clip_rows if clip.status == "rendered")
    return {
        "project": {
            "id": payload["id"],
            "title": payload["title"],
            "channel": payload["channel"],
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
        "word_count": payload["word_count"],
        "segment_count": payload["segment_count"],
        "speakers": payload["speakers"],
        "segments": segments,
        "count": len(segments),
    }


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


def cleanup_projects(*, keep_days: int | None = None, remove_renders_only: bool = False) -> dict[str, Any]:
    """Housekeeping used by the Storage settings panel."""
    settings = get_settings()
    days = keep_days if keep_days is not None else settings.cleanup_days
    cutoff = utcnow().timestamp() - max(days, 1) * 86400
    removed_cache = 0
    freed = 0

    cache_root = settings.resolved_export_dir().parent / "cache"
    if cache_root.exists():
        for path in cache_root.rglob("*"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    freed += path.stat().st_size
                    path.unlink()
                    removed_cache += 1
            except OSError:
                continue

    removed_projects: list[str] = []
    if not remove_renders_only:
        with session_scope() as session:
            stale = [
                project.to_dict()
                for project in session.execute(select(Project)).scalars().all()
                if project.updated_at and project.updated_at.timestamp() < cutoff and project.status in {"draft", "failed", "cancelled"}
            ]
        for snapshot in stale:
            delete_project(snapshot["id"])
            removed_projects.append(snapshot["id"])

    return {
        "cache_files_removed": removed_cache,
        "bytes_freed": freed,
        "projects_removed": removed_projects,
        "cutoff_days": days,
    }


__all__ = [
    "cleanup_projects",
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
