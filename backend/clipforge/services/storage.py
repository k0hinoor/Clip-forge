"""Storage housekeeping: the download cache budget, stale projects and old renders.

Driven by the Storage settings:

* ``cleanup_days`` - age after which cached downloads, unfinished projects and
  (on request) rendered files are removed,
* ``max_cache_gb`` - size budget for the download cache, trimmed oldest first,
* ``keep_source_video`` - when off, the source video and extracted audio of an
  old project whose clips are all rendered are deleted too.

Everything here is conservative: projects with queued or running jobs are never
touched, and ``dry_run`` reports what *would* be removed.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select

from ..config import AppSettings, Env, get_settings
from ..db import Clip, Project, session_scope
from ..logging_setup import get_logger
from . import events

log = get_logger("clipforge.worker")

UPLOAD_LEFTOVER_SECONDS = 24 * 3600  # an upload still being written is never this old
FINISHED_PROJECT_STATUSES = {"ready"}
DISPOSABLE_PROJECT_STATUSES = {"draft", "failed", "cancelled"}


def _file_size(path: Path) -> tuple[int, int]:
    """``(size, bytes freed by deleting it)`` - hard links free nothing."""
    try:
        stat = path.stat()
    except OSError:
        return 0, 0
    return stat.st_size, (stat.st_size if stat.st_nlink <= 1 else 0)


def _remove(path: Path, *, dry_run: bool) -> int:
    size, freed = _file_size(path)
    if dry_run:
        return freed
    try:
        path.unlink()
    except OSError as exc:
        log.warning("could not remove %s: %s", path, exc)
        return 0
    return freed


def folder_size(folder: Path) -> int:
    total = 0
    if folder.exists():
        for path in folder.rglob("*"):
            if path.is_file():
                total += _file_size(path)[0]
    return total


# --------------------------------------------------------------------------- #
# Download cache
# --------------------------------------------------------------------------- #


def cleanup_cache(settings: AppSettings | None = None, *, days: int | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Expire cached downloads older than ``days`` and keep the cache under budget."""
    settings = settings or get_settings()
    days = settings.cleanup_days if days is None else max(0, int(days))
    cache = Env.DATA_DIR / "cache"
    now = time.time()
    cutoff = now - days * 86400
    budget = max(0.0, float(settings.max_cache_gb)) * 1024**3

    entries: list[tuple[float, Path, int]] = []
    removed = 0
    freed = 0
    if cache.exists():
        for path in cache.rglob("*"):
            if not path.is_file():
                continue
            try:
                modified = path.stat().st_mtime
            except OSError:
                continue
            is_upload_leftover = path.parent == cache and path.name.startswith(("tmp", "upload_"))
            if is_upload_leftover:
                if now - modified > UPLOAD_LEFTOVER_SECONDS:
                    freed += _remove(path, dry_run=dry_run)
                    removed += 1
                continue
            if modified < cutoff:
                freed += _remove(path, dry_run=dry_run)
                removed += 1
                continue
            entries.append((modified, path, _file_size(path)[0]))

    total = sum(size for _modified, _path, size in entries)
    for _modified, path, size in sorted(entries):  # oldest first
        if total <= budget:
            break
        freed += _remove(path, dry_run=dry_run)
        removed += 1
        total -= size

    return {"removed_files": removed, "freed_bytes": freed, "remaining_bytes": total, "dry_run": dry_run}


# --------------------------------------------------------------------------- #
# Projects
# --------------------------------------------------------------------------- #


def _older_than(stamp: str | None, cutoff: float) -> bool:
    if not stamp:
        return True
    try:
        moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return False
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp() < cutoff


def cleanup_storage(*, days: int | None = None, include_renders: bool = False, dry_run: bool = False) -> dict[str, Any]:
    """Run every cleanup rule; returns a report of what was (or would be) removed."""
    from ..jobs import queue as job_queue
    from ..pipeline.context import ProjectPaths
    from .projects import delete_project

    settings = get_settings()
    days = settings.cleanup_days if days is None else max(0, int(days))
    cutoff = time.time() - days * 86400
    cache_report = cleanup_cache(settings, days=days, dry_run=dry_run)

    busy = {job["project_id"] for job in job_queue.active_jobs() if job.get("project_id")}
    with session_scope() as session:
        projects = [project.to_dict() for project in session.execute(select(Project)).scalars().all()]

    removed_projects: list[dict[str, Any]] = []
    renders_removed = 0
    sources_removed = 0
    freed = cache_report["freed_bytes"]

    for snapshot in projects:
        project_id = snapshot["id"]
        if project_id in busy or not _older_than(snapshot.get("updated_at"), cutoff):
            continue
        paths = ProjectPaths.for_snapshot(snapshot)

        if snapshot.get("status") in DISPOSABLE_PROJECT_STATUSES:
            size = folder_size(paths.root)
            if not dry_run:
                delete_project(project_id)
            removed_projects.append({"id": project_id, "title": snapshot.get("title", ""), "status": snapshot.get("status"), "bytes": size})
            freed += size
            continue

        if snapshot.get("status") not in FINISHED_PROJECT_STATUSES:
            continue

        with session_scope() as session:
            clips = session.execute(select(Clip).where(Clip.project_id == project_id)).scalars().all()
            states = [(clip.id, clip.status, clip.file_path, clip.rendered_at) for clip in clips]

        if include_renders:
            for clip_id, status, file_path, rendered_at in states:
                if status != "rendered" or not file_path:
                    continue
                if rendered_at is not None and rendered_at.replace(tzinfo=rendered_at.tzinfo or timezone.utc).timestamp() >= cutoff:
                    continue
                freed += _remove(Path(file_path), dry_run=dry_run)
                renders_removed += 1
                if not dry_run:
                    with session_scope() as session:
                        clip = session.get(Clip, clip_id)
                        if clip is not None:
                            clip.status, clip.stage, clip.progress = "pending", "pending", 0.0
                            clip.file_path = ""
                            clip.file_size = 0

        all_rendered = bool(states) and all(status == "rendered" for _id, status, _path, _at in states)
        if not settings.keep_source_video and all_rendered:
            for folder in (paths.source, paths.audio):
                if not folder.exists():
                    continue
                for path in folder.iterdir():
                    if path.is_file():
                        freed += _remove(path, dry_run=dry_run)
                        sources_removed += 1

    report = {
        "days": days,
        "dry_run": dry_run,
        "cache": cache_report,
        "projects_removed": removed_projects,
        "renders_removed": renders_removed,
        "source_files_removed": sources_removed,
        "freed_bytes": freed,
    }
    if not dry_run:
        log.info(
            "storage cleanup: %d projects, %d renders, %d source files, %d cache files - %.1f MB freed",
            len(removed_projects), renders_removed, sources_removed, cache_report["removed_files"], freed / 1e6,
        )
        events.publish("storage.cleaned", {"freed_bytes": freed, "projects_removed": len(removed_projects)})
    return report


def storage_usage() -> dict[str, Any]:
    """Bytes used per area of the data folder (for the Storage panel)."""
    areas = {
        "projects": Env.DATA_DIR / "projects",
        "cache": Env.DATA_DIR / "cache",
        "assets": Env.DATA_DIR / "assets",
        "exports": get_settings().resolved_export_dir(),
        "logs": Env.LOG_DIR,
    }
    return {name: folder_size(path) for name, path in areas.items()}


__all__ = ["cleanup_cache", "cleanup_storage", "folder_size", "storage_usage"]
