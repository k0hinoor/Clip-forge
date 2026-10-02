"""Scheduled, idempotent cleanup (TRD §33; PRD §19 retention policy).

1. expired source files
2. abandoned uploads
3. temporary artifacts (work dirs) of finished / orphaned jobs
4. expired exports (renders, thumbnails)
5. orphan database records (stale sessions, idempotency keys, deleted jobs)
6. cleanup metrics (persisted in ``system_settings['cleanup.last_run']``)
"""

from __future__ import annotations

import shutil
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from clipforge.core.config import Settings
from clipforge.core.logging import get_logger
from clipforge.core.states import (
    IN_FLIGHT_STATES,
    AssetStatus,
    ClipStatus,
    JobStatus,
    RenderStatus,
)
from clipforge.core.timeutil import iso, utcnow
from clipforge.db.models import (
    Asset,
    Clip,
    IdempotencyKey,
    Job,
    JobEvent,
    Render,
    SystemSetting,
    UserSession,
)
from clipforge.db.session import session_scope
from clipforge.services.jobs import purge_job_files
from clipforge.storage import StorageProvider

log = get_logger(__name__)


class CleanupService:
    def __init__(self, settings: Settings, session_factory: sessionmaker[Session], storage: StorageProvider):
        self.settings = settings
        self.session_factory = session_factory
        self.storage = storage
        self.work_root = settings.STORAGE_ROOT / "work"

    def _delete_key(self, key: str | None) -> int:
        if not key:
            return 0
        st = self.storage.stat(key)
        self.storage.delete(key)
        return st.size if st else 0

    def run_once(self) -> dict[str, Any]:
        started = time.monotonic()
        now = utcnow()
        m: dict[str, Any] = {"expired_sources": 0, "abandoned_uploads": 0, "work_dirs_removed": 0,
                             "expired_renders": 0, "expired_jobs": 0, "purged_deleted_jobs": 0,
                             "stale_sessions": 0, "stale_idempotency_keys": 0, "bytes_freed": 0}
        with session_scope(self.session_factory) as s:
            self._expired_sources(s, now, m)
            self._abandoned_uploads(s, now, m)
            self._expired_exports(s, now, m)
            self._failed_job_artifacts(s, now, m)
            self._deleted_jobs(s, m)
            self._orphans(s, now, m)
        m["orphan_work_dirs_removed"] = self._orphan_work_dirs()
        m["duration_ms"] = int((time.monotonic() - started) * 1000)
        m["ran_at"] = iso(now)
        with session_scope(self.session_factory) as s:
            row = s.get(SystemSetting, "cleanup.last_run")
            if row is None:
                s.add(SystemSetting(key="cleanup.last_run", value=m))
            else:
                row.value = m
        log.info("cleanup completed", extra={"metrics": m})
        return m

    # ------------------------------------------------------------------ 1
    def _expired_sources(self, s: Session, now, m: dict[str, Any]) -> None:
        assets = s.scalars(select(Asset).where(
            Asset.status == AssetStatus.READY.value, Asset.expires_at.is_not(None), Asset.expires_at < now)).all()
        for asset in assets:
            busy = s.scalar(select(Job.id).where(
                Job.source_asset_id == asset.id, Job.status.in_([st.value for st in IN_FLIGHT_STATES]),
                Job.deleted_at.is_(None)).limit(1))
            if busy:
                continue
            m["bytes_freed"] += self._delete_key(asset.storage_key)
            asset.status = AssetStatus.EXPIRED.value
            m["expired_sources"] += 1

    # ------------------------------------------------------------------ 2
    def _abandoned_uploads(self, s: Session, now, m: dict[str, Any]) -> None:
        assets = s.scalars(select(Asset).where(
            Asset.status == AssetStatus.UPLOADING.value, Asset.upload_expires_at.is_not(None),
            Asset.upload_expires_at < now)).all()
        for asset in assets:
            m["bytes_freed"] += self._delete_key(asset.storage_key)
            asset.status = AssetStatus.EXPIRED.value
            m["abandoned_uploads"] += 1

    # ------------------------------------------------------------------ 4
    def _expired_exports(self, s: Session, now, m: dict[str, Any]) -> None:
        renders = s.scalars(select(Render).where(
            Render.status == RenderStatus.READY.value, Render.expires_at.is_not(None), Render.expires_at < now)).all()
        touched_jobs: set[str] = set()
        for r in renders:
            for key in (r.storage_key, r.thumbnail_key, r.subtitles_key):
                m["bytes_freed"] += self._delete_key(key)
            r.status = RenderStatus.EXPIRED.value
            m["expired_renders"] += 1
            touched_jobs.add(r.job_id)
        clips = s.scalars(select(Clip).where(
            Clip.status.in_([ClipStatus.READY.value, ClipStatus.FAILED.value, ClipStatus.PENDING.value]),
            Clip.expires_at.is_not(None), Clip.expires_at < now)).all()
        for c in clips:
            c.status = ClipStatus.EXPIRED.value
            touched_jobs.add(c.job_id)
        for job_id in touched_jobs:
            job = s.get(Job, job_id)
            if job is None or job.status != JobStatus.COMPLETED.value:
                continue
            live = s.scalar(select(Clip.id).where(Clip.job_id == job_id, Clip.status.in_(
                [ClipStatus.READY.value, ClipStatus.RENDERING.value, ClipStatus.PENDING.value])).limit(1))
            if live is None:
                job.status = JobStatus.EXPIRED.value
                s.add(JobEvent(job_id=job.id, event_type="job.expired", status=job.status, progress=job.progress,
                               message="Exports expired and were removed"))
                self._remove_work_dir(job.id)
                m["expired_jobs"] += 1

    # ------------------------------------------------------------------ 3
    def _failed_job_artifacts(self, s: Session, now, m: dict[str, Any]) -> None:
        cutoff = now - timedelta(hours=self.settings.FAILED_SOURCE_RETENTION_HOURS)
        jobs = s.scalars(select(Job).where(
            Job.status.in_([JobStatus.FAILED.value, JobStatus.CANCELLED.value]),
            Job.updated_at < cutoff, Job.deleted_at.is_(None))).all()
        for job in jobs:
            if self._remove_work_dir(job.id):
                m["work_dirs_removed"] += 1
            job.status = JobStatus.EXPIRED.value
            s.add(JobEvent(job_id=job.id, event_type="job.expired", status=job.status, progress=job.progress,
                           message="Intermediate artifacts expired"))
            m["expired_jobs"] += 1
        # Completed jobs should have no work dir left except the manifest.
        done = s.scalars(select(Job.id).where(Job.status == JobStatus.COMPLETED.value,
                                              Job.completed_at < now - timedelta(hours=1))).all()
        for job_id in done:
            work = self.work_root / job_id
            if (work.exists() and any(p.name != "manifest.json" for p in work.rglob("*") if p.is_file())
                    and not self._has_pending_render(s, job_id)):
                for p in work.rglob("*"):
                    if p.is_file() and p.name not in ("manifest.json",):
                        p.unlink(missing_ok=True)
                m["work_dirs_removed"] += 1

    def _has_pending_render(self, s: Session, job_id: str) -> bool:
        return s.scalar(select(Render.id).where(Render.job_id == job_id, Render.status.in_(
            [RenderStatus.QUEUED.value, RenderStatus.RENDERING.value, RenderStatus.QA.value])).limit(1)) is not None

    # ------------------------------------------------------------------ 5
    def _deleted_jobs(self, s: Session, m: dict[str, Any]) -> None:
        jobs = s.scalars(select(Job).where(Job.deleted_at.is_not(None), Job.lock_owner.is_(None))).all()
        for job in jobs:
            meta = job.manifest or {}
            if meta.get("purged"):
                continue
            purge_job_files(self.storage, job.id, self.work_root)
            job.manifest = {**meta, "purged": True}
            m["purged_deleted_jobs"] += 1

    def _orphans(self, s: Session, now, m: dict[str, Any]) -> None:
        res = s.execute(delete(UserSession).where(
            (UserSession.expires_at < now - timedelta(days=7))
            | (UserSession.revoked_at < now - timedelta(days=7))))
        m["stale_sessions"] = res.rowcount or 0
        res = s.execute(delete(IdempotencyKey).where(IdempotencyKey.created_at < now - timedelta(days=1)))
        m["stale_idempotency_keys"] = res.rowcount or 0
        # Queued renders for deleted clips can never run.
        for r in s.scalars(select(Render).join(Clip, Clip.id == Render.clip_id).where(
                Render.status == RenderStatus.QUEUED.value, Clip.deleted_at.is_not(None))).all():
            r.status = RenderStatus.CANCELLED.value

    def _orphan_work_dirs(self) -> int:
        if not self.work_root.exists():
            return 0
        removed = 0
        with session_scope(self.session_factory) as s:
            for child in self.work_root.iterdir():
                if (child.is_dir() and s.get(Job, child.name) is None
                        and time.time() - child.stat().st_mtime > 3600):
                    shutil.rmtree(child, ignore_errors=True)
                    removed += 1
        return removed

    def _remove_work_dir(self, job_id: str) -> bool:
        work = (self.work_root / job_id).resolve()
        if work.parent != self.work_root.resolve() or not work.exists():
            return False
        shutil.rmtree(work, ignore_errors=True)
        return True


def directory_size(path: Path) -> int:
    total = 0
    if path.exists():
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue
    return total
