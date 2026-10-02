"""Job lifecycle service (TRD §7, §29, §31, §43; PRD §12-13).

The API only creates/cancels/retries jobs — it never processes media itself
(critical decision #1). All transitions are validated and produce events.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from clipforge.core.config import Settings
from clipforge.core.errors import AppError, ErrorCode, not_found
from clipforge.core.presets import ASPECT_RATIOS, FRAMING_MODES, caption_presets
from clipforge.core.states import (
    TERMINAL_STATES,
    AssetStatus,
    ClipStatus,
    JobStatus,
    RenderStatus,
    assert_transition,
    is_active,
)
from clipforge.core.timeutil import utcnow
from clipforge.core.versioning import PIPELINE_VERSION
from clipforge.db.models import Asset, Clip, Job, Render, User
from clipforge.queue.base import QueueProvider
from clipforge.services import idempotency
from clipforge.services.analytics import track
from clipforge.services.events import record_job_event
from clipforge.services.quotas import EntitlementService
from clipforge.storage import Keys, StorageProvider

PLATFORM_DEFAULTS: dict[str, dict[str, Any]] = {
    "tiktok": {"aspect_ratio": "9:16"},
    "instagram_reels": {"aspect_ratio": "9:16"},
    "youtube_shorts": {"aspect_ratio": "9:16"},
    "youtube": {"aspect_ratio": "16:9"},
    "linkedin": {"aspect_ratio": "1:1"},
    "generic": {"aspect_ratio": "9:16"},
}
WHISPER_MODELS = ("auto", "tiny", "base", "small", "medium", "large-v3")


def transition(session: Session, job: Job, new_status: JobStatus | str, *, message: str | None = None,
               stage: str | None = None, error_code: str | None = None,
               metadata: dict[str, Any] | None = None) -> None:
    new = JobStatus(new_status)
    assert_transition(job.status, new.value)
    changed = job.status != new.value
    job.status = new.value
    if stage is not None:
        job.current_stage = stage
    if changed or message:
        record_job_event(session, job, "job.status", message=message or f"Status changed to {new.value}",
                         stage=stage, error_code=error_code, metadata=metadata)


def normalize_job_settings(raw: dict[str, Any], settings: Settings, max_clips: int) -> dict[str, Any]:
    platform = (raw.get("target_platform") or "generic").lower()
    if platform not in PLATFORM_DEFAULTS:
        raise AppError(ErrorCode.VALIDATION_ERROR, f"Unknown target platform '{platform}'.")
    aspect = raw.get("aspect_ratio") or PLATFORM_DEFAULTS[platform]["aspect_ratio"]
    if aspect not in ASPECT_RATIOS:
        raise AppError(ErrorCode.VALIDATION_ERROR, f"Aspect ratio must be one of {', '.join(ASPECT_RATIOS)}.")
    num = int(raw.get("num_clips") or settings.DEFAULT_CLIPS_PER_JOB)
    if not 1 <= num <= max_clips:
        raise AppError(ErrorCode.VALIDATION_ERROR, f"Number of clips must be between 1 and {max_clips}.")
    preset = (raw.get("caption_preset") or "clean").lower()
    if preset not in caption_presets(settings):
        raise AppError(ErrorCode.VALIDATION_ERROR, f"Unknown caption preset '{preset}'.")
    framing = (raw.get("framing_mode") or "auto").lower()
    if framing not in FRAMING_MODES:
        raise AppError(ErrorCode.VALIDATION_ERROR, f"Framing mode must be one of {', '.join(FRAMING_MODES)}.")
    min_s = float(raw.get("min_clip_seconds") or settings.TARGET_MIN_SECONDS)
    max_s = float(raw.get("max_clip_seconds") or settings.TARGET_MAX_SECONDS)
    if not (settings.MIN_CLIP_SECONDS <= min_s <= max_s <= settings.MAX_CLIP_SECONDS):
        raise AppError(ErrorCode.VALIDATION_ERROR,
                       f"Clip length must satisfy {settings.MIN_CLIP_SECONDS:g}s <= min <= max <= "
                       f"{settings.MAX_CLIP_SECONDS:g}s.")
    model = (raw.get("whisper_model") or settings.WHISPER_MODEL or "auto").lower()
    if model not in WHISPER_MODELS:
        raise AppError(ErrorCode.VALIDATION_ERROR, f"Whisper model must be one of {', '.join(WHISPER_MODELS)}.")
    language = (raw.get("language") or "").strip().lower() or None
    if language and (len(language) > 8 or not language.replace("-", "").isalpha()):
        raise AppError(ErrorCode.VALIDATION_ERROR, "Language must be an ISO code such as 'en'.")
    return {
        "target_platform": platform,
        "aspect_ratio": aspect,
        "num_clips": num,
        "caption_preset": preset,
        "captions_enabled": bool(raw.get("captions_enabled", True)),
        "framing_mode": framing,
        "min_clip_seconds": min_s,
        "max_clip_seconds": max_s,
        "whisper_model": model,
        "language": language,
    }


class JobService:
    def __init__(self, session: Session, settings: Settings, queue: QueueProvider | None = None,
                 storage: StorageProvider | None = None) -> None:
        self.session = session
        self.settings = settings
        self.queue = queue
        self.storage = storage

    # -------------------------------------------------------------- queries
    def get(self, user: User, job_id: str) -> Job:
        job = self.session.scalar(select(Job).where(Job.id == job_id, Job.user_id == user.id,
                                                    Job.deleted_at.is_(None)))
        if job is None:
            raise not_found("Job")
        return job

    def list(self, user: User, *, status: str | None = None, limit: int = 20, offset: int = 0) -> tuple[list[Job], int]:
        q = select(Job).where(Job.user_id == user.id, Job.deleted_at.is_(None))
        if status:
            q = q.where(Job.status == status.upper())
        total = int(self.session.scalar(select(func.count()).select_from(q.subquery())) or 0)
        rows = self.session.scalars(q.order_by(Job.created_at.desc()).limit(limit).offset(offset)).all()
        return list(rows), total

    # -------------------------------------------------------------- create
    def create(self, user: User, asset_id: str, raw_settings: dict[str, Any],
               idempotency_key: str | None = None) -> tuple[Job, bool]:
        """Create a job inside the caller's transaction. Returns (job, created)."""
        key = idempotency.validate_key(idempotency_key)
        req_hash = idempotency.request_hash({"asset_id": asset_id, "settings": raw_settings})
        existing_id = idempotency.lookup(self.session, user.id, "jobs.create", key, req_hash)
        if existing_id:
            return self.get(user, existing_id), False

        asset = self.session.scalar(select(Asset).where(Asset.id == asset_id, Asset.user_id == user.id,
                                                        Asset.deleted_at.is_(None)))
        if asset is None:
            raise not_found("Upload")
        if asset.status != AssetStatus.READY.value:
            raise AppError(ErrorCode.CONFLICT, "The upload is not ready for processing.")
        if not self.storage or not self.storage.exists(asset.storage_key):
            raise AppError(ErrorCode.SOURCE_EXPIRED)

        entitlements = EntitlementService(self.session, self.settings)
        ent = entitlements.entitlement(user)
        job_settings = normalize_job_settings(raw_settings, self.settings, ent.max_clips_per_job)
        duration = float(asset.duration_seconds or 0.0)
        entitlements.can_process(user, duration).raise_if_denied()

        job = Job(
            user_id=user.id, project_id=asset.project_id, source_asset_id=asset.id,
            status=JobStatus.QUEUED.value, progress=0, current_stage=None, settings=job_settings,
            attempt=0, max_attempts=self.settings.MAX_JOB_ATTEMPTS, priority=ent.priority,
            source_seconds=duration, pipeline_version=PIPELINE_VERSION,
            versions={"pipeline_version": PIPELINE_VERSION}, next_attempt_at=utcnow(),
        )
        self.session.add(job)
        self.session.flush()
        # Keep the source while the job exists; retention restarts on completion.
        asset.expires_at = None
        record_job_event(self.session, job, "job.created", message="Job queued",
                         metadata={"settings": job_settings})
        idempotency.store(self.session, user.id, "jobs.create", key, req_hash, "job", job.id)
        track(self.session, "job_created", user_id=user.id, job_id=job.id,
              properties={"duration": duration, "aspect_ratio": job_settings["aspect_ratio"],
                          "clips": job_settings["num_clips"], "caption_preset": job_settings["caption_preset"]})
        return job, True

    def notify_queue(self, job: Job) -> None:
        if self.queue is not None:
            self.queue.enqueue(job.id)

    # --------------------------------------------------------------- cancel
    def cancel(self, user: User, job: Job) -> Job:
        status = JobStatus(job.status)
        if status in TERMINAL_STATES:
            if status == JobStatus.CANCELLED:
                return job
            raise AppError(ErrorCode.INVALID_STATE_TRANSITION, f"A {status.value.lower()} job cannot be cancelled.")
        job.cancel_requested = True
        if status == JobStatus.QUEUED or (is_active(job.status) and not job.lock_owner):
            transition(self.session, job, JobStatus.CANCELLED, message="Cancelled by user",
                       error_code=ErrorCode.JOB_CANCELLED.value)
            job.completed_at = utcnow()
            job.error_code = ErrorCode.JOB_CANCELLED.value
            job.error_message = "The job was cancelled."
            job.retryable = True
        else:
            # The worker observes ``cancel_requested`` and kills FFmpeg/transcription.
            record_job_event(self.session, job, "job.cancel_requested", message="Cancellation requested")
        track(self.session, "job_cancelled", user_id=user.id, job_id=job.id, properties={"stage": job.current_stage})
        return job

    # ---------------------------------------------------------------- retry
    def retry(self, user: User, job: Job) -> Job:
        status = JobStatus(job.status)
        if status not in (JobStatus.FAILED, JobStatus.CANCELLED):
            raise AppError(ErrorCode.INVALID_STATE_TRANSITION, "Only failed or cancelled jobs can be retried.")
        if job.retryable is False and job.error_code not in (ErrorCode.QUOTA_EXCEEDED.value,):
            raise AppError(ErrorCode.CONFLICT, "This job failed with a non-retryable error. Upload a new video.",
                           details={"error_code": job.error_code})
        asset = job.source_asset
        if asset is None or asset.deleted_at is not None or not (self.storage and self.storage.exists(asset.storage_key)):
            raise AppError(ErrorCode.SOURCE_EXPIRED)
        EntitlementService(self.session, self.settings).can_process(
            user, float(job.source_seconds or 0), exclude_job_id=job.id).raise_if_denied()
        transition(self.session, job, JobStatus.QUEUED, message="Retry requested by user")
        job.cancel_requested = False
        job.attempt = 0  # manual retry gets a fresh automatic-retry budget
        job.error_code = job.error_message = job.diagnostic_id = None
        job.retryable = None
        job.completed_at = None
        job.next_attempt_at = utcnow()
        job.lock_owner = job.lock_expires_at = None
        asset.expires_at = None
        track(self.session, "job_retried", user_id=user.id, job_id=job.id)
        return job

    # --------------------------------------------------------------- delete
    def delete(self, user: User, job: Job) -> None:
        """User-initiated immediate deletion (PRD §19)."""
        if JobStatus(job.status) not in TERMINAL_STATES:
            job.cancel_requested = True
            if job.status == JobStatus.QUEUED.value or not job.lock_owner:
                transition(self.session, job, JobStatus.CANCELLED, message="Cancelled (job deleted)")
        now = utcnow()
        job.deleted_at = now
        self.session.execute(update(Clip).where(Clip.job_id == job.id).values(
            status=ClipStatus.DELETED.value, deleted_at=now))
        self.session.execute(update(Render).where(Render.job_id == job.id, Render.status.in_(
            [RenderStatus.QUEUED.value])).values(status=RenderStatus.CANCELLED.value))
        if self.storage is not None and not job.lock_owner:
            purge_job_files(self.storage, job.id, self.settings.STORAGE_ROOT / "work")
        track(self.session, "job_deleted", user_id=user.id, job_id=job.id)


def purge_job_files(storage: StorageProvider, job_id: str, local_work_root: Path | None = None) -> int:
    """Delete every artifact of a job (work dir, outputs, thumbnails, transcript)."""
    removed = 0
    if local_work_root is not None:
        work = (local_work_root / job_id).resolve()
        if work.parent == local_work_root.resolve() and work.exists():
            removed += sum(1 for p in work.rglob("*") if p.is_file())
            shutil.rmtree(work, ignore_errors=True)
    for prefix in (Keys.work_prefix(job_id), Keys.output_prefix(job_id), Keys.thumbnail_prefix(job_id),
                   Keys.transcript_prefix(job_id)):
        removed += storage.delete_prefix(prefix)
    return removed
