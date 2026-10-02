"""Pipeline orchestration: resumable stage execution + failure policy (TRD §7, §8, §31, §45)."""

from __future__ import annotations

import threading
import time
import traceback
from datetime import timedelta
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from clipforge.core.config import Settings
from clipforge.core.errors import AppError, ErrorCode
from clipforge.core.logging import get_logger, log_context
from clipforge.core.states import STAGE_BY_NAME, JobStatus, overall_progress
from clipforge.core.timeutil import hours_from_now, utcnow
from clipforge.db.models import Asset, Job
from clipforge.db.session import session_scope
from clipforge.queue.base import QueueProvider
from clipforge.services.analytics import track
from clipforge.services.events import record_job_event
from clipforge.services.jobs import purge_job_files, transition
from clipforge.storage import StorageProvider
from clipforge.worker.models.manager import ModelManager
from clipforge.worker.pipeline.audio import AudioExtractStage
from clipforge.worker.pipeline.base import LeaseLost, Manifest, PipelineContext, Stage
from clipforge.worker.pipeline.candidates import CandidateGenerationStage
from clipforge.worker.pipeline.captions import CaptionStage
from clipforge.worker.pipeline.finalize import FinalizeStage
from clipforge.worker.pipeline.framing import VisualAnalysisStage
from clipforge.worker.pipeline.ingest import IngestStage, ValidateStage
from clipforge.worker.pipeline.qa import QAStage
from clipforge.worker.pipeline.rendering import RenderStage
from clipforge.worker.pipeline.scoring import BoundaryOptimizationStage, CandidateScoringStage
from clipforge.worker.pipeline.segmentation import SegmentStage
from clipforge.worker.pipeline.transcription import TranscribeStage
from clipforge.worker.resources.guard import ResourceGuard

log = get_logger(__name__)


def build_stages() -> list[Stage]:
    return [ValidateStage(), IngestStage(), AudioExtractStage(), TranscribeStage(), SegmentStage(),
            CandidateGenerationStage(), CandidateScoringStage(), BoundaryOptimizationStage(),
            VisualAnalysisStage(), CaptionStage(), RenderStage(), QAStage(), FinalizeStage()]


class _ProgressReporter:
    """Throttled persistence of progress so SSE clients see live updates."""

    def __init__(self, session_factory: sessionmaker[Session], job_id: str) -> None:
        self.session_factory = session_factory
        self.job_id = job_id
        self.last_value = -1
        self.last_time = 0.0
        self.lock = threading.Lock()

    def __call__(self, stage: str, fraction: float, message: str | None = None) -> None:
        value = overall_progress(stage, fraction)
        now = time.monotonic()
        with self.lock:
            if value <= self.last_value or (now - self.last_time < 1.0 and value - self.last_value < 3):
                return
            self.last_value, self.last_time = value, now
        with session_scope(self.session_factory) as s:
            job = s.get(Job, self.job_id)
            if job is not None and job.progress < value:
                job.progress = value


class PipelineRunner:
    def __init__(self, settings: Settings, session_factory: sessionmaker[Session], storage: StorageProvider,
                 queue: QueueProvider, models: ModelManager, guard: ResourceGuard, worker_id: str) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.storage = storage
        self.queue = queue
        self.models = models
        self.guard = guard
        self.worker_id = worker_id

    # ----------------------------------------------------------------- run
    def run_job(self, job_id: str) -> str:
        """Process a leased job. Returns the final status."""
        with session_scope(self.session_factory) as s:
            job = s.get(Job, job_id)
            if job is None or job.lock_owner != self.worker_id:
                return "SKIPPED"
            if job.cancel_requested or job.deleted_at is not None:
                transition(s, job, JobStatus.CANCELLED, message="Cancelled before start",
                           error_code=ErrorCode.JOB_CANCELLED.value)
                job.completed_at = utcnow()
                return JobStatus.CANCELLED.value
            job.attempt = (job.attempt or 0) + 1
            job.started_at = job.started_at or utcnow()
            job.next_attempt_at = None
            record_job_event(s, job, "job.started", message=f"Attempt {job.attempt} started on {self.worker_id}",
                             metadata={"attempt": job.attempt, "worker_id": self.worker_id})
            ctx_args = dict(user_id=job.user_id, asset_id=job.source_asset_id, job_settings=dict(job.settings or {}),
                            manifest_db=job.manifest)

        manifest = Manifest.load(job_id, self.settings.STORAGE_ROOT, ctx_args.pop("manifest_db"))
        ctx = PipelineContext(job_id=job_id, settings=self.settings, storage=self.storage,
                              session_factory=self.session_factory, models=self.models, guard=self.guard,
                              work_dir=self.settings.STORAGE_ROOT / "work" / job_id, manifest=manifest,
                              worker_id=self.worker_id, **ctx_args)
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        ctx.cancel_flag = lambda: self._cancel_requested(job_id)
        ctx.progress_cb = _ProgressReporter(self.session_factory, job_id)

        stop = threading.Event()
        heartbeat = threading.Thread(target=self._renew_lease, args=(job_id, ctx, stop), daemon=True)
        heartbeat.start()
        try:
            with log_context(job_id=job_id):
                self._execute(ctx)
            return JobStatus.COMPLETED.value
        except LeaseLost:
            log.warning("lease lost; abandoning job", extra={"job_id": job_id})
            return "LEASE_LOST"
        except AppError as err:
            if err.code == ErrorCode.JOB_CANCELLED:
                return self._mark_cancelled(job_id)
            return self._handle_failure(job_id, err)
        except Exception as exc:  # unexpected bug: never leak traces to users
            err = AppError(ErrorCode.UNKNOWN_ERROR, internal="".join(traceback.format_exception(exc))[-4000:])
            log.error("unexpected pipeline error", exc_info=True,
                      extra={"job_id": job_id, "diagnostic_id": err.diagnostic_id})
            return self._handle_failure(job_id, err)
        finally:
            stop.set()
            heartbeat.join(timeout=5)
            self.queue.release_job(job_id, self.worker_id)

    def _execute(self, ctx: PipelineContext) -> None:
        resumed = bool(ctx.manifest.data.get("stages"))
        for stage in build_stages():
            ctx.check_cancel()
            info = STAGE_BY_NAME[stage.name]
            ctx.current_stage = stage.name
            with log_context(stage=stage.name):
                if stage.name != "finalize" and ctx.manifest.stage_done(stage.name):
                    stage.load(ctx)
                    if resumed:
                        self._event(ctx, "stage.skipped", stage.name, message="Reused valid artifacts")
                    continue
                with session_scope(self.session_factory) as s:
                    job = s.get(Job, ctx.job_id)
                    transition(s, job, info.status, stage=stage.name)
                    job.progress = max(job.progress, overall_progress(stage.name, 0.0))
                    record_job_event(s, job, "stage.started", stage=stage.name)
                started = time.monotonic()
                try:
                    stage.validate(ctx)
                    output = stage.execute(ctx)
                    outputs = stage.persist(ctx, output)
                    stage.validate_output(ctx, output)
                finally:
                    try:
                        stage.cleanup(ctx)
                    except Exception:
                        log.warning("stage cleanup failed", exc_info=True)
                duration_ms = int((time.monotonic() - started) * 1000)
                if stage.name != "finalize":
                    ctx.manifest.mark_stage(stage.name, outputs, duration_ms)
                ctx.manifest.save()
                with session_scope(self.session_factory) as s:
                    job = s.get(Job, ctx.job_id)
                    job.manifest = ctx.manifest.data
                    if stage.name != "finalize":
                        job.progress = max(job.progress, overall_progress(stage.name, 1.0))
                    record_job_event(s, job, "stage.completed", stage=stage.name, duration_ms=duration_ms)
                log.info("stage completed", extra={"duration_ms": duration_ms})

    # -------------------------------------------------------------- helpers
    def _event(self, ctx: PipelineContext, event_type: str, stage: str, message: str | None = None) -> None:
        with session_scope(self.session_factory) as s:
            job = s.get(Job, ctx.job_id)
            record_job_event(s, job, event_type, stage=stage, message=message)

    def _cancel_requested(self, job_id: str) -> bool:
        with session_scope(self.session_factory) as s:
            job = s.get(Job, job_id)
            return job is None or bool(job.cancel_requested) or job.deleted_at is not None

    def _renew_lease(self, job_id: str, ctx: PipelineContext, stop: threading.Event) -> None:
        interval = max(1.0, self.settings.WORKER_LEASE_SECONDS / 3)
        while not stop.wait(interval):
            try:
                if not self.queue.renew_job(job_id, self.worker_id, self.settings.WORKER_LEASE_SECONDS):
                    ctx.lease_lost.set()
                    return
            except Exception:  # transient DB issue: keep trying until the lease expires
                log.warning("lease renewal failed", exc_info=True)

    def _mark_cancelled(self, job_id: str) -> str:
        with session_scope(self.session_factory) as s:
            job = s.get(Job, job_id)
            if job is None:
                return JobStatus.CANCELLED.value
            if job.status != JobStatus.CANCELLED.value:
                transition(s, job, JobStatus.CANCELLED, message="Cancelled", error_code=ErrorCode.JOB_CANCELLED.value)
            job.completed_at = utcnow()
            job.error_code = ErrorCode.JOB_CANCELLED.value
            job.error_message = "The job was cancelled."
            job.retryable = True
            deleted = job.deleted_at is not None
        if deleted:
            purge_job_files(self.storage, job_id, self.settings.STORAGE_ROOT / "work")
        return JobStatus.CANCELLED.value

    def _handle_failure(self, job_id: str, err: AppError) -> str:
        """TRD §31: attempt 1 → immediate retry, attempt 2 → backoff, attempt 3 → failed."""
        log.warning("job attempt failed", extra={"job_id": job_id, "error_code": err.code.value,
                                                 "diagnostic_id": err.diagnostic_id,
                                                 "internal": (err.internal or "")[-1500:]})
        with session_scope(self.session_factory) as s:
            job = s.get(Job, job_id)
            if job is None:
                return "GONE"
            if err.retryable and job.attempt < job.max_attempts and not job.cancel_requested:
                delay = 0 if job.attempt <= 1 else self.settings.RETRY_BACKOFF_BASE_SECONDS * 2 ** (job.attempt - 2)
                transition(s, job, JobStatus.QUEUED, message=f"Retrying after {err.code.value} (attempt {job.attempt})",
                           error_code=err.code.value,
                           metadata={"retry_in_seconds": delay, "diagnostic_id": err.diagnostic_id})
                job.next_attempt_at = utcnow() + timedelta(seconds=delay)
                return JobStatus.QUEUED.value
            transition(s, job, JobStatus.FAILED, message=err.user_message, error_code=err.code.value,
                       metadata={"diagnostic_id": err.diagnostic_id})
            job.error_code = err.code.value
            job.error_message = err.user_message[:500]
            job.diagnostic_id = err.diagnostic_id
            job.retryable = err.retryable
            job.completed_at = utcnow()
            asset = s.get(Asset, job.source_asset_id)
            if asset is not None:
                asset.expires_at = hours_from_now(self.settings.FAILED_SOURCE_RETENTION_HOURS)
            track(s, "job_failed", user_id=job.user_id, job_id=job.id,
                  properties={"error_code": err.code.value, "stage": job.current_stage, "attempt": job.attempt})
        return JobStatus.FAILED.value


def job_snapshot(session: Session, job_id: str) -> dict[str, Any]:  # pragma: no cover - debug helper
    job = session.get(Job, job_id)
    return {"status": job.status, "stage": job.current_stage, "progress": job.progress} if job else {}
