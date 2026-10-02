"""User-requested re-renders (PRD §12 Flow B).

Only the affected stages run (framing → captions → render → QA); the stored
transcript and analysis are reused (TRD §49).
"""

from __future__ import annotations

import threading
import traceback

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from clipforge.core.config import Settings
from clipforge.core.errors import AppError, ErrorCode
from clipforge.core.logging import get_logger, log_context
from clipforge.core.states import ClipStatus, RenderStatus
from clipforge.core.timeutil import utcnow
from clipforge.db.models import Asset, Clip, Job, Render, Transcript
from clipforge.db.session import session_scope
from clipforge.queue.base import QueueProvider
from clipforge.scoring.segmentation import flatten_words
from clipforge.services.events import record_job_event
from clipforge.services.quotas import UsageService
from clipforge.storage import StorageProvider
from clipforge.worker.models.manager import ModelManager
from clipforge.worker.pipeline.rendering import ClipRenderer, RenderInputs

log = get_logger(__name__)


class RerenderProcessor:
    def __init__(self, settings: Settings, session_factory: sessionmaker[Session], storage: StorageProvider,
                 queue: QueueProvider, models: ModelManager, worker_id: str) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.storage = storage
        self.queue = queue
        self.models = models
        self.worker_id = worker_id

    def _cancelled(self, render_id: str) -> bool:
        with session_scope(self.session_factory) as s:
            r = s.get(Render, render_id)
            if r is None or r.status == RenderStatus.CANCELLED.value:
                return True
            clip = s.get(Clip, r.clip_id)
            return clip is None or clip.deleted_at is not None

    def run(self, render_id: str) -> str:
        stop = threading.Event()

        def renew() -> None:
            while not stop.wait(max(1.0, self.settings.WORKER_LEASE_SECONDS / 3)):
                self.queue.renew_render(render_id, self.worker_id, self.settings.WORKER_LEASE_SECONDS)

        t = threading.Thread(target=renew, daemon=True)
        t.start()
        try:
            with log_context(render_id=render_id):
                return self._run(render_id)
        finally:
            stop.set()
            t.join(timeout=5)
            self.queue.release_render(render_id, self.worker_id)

    def _run(self, render_id: str) -> str:
        with session_scope(self.session_factory) as s:
            render = s.get(Render, render_id)
            if render is None or render.lock_owner != self.worker_id or render.status != RenderStatus.QUEUED.value:
                return "SKIPPED"
            clip = s.get(Clip, render.clip_id)
            job = s.get(Job, render.job_id)
            asset = s.get(Asset, job.source_asset_id) if job else None
            if clip is None or clip.deleted_at is not None or job is None:
                render.status = RenderStatus.CANCELLED.value
                return RenderStatus.CANCELLED.value
            transcript = s.scalar(select(Transcript).where(Transcript.job_id == job.id))
            words = flatten_words({"segments": [
                {"id": seg.idx, "start": seg.start, "end": seg.end, "text": seg.text, "words": seg.words,
                 "speaker_id": seg.speaker_id} for seg in (transcript.segments if transcript else [])]})
            if asset is None or asset.deleted_at is not None or not self.storage.exists(asset.storage_key):
                self._fail(s, render, clip, job, AppError(ErrorCode.SOURCE_EXPIRED))
                return RenderStatus.FAILED.value
            media = {"display_width": asset.width, "display_height": asset.height,
                     "has_audio": bool(asset.has_audio), "duration": asset.duration_seconds}
            key, ext, job_id, clip_id, user_id = asset.storage_key, asset.extension, job.id, clip.id, job.user_id

        work_dir = self.settings.STORAGE_ROOT / "work" / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        source = self.storage.local_path(key)
        if source is None:
            source = work_dir / f"source{ext}"
            if not source.exists():
                self.storage.get_to_path(key, source)
        renderer = ClipRenderer(self.settings, self.storage, self.session_factory, self.models.vision,
                                cancel_check=lambda: self._cancelled(render_id))
        try:
            renderer.render(clip_id, render_id, RenderInputs(source_path=source, media=media, words=words,
                                                             work_dir=work_dir))
        except AppError as err:
            with session_scope(self.session_factory) as s:
                render, clip, job = s.get(Render, render_id), s.get(Clip, clip_id), s.get(Job, job_id)
                if err.code == ErrorCode.JOB_CANCELLED:
                    render.status = RenderStatus.CANCELLED.value
                    return RenderStatus.CANCELLED.value
                if render.status != RenderStatus.FAILED.value:
                    self._fail(s, render, clip, job, err)
                else:
                    record_job_event(s, job, "render.failed", stage="render", error_code=err.code.value,
                                     message=f"Re-render of clip #{clip.rank} failed",
                                     metadata={"clip_id": clip_id, "render_id": render_id})
            return RenderStatus.FAILED.value
        except Exception as exc:
            log.error("unexpected re-render error", exc_info=True)
            with session_scope(self.session_factory) as s:
                self._fail(s, s.get(Render, render_id), s.get(Clip, clip_id), s.get(Job, job_id),
                           AppError(ErrorCode.UNKNOWN_ERROR, internal="".join(traceback.format_exception(exc))))
            return RenderStatus.FAILED.value
        finally:
            if self.settings.STORAGE_BACKEND != "local":
                (work_dir / f"source{ext}").unlink(missing_ok=True)
        with session_scope(self.session_factory) as s:
            job, clip = s.get(Job, job_id), s.get(Clip, clip_id)
            UsageService(s).record_rerender(user_id)
            record_job_event(s, job, "render.completed", stage="render",
                             message=f"Re-render of clip #{clip.rank} completed",
                             metadata={"clip_id": clip_id, "render_id": render_id})
        return RenderStatus.READY.value

    def _fail(self, s: Session, render: Render, clip: Clip, job: Job, err: AppError) -> None:
        render.status = RenderStatus.FAILED.value
        render.error_code = err.code.value
        render.error_message = err.user_message
        render.completed_at = utcnow()
        prev = s.get(Render, clip.current_render_id) if clip.current_render_id else None
        clip.status = ClipStatus.READY.value if prev and prev.status == RenderStatus.READY.value \
            else ClipStatus.FAILED.value
        record_job_event(s, job, "render.failed", stage="render", error_code=err.code.value,
                         message=f"Re-render of clip #{clip.rank} failed",
                         metadata={"clip_id": clip.id, "render_id": render.id, "diagnostic_id": err.diagnostic_id})
