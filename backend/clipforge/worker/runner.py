"""Worker main loop.

* registers a heartbeat (``worker_heartbeats``) so the API can report health
* crash recovery on startup and periodically (TRD §45)
* resource checks before claiming work (TRD §32)
* bounded concurrency (``WORKER_CONCURRENCY``)
* scheduled cleanup (TRD §33)
"""

from __future__ import annotations

import os
import signal
import socket
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor

from sqlalchemy.orm import Session, sessionmaker

from clipforge.core.config import Settings
from clipforge.core.logging import get_logger
from clipforge.core.timeutil import utcnow
from clipforge.db.models import WorkerHeartbeat
from clipforge.db.session import session_scope
from clipforge.media.ffmpeg import configure_ffmpeg_concurrency
from clipforge.queue.base import QueueProvider
from clipforge.services.cleanup import CleanupService
from clipforge.storage import StorageProvider
from clipforge.worker.models.manager import ModelManager, detect_hardware
from clipforge.worker.pipeline.orchestrator import PipelineRunner
from clipforge.worker.rerender import RerenderProcessor
from clipforge.worker.resources.guard import ResourceGuard

log = get_logger(__name__)


class Worker:
    def __init__(self, settings: Settings, session_factory: sessionmaker[Session], storage: StorageProvider,
                 queue: QueueProvider, models: ModelManager | None = None, worker_id: str | None = None) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.storage = storage
        self.queue = queue
        self.models = models or ModelManager(settings)
        self.worker_id = worker_id or settings.WORKER_ID or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.guard = ResourceGuard(settings)
        self.pipeline = PipelineRunner(settings, session_factory, storage, queue, self.models, self.guard,
                                       self.worker_id)
        self.rerenders = RerenderProcessor(settings, session_factory, storage, queue, self.models, self.worker_id)
        self.cleanup = CleanupService(settings, session_factory, storage)
        self.stop_event = threading.Event()
        self._active: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._last_cleanup = 0.0
        self._last_recovery = 0.0
        self._last_heartbeat = 0.0
        configure_ffmpeg_concurrency(settings.MAX_CONCURRENT_FFMPEG)

    # ------------------------------------------------------------ heartbeat
    def heartbeat(self, status: str | None = None) -> None:
        with self._lock:
            active = list(self._active)
        status = status or ("busy" if active else "idle")
        with session_scope(self.session_factory) as s:
            hb = s.get(WorkerHeartbeat, self.worker_id)
            if hb is None:
                hb = WorkerHeartbeat(worker_id=self.worker_id, hostname=socket.gethostname(), pid=os.getpid(),
                                     status=status, started_at=utcnow(), last_seen_at=utcnow(), active_jobs=active,
                                     capabilities={"concurrency": self.settings.WORKER_CONCURRENCY,
                                                   "cuda": detect_hardware().get("cuda_available", False),
                                                   "transcription": self.models.transcriber.name,
                                                   "vision": self.models.vision.name})
                s.add(hb)
            else:
                hb.status = status
                hb.last_seen_at = utcnow()
                hb.active_jobs = active
        self._last_heartbeat = time.monotonic()

    # ---------------------------------------------------------------- loop
    def start(self) -> None:
        log.info("worker starting", extra={"worker_id": self.worker_id})
        self.models.warm_up()
        self.queue.recover_abandoned()
        self.heartbeat("idle")

    def run_forever(self) -> None:
        self.start()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, lambda *_: self.stop_event.set())
            except ValueError:  # not in main thread
                pass
        with ThreadPoolExecutor(max_workers=max(1, self.settings.WORKER_CONCURRENCY),
                                thread_name_prefix="clipforge-job") as pool:
            while not self.stop_event.is_set():
                claimed = self.tick(pool)
                if not claimed:
                    self.queue.wait_for_work(self.settings.WORKER_POLL_INTERVAL_SECONDS)
            log.info("worker stopping; waiting for active jobs")
        self.heartbeat("stopped")

    def tick(self, pool: ThreadPoolExecutor | None = None) -> bool:
        """One scheduling iteration. Returns True if work was claimed."""
        now = time.monotonic()
        self._reap()
        if now - self._last_heartbeat >= self.settings.WORKER_HEARTBEAT_SECONDS:
            self.heartbeat()
        if now - self._last_recovery >= self.settings.WORKER_LEASE_SECONDS:
            self._last_recovery = now
            self.queue.recover_abandoned()
        if now - self._last_cleanup >= self.settings.CLEANUP_INTERVAL_SECONDS:
            self._last_cleanup = now
            try:
                self.cleanup.run_once()
            except Exception:
                log.error("cleanup failed", exc_info=True)
        check = self.guard.can_start()
        if not check.ok:
            if check.reason != "worker_at_capacity":
                log.warning("insufficient resources; leaving jobs queued", extra={"reason": check.reason})
            return False
        job_id = self.queue.claim_job(self.worker_id, self.settings.WORKER_LEASE_SECONDS)
        if job_id:
            self._submit(pool, job_id, self._run_job)
            return True
        render_id = self.queue.claim_render(self.worker_id, self.settings.WORKER_LEASE_SECONDS)
        if render_id:
            self._submit(pool, render_id, self._run_render)
            return True
        return False

    def _submit(self, pool: ThreadPoolExecutor | None, key: str, fn) -> None:
        self.guard.acquire()
        if pool is None:
            try:
                fn(key)
            finally:
                self.guard.release()
            return
        fut = pool.submit(fn, key)
        with self._lock:
            self._active[key] = fut
        self.heartbeat("busy")

    def _reap(self) -> None:
        with self._lock:
            done = [k for k, f in self._active.items() if f.done()]
            for k in done:
                self._active.pop(k)
        for _ in done:
            self.guard.release()

    def _run_job(self, job_id: str) -> None:
        status = self.pipeline.run_job(job_id)
        log.info("job finished", extra={"job_id": job_id, "status": status})

    def _run_render(self, render_id: str) -> None:
        status = self.rerenders.run(render_id)
        log.info("render finished", extra={"render_id": render_id, "status": status})

    def run_until_idle(self, max_iterations: int = 100) -> None:
        """Synchronously process all currently runnable work (tests / ``--once``)."""
        self.start()
        idle_rounds = 0
        for _ in range(max_iterations):
            if self.tick(None):
                idle_rounds = 0
                continue
            idle_rounds += 1
            if idle_rounds >= 1:
                break
