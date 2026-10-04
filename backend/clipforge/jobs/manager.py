"""Job manager: owns the worker pool and is the API's way to schedule work.

Concurrency is deliberately conservative on local hardware:

* one analysis job at a time (it already saturates the CPU with Whisper),
* renders limited by ``max_concurrent_renders`` from settings,
* everything else queues politely behind.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from ..config import Env, get_settings
from ..db import Clip, Project, session_scope
from ..errors import ClipForgeError, ErrorCode
from ..logging_setup import get_logger
from ..services.events import BUS
from . import queue as job_queue
from .worker import Worker

log = get_logger("clipforge.worker")


class JobManager:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._workers: list[Worker] = []
        self._lock = threading.Lock()
        self._started_at = 0.0

    # ------------------------------------------------------------ lifecycle
    def start(self, *, workers: int | None = None) -> dict[str, Any]:
        started: list[Worker] = []
        with self._lock:
            if self._workers:
                pass  # already running - fall through and report the current state
            else:
                # This process owns the queue now: anything still marked running
                # was interrupted (crash, deploy, workers stopped) - resume it.
                job_queue.recover_stale_jobs()
                settings = get_settings()
                # Limits per job kind are enforced when a job is claimed (see
                # queue.claim_next); the pool only needs enough threads.
                count = max(1, min(workers if workers is not None else settings.concurrency + settings.max_concurrent_renders - 1, 8))
                self._stop.clear()
                self._workers = [Worker(f"worker-{index + 1}", stop_event=self._stop) for index in range(count)]
                for worker in self._workers:
                    worker.start()
                self._started_at = time.time()
                started = list(self._workers)
        # Never call status() from inside the lock: it takes the lock itself and a
        # plain Lock is not reentrant (the API hung forever on a second start).
        if started:
            log.info("job manager started with %d worker(s)", len(started))
            BUS.publish("worker.started", {"workers": len(started)})
        return self.status()

    def stop(self, *, timeout: float = 6.0) -> None:
        """Stop the pool. Interrupted jobs stay ``running`` and resume on the next start."""
        from ..media.runner import kill_all

        with self._lock:
            workers = list(self._workers)
            self._workers = []
        if not workers:
            return
        self._stop.set()
        kill_all()  # encoders stop now instead of finishing a clip nobody waits for
        for worker in workers:
            worker.join(timeout=timeout)
        log.info("job manager stopped")

    def restart(self) -> dict[str, Any]:
        self.stop()
        return self.start()

    def status(self) -> dict[str, Any]:
        with self._lock:
            workers = list(self._workers)
        return {
            "running": bool(workers),
            "workers": len(workers),
            "worker_names": [worker.name for worker in workers],
            "busy": [worker.current_job for worker in workers if worker.current_job],
            "uptime_seconds": round(time.time() - self._started_at, 1) if self._started_at else 0.0,
        }

    # ------------------------------------------------------------- scheduling
    def enqueue_analysis(self, project_id: str, *, priority: int = 1, force: bool = False) -> dict[str, Any]:
        """Queue an analysis; an analysis that is already queued/running is returned as-is."""
        self.ensure_started()
        active = job_queue.active_jobs(project_id=project_id, kinds=("analyze",))
        if active:
            return active[0]
        with session_scope() as session:
            project = session.get(Project, project_id)
            if project is None:
                raise ClipForgeError(code=ErrorCode.NOT_FOUND, message="Project not found.", status_code=404)
            project.status = "queued"
            project.stage = "queued"
            project.progress = 0.0
            project.status_message = "queued for analysis"
            project.error_code = ""
            project.error_message = ""
            project.error_hint = ""
        job = job_queue.enqueue("analyze", project_id=project_id, payload={"force": force} if force else None, priority=priority)
        BUS.publish("job.queued", {"job_id": job["id"], "kind": "analyze", "status": job["status"]}, project_id=project_id, job_id=job["id"])
        BUS.publish("project.updated", {"project_id": project_id, "status": "queued", "stage": "queued"}, project_id=project_id)
        return job

    def enqueue_render(self, clip_id: str, *, export: bool = False, overrides: dict[str, Any] | None = None, priority: int = 6) -> dict[str, Any]:
        """Queue a final render; a render already queued/running for the clip is returned as-is."""
        self.ensure_started()
        active = job_queue.active_jobs(clip_id=clip_id, kinds=("render_clip",))
        if active:
            return active[0]
        with session_scope() as session:
            clip = session.get(Clip, clip_id)
            if clip is None:
                raise ClipForgeError(code=ErrorCode.NOT_FOUND, message="Clip not found.", status_code=404)
            clip.status = "queued"
            clip.progress = 0.0
            clip.stage = "queued"
            clip.error_code = ""
            clip.error_message = ""
            project_id = clip.project_id
        job = job_queue.enqueue(
            "render_clip",
            project_id=project_id,
            clip_id=clip_id,
            payload={"export": export, "overrides": overrides or {}},
            priority=priority,
        )
        BUS.publish("job.queued", {"job_id": job["id"], "kind": "render_clip", "clip_id": clip_id}, project_id=project_id, job_id=job["id"])
        return job

    def enqueue_preview(self, clip_id: str, *, priority: int = 2) -> dict[str, Any]:
        self.ensure_started()
        with session_scope() as session:
            clip = session.get(Clip, clip_id)
            if clip is None:
                raise ClipForgeError(code=ErrorCode.NOT_FOUND, message="Clip not found.", status_code=404)
            project_id = clip.project_id
        job = job_queue.enqueue("render_preview", project_id=project_id, clip_id=clip_id, priority=priority)
        BUS.publish("job.queued", {"job_id": job["id"], "kind": "render_preview", "clip_id": clip_id}, project_id=project_id, job_id=job["id"])
        return job

    def enqueue_all(self, project_id: str, clip_ids: list[str], *, export: bool = False, priority: int = 6) -> dict[str, Any]:
        """Queue several renders, best-scored first. ``queued`` is the number of jobs."""
        self.ensure_started()
        queued = [
            self.enqueue_render(clip_id, export=export, priority=priority + min(index, 5))
            for index, clip_id in enumerate(clip_ids)
        ]
        return {"queued": len(queued), "jobs": [job["id"] for job in queued]}

    def enqueue_scan(self, *, priority: int = 8) -> dict[str, Any]:
        self.ensure_started()
        return job_queue.enqueue("scan_assets", priority=priority, dedupe=True)

    def ensure_started(self) -> None:
        """Start the embedded pool on demand (a separate ``clipforge worker`` owns it otherwise)."""
        if Env.EMBED_WORKER and not self.status()["running"]:
            self.start()

    # ---------------------------------------------------------------- control
    def cancel(self, job_id: str, *, kill: bool = True) -> dict[str, Any] | None:
        """Cancel a job. Queued jobs stop immediately; running ones at their next checkpoint."""
        from ..media.runner import kill_process
        from .worker import settle_cancelled

        before = job_queue.get(job_id)
        job = job_queue.cancel(job_id)
        if job is None:
            return None
        if before and before["status"] == "queued" and job["status"] == "cancelled":
            settle_cancelled(job)  # nothing will run, so put the clip/project back now
            BUS.publish("job.cancelled", {"job_id": job_id, "kind": job["kind"], "clip_id": job.get("clip_id", "")}, project_id=job.get("project_id", ""), job_id=job_id)
            return job
        if kill and job["status"] == "running":
            for key in (job.get("clip_id"), job_id):
                if key:
                    kill_process(key)
        if job["status"] == "running":
            BUS.publish("job.cancelling", {"job_id": job_id}, project_id=job.get("project_id", ""), job_id=job_id)
        return job

    def cancel_project(self, project_id: str) -> int:
        jobs = job_queue.list_jobs(project_id=project_id, statuses=("queued", "running"), limit=200)
        for job in jobs:
            self.cancel(job["id"])
        return len(jobs)

    def retry(self, job_id: str) -> dict[str, Any] | None:
        """Re-queue a failed or cancelled job and mark its clip/project as queued again."""
        self.ensure_started()
        job = job_queue.retry(job_id)
        if job is None:
            return None
        with session_scope() as session:
            if job["kind"] == "render_clip" and job.get("clip_id"):
                clip = session.get(Clip, job["clip_id"])
                if clip is not None:
                    clip.status, clip.stage, clip.progress = "queued", "queued", 0.0
                    clip.error_code = clip.error_message = ""
            elif job["kind"] == "analyze" and job.get("project_id"):
                project = session.get(Project, job["project_id"])
                if project is not None:
                    project.status = project.stage = "queued"
                    project.progress = 0.0
                    project.status_message = "queued for analysis"
                    project.error_code = project.error_message = project.error_hint = ""
        BUS.publish("job.queued", {"job_id": job_id, "kind": job["kind"], "retried": True}, project_id=job.get("project_id", ""), job_id=job_id)
        return job

    def retry_failed(self, project_id: str = "") -> int:
        """Retry every failed job whose clip/project still exists. Returns the count."""
        retried = 0
        for job in job_queue.list_jobs(project_id=project_id, statuses=("failed",), limit=200):
            if (job.get("error") or {}).get("code") == ErrorCode.CANCELLED:
                continue
            if not self._target_exists(job):
                continue
            if self.retry(job["id"]) is not None:
                retried += 1
        return retried

    @staticmethod
    def _target_exists(job: dict[str, Any]) -> bool:
        with session_scope() as session:
            if job.get("clip_id"):
                return session.get(Clip, job["clip_id"]) is not None
            if job.get("project_id"):
                return session.get(Project, job["project_id"]) is not None
        return True


MANAGER = JobManager()


def manager() -> JobManager:
    return MANAGER


__all__ = ["MANAGER", "JobManager", "manager"]
