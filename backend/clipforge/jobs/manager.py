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

from ..config import get_settings
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
        self._analysis_lock = threading.Semaphore(1)
        self._render_lock = threading.Semaphore(1)
        self._progress_lock = threading.Lock()

    # ------------------------------------------------------------ lifecycle
    def start(self, *, workers: int | None = None) -> dict[str, Any]:
        started: list[Worker] = []
        with self._lock:
            if self._workers:
                pass  # already running - fall through and report the current state
            else:
                settings = get_settings()
                count = max(1, min(workers if workers is not None else settings.concurrency + settings.max_concurrent_renders - 1, 8))
                self._render_lock = threading.Semaphore(max(1, settings.max_concurrent_renders))
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
        self._stop.set()
        for worker in self._workers:
            worker.join(timeout=timeout)
        with self._lock:
            self._workers = []
        from ..media.runner import kill_all

        kill_all()
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
    def enqueue_analysis(self, project_id: str, *, priority: int = 1) -> dict[str, Any]:
        self.ensure_started()
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
        job = job_queue.enqueue("analyze", project_id=project_id, priority=priority)
        BUS.publish("job.queued", {"job_id": job["id"], "kind": "analyze", "status": job["status"]}, project_id=project_id, job_id=job["id"])
        return job

    def enqueue_render(self, clip_id: str, *, export: bool = False, overrides: dict[str, Any] | None = None, priority: int = 6) -> dict[str, Any]:
        self.ensure_started()
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
        if not self.status()["running"]:
            self.start()

    # ---------------------------------------------------------------- control
    def cancel(self, job_id: str, *, kill: bool = True) -> dict[str, Any] | None:
        job = job_queue.cancel(job_id)
        if kill and job and job.get("clip_id"):
            from ..media.runner import kill_process

            kill_process(job["clip_id"])
        if kill:
            from ..media.runner import kill_process

            kill_process(job_id)
        if job:
            BUS.publish("job.cancelling", {"job_id": job_id}, project_id=job.get("project_id", ""), job_id=job_id)
        return job

    def cancel_project(self, project_id: str) -> int:
        jobs = job_queue.list_jobs(project_id=project_id, statuses=("queued", "running"), limit=200)
        for job in jobs:
            self.cancel(job["id"])
        return len(jobs)

    def retry(self, job_id: str) -> dict[str, Any] | None:
        job = job_queue.retry(job_id)
        if job is None:
            return None
        BUS.publish("job.queued", {"job_id": job_id, "retried": True}, project_id=job.get("project_id", ""), job_id=job_id)
        return job

    def retry_failed(self, project_id: str = "") -> int:
        jobs = job_queue.list_jobs(project_id=project_id, statuses=("failed",), limit=200)
        retried = 0
        for job in jobs:
            if job.get("error", {}).get("code") == ErrorCode.CANCELLED:
                continue
            self.retry(job["id"])
            retried += 1
        return retried


MANAGER = JobManager()


def manager() -> JobManager:
    return MANAGER


__all__ = ["MANAGER", "JobManager", "manager"]
