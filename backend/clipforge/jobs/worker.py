"""Worker threads: claim a job, run the real pipeline, report real progress."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from ..constants import progress_for
from ..db import Project, session_scope
from ..errors import ClipForgeError, ErrorCode
from ..logging_setup import get_logger
from ..services.events import BUS
from . import queue as job_queue

log = get_logger("clipforge.worker")

POLL_SECONDS = 1.0


class JobReporter:
    """Bridges pipeline progress into the job row, the project row and the event bus."""

    def __init__(self, job_id: str, *, project_id: str = "", clip_id: str = "") -> None:
        self.job_id = job_id
        self.project_id = project_id
        self.clip_id = clip_id
        self.stage_key = ""
        self.stage_fraction = 0.0
        self.last_publish = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------ interface
    def stage(self, key: str, message: str, *, fraction: float = 0.0) -> None:
        with self._lock:
            self.stage_key = key
            self.stage_fraction = max(0.0, min(1.0, fraction))
        overall = progress_for(key, fraction)
        self._push(overall, key, message, log_entry=True)

    def sub(self, fraction: float, message: str = "") -> None:
        with self._lock:
            self.stage_fraction = max(0.0, min(1.0, fraction))
            key = self.stage_key
        if not key:
            return
        overall = progress_for(key, fraction)
        self._push(overall, key, message or key.replace("_", " ").title())

    def log(self, message: str) -> None:
        self._push(None, self.stage_key or "log", message, log_entry=True)

    def cancelled(self) -> bool:
        return job_queue.is_cancelled(self.job_id)

    # -------------------------------------------------------------- internals
    def _push(self, overall: float | None, stage: str, message: str, *, log_entry: bool = False) -> None:
        now = time.time()
        throttled = overall is not None and (now - self.last_publish) < 0.4 and not log_entry
        if throttled:
            return
        self.last_publish = now

        payload: dict[str, Any] = {"stage": stage, "message": message}
        if overall is not None:
            job_queue.update_progress(self.job_id, progress=overall, stage=stage, message=message, log_entry=log_entry)
            payload["progress"] = round(overall, 4)
            self._update_project(overall, message)
        else:
            job_queue.update_progress(self.job_id, progress=_current_progress(self.job_id), stage=stage, message=message, log_entry=True)
        BUS.publish("job.progress", {**payload, "job_id": self.job_id, "clip_id": self.clip_id}, project_id=self.project_id, job_id=self.job_id)

    def _update_project(self, progress: float, message: str) -> None:
        if not self.project_id:
            return
        with session_scope() as session:
            project = session.get(Project, self.project_id)
            if project is None:
                return
            project.progress = progress
            project.stage = self.stage_key
            project.status = "running"
            project.status_message = message[:400]


def _current_progress(job_id: str) -> float:
    job = job_queue.get(job_id)
    return float(job["progress"]) if job else 0.0


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #


def run_job(job: dict[str, Any], *, worker_name: str) -> dict[str, Any]:
    """Execute one claimed job. Raises :class:`ClipForgeError` on failure."""
    kind = job["kind"]
    project_id = job.get("project_id") or ""
    clip_id = job.get("clip_id") or ""
    payload = job.get("payload") or {}
    reporter = JobReporter(job["id"], project_id=project_id, clip_id=clip_id)

    log.info("worker %s running %s (%s)", worker_name, kind, job["id"])

    if kind == "analyze":
        from ..pipeline.analyze import analyze

        outcome = analyze(project_id, report=reporter)
        return outcome.to_dict()

    if kind in {"render_clip", "render_preview"}:
        from ..pipeline.render import render_clip

        result = render_clip(
            clip_id,
            report=reporter,
            quality="preview" if kind == "render_preview" else "final",
            overrides=payload.get("overrides") or None,
            export=bool(payload.get("export", False)),
            cancel_key=job["id"],
            should_cancel=reporter.cancelled,
        )
        BUS.publish("clip.updated", {"clip_id": clip_id, "status": "rendered" if kind == "render_clip" else "preview"}, project_id=project_id, job_id=job["id"])
        return result

    if kind == "render_all":
        ids = payload.get("clip_ids") or []
        for position, target in enumerate(ids, start=1):
            job_queue.enqueue(
                "render_clip",
                project_id=project_id,
                clip_id=target,
                payload={"export": payload.get("export", False)},
                priority=6,
            )
            reporter.sub(position / max(len(ids), 1), f"queued {position}/{len(ids)} clips")
        return {"queued": len(ids)}

    if kind == "scan_assets":
        from ..media.assets import ensure_seed_assets

        imported = ensure_seed_assets()
        return {"imported": imported}

    raise ClipForgeError(
        code=ErrorCode.INVALID_INPUT,
        message=f"Unknown job kind: {kind}",
        status_code=400,
    )


class Worker(threading.Thread):
    """One worker thread: claim jobs until stopped."""

    def __init__(self, name: str, *, stop_event: threading.Event, on_idle: Callable[[], None] | None = None) -> None:
        super().__init__(name=name, daemon=True)
        self.stop_event = stop_event
        self.on_idle = on_idle
        self.current_job: str = ""

    def run(self) -> None:  # noqa: D102 - thread body
        log.debug("%s started", self.name)
        while not self.stop_event.is_set():
            job = job_queue.claim_next(self.name)
            if job is None:
                if self.on_idle:
                    self.on_idle()
                self.stop_event.wait(POLL_SECONDS)
                continue
            self.current_job = job["id"]
            BUS.publish(
                "job.started",
                {"job_id": job["id"], "kind": job["kind"], "worker": self.name},
                project_id=job.get("project_id", ""),
                job_id=job["id"],
            )
            from ..media import runner as media_runner

            started = time.time()
            try:
                result = run_job(job, worker_name=self.name)
                if job_queue.is_cancelled(job["id"]):
                    job_queue.mark_cancelled(job["id"])
                    BUS.publish("job.cancelled", {"job_id": job["id"]}, project_id=job.get("project_id", ""), job_id=job["id"])
                else:
                    job_queue.finish(job["id"], {**result, "duration_seconds": round(time.time() - started, 2)})
                    BUS.publish(
                        "job.finished",
                        {"job_id": job["id"], "kind": job["kind"], "result": result, "duration": round(time.time() - started, 2)},
                        project_id=job.get("project_id", ""),
                        job_id=job["id"],
                    )
            except ClipForgeError as exc:
                if exc.code == ErrorCode.CANCELLED:
                    job_queue.mark_cancelled(job["id"])
                    BUS.publish("job.cancelled", {"job_id": job["id"], "message": exc.message}, project_id=job.get("project_id", ""), job_id=job["id"])
                else:
                    log.error("job %s failed: %s", job["id"], exc.message)
                    job_queue.fail(job["id"], code=exc.code, message=exc.message, hint=exc.hint)
                    _mark_project_failed(job, exc)
                    BUS.publish(
                        "job.failed",
                        {"job_id": job["id"], "kind": job["kind"], "error": {"code": exc.code, "message": exc.message, "hint": exc.hint}},
                        project_id=job.get("project_id", ""),
                        job_id=job["id"],
                    )
            except Exception as exc:  # noqa: BLE001 - unexpected errors must not kill the worker
                log.exception("unexpected failure in job %s", job["id"])
                message = f"Unexpected error: {type(exc).__name__}: {exc}"[:400]
                job_queue.fail(job["id"], code=ErrorCode.INTERNAL, message=message, hint="See logs/worker.log for the full traceback.")
                _mark_project_failed(job, ClipForgeError(code=ErrorCode.INTERNAL, message=message))
                BUS.publish(
                    "job.failed",
                    {"job_id": job["id"], "kind": job["kind"], "error": {"code": ErrorCode.INTERNAL, "message": message}},
                    project_id=job.get("project_id", ""),
                    job_id=job["id"],
                )
            finally:
                self.current_job = ""
                media_runner.kill_all()
        log.debug("%s stopped", self.name)


def _mark_project_failed(job: dict[str, Any], error: ClipForgeError) -> None:
    project_id = job.get("project_id")
    if not project_id or job.get("kind") != "analyze":
        return
    with session_scope() as session:
        project = session.get(Project, project_id)
        if project is None:
            return
        project.status = "failed"
        project.stage = "failed"
        project.error_code = error.code
        project.error_message = error.message
        project.error_hint = error.hint
        project.status_message = error.message
        BUS.publish(
            "project.failed",
            {"project_id": project_id, "error": {"code": error.code, "message": error.message, "hint": error.hint}},
            project_id=project_id,
        )


__all__ = ["JobReporter", "Worker", "run_job"]
