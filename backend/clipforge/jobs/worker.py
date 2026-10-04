"""Worker threads: claim a job, run the real pipeline, report real progress."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import func, select

from ..config import get_settings
from ..constants import progress_for
from ..db import Clip, Project, session_scope
from ..errors import ClipForgeError, ErrorCode
from ..logging_setup import get_logger
from ..services.events import BUS
from . import queue as job_queue

log = get_logger("clipforge.worker")

POLL_SECONDS = 1.0


class JobReporter:
    """Bridges pipeline progress into the job row, the project row and the event bus.

    Only an analysis drives the *project's* progress and status; a render
    reports on its job and clip, so rendering one clip never turns a finished
    project back into "running".
    """

    def __init__(self, job_id: str, *, project_id: str = "", clip_id: str = "", kind: str = "") -> None:
        self.job_id = job_id
        self.project_id = project_id
        self.clip_id = clip_id
        self.kind = kind
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
        if not self.project_id or self.kind != "analyze":
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
    reporter = JobReporter(job["id"], project_id=project_id, clip_id=clip_id, kind=kind)

    log.info("worker %s running %s (%s)", worker_name, kind, job["id"])

    if kind == "analyze":
        from ..pipeline.analyze import analyze

        outcome = analyze(project_id, report=reporter, force=bool(payload.get("force", False)))
        return outcome.to_dict()

    if kind in {"render_clip", "render_preview"}:
        from ..pipeline.render import render_clip

        result = render_clip(
            clip_id,
            report=reporter,
            quality="preview" if kind == "render_preview" else "final",
            overrides=payload.get("overrides") or None,
            export=bool(payload.get("export", False)),
            should_cancel=reporter.cancelled,
        )
        BUS.publish(
            "clip.updated",
            {"clip_id": clip_id, "status": "rendered" if kind == "render_clip" else "preview_ready"},
            project_id=project_id,
            job_id=job["id"],
        )
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
            try:
                job = job_queue.claim_next(self.name, max_analyses=1, max_renders=get_settings().max_concurrent_renders)
            except Exception:  # noqa: BLE001 - a database hiccup must not kill the worker
                log.exception("%s could not claim a job", self.name)
                self.stop_event.wait(POLL_SECONDS * 5)
                continue
            if job is None:
                if self.on_idle:
                    self.on_idle()
                self.stop_event.wait(POLL_SECONDS)
                continue
            self.current_job = job["id"]
            try:
                self._execute(job)
            finally:
                self.current_job = ""
                _kill_job_processes(job)
        log.debug("%s stopped", self.name)

    def _shutting_down(self, job: dict[str, Any]) -> bool:
        """The app is stopping: leave the job ``running`` so the next start resumes it."""
        if self.stop_event.is_set() and not job_queue.is_cancelled(job["id"]):
            log.info("%s: %s interrupted by shutdown - it will resume on the next start", self.name, job["id"])
            return True
        return False

    def _execute(self, job: dict[str, Any]) -> None:
        BUS.publish(
            "job.started",
            {"job_id": job["id"], "kind": job["kind"], "worker": self.name, "clip_id": job.get("clip_id", "")},
            project_id=job.get("project_id", ""),
            job_id=job["id"],
        )
        started = time.time()
        try:
            result = run_job(job, worker_name=self.name)
        except ClipForgeError as exc:
            if self._shutting_down(job):
                return
            if exc.code in {ErrorCode.CANCELLED, ErrorCode.RENDER_CANCELLED}:
                _on_cancelled(job, exc.message)
            else:
                _on_failed(job, exc)
        except Exception as exc:  # noqa: BLE001 - unexpected errors must not kill the worker
            if self._shutting_down(job):
                return
            log.exception("unexpected failure in job %s", job["id"])
            message = f"Unexpected error: {type(exc).__name__}: {exc}"[:400]
            _on_failed(job, ClipForgeError(code=ErrorCode.INTERNAL, message=message, hint="See logs/worker.log for the full traceback."))
        else:
            if job_queue.is_cancelled(job["id"]):
                _on_cancelled(job, "cancelled")
            else:
                _on_succeeded(job, result, time.time() - started)


# --------------------------------------------------------------------------- #
# Terminal states - every path leaves the job, its clip and its project consistent
# --------------------------------------------------------------------------- #


def _on_succeeded(job: dict[str, Any], result: dict[str, Any], seconds: float) -> None:
    job_queue.finish(job["id"], {**result, "duration_seconds": round(seconds, 2)})
    _settle_project(job)
    BUS.publish(
        "job.finished",
        {"job_id": job["id"], "kind": job["kind"], "clip_id": job.get("clip_id", ""), "result": result, "duration": round(seconds, 2)},
        project_id=job.get("project_id", ""),
        job_id=job["id"],
    )


def _on_failed(job: dict[str, Any], error: ClipForgeError) -> None:
    log.error("job %s (%s) failed: %s", job["id"], job["kind"], error.message)
    job_queue.fail(job["id"], code=error.code, message=error.message, hint=error.hint)
    _mark_project_failed(job, error)
    _mark_clip_failed(job, error)
    _settle_project(job)
    BUS.publish(
        "job.failed",
        {
            "job_id": job["id"],
            "kind": job["kind"],
            "clip_id": job.get("clip_id", ""),
            "error": {"code": error.code, "message": error.message, "hint": error.hint},
        },
        project_id=job.get("project_id", ""),
        job_id=job["id"],
    )


def _on_cancelled(job: dict[str, Any], message: str) -> None:
    job_queue.mark_cancelled(job["id"])
    settle_cancelled(job)
    BUS.publish(
        "job.cancelled",
        {"job_id": job["id"], "kind": job["kind"], "clip_id": job.get("clip_id", ""), "message": message},
        project_id=job.get("project_id", ""),
        job_id=job["id"],
    )


def settle_cancelled(job: dict[str, Any]) -> None:
    """Put the clip/project of a cancelled job back into a truthful state.

    Used by the worker (running jobs) and by the manager (jobs cancelled while
    still queued). A cancelled render falls back to its previous file if there
    is one; a cancelled analysis leaves the project ``ready`` when it still has
    clips from an earlier run, otherwise ``cancelled``.
    """
    project_id = job.get("project_id") or ""
    clip_id = job.get("clip_id") or ""
    if job.get("kind") == "render_clip" and clip_id:
        _release_clip(clip_id, project_id=project_id, job_id=job.get("id", ""))
    elif job.get("kind") == "analyze" and project_id:
        with session_scope() as session:
            project = session.get(Project, project_id)
            if project is None:
                return
            has_clips = session.execute(select(func.count(Clip.id)).where(Clip.project_id == project_id)).scalar_one() > 0
            project.status = "ready" if has_clips else "cancelled"
            project.stage = project.status
            project.status_message = "analysis cancelled" + (" - showing the clips from the previous run" if has_clips else "")
            status = project.status
        BUS.publish("project.updated", {"project_id": project_id, "status": status, "stage": status}, project_id=project_id)
    _settle_project(job)


def _release_clip(clip_id: str, *, project_id: str, job_id: str) -> None:
    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        if clip is None or clip.status not in {"queued", "rendering"}:
            return
        rendered = bool(clip.file_path) and Path(clip.file_path).exists()
        clip.status = "rendered" if rendered else "pending"
        clip.stage = "cancelled"
        clip.progress = 1.0 if rendered else 0.0
        status = clip.status
    BUS.publish("clip.updated", {"clip_id": clip_id, "status": status}, project_id=project_id, job_id=job_id)


def _kill_job_processes(job: dict[str, Any]) -> None:
    """Stop anything this job left running - and nothing that belongs to another worker."""
    from ..media.runner import kill_process

    for key in (job.get("id"), job.get("clip_id")):
        if key:
            kill_process(key)


def _settle_project(job: dict[str, Any]) -> None:
    """Make sure a project is not left ``running``/``queued`` with nothing running.

    Analyses write their own final state. This is the safety net for every
    other job - and for projects touched by older versions that let render
    progress flip the project to "running".
    """
    project_id = job.get("project_id")
    if not project_id or job.get("kind") == "analyze":
        return
    if job_queue.active_jobs(project_id=project_id, kinds=("analyze",)):
        return
    with session_scope() as session:
        project = session.get(Project, project_id)
        if project is None or project.status not in {"running", "queued"}:
            return
        project.status = "ready"
        project.stage = "ready"
        project.status_message = "ready"
    BUS.publish(
        "project.updated",
        {"project_id": project_id, "status": "ready", "stage": "ready"},
        project_id=project_id,
        job_id=job.get("id", ""),
    )


def _mark_clip_failed(job: dict[str, Any], error: ClipForgeError) -> None:
    """Flip a clip out of ``queued``/``rendering`` when its final render fails.

    A failed *preview* leaves the clip's status alone (a rendered clip stays
    rendered) and only records the error so the editor can show it.
    """
    clip_id = job.get("clip_id")
    kind = job.get("kind")
    if kind not in job_queue.RENDER_KINDS or not clip_id:
        return
    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            return
        if kind == "render_clip":
            clip.status = "failed"
            clip.stage = "failed"
        clip.error_code = error.code
        clip.error_message = error.message if kind == "render_clip" else f"Preview failed: {error.message}"
        status = clip.status
    BUS.publish(
        "clip.updated",
        {"clip_id": clip_id, "status": status, "error_message": error.message, "preview": kind == "render_preview"},
        project_id=job.get("project_id", ""),
        job_id=job.get("id", ""),
    )


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


__all__ = ["JobReporter", "Worker", "run_job", "settle_cancelled"]
