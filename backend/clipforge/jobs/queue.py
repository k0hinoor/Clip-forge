"""Database-backed job queue.

Jobs live in SQLite so a crash, a restart or a closed browser never loses work:
``queued`` jobs are picked up again on the next launch, ``running`` jobs left
over from a previous process are marked failed with a clear message.

Kinds:
* ``analyze``        - the full transcript → clips pipeline
* ``render_clip``    - one clip, final quality
* ``render_preview`` - one clip, fast preview
* ``retry_failed``   - fan-out job that re-queues failed renders
* ``scan_assets``    - index files dropped into the asset folders
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import delete, func, select

from ..db import Clip, Job, Project, session_scope, utcnow
from ..errors import ClipForgeError, ErrorCode
from ..logging_setup import get_logger

log = get_logger("clipforge.worker")

STALE_RUNNING_MESSAGE = "Interrupted: CLIPFORGE was closed while this job was running. Retry to continue."
MAX_AUTO_RESUME_ATTEMPTS = 2
"""A job interrupted by a restart resumes automatically until it has been started this often.

The cap matters: a job that crashes the whole process (e.g. out of memory) would
otherwise be re-run - and crash it again - on every restart.
"""
ACTIVE_STATUSES = ("queued", "running")
FINISHED_STATUSES = ("succeeded", "failed", "cancelled")
RETRYABLE_STATUSES = ("failed", "cancelled")
RENDER_KINDS = ("render_clip", "render_preview")
_claim_lock = threading.Lock()


def enqueue(
    kind: str,
    *,
    project_id: str = "",
    clip_id: str = "",
    payload: dict[str, Any] | None = None,
    priority: int = 5,
    dedupe: bool = True,
) -> dict[str, Any]:
    """Add a job. Returns the job dict. Deduplicates identical pending work."""
    with session_scope() as session:
        if dedupe:
            existing = session.execute(
                select(Job).where(
                    Job.kind == kind,
                    Job.project_id == project_id,
                    Job.clip_id == clip_id,
                    Job.status.in_(("queued", "running")),
                )
            ).scalars().first()
            if existing is not None:
                return existing.to_dict()
        job = Job(
            kind=kind,
            project_id=project_id,
            clip_id=clip_id,
            payload_json=json.dumps(payload or {}),
            priority=priority,
        )
        session.add(job)
        session.flush()
        result = job.to_dict()
    log.debug("queued %s job %s (project=%s clip=%s)", kind, result["id"], project_id, clip_id)
    return result


def claim_next(worker_name: str, *, max_analyses: int = 1, max_renders: int = 1) -> dict[str, Any] | None:
    """Atomically move the highest-priority *runnable* queued job to ``running``.

    Concurrency limits are enforced here, inside the claim lock, from the rows
    that are actually running: at most ``max_analyses`` analyses (Whisper
    already saturates the CPU) and ``max_renders`` encodes at a time. A job
    whose kind is at its limit stays queued while other kinds may still run.
    """
    with _claim_lock, session_scope() as session:
        running = dict(
            session.execute(select(Job.kind, func.count(Job.id)).where(Job.status == "running").group_by(Job.kind)).all()
        )
        blocked: list[str] = []
        if running.get("analyze", 0) >= max(1, max_analyses):
            blocked.append("analyze")
        if sum(running.get(kind, 0) for kind in RENDER_KINDS) >= max(1, max_renders):
            blocked.extend(RENDER_KINDS)
        query = select(Job).where(Job.status == "queued")
        if blocked:
            query = query.where(Job.kind.not_in(blocked))
        job = session.execute(query.order_by(Job.priority.asc(), Job.created_at.asc()).limit(1)).scalars().first()
        if job is None:
            return None
        job.status = "running"
        job.started_at = utcnow()
        job.worker = worker_name
        job.attempts = (job.attempts or 0) + 1
        job.stage = "starting"
        job.message = "starting"
        session.flush()
        payload = job.to_dict()
    payload["payload"] = get_payload(payload["id"])
    return payload


def get_payload(job_id: str) -> dict[str, Any]:
    """The job's input payload (e.g. render overrides)."""
    with session_scope() as session:
        job = session.get(Job, job_id)
        return job.payload() if job else {}



def finish(job_id: str, result: dict[str, Any] | None = None) -> None:
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        job.status = "succeeded"
        job.progress = 1.0
        job.stage = "done"
        job.message = "finished"
        job.result_json = json.dumps(result or {}, default=str)
        job.finished_at = utcnow()


def fail(job_id: str, *, code: str, message: str, hint: str = "") -> None:
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        job.status = "failed"
        job.stage = "failed"
        job.message = message[:900]
        job.error_code = code
        job.error_message = message
        job.error_hint = hint
        job.finished_at = utcnow()


def cancel(job_id: str) -> dict[str, Any] | None:
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            return None
        if job.status in {"succeeded", "failed", "cancelled"}:
            return job.to_dict()
        job.cancel_requested = True
        if job.status == "queued":
            job.status = "cancelled"
            job.stage = "cancelled"
            job.message = "cancelled before it started"
            job.finished_at = utcnow()
        else:
            job.message = "cancelling…"
        session.flush()
        return job.to_dict()


def mark_cancelled(job_id: str) -> None:
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        job.status = "cancelled"
        job.stage = "cancelled"
        job.message = "cancelled"
        job.finished_at = utcnow()


def is_cancelled(job_id: str) -> bool:
    """True when cancellation was requested - or the job row is gone.

    Deleting a project removes its jobs; a pipeline still running for it must
    stop instead of writing into a folder that no longer exists.
    """
    with session_scope() as session:
        job = session.get(Job, job_id)
        return job is None or bool(job.cancel_requested)


def retry(job_id: str) -> dict[str, Any] | None:
    """Re-queue a failed or cancelled job. Returns ``None`` when it does not exist.

    Raises :class:`ClipForgeError` (409) for jobs that are still queued or
    running, or that already succeeded - re-running those would duplicate work.
    """
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            return None
        if job.status not in RETRYABLE_STATUSES:
            raise ClipForgeError(
                code=ErrorCode.CONFLICT,
                message=f"This job is {job.status}; only failed or cancelled jobs can be retried.",
                hint="Queue a new render or analysis instead." if job.status == "succeeded" else "Wait for it to finish.",
                status_code=409,
            )
        job.status = "queued"
        job.cancel_requested = False
        job.progress = 0.0
        job.stage = "queued"
        job.message = "re-queued"
        job.error_code = ""
        job.error_message = ""
        job.error_hint = ""
        job.started_at = None
        job.finished_at = None
        session.flush()
        return job.to_dict()


def purge_finished(project_id: str = "") -> int:
    with session_scope() as session:
        query = delete(Job).where(Job.status.in_(("succeeded", "cancelled")))
        if project_id:
            query = query.where(Job.project_id == project_id)
        result = session.execute(query)
        return int(result.rowcount or 0)


def recover_stale_jobs() -> int:
    """Repair state left behind by a previous process (crash, restart, deploy).

    * jobs still ``running`` are marked failed ("interrupted") so they can be retried;
    * projects stuck ``running``/``queued`` without a live analysis are settled;
    * clips stuck ``queued``/``rendering`` without a live render are released.

    Queued jobs stay queued and run again on this launch.
    """
    recovered = 0
    resumed = 0
    with session_scope() as session:
        rows = session.execute(select(Job).where(Job.status == "running")).scalars().all()
        for job in rows:
            recovered += 1
            if (job.attempts or 0) < MAX_AUTO_RESUME_ATTEMPTS and not job.cancel_requested:
                job.status = "queued"
                job.stage = "queued"
                job.progress = 0.0
                job.message = "resuming after a restart"
                job.started_at = None
                job.worker = ""
                if job.kind == "render_clip" and job.clip_id:
                    clip = session.get(Clip, job.clip_id)
                    if clip is not None:
                        clip.status, clip.stage, clip.progress = "queued", "queued", 0.0
                elif job.kind == "analyze" and job.project_id:
                    project = session.get(Project, job.project_id)
                    if project is not None:
                        project.status, project.stage, project.progress = "queued", "queued", 0.0
                        project.status_message = "resuming after a restart"
                resumed += 1
                continue
            job.status = "failed"
            job.stage = "failed"
            job.error_code = "interrupted"
            job.error_message = STALE_RUNNING_MESSAGE
            job.error_hint = "Open the queue and press Retry."
            job.finished_at = utcnow()
        session.flush()

        live = session.execute(select(Job.kind, Job.project_id, Job.clip_id).where(Job.status.in_(ACTIVE_STATUSES))).all()
        analysing = {project_id for kind, project_id, _clip in live if kind == "analyze"}
        rendering = {clip_id for kind, _project, clip_id in live if kind == "render_clip"}

        for project in session.execute(select(Project).where(Project.status.in_(("running", "queued")))).scalars():
            if project.id in analysing:
                continue
            has_clips = session.execute(select(func.count(Clip.id)).where(Clip.project_id == project.id)).scalar_one() > 0
            if has_clips:
                project.status, project.stage, project.status_message = "ready", "ready", "ready"
            else:
                project.status, project.stage = "failed", "failed"
                project.error_code = "interrupted"
                project.error_message = "The analysis was interrupted when CLIPFORGE stopped."
                project.error_hint = "Press Analyze to run it again."
                project.status_message = project.error_message
        for clip in session.execute(select(Clip).where(Clip.status.in_(("queued", "rendering")))).scalars():
            if clip.id in rendering:
                continue
            clip.status = "rendered" if clip.file_path and Path(clip.file_path).exists() else "pending"
            clip.stage = clip.status
            clip.progress = 1.0 if clip.status == "rendered" else 0.0
        queued = session.execute(select(func.count(Job.id)).where(Job.status == "queued")).scalar_one()
    if recovered:
        log.warning("recovered %d interrupted jobs (%d resumed automatically), %d queued", recovered, resumed, queued)
    return recovered


def active_jobs(*, project_id: str = "", clip_id: str = "", kinds: Sequence[str] = ()) -> list[dict[str, Any]]:
    """Queued or running jobs matching the filters (running first)."""
    with session_scope() as session:
        query = select(Job).where(Job.status.in_(ACTIVE_STATUSES))
        if project_id:
            query = query.where(Job.project_id == project_id)
        if clip_id:
            query = query.where(Job.clip_id == clip_id)
        if kinds:
            query = query.where(Job.kind.in_(tuple(kinds)))
        rows = session.execute(query.order_by(Job.created_at.asc())).scalars().all()
        jobs = [row.to_dict(with_log=False) for row in rows]
    return sorted(jobs, key=lambda job: 0 if job["status"] == "running" else 1)


def get(job_id: str) -> dict[str, Any] | None:
    with session_scope() as session:
        job = session.get(Job, job_id)
        return job.to_dict() if job else None


def list_jobs(
    *,
    project_id: str = "",
    statuses: Sequence[str] = (),
    limit: int = 100,
    include_finished: bool = True,
    with_log: bool = False,
) -> list[dict[str, Any]]:
    with session_scope() as session:
        query = select(Job)
        if project_id:
            query = query.where(Job.project_id == project_id)
        if statuses:
            query = query.where(Job.status.in_(tuple(statuses)))
        elif not include_finished:
            query = query.where(Job.status.in_(("queued", "running")))
        rows = session.execute(query.order_by(Job.created_at.desc()).limit(limit)).scalars().all()
        return [row.to_dict(with_log=with_log) for row in rows]


def queue_state(project_id: str = "") -> dict[str, Any]:
    """Everything the queue panel needs: running, waiting, recent failures."""
    jobs = list_jobs(project_id=project_id, limit=200)
    running = [job for job in jobs if job["status"] == "running"]
    queued = sorted([job for job in jobs if job["status"] == "queued"], key=lambda job: (job["priority"], job["created_at"]))
    failed = [job for job in jobs if job["status"] == "failed"][:25]
    cancelled = [job for job in jobs if job["status"] == "cancelled"][:15]
    done = [job for job in jobs if job["status"] == "succeeded"][:15]

    clip_titles: dict[str, str] = {}
    clip_indexes: dict[str, int] = {}
    project_titles: dict[str, str] = {}
    clip_ids = {job["clip_id"] for job in jobs if job.get("clip_id")}
    project_ids = {job["project_id"] for job in jobs if job.get("project_id")}
    with session_scope() as session:
        if clip_ids:
            for clip in session.execute(select(Clip).where(Clip.id.in_(tuple(clip_ids)))).scalars():
                clip_titles[clip.id] = clip.title
                clip_indexes[clip.id] = clip.index
        if project_ids:
            for project_id, title in session.execute(select(Project.id, Project.title).where(Project.id.in_(tuple(project_ids)))).all():
                project_titles[project_id] = title

    def decorate(job: dict[str, Any]) -> dict[str, Any]:
        clip_id = job.get("clip_id") or ""
        return {
            **job,
            "clip_title": clip_titles.get(clip_id, ""),
            "clip_index": clip_indexes.get(clip_id, 0),
            "project_title": project_titles.get(job.get("project_id") or "", ""),
        }

    return {
        "running": [decorate(job) for job in running],
        "queued": [decorate(job) for job in queued],
        "failed": [decorate(job) for job in failed],
        "cancelled": [decorate(job) for job in cancelled],
        "recent": [decorate(job) for job in done],
        "counts": {
            "running": len(running),
            "queued": len(queued),
            "failed": len(failed),
            "total": len(jobs),
        },
    }


def active_for_project(project_id: str) -> dict[str, Any] | None:
    jobs = list_jobs(project_id=project_id, statuses=("running", "queued"), limit=10)
    if not jobs:
        return None
    running = next((job for job in jobs if job["status"] == "running"), None)
    return running or jobs[0]


def project_job_summary(project_id: str) -> dict[str, Any]:
    jobs = list_jobs(project_id=project_id, limit=200, include_finished=True)
    return {
        "queued": sum(1 for job in jobs if job["status"] == "queued"),
        "running": sum(1 for job in jobs if job["status"] == "running"),
        "failed": sum(1 for job in jobs if job["status"] == "failed"),
        "succeeded": sum(1 for job in jobs if job["status"] == "succeeded"),
    }


def render_queue_for_project(project_id: str) -> list[dict[str, Any]]:
    """Ordered render queue with clip metadata (for the batch render panel)."""
    jobs = [
        job for job in list_jobs(project_id=project_id, statuses=("queued", "running", "failed", "succeeded"), limit=500)
        if job["kind"] in {"render_clip", "render_preview"}
    ]
    jobs.sort(key=lambda job: (0 if job["status"] == "running" else 1 if job["status"] == "queued" else 2, job["created_at"]))
    return jobs


def update_progress(job_id: str, *, progress: float, stage: str, message: str, log_entry: bool = False) -> dict[str, Any] | None:
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            return None
        job.progress = max(0.0, min(1.0, progress))
        job.stage = stage
        job.message = message[:900]
        if log_entry:
            entries = job.stage_log()
            entries.append({"stage": stage, "message": message, "progress": round(job.progress, 3)})
            job.log_json = json.dumps(entries[-120:])
        session.flush()
        return job.to_dict(with_log=False)


__all__ = [
    "ACTIVE_STATUSES",
    "MAX_AUTO_RESUME_ATTEMPTS",
    "RENDER_KINDS",
    "STALE_RUNNING_MESSAGE",
    "active_for_project",
    "active_jobs",
    "cancel",
    "claim_next",
    "enqueue",
    "fail",
    "finish",
    "get",
    "get_payload",
    "is_cancelled",
    "list_jobs",
    "mark_cancelled",
    "project_job_summary",
    "purge_finished",
    "queue_state",
    "recover_stale_jobs",
    "render_queue_for_project",
    "retry",
    "update_progress",
]
