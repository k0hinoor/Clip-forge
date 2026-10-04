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
from typing import Any, Iterable, Sequence

from sqlalchemy import delete, func, select

from ..db import Job, Project, session_scope, utcnow
from ..logging_setup import get_logger

log = get_logger("clipforge.worker")

STALE_RUNNING_MESSAGE = "Interrupted: CLIPFORGE was closed while this job was running. Retry to continue."
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


def claim_next(worker_name: str) -> dict[str, Any] | None:
    """Atomically move the highest-priority queued job to ``running``."""
    with _claim_lock, session_scope() as session:
        job = session.execute(
            select(Job)
            .where(Job.status == "queued")
            .order_by(Job.priority.asc(), Job.created_at.asc())
            .limit(1)
        ).scalars().first()
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
    payload["payload"] = _payload_of(payload["id"])
    return payload


def _payload_of(job_id: str) -> dict[str, Any]:
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
    with session_scope() as session:
        job = session.get(Job, job_id)
        return bool(job and job.cancel_requested)


def retry(job_id: str) -> dict[str, Any] | None:
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            return None
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
    """Mark jobs left 'running' by a previous process so the UI is honest."""
    recovered = 0
    with session_scope() as session:
        rows = session.execute(select(Job).where(Job.status == "running")).scalars().all()
        for job in rows:
            job.status = "failed"
            job.stage = "failed"
            job.error_code = "interrupted"
            job.error_message = STALE_RUNNING_MESSAGE
            job.error_hint = "Open the queue and press Retry."
            job.finished_at = utcnow()
            recovered += 1
        queued = session.execute(select(func.count(Job.id)).where(Job.status == "queued")).scalar_one()
    if recovered:
        log.warning("recovered %d interrupted jobs, %d still queued", recovered, queued)
    return recovered


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
    done = [job for job in jobs if job["status"] == "succeeded"][:15]

    clip_titles: dict[str, str] = {}
    clip_indexes: dict[str, int] = {}
    clip_ids = {job["clip_id"] for job in jobs if job.get("clip_id")}
    if clip_ids:
        from ..db import Clip

        with session_scope() as session:
            for clip in session.execute(select(Clip).where(Clip.id.in_(tuple(clip_ids)))).scalars():
                clip_titles[clip.id] = clip.title
                clip_indexes[clip.id] = clip.index

    def decorate(job: dict[str, Any]) -> dict[str, Any]:
        clip_id = job.get("clip_id") or ""
        return {
            **job,
            "clip_title": clip_titles.get(clip_id, ""),
            "clip_index": clip_indexes.get(clip_id, 0),
        }

    return {
        "running": [decorate(job) for job in running],
        "queued": [decorate(job) for job in queued],
        "failed": [decorate(job) for job in failed],
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
    "STALE_RUNNING_MESSAGE",
    "active_for_project",
    "cancel",
    "claim_next",
    "enqueue",
    "fail",
    "finish",
    "get",
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
