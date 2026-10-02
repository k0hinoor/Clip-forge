"""Job routes (TRD §28.3) + SSE progress stream (TRD §29)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from clipforge.api.deps import current_user, get_db, get_queue_dep, get_settings_dep, get_storage_dep
from clipforge.api.schemas import ClipOut, JobCreateRequest, JobEventOut, JobListOut, JobOut
from clipforge.api.serializers import clip_out, event_out, job_out
from clipforge.core.config import Settings
from clipforge.core.errors import not_found
from clipforge.core.states import TERMINAL_STATES, JobStatus
from clipforge.db.models import Job, JobEvent, Transcript, TranscriptSegment, User
from clipforge.db.session import session_scope
from clipforge.queue.base import QueueProvider
from clipforge.services.clips import ClipService
from clipforge.services.jobs import JobService
from clipforge.storage import StorageProvider

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _svc(db: Session, settings: Settings, queue: QueueProvider, storage: StorageProvider) -> JobService:
    return JobService(db, settings, queue, storage)


@router.post("", response_model=JobOut, status_code=201)
def create_job(body: JobCreateRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
               user: User = Depends(current_user), db: Session = Depends(get_db),
               settings: Settings = Depends(get_settings_dep), queue: QueueProvider = Depends(get_queue_dep),
               storage: StorageProvider = Depends(get_storage_dep)) -> Any:
    svc = _svc(db, settings, queue, storage)
    job, created = svc.create(user, body.upload_id, body.settings.model_dump(exclude_none=True), idempotency_key)
    db.commit()
    if created:
        svc.notify_queue(job)  # after commit so the worker can see the row
    return JSONResponse(job_out(job, db), status_code=201 if created else 200)


@router.get("", response_model=JobListOut)
def list_jobs(status: str | None = Query(default=None, max_length=30), limit: int = Query(default=20, ge=1, le=100),
              offset: int = Query(default=0, ge=0), user: User = Depends(current_user),
              db: Session = Depends(get_db), settings: Settings = Depends(get_settings_dep)) -> dict[str, Any]:
    rows, total = JobService(db, settings).list(user, status=status, limit=limit, offset=offset)
    return {"items": [job_out(j, db) for j in rows], "total": total, "limit": limit, "offset": offset}


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str, user: User = Depends(current_user), db: Session = Depends(get_db),
            settings: Settings = Depends(get_settings_dep)) -> dict[str, Any]:
    return job_out(JobService(db, settings).get(user, job_id), db)


@router.post("/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: str, user: User = Depends(current_user), db: Session = Depends(get_db),
               settings: Settings = Depends(get_settings_dep), queue: QueueProvider = Depends(get_queue_dep),
               storage: StorageProvider = Depends(get_storage_dep)) -> dict[str, Any]:
    svc = _svc(db, settings, queue, storage)
    job = svc.cancel(user, svc.get(user, job_id))
    db.commit()
    return job_out(job, db)


@router.post("/{job_id}/retry", response_model=JobOut)
def retry_job(job_id: str, user: User = Depends(current_user), db: Session = Depends(get_db),
              settings: Settings = Depends(get_settings_dep), queue: QueueProvider = Depends(get_queue_dep),
              storage: StorageProvider = Depends(get_storage_dep)) -> dict[str, Any]:
    svc = _svc(db, settings, queue, storage)
    job = svc.retry(user, svc.get(user, job_id))
    db.commit()
    svc.notify_queue(job)
    return job_out(job, db)


@router.delete("/{job_id}", status_code=204)
def delete_job(job_id: str, user: User = Depends(current_user), db: Session = Depends(get_db),
               settings: Settings = Depends(get_settings_dep), queue: QueueProvider = Depends(get_queue_dep),
               storage: StorageProvider = Depends(get_storage_dep)) -> None:
    svc = _svc(db, settings, queue, storage)
    svc.delete(user, svc.get(user, job_id))
    db.commit()


@router.get("/{job_id}/clips", response_model=list[ClipOut])
def job_clips(job_id: str, user: User = Depends(current_user), db: Session = Depends(get_db),
              settings: Settings = Depends(get_settings_dep),
              storage: StorageProvider = Depends(get_storage_dep)) -> list[dict[str, Any]]:
    job = JobService(db, settings).get(user, job_id)
    return [clip_out(c, db, settings) for c in ClipService(db, settings, storage).list_for_job(user, job)]


@router.get("/{job_id}/events", response_model=list[JobEventOut])
def job_events(job_id: str, after: int = Query(default=0, ge=0), limit: int = Query(default=200, ge=1, le=1000),
               user: User = Depends(current_user), db: Session = Depends(get_db),
               settings: Settings = Depends(get_settings_dep)) -> list[dict[str, Any]]:
    JobService(db, settings).get(user, job_id)
    rows = db.scalars(select(JobEvent).where(JobEvent.job_id == job_id, JobEvent.id > after)
                      .order_by(JobEvent.id).limit(limit)).all()
    return [event_out(e) for e in rows]


@router.get("/{job_id}/transcript")
def job_transcript(job_id: str, user: User = Depends(current_user), db: Session = Depends(get_db),
                   settings: Settings = Depends(get_settings_dep)) -> dict[str, Any]:
    JobService(db, settings).get(user, job_id)
    t = db.scalar(select(Transcript).where(Transcript.job_id == job_id))
    if t is None:
        raise not_found("Transcript")
    segs = db.scalars(select(TranscriptSegment).where(TranscriptSegment.transcript_id == t.id)
                      .order_by(TranscriptSegment.idx)).all()
    return {"language": t.language, "provider": t.provider, "model": t.model, "duration_seconds": t.duration_seconds,
            "word_count": t.word_count,
            "segments": [{"id": s.idx, "start": s.start, "end": s.end, "text": s.text, "speaker_id": s.speaker_id,
                          "topic_id": s.topic_id, "words": s.words} for s in segs]}


# ------------------------------------------------------------------- SSE
def _sse(event: str, data: dict[str, Any], event_id: int | None = None) -> str:
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append("data: " + json.dumps(data, separators=(",", ":")))
    return "\n".join(lines) + "\n\n"


@router.get("/{job_id}/stream")
async def job_stream(job_id: str, request: Request, user: User = Depends(current_user),
                     last_event_id: str | None = Header(default=None, alias="Last-Event-ID")) -> StreamingResponse:
    """Server-Sent Events: ``job.progress`` snapshots plus raw ``job.event`` rows.

    Reconnects resume from ``Last-Event-ID``. The stream closes once the job
    reaches a terminal state.
    """
    factory = request.app.state.session_factory
    settings: Settings = request.app.state.settings
    user_id = user.id

    def load(after: int) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        with session_scope(factory) as s:
            job = s.scalar(select(Job).where(Job.id == job_id, Job.user_id == user_id, Job.deleted_at.is_(None)))
            if job is None:
                return None, []
            events = s.scalars(select(JobEvent).where(JobEvent.job_id == job_id, JobEvent.id > after)
                               .order_by(JobEvent.id).limit(500)).all()
            return job_out(job), [event_out(e) for e in events]

    first, _ = await run_in_threadpool(load, 0)
    if first is None:
        raise not_found("Job")
    try:
        cursor = int(last_event_id) if last_event_id else 0
    except ValueError:
        cursor = 0

    async def gen() -> AsyncIterator[str]:
        nonlocal cursor
        yield f"retry: {settings.SSE_RETRY_MS}\n\n"
        last_snapshot: tuple[Any, ...] | None = None
        idle = 0.0
        while True:
            if await request.is_disconnected():
                return
            job, events = await run_in_threadpool(load, cursor)
            if job is None:
                yield _sse("job.deleted", {"job_id": job_id})
                return
            for e in events:
                cursor = e["id"]
                yield _sse("job.event", {"job_id": job_id, **e}, cursor)
            snapshot = (job["status"], job["progress"], job["current_stage"])
            if snapshot != last_snapshot:
                last_snapshot = snapshot
                idle = 0.0
                yield _sse("job.progress", {"job_id": job_id, "status": job["status"], "progress": job["progress"],
                                            "stage": job["current_stage"], "error": job["error"]})
            if JobStatus(job["status"]) in TERMINAL_STATES and not events:
                yield _sse("job.finished", {"job_id": job_id, "status": job["status"]})
                return
            await asyncio.sleep(settings.SSE_POLL_SECONDS)
            idle += settings.SSE_POLL_SECONDS
            if idle >= 15:
                idle = 0.0
                yield ": keep-alive\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
