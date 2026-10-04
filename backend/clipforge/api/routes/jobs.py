"""Job/queue routes and the Server-Sent Events stream for live progress."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Body, Query
from fastapi.responses import StreamingResponse

from ...db import Job, session_scope
from ...errors import ClipForgeError, ErrorCode
from ...jobs import queue as job_queue
from ...jobs.manager import manager
from ...logging_setup import get_logger
from ...services.events import BUS
from ...system import hardware, process_snapshot

router = APIRouter(tags=["jobs"])
log = get_logger(__name__)

HEARTBEAT_SECONDS = 15.0
"""Idle interval between keep-alive frames (proxies drop silent connections)."""
USAGE_INTERVAL_SECONDS = 10.0


def _body_value(body: dict[str, Any] | None, key: str, fallback: Any) -> Any:
    """Read ``key`` from an optional JSON body, falling back to the query value."""
    if isinstance(body, dict) and body.get(key) not in (None, ""):
        return body[key]
    return fallback


@router.get("/jobs")
def list_jobs(
    project_id: str = Query(""),
    statuses: str = Query("", description="Comma separated"),
    limit: int = Query(100, ge=1, le=500),
    with_log: bool = Query(False, description="Include the per-stage log of each job"),
):
    parsed = tuple(item for item in statuses.split(",") if item)
    return {"jobs": job_queue.list_jobs(project_id=project_id, statuses=parsed, limit=limit, with_log=with_log)}


@router.get("/jobs/queue")
def get_queue(project_id: str = Query("")):
    return {
        "queue": job_queue.queue_state(project_id),
        "workers": manager().status(),
    }


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    """One job with its per-stage log (poll this to follow a job without SSE)."""
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise ClipForgeError(code=ErrorCode.NOT_FOUND, message="Job not found.", status_code=404)
        payload = job.to_dict(with_log=True)
    payload["payload"] = job_queue.get_payload(job_id)
    return payload


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = manager().cancel(job_id)
    if job is None:
        raise ClipForgeError(code=ErrorCode.NOT_FOUND, message="Job not found.", status_code=404)
    return job


@router.post("/jobs/{job_id}/retry")
def retry_job(job_id: str):
    job = manager().retry(job_id)
    if job is None:
        raise ClipForgeError(code=ErrorCode.NOT_FOUND, message="Job not found.", status_code=404)
    return job


@router.post("/jobs/retry-failed")
def retry_failed(
    project_id: str = Query("", description="Limit to one project (also accepted in a JSON body)"),
    body: dict[str, Any] | None = Body(None),  # noqa: B008
):
    count = manager().retry_failed(str(_body_value(body, "project_id", project_id) or ""))
    return {"retried": count}


@router.post("/jobs/purge")
def purge(
    project_id: str = Query("", description="Limit to one project (also accepted in a JSON body)"),
    body: dict[str, Any] | None = Body(None),  # noqa: B008
):
    removed = job_queue.purge_finished(str(_body_value(body, "project_id", project_id) or ""))
    return {"removed": removed}


@router.get("/workers")
def workers():
    return manager().status()


@router.post("/workers/start")
def start_workers(
    workers: int = Query(0, ge=0, le=8, description="0 = use the configured count (also accepted in a JSON body)"),
    body: dict[str, Any] | None = Body(None),  # noqa: B008
):
    try:
        count = int(_body_value(body, "workers", workers) or 0)
    except (TypeError, ValueError) as exc:
        raise ClipForgeError(code=ErrorCode.INVALID_INPUT, message="workers must be a whole number.", status_code=422) from exc
    return manager().start(workers=max(0, min(count, 8)) or None)


@router.post("/workers/stop")
def stop_workers():
    manager().stop()
    return manager().status()


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


def _usage_payload(project_id: str) -> dict[str, Any]:
    return {"type": "usage", "usage": process_snapshot(), "queue": job_queue.queue_state(project_id)["counts"]}


@router.get("/events")
async def events(
    project_id: str = Query(""),
    include_hardware: bool = Query(True, description="Send a hardware report first and live usage while idle"),
):
    """Server-Sent Events stream of pipeline progress, queue changes and errors.

    Every frame is ``data: {"type": ..., ...}``. While nothing happens the stream
    sends a ``usage`` frame (or a ``: ping`` comment when ``include_hardware`` is
    off) every 15 seconds so proxies keep the connection open.
    """

    async def generator():
        subscription = BUS.subscribe_async(project_id)
        try:
            yield "retry: 3000\n\n"  # EventSource reconnect delay
            if include_hardware:
                report, usage = await asyncio.gather(asyncio.to_thread(hardware), asyncio.to_thread(process_snapshot))
                yield _sse({"type": "hardware", "hardware": report, "usage": usage})
            last_usage = 0.0
            loop = asyncio.get_running_loop()
            while True:
                event = await subscription.next(timeout=HEARTBEAT_SECONDS)
                if event is not None:
                    yield _sse(event.to_payload())
                    continue
                if include_hardware and loop.time() - last_usage >= USAGE_INTERVAL_SECONDS:
                    last_usage = loop.time()
                    yield _sse(await asyncio.to_thread(_usage_payload, project_id))
                else:
                    yield ": ping\n\n"
        except asyncio.CancelledError:
            raise  # the client went away; nothing to report
        except Exception:  # noqa: BLE001 - a broken stream must not take the API down
            log.exception("event stream failed")
        finally:
            subscription.close()

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


__all__ = ["router"]
