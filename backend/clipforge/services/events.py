"""Job event recording (TRD §29). Never store stack traces in events."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from clipforge.db.models import Job, JobEvent


def record_job_event(
    session: Session,
    job: Job,
    event_type: str,
    *,
    message: str | None = None,
    stage: str | None = None,
    duration_ms: int | None = None,
    error_code: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> JobEvent:
    event = JobEvent(
        job_id=job.id,
        event_type=event_type,
        stage=stage or job.current_stage,
        status=job.status,
        progress=job.progress,
        duration_ms=duration_ms,
        message=(message or "")[:500] or None,
        error_code=error_code,
        metadata_json=metadata or None,
    )
    session.add(event)
    return event
