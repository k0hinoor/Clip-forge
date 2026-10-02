"""Product analytics events (TRD §56).

Only aggregated, non-sensitive metadata is recorded — never transcript text,
filenames, media or credentials. Properties are whitelisted.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from clipforge.db.models import AnalyticsEvent

ALLOWED_EVENTS = {
    "upload_completed", "upload_rejected", "job_created", "job_completed", "job_failed",
    "job_cancelled", "job_retried", "clip_downloaded", "clip_rerendered", "clip_edited",
    "clip_deleted", "job_deleted", "user_registered",
}
ALLOWED_PROPERTIES = {
    "duration", "aspect_ratio", "caption_preset", "clips", "source_seconds", "error_code",
    "stage", "attempt", "size_bytes", "framing_mode", "fields", "plan", "processing_seconds",
}


def track(
    session: Session,
    event: str,
    *,
    user_id: str | None = None,
    job_id: str | None = None,
    clip_id: str | None = None,
    properties: dict[str, Any] | None = None,
) -> None:
    if event not in ALLOWED_EVENTS:
        raise ValueError(f"Unknown analytics event: {event}")
    props = {k: v for k, v in (properties or {}).items() if k in ALLOWED_PROPERTIES}
    session.add(AnalyticsEvent(event=event, user_id=user_id, job_id=job_id, clip_id=clip_id, properties=props))
