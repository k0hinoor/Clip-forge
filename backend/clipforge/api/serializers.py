"""ORM → API response conversion (keeps internal fields out of responses)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from clipforge.api.schemas import iso
from clipforge.core.config import Settings
from clipforge.core.states import RenderStatus
from clipforge.db.models import Asset, CandidateFeatures, Clip, Job, JobEvent, Render, User
from clipforge.security.tokens import sign_resource


def user_out(u: User) -> dict[str, Any]:
    return {"id": u.id, "email": u.email, "display_name": u.display_name, "role": u.role, "plan": u.plan,
            "created_at": iso(u.created_at)}


def asset_out(a: Asset, duplicate_of: str | None = None) -> dict[str, Any]:
    return {"id": a.id, "status": a.status, "filename": a.original_filename, "size_bytes": a.size_bytes,
            "bytes_received": a.bytes_received, "duration_seconds": a.duration_seconds, "width": a.width,
            "height": a.height, "fps": a.fps, "video_codec": a.video_codec, "audio_codec": a.audio_codec,
            "has_audio": a.has_audio, "checksum_sha256": a.checksum_sha256, "error_code": a.error_code,
            "duplicate_of": duplicate_of, "created_at": iso(a.created_at), "expires_at": iso(a.expires_at)}


def job_out(j: Job, session: Session | None = None) -> dict[str, Any]:
    error = None
    if j.error_code:
        error = {"code": j.error_code, "message": j.error_message, "retryable": bool(j.retryable),
                 "diagnostic_id": j.diagnostic_id}
    clip_count = None
    if session is not None:
        clip_count = int(session.scalar(select(func.count()).select_from(Clip).where(
            Clip.job_id == j.id, Clip.deleted_at.is_(None))) or 0)
    return {"id": j.id, "status": j.status, "progress": j.progress, "current_stage": j.current_stage,
            "upload_id": j.source_asset_id, "settings": j.settings or {}, "attempt": j.attempt,
            "max_attempts": j.max_attempts, "error": error, "cancel_requested": bool(j.cancel_requested),
            "source_seconds": j.source_seconds, "versions": j.versions, "clip_count": clip_count,
            "created_at": iso(j.created_at), "started_at": iso(j.started_at), "completed_at": iso(j.completed_at),
            "next_attempt_at": iso(j.next_attempt_at)}


def event_out(e: JobEvent) -> dict[str, Any]:
    return {"id": e.id, "event_type": e.event_type, "stage": e.stage, "status": e.status, "progress": e.progress,
            "timestamp": iso(e.timestamp), "duration_ms": e.duration_ms, "message": e.message,
            "error_code": e.error_code, "metadata": e.metadata_json}


def render_out(r: Render | None) -> dict[str, Any] | None:
    if r is None:
        return None
    return {"id": r.id, "status": r.status, "source": r.source, "profile": r.render_profile_name,
            "width": r.width, "height": r.height, "duration_seconds": r.duration_seconds, "size_bytes": r.size_bytes,
            "filename": r.filename, "error_code": r.error_code, "error_message": r.error_message,
            "created_at": iso(r.created_at), "completed_at": iso(r.completed_at)}


def signed_path(settings: Settings, clip_id: str, variant: str, render_id: str) -> str:
    expires, sig = sign_resource(settings.JWT_SECRET, "clip", clip_id, f"{variant}:{render_id}",
                                 settings.SIGNED_URL_TTL_SECONDS)
    base = settings.PUBLIC_BASE_URL.rstrip("/")
    return f"{base}{settings.API_PREFIX}/clips/{clip_id}/{variant}?r={render_id}&expires={expires}&sig={sig}"


def clip_out(c: Clip, session: Session, settings: Settings, *, with_features: bool = True) -> dict[str, Any]:
    current = session.get(Render, c.current_render_id) if c.current_render_id else None
    pending = session.scalar(select(Render).where(
        Render.clip_id == c.id, Render.status.in_([RenderStatus.QUEUED.value, RenderStatus.RENDERING.value]))
        .order_by(Render.created_at.desc()).limit(1))
    ready = current is not None and current.status == RenderStatus.READY.value and current.storage_key
    features = None
    if with_features and c.candidate_id:
        cf = session.scalar(select(CandidateFeatures).where(CandidateFeatures.candidate_id == c.candidate_id))
        features = cf.features if cf else None
    crop = c.crop or {}
    framing = {"mode": crop.get("mode"), "requested_mode": crop.get("requested_mode"),
               "face_ratio": crop.get("face_ratio")} if crop else None
    return {
        "id": c.id, "job_id": c.job_id, "rank": c.rank, "status": c.status, "start": c.start, "end": c.end,
        "duration": round(c.end - c.start, 3), "title": c.title, "hook": c.hook, "description": c.description,
        "keywords": c.keywords or [], "reasoning": c.reasoning, "transcript_excerpt": c.transcript_excerpt,
        "score": c.score, "reasons": c.reasons or [], "features": features, "aspect_ratio": c.aspect_ratio,
        "caption_preset": c.caption_preset, "captions_enabled": c.captions_enabled, "framing_mode": c.framing_mode,
        "framing": framing, "metadata_source": c.metadata_source, "render": render_out(current),
        "pending_render": render_out(pending),
        "download_url": signed_path(settings, c.id, "download", current.id) if ready else None,
        "thumbnail_url": signed_path(settings, c.id, "thumbnail", current.id)
        if ready and current.thumbnail_key else None,
        "subtitles_url": signed_path(settings, c.id, "subtitles", current.id)
        if ready and current.subtitles_key else None,
        "expires_at": iso(c.expires_at), "created_at": iso(c.created_at), "updated_at": iso(c.updated_at),
    }
