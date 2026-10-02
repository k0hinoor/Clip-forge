"""Health, readiness, worker health, metrics and public config (TRD §28.5, §36)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from clipforge import __version__
from clipforge.api.deps import get_db, get_settings_dep
from clipforge.core.config import Settings
from clipforge.core.presets import ASPECT_RATIOS, FRAMING_MODES, caption_presets, plans_config
from clipforge.core.versioning import PIPELINE_VERSION
from clipforge.media.ffmpeg import ffmpeg_version
from clipforge.security.tokens import constant_time_equals
from clipforge.services import system

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__, "pipeline_version": PIPELINE_VERSION}


@router.get("/health/ready")
def ready(db: Session = Depends(get_db), settings: Settings = Depends(get_settings_dep)) -> JSONResponse:
    checks: dict[str, Any] = {}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = {"ok": True}
    except Exception as exc:  # pragma: no cover - depends on infra
        checks["database"] = {"ok": False, "error": type(exc).__name__}
    disk = system.disk_status(settings)
    checks["storage"] = {"ok": disk["free_bytes"] >= settings.MIN_FREE_DISK_BYTES, **disk}
    version = ffmpeg_version(settings.FFMPEG_PATH)
    checks["ffmpeg"] = {"ok": version is not None, "version": version}
    ok = all(c["ok"] for c in checks.values())
    return JSONResponse({"status": "ready" if ok else "degraded", "checks": checks}, status_code=200 if ok else 503)


@router.get("/health/worker")
def worker_health(db: Session = Depends(get_db), settings: Settings = Depends(get_settings_dep)) -> JSONResponse:
    workers = system.workers_status(db, settings)
    online = [w for w in workers if w.get("online")]
    counts = system.job_counts(db)
    body = {"status": "ok" if online else "offline", "workers": workers, "queued_jobs": counts.get("QUEUED", 0)}
    if not online:
        body["error"] = {"code": "WORKER_OFFLINE", "message": "No processing worker is online.", "retryable": True}
    return JSONResponse(body, status_code=200 if online else 503)


@router.get("/metrics", include_in_schema=False)
def metrics(request: Request, db: Session = Depends(get_db),
            settings: Settings = Depends(get_settings_dep)) -> PlainTextResponse:
    if settings.METRICS_TOKEN:
        supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not constant_time_equals(supplied, settings.METRICS_TOKEN):
            return PlainTextResponse("forbidden\n", status_code=403)
    data = system.collect_metrics(db, settings)
    return PlainTextResponse(system.prometheus_text(data, system.disk_status(settings)),
                             media_type="text/plain; version=0.0.4")


@router.get("/config")
def public_config(settings: Settings = Depends(get_settings_dep)) -> dict[str, Any]:
    """Options the frontend needs to build upload/settings forms."""
    presets = caption_presets(settings)
    return {
        "deployment_mode": settings.DEPLOYMENT_MODE,
        "auth_required": settings.auth_required,
        "registration_enabled": settings.ALLOW_REGISTRATION,
        "max_upload_bytes": settings.MAX_UPLOAD_BYTES,
        "max_video_seconds": settings.MAX_VIDEO_SECONDS,
        "upload_chunk_max_bytes": settings.UPLOAD_CHUNK_MAX_BYTES,
        "allowed_extensions": settings.ALLOWED_EXTENSIONS,
        "aspect_ratios": list(ASPECT_RATIOS),
        "framing_modes": list(FRAMING_MODES),
        "caption_presets": [{"id": k, "name": v.get("name", k)} for k, v in presets.items()],
        "target_platforms": ["tiktok", "youtube_shorts", "instagram_reels", "linkedin", "x", "generic"],
        "clip_seconds": {"min": settings.MIN_CLIP_SECONDS, "target_min": settings.TARGET_MIN_SECONDS,
                         "target_max": settings.TARGET_MAX_SECONDS, "max": settings.MAX_CLIP_SECONDS},
        "default_clips_per_job": settings.DEFAULT_CLIPS_PER_JOB,
        "plans": {k: {kk: vv for kk, vv in v.items() if kk != "internal"} for k, v in plans_config(settings).items()},
    }
