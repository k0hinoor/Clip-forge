"""Clip routes (TRD §28.4): review, manual correction, re-render, downloads."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse, RedirectResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from clipforge.api.deps import (
    client_ip,
    current_user,
    get_auth,
    get_db,
    get_queue_dep,
    get_settings_dep,
    get_storage_dep,
)
from clipforge.api.schemas import ClipOut, ClipPatchRequest, ClipPatchResponse, RenderOut
from clipforge.api.serializers import clip_out, render_out
from clipforge.core.config import Settings
from clipforge.core.errors import AppError, ErrorCode, not_found
from clipforge.core.states import JobStatus, RenderStatus
from clipforge.db.models import Clip, Job, Render, User
from clipforge.queue.base import QueueProvider
from clipforge.security.tokens import verify_resource_signature
from clipforge.services.analytics import track
from clipforge.services.audit import audit
from clipforge.services.clips import ClipService
from clipforge.storage import StorageProvider

router = APIRouter(prefix="/clips", tags=["clips"])


def _svc(db: Session, settings: Settings, storage: StorageProvider, queue: QueueProvider | None = None) -> ClipService:
    return ClipService(db, settings, storage, queue)


@router.get("/{clip_id}", response_model=ClipOut)
def get_clip(clip_id: str, user: User = Depends(current_user), db: Session = Depends(get_db),
             settings: Settings = Depends(get_settings_dep),
             storage: StorageProvider = Depends(get_storage_dep)) -> dict[str, Any]:
    return clip_out(_svc(db, settings, storage).get(user, clip_id), db, settings)


@router.patch("/{clip_id}", response_model=ClipPatchResponse)
def patch_clip(clip_id: str, body: ClipPatchRequest, user: User = Depends(current_user),
               db: Session = Depends(get_db), settings: Settings = Depends(get_settings_dep),
               storage: StorageProvider = Depends(get_storage_dep),
               queue: QueueProvider = Depends(get_queue_dep)) -> dict[str, Any]:
    svc = _svc(db, settings, storage, queue)
    clip = svc.get(user, clip_id)
    patch = body.model_dump(exclude_unset=True, exclude={"render"})
    clip, render_required = svc.update(user, clip, patch)
    queued = False
    job = db.get(Job, clip.job_id)
    if render_required and body.render and job is not None and job.status == JobStatus.COMPLETED.value:
        svc.request_render(user, clip)
        queued = True
    db.commit()
    return {"clip": clip_out(clip, db, settings), "render_required": render_required, "render_queued": queued}


@router.post("/{clip_id}/render", response_model=RenderOut, status_code=202)
def render_clip(clip_id: str, user: User = Depends(current_user), db: Session = Depends(get_db),
                settings: Settings = Depends(get_settings_dep), storage: StorageProvider = Depends(get_storage_dep),
                queue: QueueProvider = Depends(get_queue_dep)) -> dict[str, Any]:
    svc = _svc(db, settings, storage, queue)
    render = svc.request_render(user, svc.get(user, clip_id))
    db.commit()
    return render_out(render)


@router.get("/{clip_id}/renders", response_model=list[RenderOut])
def list_renders(clip_id: str, user: User = Depends(current_user), db: Session = Depends(get_db),
                 settings: Settings = Depends(get_settings_dep),
                 storage: StorageProvider = Depends(get_storage_dep)) -> list[dict[str, Any]]:
    clip = _svc(db, settings, storage).get(user, clip_id)
    rows = db.scalars(select(Render).where(Render.clip_id == clip.id).order_by(Render.created_at.desc())).all()
    return [render_out(r) for r in rows]


@router.delete("/{clip_id}", status_code=204)
def delete_clip(clip_id: str, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db),
                settings: Settings = Depends(get_settings_dep),
                storage: StorageProvider = Depends(get_storage_dep)) -> None:
    svc = _svc(db, settings, storage)
    clip = svc.get(user, clip_id)
    svc.delete(user, clip)
    audit(db, "clip.deleted", actor_user_id=user.id, target_type="clip", target_id=clip.id,
          ip_address=client_ip(request))
    db.commit()


# ---------------------------------------------------------------- files
VARIANTS = {"download": ("storage_key", "video/mp4"), "thumbnail": ("thumbnail_key", "image/jpeg"),
            "subtitles": ("subtitles_key", "text/x-ssa")}


def _resolve_download_user(request: Request, db: Session, clip_id: str, variant: str, render_id: str | None,
                           expires: int | None, sig: str | None) -> tuple[Clip, Render]:
    settings: Settings = request.app.state.settings
    clip = db.scalar(select(Clip).where(Clip.id == clip_id, Clip.deleted_at.is_(None)))
    if clip is None:
        raise not_found("Clip")
    if sig and expires and render_id:
        if not verify_resource_signature(settings.JWT_SECRET, "clip", clip_id, f"{variant}:{render_id}",
                                         int(expires), sig):
            raise AppError(ErrorCode.FORBIDDEN, "This link is invalid or has expired.")
    else:
        auth = get_auth(request, db)
        if clip.user_id != auth.user.id:
            raise not_found("Clip")
    rid = render_id or clip.current_render_id
    render = db.get(Render, rid) if rid else None
    if render is None or render.clip_id != clip.id or render.status != RenderStatus.READY.value:
        raise not_found("Render")
    return clip, render


def _serve(variant: str, clip_id: str, request: Request, db: Session, r: str | None, expires: int | None,
           sig: str | None, storage: StorageProvider) -> Response:
    clip, render = _resolve_download_user(request, db, clip_id, variant, r, expires, sig)
    attr, media_type = VARIANTS[variant]
    key = getattr(render, attr)
    if not key or not storage.exists(key):
        raise AppError(ErrorCode.NOT_FOUND, "This file has expired or was deleted.")
    filename = render.filename if variant == "download" else None
    if variant == "subtitles" and render.filename:
        filename = render.filename.rsplit(".", 1)[0] + ".ass"
    if variant == "download":
        track(db, "clip_downloaded", user_id=clip.user_id, job_id=clip.job_id, clip_id=clip.id)
        db.commit()
    local = storage.local_path(key)
    if local is None:
        url = storage.signed_url(key, 300, filename=filename)
        if url is None:
            raise AppError(ErrorCode.NOT_FOUND)
        return RedirectResponse(url, status_code=307)
    headers = {"Cache-Control": "private, max-age=300"}
    if variant == "download":
        return FileResponse(local, media_type=media_type, filename=filename, headers=headers)
    return FileResponse(local, media_type=media_type, headers=headers,
                        content_disposition_type="inline", filename=filename)


@router.get("/{clip_id}/download")
def download_clip(clip_id: str, request: Request, r: str | None = Query(default=None, max_length=36),
                  expires: int | None = None, sig: str | None = Query(default=None, max_length=128),
                  db: Session = Depends(get_db), storage: StorageProvider = Depends(get_storage_dep)) -> Response:
    return _serve("download", clip_id, request, db, r, expires, sig, storage)


@router.get("/{clip_id}/thumbnail")
def clip_thumbnail(clip_id: str, request: Request, r: str | None = Query(default=None, max_length=36),
                   expires: int | None = None, sig: str | None = Query(default=None, max_length=128),
                   db: Session = Depends(get_db), storage: StorageProvider = Depends(get_storage_dep)) -> Response:
    return _serve("thumbnail", clip_id, request, db, r, expires, sig, storage)


@router.get("/{clip_id}/subtitles")
def clip_subtitles(clip_id: str, request: Request, r: str | None = Query(default=None, max_length=36),
                   expires: int | None = None, sig: str | None = Query(default=None, max_length=128),
                   db: Session = Depends(get_db), storage: StorageProvider = Depends(get_storage_dep)) -> Response:
    return _serve("subtitles", clip_id, request, db, r, expires, sig, storage)
