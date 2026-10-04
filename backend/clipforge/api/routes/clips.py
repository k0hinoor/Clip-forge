"""Clip routes: detail, edit, captions, render, preview, regenerate, export."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from ...errors import ClipForgeError, ErrorCode
from ...jobs import queue as job_queue
from ...jobs.manager import manager
from ...logging_setup import get_logger
from ...services import clips as clip_service
from ...services import events
from ..media import require_file, stream_file
from ..schemas import CaptionPreviewRequest, ClipUpdate, RenderRequest

router = APIRouter(prefix="/clips", tags=["clips"])
captions_router = APIRouter(tags=["clips"])
log = get_logger(__name__)


class ClipRenderRequest(BaseModel):
    export: bool = Field(False, description="Also copy the finished file into the export folder")
    overrides: dict[str, Any] = Field(default_factory=dict, description="One-off settings for this render only")


class ClipPreviewRequest(BaseModel):
    priority: int = Field(2, ge=0, le=10)


class RegenerateRequest(BaseModel):
    window: float = Field(120.0, ge=20, le=900, description="Seconds of transcript around the clip to search")
    use_llm: bool = True


@router.post("/render-all")
def render_all(payload: RenderRequest):
    """Queue every requested clip for rendering."""
    if not payload.clip_ids:
        raise ClipForgeError(
            code=ErrorCode.INVALID_INPUT,
            message="No clips were selected.",
            hint="Pass clip_ids, or use POST /api/projects/{id}/render-all.",
            status_code=422,
        )
    project_ids = {clip_service.get_clip(clip_id)["project_id"] for clip_id in dict.fromkeys(payload.clip_ids)}
    if len(project_ids) > 1:
        raise ClipForgeError(
            code=ErrorCode.INVALID_INPUT,
            message="Those clips belong to different projects.",
            hint="Render one project's clips at a time.",
            status_code=422,
        )
    project_id = project_ids.pop()
    result = manager().enqueue_all(project_id, list(dict.fromkeys(payload.clip_ids)), export=payload.export, priority=payload.priority)
    return {**result, "queue": _queue(project_id)}


@router.get("/{clip_id}")
def get_clip(clip_id: str):
    return clip_service.clip_detail(clip_id)


@router.patch("/{clip_id}")
def update_clip(clip_id: str, payload: ClipUpdate):
    patch = payload.model_dump(exclude_none=True)
    if not patch:
        return clip_service.clip_detail(clip_id)
    updated = clip_service.update_clip(clip_id, patch)
    events.publish("clip.updated", {"clip_id": clip_id, "clip": updated}, project_id=updated["project_id"])
    return updated


@router.delete("/{clip_id}")
def delete_clip(clip_id: str, remove_files: bool = Query(True)):
    result = clip_service.delete_clip(clip_id, remove_files=remove_files)
    events.publish("clip.deleted", {"clip_id": clip_id}, project_id=result.get("project_id", ""))
    return result


@router.post("/{clip_id}/render")
def render_clip(clip_id: str, payload: ClipRenderRequest | None = Body(default=None)):  # noqa: B008
    """Queue the final render of one clip."""
    body = payload or ClipRenderRequest()
    clip = clip_service.get_clip(clip_id)
    job = manager().enqueue_render(clip_id, export=body.export, overrides=body.overrides or None)
    return {"job": job, "clip_id": clip_id, "queue": _queue(clip["project_id"])}


@router.post("/{clip_id}/preview")
def preview_clip(clip_id: str, payload: ClipPreviewRequest | None = Body(default=None)):  # noqa: B008
    """Render a fast low-resolution preview (separate from the final render)."""
    body = payload or ClipPreviewRequest()
    clip = clip_service.get_clip(clip_id)
    job = manager().enqueue_preview(clip_id, priority=body.priority)
    return {"job": job, "clip_id": clip_id, "queue": _queue(clip["project_id"])}


@router.post("/{clip_id}/cancel")
def cancel_clip_render(clip_id: str):
    """Cancel this clip's queued or running renders and previews."""
    clip_service.get_clip(clip_id)  # 404 for unknown clips
    jobs = job_queue.active_jobs(clip_id=clip_id)
    for job in jobs:
        manager().cancel(job["id"])
    return {"cancelled": len(jobs), "clip_id": clip_id}


@router.post("/{clip_id}/regenerate")
def regenerate_clip(clip_id: str, payload: RegenerateRequest | None = Body(default=None)):  # noqa: B008
    """Re-examine the neighbourhood and keep the strongest moment in it."""
    body = payload or RegenerateRequest()
    result = clip_service.regenerate_clip(clip_id, window=body.window, use_llm=body.use_llm)
    events.publish("clip.regenerated", {"clip_id": clip_id, "clip": result}, project_id=result["project_id"])
    return result


@router.post("/{clip_id}/duplicate", status_code=201)
def duplicate_clip(clip_id: str):
    """Copy a clip (with its edits) as a new clip that can be changed independently."""
    clone = clip_service.duplicate_clip(clip_id)
    events.publish("clip.created", {"clip_id": clone["id"], "source": clip_id}, project_id=clone["project_id"])
    return clone


@router.get("/{clip_id}/captions")
def get_captions(clip_id: str, preset: str = Query(""), theme: str = Query("", description="JSON object of caption theme overrides")):
    parsed: dict[str, Any] | None = None
    if theme:
        try:
            parsed = json.loads(theme)
        except ValueError as exc:
            raise ClipForgeError(code=ErrorCode.INVALID_INPUT, message="theme must be a JSON object.", status_code=422) from exc
        if not isinstance(parsed, dict):
            raise ClipForgeError(code=ErrorCode.INVALID_INPUT, message="theme must be a JSON object.", status_code=422)
    return clip_service.caption_preview(clip_id, preset=preset or None, theme=parsed)


@router.post("/{clip_id}/captions/preview")
def preview_captions(clip_id: str, payload: CaptionPreviewRequest):
    return clip_service.caption_preview(clip_id, preset=payload.preset, theme=payload.theme)


@captions_router.post("/captions/preview")
def preview_caption_style(payload: CaptionPreviewRequest):
    """Caption lines in a style - for ``clip_id``'s words, or sample text when it is empty."""
    return clip_service.caption_preview(payload.clip_id, preset=payload.preset, theme=payload.theme)


@router.get("/{clip_id}/captions/srt", response_class=PlainTextResponse)
def download_srt(clip_id: str):
    path = clip_service.write_clip_srt(clip_id)
    clip = clip_service.get_clip(clip_id)
    filename = f"{_safe_filename(clip.get('title') or clip_id)}.srt"
    return PlainTextResponse(
        path.read_text(encoding="utf-8-sig"),
        media_type="application/x-subrip; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{clip_id}/command")
def get_command(clip_id: str, quality: str = Query("final", pattern="^(final|preview)$")):
    """The exact ffmpeg command this clip's render would run."""
    return {"clip_id": clip_id, "quality": quality, "command": clip_service.render_spec_preview(clip_id, quality=quality)}


@router.get("/{clip_id}/preview")
def stream_preview(clip_id: str, request: Request):
    """The low-res preview - or the final render when there is no preview."""
    target = _existing(_path_of(clip_id, "preview_path")) or _existing(_path_of(clip_id, "file_path"))
    if not target:
        raise ClipForgeError(
            code=ErrorCode.NOT_FOUND,
            message="This clip has not been rendered yet.",
            hint="Press Render (or Preview) to create the video.",
            status_code=404,
        )
    return stream_file(require_file(target), request, cache_seconds=600)


@router.get("/{clip_id}/file")
def stream_file_route(clip_id: str, request: Request, download: bool = Query(False)):
    target = _path_of(clip_id, "file_path")
    if not target:
        raise ClipForgeError(
            code=ErrorCode.NOT_FOUND,
            message="This clip has not been rendered yet.",
            hint="Press Render to create the final video.",
            status_code=404,
        )
    path = require_file(target)
    return stream_file(path, request, download_name=path.name if download else None, cache_seconds=3600)


@router.get("/{clip_id}/thumbnail")
def stream_thumbnail(clip_id: str, request: Request):
    target = _path_of(clip_id, "thumb_path")
    if not target:
        raise ClipForgeError(code=ErrorCode.NOT_FOUND, message="No thumbnail for this clip yet.", status_code=404)
    return stream_file(require_file(target), request, cache_seconds=86400)


@router.get("/{clip_id}/subtitles")
def download_subtitles(clip_id: str):
    target = _path_of(clip_id, "subtitle_path")
    if not target:
        raise ClipForgeError(code=ErrorCode.NOT_FOUND, message="No subtitle file for this clip yet.", status_code=404)
    path = require_file(target)
    return PlainTextResponse(path.read_text(encoding="utf-8", errors="replace"), media_type="text/plain; charset=utf-8")


@router.get("/{clip_id}/assets")
def clip_asset_options(clip_id: str):
    """Assets this clip can use (enabled library items per kind)."""
    clip_service.get_clip(clip_id)  # 404 for unknown clips
    return clip_service.clip_assets_options()


def _existing(path_text: str) -> str:
    from pathlib import Path

    return path_text if path_text and Path(path_text).is_file() else ""


def _safe_filename(text: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in " -_" else "_" for char in text).strip()
    return (cleaned or "captions")[:80]


def _path_of(clip_id: str, field: str) -> str:
    from ...db import Clip, session_scope

    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise ClipForgeError(code=ErrorCode.NOT_FOUND, message="Clip not found.", status_code=404)
        return str(getattr(clip, field) or "")


def _queue(project_id: str) -> dict:
    from ...jobs import queue as job_queue

    return job_queue.queue_state(project_id)


__all__ = ["captions_router", "router"]
