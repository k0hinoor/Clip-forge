"""Clip routes: detail, edit, captions, render, preview, regenerate, export."""

from __future__ import annotations

from fastapi import APIRouter, Body, Query, Request
from fastapi.responses import PlainTextResponse

from ...errors import ClipForgeError, ErrorCode
from ...jobs.manager import manager
from ...logging_setup import get_logger
from ...services import clips as clip_service
from ...services import events
from ..media import require_file, stream_file
from ..schemas import CaptionPreviewRequest, ClipUpdate, RenderRequest

router = APIRouter(prefix="/clips", tags=["clips"])
log = get_logger(__name__)


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
    project_id = ""
    with_render = []
    for clip_id in payload.clip_ids:
        clip = clip_service.get_clip(clip_id)
        project_id = project_id or clip["project_id"]
        with_render.append(clip_id)
    result = manager().enqueue_all(project_id, with_render, export=payload.export)
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
def render_clip(clip_id: str, payload: dict | None = Body(default=None)):  # noqa: B008
    body = payload or {}
    job = manager().enqueue_render(
        clip_id,
        export=bool(body.get("export", False)),
        overrides=body.get("overrides") or None,
    )
    clip = clip_service.get_clip(clip_id)
    return {"job": job, "clip_id": clip_id, "queue": _queue(clip["project_id"])}


@router.post("/{clip_id}/preview")
def preview_clip(clip_id: str, payload: dict | None = Body(default=None)):  # noqa: B008
    """Render a fast low-resolution preview (separate from the final render)."""
    body = payload or {}
    job = manager().enqueue_preview(clip_id, priority=int(body.get("priority", 2)))
    clip = clip_service.get_clip(clip_id)
    return {"job": job, "clip_id": clip_id, "queue": _queue(clip["project_id"])}


@router.post("/{clip_id}/cancel")
def cancel_clip_render(clip_id: str):
    clip = clip_service.get_clip(clip_id)
    from ...jobs import queue as job_queue

    jobs = job_queue.list_jobs(project_id=clip["project_id"], statuses=("queued", "running"), limit=100)
    cancelled = 0
    for job in jobs:
        if job.get("clip_id") == clip_id:
            manager().cancel(job["id"])
            cancelled += 1
    return {"cancelled": cancelled, "clip_id": clip_id}


@router.post("/{clip_id}/regenerate")
def regenerate_clip(clip_id: str, payload: dict | None = Body(default=None)):  # noqa: B008
    """Re-examine the neighbourhood and keep the strongest moment in it."""
    body = payload or {}
    result = clip_service.regenerate_clip(
        clip_id,
        window=float(body.get("window", 120.0)),
        use_llm=bool(body.get("use_llm", True)),
    )
    events.publish("clip.regenerated", {"clip_id": clip_id, "clip": result}, project_id=result["project_id"])
    return result


@router.post("/{clip_id}/duplicate")
def duplicate_clip(clip_id: str):
    from ...db import Clip, session_scope

    detail = clip_service.clip_detail(clip_id)
    with session_scope() as session:
        original = session.get(Clip, clip_id)
        if original is None:
            raise ClipForgeError(code=ErrorCode.NOT_FOUND, message="Clip not found.", status_code=404)
        clone = Clip(
            project_id=original.project_id,
            candidate_id=original.candidate_id,
            index=original.index,
            title=f"{original.title} (copy)"[:300],
            hook=original.hook,
            summary=original.summary,
            category=original.category,
            score=original.score,
            why_json=original.why_json,
            factors_json=original.factors_json,
            start=original.start,
            end=original.end,
            duration=original.duration,
            words_json=original.words_json,
            segments_json=original.segments_json,
            trim_json=original.trim_json,
            layout_json=original.layout_json,
            status="pending",
        )
        session.add(clone)
        session.flush()
        clone_id = clone.id
    return clip_service.clip_detail(clone_id)


@router.get("/{clip_id}/captions")
def get_captions(clip_id: str, preset: str = Query(""), theme: str = Query("")):
    import json

    parsed = json.loads(theme) if theme else None
    return clip_service.caption_preview(clip_id, preset=preset or None, theme=parsed)


@router.post("/{clip_id}/captions/preview")
def preview_captions(clip_id: str, payload: CaptionPreviewRequest):
    return clip_service.caption_preview(clip_id, preset=payload.preset, theme=payload.theme)


@router.get("/{clip_id}/captions/srt", response_class=PlainTextResponse)
def download_srt(clip_id: str):
    path = clip_service.write_clip_srt(clip_id)
    return PlainTextResponse(path.read_text(encoding="utf-8-sig"), media_type="application/x-subrip")


@router.get("/{clip_id}/command")
def get_command(clip_id: str, quality: str = Query("final", pattern="^(final|preview)$")):
    """The exact ffmpeg command this clip's render would run."""
    return {"clip_id": clip_id, "quality": quality, "command": clip_service.render_spec_preview(clip_id, quality=quality)}


@router.get("/{clip_id}/preview")
def stream_preview(clip_id: str, request: Request):
    detail = clip_service.get_clip(clip_id)
    path = detail.get("has_preview") and None
    with_path = clip_service.get_clip(clip_id)
    target = _path_of(clip_id, "preview_path") or (with_path.get("render_url") and _path_of(clip_id, "file_path"))
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
    from pathlib import Path

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
    return clip_service.clip_assets_options()


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


__all__ = ["router"]
