"""Project routes: create from a URL (or upload), analyse, inspect, delete."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, File, Form, Query, Request, UploadFile

from ...config import Env, get_settings
from ...errors import ClipForgeError, ErrorCode
from ...jobs import queue as job_queue
from ...jobs.manager import manager
from ...logging_setup import get_logger
from ...services import events
from ...services import projects as project_service
from ...system import open_in_file_manager
from ..media import stream_file
from ..schemas import AnalyzeRequest, ClipOptions, ProjectAnalyzeRequest, ProjectCreate, RenderAllRequest, local_media_path

router = APIRouter(prefix="/projects", tags=["projects"])
# Short aliases documented in the README (same handlers as the resource routes).
shortcuts = APIRouter(tags=["projects"])
log = get_logger(__name__)


def _create_from_link(url: str, *, title: str = "", options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a project from a pasted link - or, on the desktop app, a local file path."""
    local = local_media_path(url)
    if local is not None:
        from ...media.download import ALLOWED_UPLOAD_EXTENSIONS

        if local.suffix.lower() not in ALLOWED_UPLOAD_EXTENSIONS:
            raise ClipForgeError(
                code=ErrorCode.UNSUPPORTED_FORMAT,
                message=f"{local.suffix or 'That file type'} cannot be analysed.",
                hint="Supported: " + ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS)),
                status_code=422,
            )
        return project_service.create_project(title=title or local.stem, source_type="upload", options=options, source_path=local)

    from ...media.download import classify_url

    source = classify_url(url)
    return project_service.create_project(
        url=source.url,
        title=title,
        source_type="youtube" if source.is_youtube else "url",
        options=options,
    )


@router.post("", status_code=201)
def create_project(payload: ProjectCreate):
    """Create a project from any video link and (by default) start analysing it."""
    if not payload.url:
        raise ClipForgeError(
            code=ErrorCode.INVALID_URL,
            message="A video link is required.",
            hint="Paste a YouTube, Vimeo or direct .mp4 link - or upload a file with POST /api/projects/upload.",
            status_code=422,
        )
    project = _create_from_link(payload.url, title=payload.title, options=payload.options.to_settings_patch())
    job = manager().enqueue_analysis(project["id"], priority=payload.priority) if payload.analyze else None
    return {"project": project, "job": job}


@router.post("/upload", status_code=201)
async def upload_project(
    file: UploadFile = File(...),
    title: str = Form(""),
    options: str = Form(""),
    analyze: str = Form("true"),
):
    """Analyse a local file instead of a link (multipart upload)."""
    from ...media.download import ALLOWED_UPLOAD_EXTENSIONS

    settings = get_settings()
    if not settings.uploads_enabled:
        await file.close()
        raise ClipForgeError(
            code=ErrorCode.INVALID_INPUT,
            message="Uploads are disabled in Settings -> Downloads & uploads.",
            status_code=403,
        )

    parsed_options = ClipOptions()
    if options:
        try:
            parsed_options = ClipOptions.model_validate(json.loads(options))
        except (ValueError, TypeError) as exc:
            await file.close()
            raise ClipForgeError(code=ErrorCode.INVALID_INPUT, message=f"Invalid options: {exc}", status_code=422) from exc

    filename = Path(file.filename or "upload.mp4").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        await file.close()
        raise ClipForgeError(
            code=ErrorCode.UNSUPPORTED_FORMAT,
            message=f"{suffix or 'That file type'} cannot be analysed.",
            hint="Supported: " + ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS)),
            status_code=422,
        )

    limit = int(settings.max_upload_gb * 1024**3)
    cache = Env.DATA_DIR / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    written = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(cache), prefix="upload_") as handle:
        temp_path = Path(handle.name)
        try:
            while chunk := await file.read(1024 * 1024 * 4):
                written += len(chunk)
                if written > limit:
                    raise ClipForgeError(
                        code=ErrorCode.UPLOAD_TOO_LARGE,
                        message=f"That file is larger than the {settings.max_upload_gb:g} GB upload limit.",
                        hint="Raise the limit in Settings -> Downloads & uploads, or upload a smaller file.",
                        status_code=413,
                    )
                handle.write(chunk)
        except BaseException:
            handle.close()
            temp_path.unlink(missing_ok=True)
            await file.close()
            raise

    # Give the stored file its real name (the temp name is meaningless).
    named = temp_path.with_name(f"{temp_path.stem}_{filename}")
    try:
        temp_path.rename(named)
        project = project_service.create_project(
            title=title or Path(filename).stem,
            source_type="upload",
            options=parsed_options.to_settings_patch(),
            source_path=named,
            move_source=True,
        )
    finally:
        temp_path.unlink(missing_ok=True)
        named.unlink(missing_ok=True)
        await file.close()

    job = None
    if analyze.strip().lower() not in {"false", "0", "no", "off"}:
        job = manager().enqueue_analysis(project["id"])
    return {"project": project, "job": job}


@router.get("")
def list_projects(search: str = Query(""), limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
    return project_service.list_projects(search=search, limit=limit, offset=offset)


@router.get("/{project_id}")
def get_project(project_id: str):
    return project_service.project_detail(project_id)


@router.get("/{project_id}/status")
def get_status(project_id: str):
    return project_service.project_status(project_id)


@router.get("/{project_id}/transcript")
def get_transcript(
    project_id: str,
    with_words: bool = Query(True),
    start: float | None = Query(None),
    end: float | None = Query(None),
    limit: int = Query(5000, ge=1, le=20000),
):
    return project_service.project_transcript(project_id, with_words=with_words, start=start, end=end, limit=limit)


@router.post("/{project_id}/transcript")
async def upload_transcript(project_id: str, file: UploadFile = File(...)):
    """Attach an authoritative transcript cue file for this project."""
    from ...media.download import TRANSCRIPT_EXTENSIONS, parse_transcript_segments
    from ...pipeline.context import ProjectPaths

    detail = project_service.project_detail(project_id)
    active = job_queue.active_jobs(project_id=project_id, kinds=("analyze",))
    if active:
        await file.close()
        raise ClipForgeError(
            code=ErrorCode.INVALID_INPUT,
            message="A transcript cannot be replaced while analysis is running.",
            hint="Cancel or wait for the current analysis, then attach the new transcript.",
            status_code=409,
        )

    raw_filename = (file.filename or "transcript.srt").replace("\\", "/")
    filename = Path(raw_filename).name or "transcript.srt"
    suffix = Path(filename).suffix.lower()
    if suffix not in TRANSCRIPT_EXTENSIONS:
        await file.close()
        raise ClipForgeError(
            code=ErrorCode.INVALID_INPUT,
            message=f"'{suffix or 'that file'}' is not a supported transcript format.",
            hint="Supported formats: .srt, .vtt, .json3, and timestamped .txt.",
            status_code=422,
        )

    paths = ProjectPaths.for_snapshot(detail).ensure()
    size = 0
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=".incoming_", dir=paths.transcript) as handle:
            temp_path = Path(handle.name)
            while chunk := await file.read(1024 * 512):
                size += len(chunk)
                if size > 40 * 1024 * 1024:
                    raise ClipForgeError(
                        code=ErrorCode.UPLOAD_TOO_LARGE,
                        message="That transcript is larger than 40 MB.",
                        status_code=413,
                    )
                handle.write(chunk)
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    try:
        cues = parse_transcript_segments(temp_path)
        if not cues:
            raise ClipForgeError(
                code=ErrorCode.INVALID_INPUT,
                message="That file contained no valid timestamped cues.",
                hint="Check its SRT/VTT timestamps or the timestamped-text format, then attach it again.",
                status_code=422,
            )

        upload_dir = paths.transcript / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_stem = re.sub(r"[^\w.-]+", "_", Path(filename).stem, flags=re.UNICODE).strip("._")[:100] or "transcript"
        archived = upload_dir / f"{safe_stem}{suffix}"
        shutil.copy2(temp_path, archived)
        canonical = paths.transcript / f"provided{suffix}"
        for old in paths.transcript.glob("provided.*"):
            if old.is_file():
                old.unlink(missing_ok=True)
        shutil.copy2(temp_path, canonical)
        result = project_service.attach_transcript(project_id, filename=filename, suffix=suffix, cues=cues)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    cue_count = len(cues)
    word_count = sum(len(str(cue["text"]).split()) for cue in cues)
    covers = max(float(cue["end"]) for cue in cues)
    return {
        "project_id": project_id,
        "file": filename,
        "source": result["transcript_source"],
        "transcript_filename": result["transcript_filename"],
        "timing_granularity": "cue",
        "segment_count": cue_count,
        "word_count": word_count,
        "covers": round(covers, 3),
        "message": f"Transcript attached: {filename} ({cue_count} cues). It will be used as-is; Whisper will not run unless you explicitly choose Use Whisper instead.",
    }


@router.post("/{project_id}/transcript/use-whisper")
def use_whisper_for_transcript(project_id: str):
    """Explicitly switch the next analysis to Whisper instead of the uploaded transcript."""
    project_service.project_detail(project_id)
    if job_queue.active_jobs(project_id=project_id, kinds=("analyze",)):
        raise ClipForgeError(
            code=ErrorCode.INVALID_INPUT,
            message="Analysis is already running.",
            hint="Wait for it to finish before switching transcript sources.",
            status_code=409,
        )
    project_service.update_project(project_id, transcript_preference="whisper")
    job = manager().enqueue_analysis(project_id, force=True)
    return {
        "project_id": project_id,
        "transcript_preference": "whisper",
        "job": job,
        "message": "Whisper was explicitly selected. The uploaded file remains preserved; the displayed transcript source will change only after Whisper succeeds.",
    }


@router.get("/{project_id}/candidates")
def get_candidates(project_id: str, status: str = Query(""), limit: int = Query(300, ge=1, le=1000)):
    return project_service.project_candidates(project_id, status=status, limit=limit)


@router.get("/{project_id}/clips")
def get_clips(
    project_id: str,
    sort: str = Query("score", pattern="^(score|duration|duration_asc|category|chronological|title)$"),
    category: str = Query(""),
    min_score: float = Query(0, ge=0, le=100),
    status: str = Query(""),
):
    return project_service.project_clips(project_id, sort=sort, category=category, min_score=min_score, status=status)


@router.post("/analyze")
@shortcuts.post("/analyze", summary="Analyze (alias of POST /api/projects/analyze)")
def analyze(payload: AnalyzeRequest):
    """Start (or restart) the analysis for ``project_id`` - or create a project from ``url`` and analyse it."""
    project_id = payload.project_id
    if project_id:
        project_service.project_detail(project_id)  # 404 for unknown ids
        if payload.options is not None:
            project_service.merge_project_options(project_id, payload.options.to_settings_patch())
    elif payload.url:
        project = _create_from_link(payload.url, title=payload.title, options=(payload.options or ClipOptions()).to_settings_patch())
        project_id = project["id"]
    else:
        raise ClipForgeError(
            code=ErrorCode.INVALID_INPUT,
            message="Provide a project_id or a url to analyse.",
            status_code=422,
        )
    job = manager().enqueue_analysis(project_id, priority=payload.priority, force=payload.force)
    return {"project_id": project_id, "job": job, "status": project_service.project_status(project_id)}


@router.post("/{project_id}/analyze")
def analyze_project(project_id: str, payload: ProjectAnalyzeRequest | None = Body(default=None)):  # noqa: B008
    """Re-run the analysis, optionally with new clip options for this project."""
    body = payload or ProjectAnalyzeRequest()
    project_service.project_detail(project_id)  # 404 for unknown ids
    if body.options is not None:
        project_service.merge_project_options(project_id, body.options.to_settings_patch())
    job = manager().enqueue_analysis(project_id, priority=body.priority, force=body.force)
    return {"project_id": project_id, "job": job}


@router.post("/{project_id}/cancel")
def cancel_project(project_id: str):
    cancelled = manager().cancel_project(project_id)
    events.publish("project.cancelled", {"project_id": project_id, "jobs": cancelled}, project_id=project_id)
    return {"cancelled_jobs": cancelled}


@router.post("/{project_id}/render-all")
def render_all(project_id: str, payload: RenderAllRequest | None = Body(default=None)):  # noqa: B008
    """Queue renders for a project's clips.

    Without ``clip_ids`` only clips that still need a render are queued
    (rendered, queued and rendering clips are skipped unless ``force``).
    """
    body = payload or RenderAllRequest()
    clips = project_service.project_clips(project_id, sort="score")["clips"]
    if not clips:
        raise ClipForgeError(
            code=ErrorCode.NO_CANDIDATES,
            message="This project has no clips to render yet.",
            hint="Run the analysis first.",
            status_code=409,
        )
    known = {clip["id"]: clip for clip in clips}
    if body.clip_ids:
        unknown = [clip_id for clip_id in body.clip_ids if clip_id not in known]
        if unknown:
            raise ClipForgeError(
                code=ErrorCode.NOT_FOUND,
                message=f"{len(unknown)} of those clips do not belong to this project.",
                status_code=404,
            )
        clip_ids = list(dict.fromkeys(body.clip_ids))
    else:
        skip = {"queued", "rendering"} | (set() if body.force else {"rendered"})
        clip_ids = [clip["id"] for clip in clips if clip.get("status") not in skip]
    result = manager().enqueue_all(project_id, clip_ids, export=body.export)
    return {**result, "skipped": len(clips) - len(clip_ids) if not body.clip_ids else 0, "queue": job_queue.queue_state(project_id)}


@router.get("/{project_id}/queue")
def project_queue(project_id: str):
    return {
        "queue": job_queue.queue_state(project_id),
        "renders": job_queue.render_queue_for_project(project_id),
        "summary": job_queue.project_job_summary(project_id),
    }


@router.get("/{project_id}/source")
def stream_source(project_id: str, request: Request):
    """Stream the source video (so the UI can scrub it in the browser)."""
    from ...pipeline.context import ProjectPaths, find_media

    detail = project_service.project_detail(project_id)
    paths = ProjectPaths.for_snapshot(detail)
    media = find_media(paths.source)
    if media is None:
        raise ClipForgeError(code=ErrorCode.NOT_FOUND, message="No source video is stored for this project.", status_code=404)
    return stream_file(media, request, cache_seconds=3600)


@router.get("/{project_id}/disk")
def project_disk(project_id: str):
    return project_service.project_disk_usage(project_id)


@router.delete("/{project_id}")
def delete_project(project_id: str, remove_files: bool = Query(True)):
    return project_service.delete_project(project_id, remove_files=remove_files)


@router.post("/{project_id}/open-folder")
def open_folder(project_id: str):
    """Open the project's render folder in the OS file manager (desktop app only)."""
    from ...pipeline.context import ProjectPaths

    detail = project_service.project_detail(project_id)
    target = ProjectPaths.for_snapshot(detail).ensure().renders
    open_in_file_manager(target)
    return {"opened": str(target)}


__all__ = ["router", "shortcuts"]
