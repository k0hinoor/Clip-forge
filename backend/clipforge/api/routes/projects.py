"""Project routes: create from a URL (or upload), analyse, inspect, delete."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Body, File, Form, Query, Request, UploadFile

from ...config import Env, get_settings
from ...errors import ClipForgeError, ErrorCode
from ...jobs import queue as job_queue
from ...jobs.manager import manager
from ...logging_setup import get_logger
from ...services import events
from ...services import projects as project_service
from ..schemas import AnalyzeRequest, ClipOptions, ProjectCreate
from ..media import stream_file

router = APIRouter(prefix="/projects", tags=["projects"])
log = get_logger(__name__)


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

    # A link that points at a file on this machine is imported, not downloaded.
    local = Path(payload.url).expanduser()
    if local.is_file():
        from ...media.download import ALLOWED_UPLOAD_EXTENSIONS

        if local.suffix.lower() not in ALLOWED_UPLOAD_EXTENSIONS:
            raise ClipForgeError(
                code=ErrorCode.UNSUPPORTED_FORMAT,
                message=f"{local.suffix or 'That file type'} cannot be analysed.",
                hint="Supported: " + ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS)),
                status_code=422,
            )
        project = project_service.create_project(
            title=payload.title or local.stem,
            source_type="upload",
            options=payload.options.to_settings_patch(),
            source_path=local,
        )
        job = manager().enqueue_analysis(project["id"], priority=payload.priority) if payload.analyze else None
        return {"project": project, "job": job}

    from ...media.download import classify_url

    source = classify_url(payload.url)
    options = payload.options.to_settings_patch()
    project = project_service.create_project(
        url=source.url,
        title=payload.title,
        source_type=payload.source_type or ("youtube" if source.is_youtube else "url"),
        options=options,
    )
    job = None
    if payload.analyze:
        job = manager().enqueue_analysis(project["id"], priority=payload.priority)
    return {"project": project, "job": job}


@router.post("/upload", status_code=201)
async def upload_project(
    file: UploadFile = File(...),
    title: str = Form(""),
    options: str = Form(""),
    analyze: str = Form("true"),
):
    """Analyse a local file instead of a link."""
    settings = get_settings()
    if not settings.uploads_enabled:
        raise ClipForgeError(
            code=ErrorCode.INVALID_INPUT,
            message="Uploads are disabled in Settings -> Storage.",
            status_code=403,
        )
    import json

    parsed_options = ClipOptions()
    if options:
        try:
            parsed_options = ClipOptions.model_validate(json.loads(options))
        except Exception as exc:  # noqa: BLE001
            raise ClipForgeError(code=ErrorCode.INVALID_INPUT, message=f"Invalid options: {exc}", status_code=422) from exc

    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    limit = int(settings.max_upload_gb * 1024**3)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(Env.DATA_DIR / "cache")) as handle:
        temp_path = Path(handle.name)
        written = 0
        while chunk := await file.read(1024 * 1024 * 4):
            written += len(chunk)
            if written > limit:
                handle.close()
                temp_path.unlink(missing_ok=True)
                raise ClipForgeError(
                    code=ErrorCode.UPLOAD_TOO_LARGE,
                    message=f"That file is larger than the {settings.max_upload_gb:g} GB upload limit.",
                    hint="Raise the limit in Settings -> Storage, or upload a smaller file.",
                    status_code=413,
                )
            handle.write(chunk)

    try:
        project = project_service.create_project(
            title=title or Path(file.filename or "Upload").stem,
            source_type="upload",
            options=parsed_options.to_settings_patch(),
            source_path=temp_path,
        )
    finally:
        temp_path.unlink(missing_ok=True)
        await file.close()

    job = None
    if analyze.lower() not in {"false", "0", "no"}:
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
    """Attach a transcript/caption file (``.json3``, ``.srt``, ``.vtt``, ``.txt``).

    When a project has one of these, the analysis pipeline uses it instead of
    speech recognition - useful when the user already owns accurate captions.
    """
    from ...media.download import TRANSCRIPT_EXTENSIONS
    from ...pipeline.context import ProjectPaths

    detail = project_service.project_detail(project_id)
    suffix = Path(file.filename or "transcript.srt").suffix.lower()
    if suffix not in TRANSCRIPT_EXTENSIONS:
        raise ClipForgeError(
            code=ErrorCode.INVALID_INPUT,
            message=f"'{suffix or 'that file'}' is not a supported transcript format.",
            hint="Supported formats: .json3 (YouTube), .srt, .vtt, .txt with [hh:mm:ss] stamps.",
            status_code=422,
        )

    paths = ProjectPaths.for_snapshot(detail).ensure()
    target = paths.transcript / f"provided{suffix}"
    size = 0
    with target.open("wb") as handle:
        while chunk := await file.read(1024 * 512):
            size += len(chunk)
            if size > 40 * 1024 * 1024:
                handle.close()
                target.unlink(missing_ok=True)
                raise ClipForgeError(
                    code=ErrorCode.UPLOAD_TOO_LARGE,
                    message="That transcript is larger than 40 MB.",
                    status_code=413,
                )
            handle.write(chunk)
    await file.close()

    from ...media.download import parse_transcript_file

    words = parse_transcript_file(target)
    if not words:
        target.unlink(missing_ok=True)
        raise ClipForgeError(
            code=ErrorCode.INVALID_INPUT,
            message="That file contained no readable timings.",
            hint="Export an SRT/VTT from your editor, or use a YouTube .json3 caption file.",
            status_code=422,
        )
    return {
        "project_id": project_id,
        "file": target.name,
        "words": len(words),
        "covers": round(words[-1]["end"], 2),
        "message": f"Transcript attached ({len(words)} words). It will be used instead of speech recognition on the next analysis.",
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
def analyze(payload: AnalyzeRequest):
    """Start (or restart) the analysis pipeline for a project."""
    project_id = payload.project_id
    if not project_id and payload.url:
        project = project_service.create_project(
            url=payload.url,
            options=(payload.options or ClipOptions()).to_settings_patch(),
        )
        project_id = project["id"]
    if not project_id:
        raise ClipForgeError(
            code=ErrorCode.INVALID_INPUT,
            message="Provide a project_id or a url to analyse.",
            status_code=422,
        )
    if payload.options is not None:
        project_service.merge_project_options(project_id, payload.options.to_settings_patch())
    job = manager().enqueue_analysis(project_id, priority=payload.priority)
    return {"project_id": project_id, "job": job, "status": project_service.project_status(project_id)}


@router.post("/{project_id}/analyze")
def analyze_project(project_id: str, payload: dict | None = Body(default=None)):  # noqa: B008
    body = payload or {}
    options = body.get("options")
    if options:
        project_service.merge_project_options(project_id, options)
    job = manager().enqueue_analysis(project_id, priority=int(body.get("priority", 1)))
    return {"project_id": project_id, "job": job}


@router.post("/{project_id}/cancel")
def cancel_project(project_id: str):
    cancelled = manager().cancel_project(project_id)
    events.publish("project.cancelled", {"project_id": project_id, "jobs": cancelled}, project_id=project_id)
    return {"cancelled_jobs": cancelled}


@router.post("/{project_id}/render-all")
def render_all(project_id: str, payload: dict | None = Body(default=None)):  # noqa: B008
    body = payload or {}
    clips = project_service.project_clips(project_id, sort="score")["clips"]
    clip_ids = body.get("clip_ids") or [clip["id"] for clip in clips]
    if not clip_ids:
        raise ClipForgeError(
            code=ErrorCode.NO_CANDIDATES,
            message="This project has no clips to render yet.",
            hint="Run the analysis first.",
            status_code=409,
        )
    result = manager().enqueue_all(project_id, clip_ids, export=bool(body.get("export", False)))
    return {**result, "queue": job_queue.queue_state(project_id)}


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
    """Open the project's render folder in the OS file manager (local app nicety)."""
    import subprocess
    import sys

    from ...pipeline.context import ProjectPaths

    detail = project_service.project_detail(project_id)
    paths = ProjectPaths.for_snapshot(detail).ensure()
    target = paths.renders
    try:
        if sys.platform.startswith("win"):
            import os

            os.startfile(str(target))  # type: ignore[attr-defined]  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])  # noqa: S603,S607
        else:
            subprocess.Popen(["xdg-open", str(target)])  # noqa: S603,S607
    except Exception as exc:  # noqa: BLE001 - headless environments
        raise ClipForgeError(
            code=ErrorCode.INTERNAL,
            message="Could not open the folder automatically.",
            hint=f"Open it manually: {target}",
            detail=str(exc),
            status_code=501,
        ) from exc
    return {"opened": str(target)}


__all__ = ["router"]
