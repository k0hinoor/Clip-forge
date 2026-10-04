"""System routes: hardware, capabilities, diagnostics, Ollama and log access."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Body, Query
from pydantic import BaseModel, Field

from ...config import Env, get_settings
from ...errors import ClipForgeError, ErrorCode
from ...logging_setup import recent_errors, tail_log
from ...media.assets import ensure_seed_assets, library_summary
from ...media.download import classify_url, fetch_metadata
from ...services import events
from ...system import (
    ai_stack,
    diagnostics,
    disk_report,
    features,
    ffmpeg_info,
    fit_whisper_model,
    hardware,
    process_snapshot,
    require_local_paths,
)
from ..schemas import OllamaTestRequest

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/hardware")
def get_hardware(refresh: bool = Query(False, description="Re-detect instead of using the 2 minute cache")):
    """CPU, RAM, GPU, VRAM, CUDA, ffmpeg and AI-stack report."""
    return {"hardware": hardware(refresh=refresh), "ai": ai_stack(refresh=refresh), "ffmpeg": ffmpeg_info(refresh=refresh).to_dict()}


@router.get("/status")
def get_status():
    """Lightweight health payload for the app shell."""
    from ...jobs import queue as job_queue
    from ...jobs.manager import manager

    settings = get_settings()
    ffmpeg = ffmpeg_info()
    stack = ai_stack()
    with_busy = job_queue.list_jobs(statuses=("queued", "running"), limit=50)
    return {
        "app": "CLIPFORGE AI",
        "version": __import__("clipforge").__version__,
        "ready": ffmpeg.available,
        "data_dir": str(Env.DATA_DIR),
        "ffmpeg": ffmpeg.to_dict(),
        "ai": stack,
        "hardware": hardware(),
        "workers": manager().status(),
        "queue": {
            "queued": sum(1 for job in with_busy if job["status"] == "queued"),
            "running": sum(1 for job in with_busy if job["status"] == "running"),
        },
        "usage": process_snapshot(),
        "features": features(),
        "notes": _status_notes(ffmpeg, stack, settings),
    }


def _status_notes(ffmpeg, stack, settings) -> list[dict[str, str]]:
    notes: list[dict[str, str]] = []
    if not ffmpeg.available:
        notes.append(
            {
                "level": "error",
                "title": "ffmpeg not found",
                "detail": "Rendering and audio extraction are unavailable until ffmpeg is installed or a path is set in Settings -> Video.",
            }
        )
    elif not ffmpeg.has_libass:
        notes.append(
            {
                "level": "warning",
                "title": "ffmpeg has no libass",
                "detail": "Captions cannot be burned in. Install a full ffmpeg build (with libass) or use the bundled binary.",
            }
        )
    if not stack.get("transcription_available"):
        notes.append(
            {
                "level": "warning",
                "title": "Local speech-to-text not installed",
                "detail": 'Install the AI extras to transcribe offline:  pip install -r requirements-ai.txt   (faster-whisper).',
            }
        )
    if not stack.get("opencv"):
        notes.append(
            {
                "level": "info",
                "title": "Face-aware reframing unavailable",
                "detail": "Install opencv-python-headless for face detection during vertical reframing; motion/centre framing is used meanwhile.",
            }
        )
    if not settings.llm_enabled:
        notes.append(
            {
                "level": "info",
                "title": "Local LLM disabled",
                "detail": "Analysis runs in analytical mode using the built-in scoring engine.",
            }
        )
    model, downgrade = fit_whisper_model(settings.whisper_model)
    if downgrade and stack.get("transcription_available"):
        notes.append({"level": "info", "title": f"Whisper '{model}' will be used", "detail": downgrade})
    return notes


@router.get("/diagnostics")
def get_diagnostics():
    """Paths, log locations, recent errors and version information."""
    return diagnostics()


@router.get("/logs")
def get_logs(name: str = Query("app", pattern="^(app|worker|render|ai)$"), lines: int = Query(200, ge=1, le=2000)):
    return {"file": f"{name}.log", "lines": tail_log(name, lines=lines)}


@router.get("/errors")
def get_errors(limit: int = Query(30, ge=1, le=200)):
    return {"errors": recent_errors(limit)}


@router.post("/ffmpeg/detect")
def detect_ffmpeg(path: str = Body("", embed=True), ffprobe: str = Body("", embed=True)):
    """Re-run ffmpeg detection, optionally with an explicit binary path."""
    from ...system import invalidate_cache

    if path:
        require_local_paths("Choosing an ffmpeg binary")
        candidate = Path(path).expanduser()
        if not candidate.exists():
            raise ClipForgeError(
                code=ErrorCode.FFMPEG_NOT_FOUND,
                message=f"No file exists at {path}",
                hint="Pick the full path to ffmpeg.exe (Windows) or the ffmpeg binary.",
                status_code=422,
            )
        from ...config import settings_store

        settings_store().update({"ffmpeg_path": str(candidate), "ffprobe_path": ffprobe or ""})
    invalidate_cache()
    info = ffmpeg_info(refresh=True)
    events.publish("system.ffmpeg", info.to_dict())
    return {"ffmpeg": info.to_dict(), "ok": info.available}


@router.get("/ffmpeg")
def get_ffmpeg():
    return {"ffmpeg": ffmpeg_info().to_dict()}


def _ollama_settings(base_url: str = "", model: str = ""):
    """Current settings with an Ollama URL/model under test (never persisted)."""
    settings = get_settings()
    update: dict[str, str] = {}
    if base_url.strip():
        url = base_url.strip().rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ClipForgeError(
                code=ErrorCode.INVALID_INPUT,
                message="The Ollama URL must start with http:// or https://.",
                hint="For a local install use http://127.0.0.1:11434.",
                status_code=422,
            )
        update["ollama_base_url"] = url
    if model.strip():
        update["ollama_model"] = model.strip()
    return settings.model_copy(update=update) if update else settings


@router.post("/ollama/test")
def test_ollama(payload: OllamaTestRequest = Body(default_factory=OllamaTestRequest)):  # noqa: B008
    """Check whether Ollama is reachable and the model exists.

    ``base_url``/``model`` in the body are tested without being saved - save
    them through ``PUT /api/settings`` once the test passes.
    """
    from ...ai.llm import OllamaClient

    settings = _ollama_settings(payload.base_url or "", payload.model or "")
    status = OllamaClient(settings).health()
    return {"status": status.to_dict(), "models": status.models}


@router.get("/ollama/models")
def list_ollama_models(base_url: str = Query("", description="Ask this Ollama server instead of the configured one")):
    from ...ai.llm import available_models

    settings = _ollama_settings(base_url)
    return {"models": available_models(settings), "base_url": settings.ollama_base_url, "current": settings.ollama_model}


@router.post("/youtube/probe")
@router.post("/probe")
def probe_source(url: str = Body(..., embed=True)):
    """Validate any video link and return its metadata without downloading."""
    source = classify_url(url)
    meta = fetch_metadata(source.url)
    return {"source": source.to_dict(), "video_id": meta.video_id, "metadata": meta.to_dict()}


@router.get("/capabilities")
def capabilities():
    """Everything the UI needs to decide what to offer (presets, ratios, layouts)."""
    from ...constants import CAPTION_PRESETS, CATEGORIES, LAYOUTS, SPLIT_RATIOS
    from ...media.assets import BROLL_CATEGORIES, GAMEPLAY_CATEGORIES, MUSIC_MOODS

    settings = get_settings()
    ffmpeg = ffmpeg_info()
    stack = ai_stack()
    return {
        "aspect_ratios": {"9:16": [1080, 1920], "1:1": [1080, 1080], "16:9": [1920, 1080]},
        "caption_presets": {key: {"label": value["label"], "description": value["description"], "theme": value["theme"]} for key, value in CAPTION_PRESETS.items()},
        "categories": CATEGORIES,
        "layouts": LAYOUTS,
        "split_ratios": SPLIT_RATIOS,
        "gameplay_categories": GAMEPLAY_CATEGORIES,
        "broll_categories": BROLL_CATEGORIES,
        "music_moods": MUSIC_MOODS,
        "fonts": _font_list(),
        "whisper_models": ["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"],
        "computes": ["auto", "int8", "int8_float16", "float16", "float32"],
        "encoders": ffmpeg.encoders,
        "transcription_available": bool(stack.get("transcription_available")),
        "vision_available": bool(stack.get("opencv")),
        "defaults": settings.to_public_dict(),
        "assets": library_summary(),
    }


def _font_list() -> list[str]:
    from ...media.captions import _installed_fonts

    fonts = _installed_fonts()
    preferred = [
        "Inter", "Inter ExtraBold", "Arial", "Arial Black", "Roboto", "Montserrat", "DejaVu Sans",
        "Noto Sans", "Noto Sans Devanagari", "Nirmala UI", "Segoe UI", "Georgia", "Impact", "Verdana",
    ]
    available = [font for font in preferred if any(font.lower() in name.lower() for name in fonts)] if fonts else preferred
    return available or preferred


@router.post("/assets/scan")
def scan_assets():
    imported = ensure_seed_assets()
    return {"imported": imported, "library": library_summary()}


@router.get("/paths")
def get_paths():
    settings = get_settings()
    return {
        "data_dir": str(Env.DATA_DIR),
        "database": str(Env.DB_PATH),
        "logs": str(Env.LOG_DIR),
        "projects": str(Env.DATA_DIR / "projects"),
        "cache": str(Env.DATA_DIR / "cache"),
        "exports": str(settings.resolved_export_dir()),
        "env": {"host": Env.HOST, "port": Env.PORT, "embedded_worker": Env.EMBED_WORKER},
    }


class CleanupRequest(BaseModel):
    days: int | None = Field(None, ge=0, le=3650, description="Age threshold; default = Settings -> Storage -> Clean up after")
    renders: bool = Field(False, description="Also delete rendered clip files older than the threshold (clips can be re-rendered)")
    dry_run: bool = Field(False, description="Only report what would be removed")


@router.post("/cleanup")
def cleanup(payload: CleanupRequest | None = Body(default=None)):  # noqa: B008
    """Free disk space: expired download cache, stale unfinished projects, optionally old renders."""
    from ...services.storage import cleanup_storage, storage_usage

    body = payload or CleanupRequest()
    report = cleanup_storage(days=body.days, include_renders=body.renders, dry_run=body.dry_run)
    return {**report, "usage": storage_usage()}


@router.get("/storage")
def storage():
    """Bytes used by projects, cache, assets, exports and logs."""
    from ...services.storage import storage_usage

    return {"usage": storage_usage(), "disk": disk_report()}


@router.get("/health")
def health():
    ffmpeg = ffmpeg_info()
    return {"ok": True, "ffmpeg": ffmpeg.available, "version": __import__("clipforge").__version__}
