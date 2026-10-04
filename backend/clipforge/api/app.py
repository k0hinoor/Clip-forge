"""FastAPI application factory.

Local-first by design:

* binds to ``127.0.0.1`` by default (see :class:`clipforge.config.Env`),
* CORS is limited to the local frontend origins,
* the worker pool runs inside the same process unless started separately,
* the built Next.js frontend is served from the same origin when present, so the
  whole app is one URL on one port.

Every error the UI can see is a structured ``{"error": {...}}`` payload - raw
tracebacks only ever reach ``logs/``.
"""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import __version__
from ..config import Env, ensure_dirs, get_settings, settings_store
from ..db import init_db
from ..errors import ClipForgeError, ErrorCode
from ..jobs import queue as job_queue
from ..logging_setup import configure_logging, get_logger, log_environment
from ..media.assets import ensure_seed_assets
from ..services import events
from .routes import assets as assets_routes
from .routes import clips as clips_routes
from .routes import jobs as jobs_routes
from .routes import projects as projects_routes
from .routes import settings as settings_routes
from .routes import system as system_routes

log = get_logger(__name__)

LOCAL_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8317",
    "http://127.0.0.1:8317",
]


def frontend_dist() -> Path | None:
    if Env.FRONTEND_DIST and Env.FRONTEND_DIST.exists():
        return Env.FRONTEND_DIST
    for candidate in (
        Env.DATA_DIR.parent / "frontend" / "out",
        Path(__file__).resolve().parents[3] / "frontend" / "out",
    ):
        if candidate.exists() and (candidate / "index.html").exists():
            return candidate
    return None


def create_app() -> FastAPI:
    ensure_dirs()
    configure_logging()
    init_db()
    settings_store().load()
    try:
        ensure_seed_assets()
    except Exception as exc:  # noqa: BLE001 - asset folders are a nicety
        log.debug("asset seeding skipped: %s", exc)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        _startup(application)
        try:
            yield
        finally:
            _shutdown()

    application = FastAPI(
        lifespan=lifespan,
        title="CLIPFORGE AI",
        version=__version__,
        description=(
            "Local AI clipping studio: YouTube URL in, analysed and edited vertical shorts out. "
            "Everything runs on this machine - transcription, analysis, framing and rendering."
        ),
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=LOCAL_ORIGINS + [f"http://{Env.HOST}:{Env.PORT}"],
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
    )

    _install_error_handlers(application)
    _install_routes(application)
    _install_frontend(application)
    return application


def _startup(application: FastAPI) -> None:
    application.state.started_at = time.time()
    log_environment()
    settings = get_settings()
    recovered = 0
    if Env.EMBED_WORKER:
        # With a separate `clipforge worker` process the queue belongs to it.
        recovered = job_queue.recover_stale_jobs()
    if Env.EMBED_WORKER and settings.auto_start_worker:
        from ..jobs.manager import manager

        manager().start()
    else:
        log.info("embedded worker not started (CLIPFORGE_EMBED_WORKER=%s, auto_start_worker=%s)", Env.EMBED_WORKER, settings.auto_start_worker)
    # Expired download caches are trimmed in the background on every start.
    threading.Thread(target=_trim_caches, name="cache-cleanup", daemon=True).start()
    events.publish("system.ready", {"version": __version__, "recovered_jobs": recovered})


def _trim_caches() -> None:
    from ..services.storage import cleanup_cache

    try:
        report = cleanup_cache()
        if report["removed_files"]:
            log.info("startup cache cleanup freed %.1f MB (%d files)", report["freed_bytes"] / 1e6, report["removed_files"])
    except Exception:  # noqa: BLE001 - housekeeping must never block startup
        log.exception("startup cache cleanup failed")


def _shutdown() -> None:
    from ..jobs.manager import manager
    from ..media.runner import kill_all

    manager().stop()
    kill_all()
    log.info("CLIPFORGE stopped")


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

API_PREFIX = "/api"


def _install_routes(application: FastAPI) -> None:
    application.include_router(system_routes.router, prefix=API_PREFIX)
    application.include_router(settings_routes.router, prefix=API_PREFIX)
    application.include_router(projects_routes.router, prefix=API_PREFIX)
    application.include_router(projects_routes.shortcuts, prefix=API_PREFIX)
    application.include_router(clips_routes.router, prefix=API_PREFIX)
    application.include_router(clips_routes.captions_router, prefix=API_PREFIX)
    application.include_router(assets_routes.router, prefix=API_PREFIX)
    application.include_router(jobs_routes.router, prefix=API_PREFIX)

    @application.get("/api", tags=["system"])
    def api_index() -> dict[str, Any]:
        # Kept next to the routes so the UI (and the user) can discover the surface
        # without opening the OpenAPI page.
        try:
            schema = application.openapi()
            endpoints = sorted(schema.get("paths", {}))
        except Exception:  # noqa: BLE001 - never fail the index because of docs
            endpoints = []
        return {
            "name": "CLIPFORGE AI API",
            "version": __version__,
            "docs": "/api/docs",
            "endpoints": endpoints,
        }


def _install_frontend(application: FastAPI) -> None:
    dist = frontend_dist()
    if dist is None:
        @application.get("/", include_in_schema=False)
        def root() -> JSONResponse:
            settings = get_settings()
            return JSONResponse(
                {
                    "app": "CLIPFORGE AI",
                    "version": __version__,
                    "message": "The frontend build was not found, so this is the bare API.",
                    "hint": (
                        "Run the UI with 'npm run dev' inside frontend/ (http://localhost:3000) "
                        "or build it with 'npm run build' to serve it from this port."
                    ),
                    "api_docs": "/api/docs",
                    "data_dir": str(Env.DATA_DIR),
                    "exports_dir": str(settings.resolved_export_dir()),
                }
            )
        return

    application.mount("/_next", StaticFiles(directory=str(dist / "_next")), name="next-assets") if (dist / "_next").exists() else None

    root = dist.resolve()

    @application.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path == "api" or full_path.startswith("api/"):
            return JSONResponse(status_code=404, content={"error": {"code": ErrorCode.NOT_FOUND, "message": "Unknown API route.", "hint": "See /api/docs."}})
        if full_path:
            candidate = (root / full_path).resolve()
            # Only files inside the build folder - never "../" out of it.
            if candidate.is_file() and candidate.is_relative_to(root):
                return FileResponse(candidate)
            page = (root / f"{full_path.rstrip('/')}.html").resolve()
            if page.is_file() and page.is_relative_to(root):
                return FileResponse(page)
        return FileResponse(root / "index.html")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


def _install_error_handlers(application: FastAPI) -> None:
    @application.exception_handler(StarletteHTTPException)
    async def handle_http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: ErrorCode.NOT_FOUND, 405: ErrorCode.INVALID_INPUT, 409: ErrorCode.CONFLICT}.get(exc.status_code, ErrorCode.INVALID_INPUT)
        message = {404: "Unknown API route.", 405: "That method is not allowed on this route."}.get(exc.status_code, str(exc.detail))
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": code, "message": message, "hint": "See /api/docs for every route."}},
            headers=getattr(exc, "headers", None),
        )

    @application.exception_handler(ClipForgeError)
    async def handle_clipforge_error(_request: Request, exc: ClipForgeError) -> JSONResponse:
        log.warning("request failed: %s (%s)", exc.message, exc.code)
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        problems = [
            {
                "field": ".".join(str(part) for part in error.get("loc", []) if part not in {"body", "query", "path"}),
                "problem": error.get("msg", "invalid value"),
            }
            for error in exc.errors()[:8]
        ]
        first = problems[0] if problems else {"field": "", "problem": "invalid request"}
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": ErrorCode.INVALID_INPUT,
                    "message": f"{first['field'] or 'Request'}: {first['problem']}",
                    "hint": "Check the highlighted field and try again.",
                    "context": {"problems": problems},
                }
            },
        )

    @application.exception_handler(Exception)
    async def handle_unexpected(_request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": ErrorCode.INTERNAL,
                    "message": "CLIPFORGE hit an unexpected error while handling that request.",
                    "hint": "Open Settings -> Diagnostics for recent errors, or check logs/app.log.",
                    "detail": f"{type(exc).__name__}: {exc}"[:600],
                }
            },
        )


app = create_app()


__all__ = ["app", "create_app", "frontend_dist"]
