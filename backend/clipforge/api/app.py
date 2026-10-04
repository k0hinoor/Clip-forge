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

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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

    application = FastAPI(
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

    @application.on_event("startup")
    async def _startup() -> None:
        application.state.started_at = time.time()
        log_environment()
        recovered = job_queue.recover_stale_jobs()
        settings = get_settings()
        if Env.EMBED_WORKER and settings.auto_start_worker:
            from ..jobs.manager import manager

            manager().start()
        else:
            log.info("embedded worker disabled (CLIPFORGE_EMBED_WORKER=%s)", Env.EMBED_WORKER)
        events.publish("system.ready", {"version": __version__, "recovered_jobs": recovered})

    @application.on_event("shutdown")
    async def _shutdown() -> None:
        from ..jobs.manager import manager

        manager().stop()
        from ..media.runner import kill_all

        kill_all()
        log.info("CLIPFORGE stopped")

    return application


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

API_PREFIX = "/api"

# Convenience aliases straight from the product spec, mapping onto the same
# handlers as the versioned resource routes.
ALIASES: dict[str, tuple[str, tuple[str, ...]]] = {
    "/api/projects/{project_id}/clips": ("GET", ("get_clips",)),
    "/api/clips/render-all": ("POST", ("render_all",)),
    "/api/assets/gameplay": ("GET", ("get_gameplay",)),
}


def _install_routes(application: FastAPI) -> None:
    application.include_router(system_routes.router, prefix=API_PREFIX)
    application.include_router(settings_routes.router, prefix=API_PREFIX)
    application.include_router(projects_routes.router, prefix=API_PREFIX)
    application.include_router(clips_routes.router, prefix=API_PREFIX)
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

    @application.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            return JSONResponse(status_code=404, content={"error": {"code": "not_found", "message": "Unknown API route."}})
        candidate = dist / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(dist / "index.html")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


def _install_error_handlers(application: FastAPI) -> None:
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
