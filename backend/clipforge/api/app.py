"""FastAPI application factory."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from clipforge import __version__
from clipforge.api.routes import admin, auth, clips, health, jobs, me, uploads
from clipforge.core.config import Settings, get_settings
from clipforge.core.errors import AppError, ErrorCode, new_diagnostic_id
from clipforge.core.logging import configure_logging, get_logger, log_context
from clipforge.db.session import make_session_factory
from clipforge.queue import get_queue
from clipforge.security.ratelimit import RateLimiter
from clipforge.services.bootstrap import bootstrap
from clipforge.storage import get_storage

log = get_logger(__name__)

SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"strict-origin-when-cross-origin"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
    (b"cross-origin-opener-policy", b"same-origin"),
    (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'"),
]


class SecurityMiddleware:
    """Pure ASGI middleware (streaming-safe): request ids, secure headers,
    JSON body size limits and access logging."""

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        incoming = headers.get(b"x-request-id", b"").decode("latin-1")[:64]
        request_id = incoming if incoming.replace("-", "").isalnum() and incoming else uuid.uuid4().hex
        path: str = scope.get("path", "")
        is_upload = "/uploads" in path
        limit = self.settings.MAX_REQUEST_BODY_BYTES
        length = headers.get(b"content-length", b"")
        if not is_upload and length.isdigit() and int(length) > limit:
            resp = _error_response(AppError(ErrorCode.VALIDATION_ERROR, "Request body is too large.",
                                            http_status=413))
            await resp(scope, receive, send)
            return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if not is_upload and message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise AppError(ErrorCode.VALIDATION_ERROR, "Request body is too large.", http_status=413)
            return message

        status_holder = {"code": 500}
        hsts = self.settings.SESSION_COOKIE_SECURE

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["code"] = message["status"]
                hdrs = list(message.get("headers", []))
                existing = {k.lower() for k, _ in hdrs}
                for k, v in SECURITY_HEADERS:
                    if k not in existing:
                        hdrs.append((k, v))
                if hsts:
                    hdrs.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))
                hdrs.append((b"x-request-id", request_id.encode()))
                message["headers"] = hdrs
            await send(message)

        start = time.perf_counter()
        with log_context(request_id=request_id):
            try:
                await self.app(scope, limited_receive, send_wrapper)
            finally:
                if path not in ("/api/v1/health", "/metrics") and not path.endswith("/stream"):
                    log.info("request", extra={"method": scope.get("method"), "path": path,
                                               "status": status_holder["code"],
                                               "duration_ms": round((time.perf_counter() - start) * 1000, 1)})


def _error_response(err: AppError) -> JSONResponse:
    headers = {}
    if err.code == ErrorCode.RATE_LIMITED and err.details:
        headers["Retry-After"] = str(int(err.details.get("retry_after_seconds", 1)) + 1)
    return JSONResponse({"error": err.to_dict()}, status_code=err.http_status, headers=headers)


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error(request: Request, exc: AppError) -> JSONResponse:
        if exc.internal or exc.http_status >= 500:
            log.warning("request failed", extra={"error_code": exc.code.value, "diagnostic_id": exc.diagnostic_id,
                                                 "internal": (exc.internal or "")[-2000:]})
        return _error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [{"field": ".".join(str(p) for p in e.get("loc", [])[1:]), "message": e.get("msg")}
                  for e in exc.errors()][:20]
        return _error_response(AppError(ErrorCode.VALIDATION_ERROR, "Some fields are invalid.",
                                        details={"fields": fields}))

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {401: ErrorCode.UNAUTHORIZED, 403: ErrorCode.FORBIDDEN, 404: ErrorCode.NOT_FOUND,
                405: ErrorCode.VALIDATION_ERROR, 409: ErrorCode.CONFLICT}.get(exc.status_code, ErrorCode.UNKNOWN_ERROR)
        err = AppError(code, http_status=exc.status_code)
        return _error_response(err)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        diagnostic_id = new_diagnostic_id()
        log.error("unhandled error", exc_info=exc, extra={"diagnostic_id": diagnostic_id})
        err = AppError(ErrorCode.UNKNOWN_ERROR, diagnostic_id=diagnostic_id)
        return _error_response(err)


def create_app(settings: Settings | None = None, *, run_bootstrap: bool = True) -> FastAPI:
    settings = settings or get_settings()
    configure_logging("api", settings.LOG_LEVEL, settings.LOG_JSON)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if run_bootstrap:
            bootstrap(settings, app.state.session_factory)
        log.info("api started", extra={"version": __version__, "deployment_mode": settings.DEPLOYMENT_MODE})
        yield
        log.info("api stopping")

    app = FastAPI(title="ClipForge API", version=__version__, lifespan=lifespan,
                  docs_url=f"{settings.API_PREFIX}/docs", openapi_url=f"{settings.API_PREFIX}/openapi.json",
                  redoc_url=None)
    app.state.settings = settings
    app.state.session_factory = make_session_factory(settings)
    app.state.storage = get_storage(settings)
    app.state.queue = get_queue(settings, app.state.session_factory)
    app.state.limiter = RateLimiter()

    _install_error_handlers(app)
    for r in (auth.router, uploads.router, jobs.router, clips.router, me.router, admin.router, health.router):
        app.include_router(r, prefix=settings.API_PREFIX)
    app.add_api_route("/metrics", health.metrics, include_in_schema=False)

    app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_credentials=True,
                       allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
                       allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "X-CSRF-Token",
                                      "Last-Event-ID", "X-Request-ID"],
                       expose_headers=["X-Request-ID", "Retry-After", "Content-Range", "Accept-Ranges"])
    app.add_middleware(SecurityMiddleware, settings=settings)
    return app


def openapi_schema() -> dict[str, Any]:  # used by tooling / frontend codegen
    return create_app(run_bootstrap=False).openapi()
