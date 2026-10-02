"""FastAPI dependencies: DB session, auth, CSRF, rate limiting, services."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from clipforge.core.config import Settings
from clipforge.core.errors import AppError, ErrorCode
from clipforge.db.models import User, UserSession
from clipforge.queue.base import QueueProvider
from clipforge.security.ratelimit import RateLimiter
from clipforge.security.tokens import constant_time_equals
from clipforge.services.auth import AuthService
from clipforge.storage import StorageProvider

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_storage_dep(request: Request) -> StorageProvider:
    return request.app.state.storage


def get_queue_dep(request: Request) -> QueueProvider:
    return request.app.state.queue


def get_db(request: Request) -> Iterator[Session]:
    """Request-scoped session. Commits on success, rolls back on error.

    Mutating routes also commit explicitly before building their response so
    clients never observe uncommitted state.
    """
    session: Session = request.app.state.session_factory()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


def client_ip(request: Request) -> str:
    settings: Settings = request.app.state.settings
    if settings.TRUST_PROXY_HEADERS:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()[:64]
    return (request.client.host if request.client else "unknown")[:64]


def _token_from_request(request: Request, settings: Settings) -> tuple[str | None, bool]:
    """Returns (token, via_cookie)."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip(), False
    token = request.cookies.get(settings.SESSION_COOKIE_NAME)
    return token, token is not None


@dataclass
class AuthContext:
    user: User
    session: UserSession | None
    via_cookie: bool


def get_auth(request: Request, db: Session = Depends(get_db)) -> AuthContext:
    settings: Settings = request.app.state.settings
    if not settings.auth_required:  # private local deployments only
        return AuthContext(user=AuthService(db, settings).local_user(), session=None, via_cookie=False)
    token, via_cookie = _token_from_request(request, settings)
    sess = AuthService(db, settings).resolve(token)
    if sess is None:
        raise AppError(ErrorCode.UNAUTHORIZED)
    if via_cookie and request.method not in SAFE_METHODS:
        # Double-submit CSRF protection for cookie-authenticated mutations.
        header = request.headers.get("x-csrf-token")
        cookie = request.cookies.get(settings.CSRF_COOKIE_NAME)
        if not header or not constant_time_equals(header, cookie):
            raise AppError(ErrorCode.FORBIDDEN, "Missing or invalid CSRF token.")
    request.state.user_id = sess.user_id
    return AuthContext(user=sess.user, session=sess, via_cookie=via_cookie)


def current_user(auth: AuthContext = Depends(get_auth)) -> User:
    return auth.user


def admin_user(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise AppError(ErrorCode.FORBIDDEN)
    return user


def rate_limit(bucket: str, limit_attr: str, window_seconds: float):
    """Dependency factory: per-IP (and per-user when known) sliding-window limit."""

    def dep(request: Request) -> None:
        settings: Settings = request.app.state.settings
        if not settings.RATE_LIMIT_ENABLED:
            return
        limiter: RateLimiter = request.app.state.limiter
        key = f"{bucket}:{client_ip(request)}"
        allowed, retry_after = limiter.hit(key, int(getattr(settings, limit_attr)), window_seconds)
        if not allowed:
            raise AppError(ErrorCode.RATE_LIMITED, details={"retry_after_seconds": round(retry_after, 1)})

    return dep


auth_rate_limit = rate_limit("auth", "RATE_LIMIT_AUTH_PER_MINUTE", 60)
upload_rate_limit = rate_limit("uploads", "RATE_LIMIT_UPLOADS_PER_HOUR", 3600)
