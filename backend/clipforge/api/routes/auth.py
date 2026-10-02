"""Auth routes (TRD §28.1, §35): opaque session tokens in HttpOnly cookies (or Bearer)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.orm import Session

from clipforge.api.deps import AuthContext, auth_rate_limit, client_ip, get_auth, get_db, get_settings_dep
from clipforge.api.schemas import AuthResponse, ChangePasswordRequest, LoginRequest, RegisterRequest, iso
from clipforge.api.serializers import user_out
from clipforge.core.config import Settings
from clipforge.services.audit import audit
from clipforge.services.auth import AuthService, IssuedSession

router = APIRouter(prefix="/auth", tags=["auth"])


def _set_cookies(response: Response, issued: IssuedSession, settings: Settings) -> None:
    max_age = settings.SESSION_TTL_HOURS * 3600
    common: dict[str, Any] = {"max_age": max_age, "secure": settings.SESSION_COOKIE_SECURE,
                              "samesite": settings.SESSION_COOKIE_SAMESITE, "path": "/"}
    response.set_cookie(settings.SESSION_COOKIE_NAME, issued.token, httponly=True, **common)
    response.set_cookie(settings.CSRF_COOKIE_NAME, issued.csrf_token, httponly=False, **common)


def _clear_cookies(response: Response, settings: Settings) -> None:
    response.delete_cookie(settings.SESSION_COOKIE_NAME, path="/")
    response.delete_cookie(settings.CSRF_COOKIE_NAME, path="/")


def _auth_response(issued: IssuedSession) -> dict[str, Any]:
    return {"user": user_out(issued.session.user), "token": issued.token, "csrf_token": issued.csrf_token,
            "expires_at": iso(issued.session.expires_at)}


@router.post("/register", response_model=AuthResponse, status_code=201, dependencies=[Depends(auth_rate_limit)])
def register(body: RegisterRequest, request: Request, response: Response, db: Session = Depends(get_db),
             settings: Settings = Depends(get_settings_dep)) -> dict[str, Any]:
    svc = AuthService(db, settings)
    user = svc.register(body.email, body.password, body.display_name, ip=client_ip(request))
    issued = svc.issue_session(user, user_agent=request.headers.get("user-agent"), ip=client_ip(request))
    db.commit()
    _set_cookies(response, issued, settings)
    return _auth_response(issued)


@router.post("/login", response_model=AuthResponse, dependencies=[Depends(auth_rate_limit)])
def login(body: LoginRequest, request: Request, response: Response, db: Session = Depends(get_db),
          settings: Settings = Depends(get_settings_dep)) -> dict[str, Any]:
    svc = AuthService(db, settings)
    user = svc.authenticate(body.email, body.password)
    issued = svc.issue_session(user, user_agent=request.headers.get("user-agent"), ip=client_ip(request))
    audit(db, "auth.login", actor_user_id=user.id, target_type="user", target_id=user.id, ip_address=client_ip(request))
    db.commit()
    _set_cookies(response, issued, settings)
    return _auth_response(issued)


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response, auth: AuthContext = Depends(get_auth),
           db: Session = Depends(get_db), settings: Settings = Depends(get_settings_dep)) -> Response:
    if auth.session is not None:
        AuthService(db, settings).revoke(auth.session)
        audit(db, "auth.logout", actor_user_id=auth.user.id, ip_address=client_ip(request))
    db.commit()
    out = Response(status_code=204)
    _clear_cookies(out, settings)
    return out


@router.post("/refresh", response_model=AuthResponse)
def refresh(request: Request, response: Response, auth: AuthContext = Depends(get_auth),
            db: Session = Depends(get_db), settings: Settings = Depends(get_settings_dep)) -> dict[str, Any]:
    """Rotate the session token (TRD §35 session rotation)."""
    svc = AuthService(db, settings)
    if auth.session is None:
        issued = svc.issue_session(auth.user, user_agent=request.headers.get("user-agent"), ip=client_ip(request))
    else:
        issued = svc.rotate(auth.session, user_agent=request.headers.get("user-agent"), ip=client_ip(request))
    db.commit()
    _set_cookies(response, issued, settings)
    return _auth_response(issued)


@router.get("/me")
def whoami(auth: AuthContext = Depends(get_auth)) -> dict[str, Any]:
    return {"user": user_out(auth.user)}


@router.post("/password", status_code=204, dependencies=[Depends(auth_rate_limit)])
def change_password(body: ChangePasswordRequest, request: Request, auth: AuthContext = Depends(get_auth),
                    db: Session = Depends(get_db), settings: Settings = Depends(get_settings_dep)) -> Response:
    AuthService(db, settings).change_password(auth.user, body.current_password, body.new_password)
    audit(db, "auth.password_changed", actor_user_id=auth.user.id, ip_address=client_ip(request))
    db.commit()
    out = Response(status_code=204)
    _clear_cookies(out, settings)
    return out
