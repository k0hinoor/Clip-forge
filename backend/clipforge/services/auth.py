"""Authentication service (TRD §25).

* Argon2id password hashes; plaintext is never stored.
* Opaque random session tokens stored as SHA-256 hashes (revocable).
* Session rotation on refresh; logout revokes the session.
* Optional implicit local user when ``AUTH_ENABLED=false`` in local mode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from clipforge.core.config import Settings
from clipforge.core.errors import AppError, ErrorCode
from clipforge.core.timeutil import utcnow
from clipforge.db.models import Project, User, UserSession
from clipforge.security.passwords import hash_password, needs_rehash, verify_password
from clipforge.security.tokens import hash_token, new_token
from clipforge.services.audit import audit

_EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,63}$")


@dataclass
class IssuedSession:
    token: str
    csrf_token: str
    session: UserSession


def normalize_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not _EMAIL_RE.match(email):
        raise AppError(ErrorCode.VALIDATION_ERROR, "Enter a valid email address.")
    return email


def validate_password(password: str, settings: Settings) -> None:
    if len(password or "") < settings.PASSWORD_MIN_LENGTH:
        raise AppError(ErrorCode.VALIDATION_ERROR,
                       f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters.")
    if len(password) > 1024:
        raise AppError(ErrorCode.VALIDATION_ERROR, "Password is too long.")


def ensure_default_project(session: Session, user: User) -> Project:
    project = session.scalar(select(Project).where(Project.user_id == user.id, Project.is_default.is_(True)))
    if project is None:
        project = Project(user_id=user.id, name="Default", is_default=True)
        session.add(project)
        session.flush()
    return project


class AuthService:
    def __init__(self, session: Session, settings: Settings) -> None:
        self.session = session
        self.settings = settings

    def register(self, email: str, password: str, display_name: str | None = None,
                 ip: str | None = None) -> User:
        if not self.settings.ALLOW_REGISTRATION:
            raise AppError(ErrorCode.REGISTRATION_DISABLED)
        email = normalize_email(email)
        validate_password(password, self.settings)
        if self.session.scalar(select(User).where(User.email == email)):
            raise AppError(ErrorCode.CONFLICT, "An account with this email already exists.")
        first_user = self.session.scalar(select(User.id).where(User.deleted_at.is_(None)).limit(1)) is None
        role = "admin" if (email in self.settings.ADMIN_EMAILS
                           or (first_user and self.settings.DEPLOYMENT_MODE == "local")) else "user"
        user = User(email=email, password_hash=hash_password(password),
                    display_name=(display_name or "").strip()[:120] or None,
                    role=role, plan=self.settings.DEFAULT_PLAN, settings={})
        self.session.add(user)
        self.session.flush()
        ensure_default_project(self.session, user)
        audit(self.session, "user.register", actor_user_id=user.id, target_type="user", target_id=user.id,
              ip_address=ip, metadata={"role": role})
        return user

    def authenticate(self, email: str, password: str) -> User:
        try:
            email = normalize_email(email)
        except AppError:
            email = ""
        user = self.session.scalar(select(User).where(User.email == email)) if email else None
        # verify_password runs even for unknown users to keep timing uniform.
        ok = verify_password(user.password_hash if user else None, password or "")
        if not user or not ok or not user.is_active or user.deleted_at is not None:
            raise AppError(ErrorCode.UNAUTHORIZED, "Invalid email or password.")
        if user.password_hash and needs_rehash(user.password_hash):
            user.password_hash = hash_password(password)
        user.last_login_at = utcnow()
        return user

    def issue_session(self, user: User, *, user_agent: str | None = None, ip: str | None = None,
                      rotated_from: str | None = None) -> IssuedSession:
        token = new_token()
        sess = UserSession(
            user_id=user.id,
            token_hash=hash_token(token),
            expires_at=utcnow() + timedelta(hours=self.settings.SESSION_TTL_HOURS),
            user_agent=(user_agent or "")[:255] or None,
            ip_address=ip,
            rotated_from_id=rotated_from,
        )
        self.session.add(sess)
        self.session.flush()
        return IssuedSession(token=token, csrf_token=new_token(24), session=sess)

    def resolve(self, token: str | None) -> UserSession | None:
        if not token or len(token) > 256:
            return None
        sess = self.session.scalar(select(UserSession).where(UserSession.token_hash == hash_token(token)))
        now = utcnow()
        if sess is None or sess.revoked_at is not None or sess.expires_at <= now:
            return None
        user = sess.user
        if not user.is_active or user.deleted_at is not None:
            return None
        if (now - sess.last_seen_at).total_seconds() > 300:  # throttle writes
            sess.last_seen_at = now
        return sess

    def revoke(self, sess: UserSession) -> None:
        sess.revoked_at = utcnow()

    def rotate(self, sess: UserSession, *, user_agent: str | None = None, ip: str | None = None) -> IssuedSession:
        self.revoke(sess)
        return self.issue_session(sess.user, user_agent=user_agent, ip=ip, rotated_from=sess.id)

    def revoke_all(self, user_id: str) -> None:
        self.session.execute(update(UserSession).where(
            UserSession.user_id == user_id, UserSession.revoked_at.is_(None)).values(revoked_at=utcnow()))

    def local_user(self) -> User:
        """Implicit single user for private local deployments without auth."""
        email = self.settings.LOCAL_USER_EMAIL.lower()
        user = self.session.scalar(select(User).where(User.email == email))
        if user is None:
            user = User(email=email, password_hash=None, display_name="Local user", role="admin",
                        plan=self.settings.DEFAULT_PLAN, settings={})
            self.session.add(user)
            self.session.flush()
            ensure_default_project(self.session, user)
        return user

    def change_password(self, user: User, current: str, new: str) -> None:
        if not verify_password(user.password_hash, current):
            raise AppError(ErrorCode.UNAUTHORIZED, "Current password is incorrect.")
        validate_password(new, self.settings)
        user.password_hash = hash_password(new)
        self.revoke_all(user.id)
