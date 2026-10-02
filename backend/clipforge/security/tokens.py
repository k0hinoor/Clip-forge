"""Opaque session tokens, CSRF tokens and HMAC-signed resource URLs."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time


def new_token(nbytes: int = 32) -> str:
    return secrets.token_urlsafe(nbytes)


def hash_token(token: str) -> str:
    """Session tokens are stored hashed so a DB leak does not leak sessions."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def constant_time_equals(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return hmac.compare_digest(a.encode(), b.encode())


def _sign(secret: str, message: str) -> str:
    return hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()


def sign_resource(secret: str, resource: str, resource_id: str, variant: str, ttl_seconds: int,
                  now: float | None = None) -> tuple[int, str]:
    """Return (expires, signature) for an access-controlled media URL.

    The signature binds the resource type, id and variant (e.g. ``download`` or
    ``thumbnail``) so a URL cannot be replayed against other resources.
    """
    expires = int((now or time.time()) + ttl_seconds)
    return expires, _sign(secret, f"{resource}:{resource_id}:{variant}:{expires}")


def verify_resource_signature(secret: str, resource: str, resource_id: str, variant: str,
                              expires: int, signature: str, now: float | None = None) -> bool:
    if expires < int(now or time.time()):
        return False
    expected = _sign(secret, f"{resource}:{resource_id}:{variant}:{expires}")
    return constant_time_equals(expected, signature)
