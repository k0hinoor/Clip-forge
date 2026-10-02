"""Idempotency-Key support for upload/job creation (TRD §42)."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from clipforge.core.errors import AppError, ErrorCode
from clipforge.db.models import IdempotencyKey

_KEY_RE = re.compile(r"^[A-Za-z0-9_\-:.]{8,100}$")


def validate_key(key: str | None) -> str | None:
    if key is None or key == "":
        return None
    if not _KEY_RE.match(key):
        raise AppError(ErrorCode.VALIDATION_ERROR, "Idempotency-Key must be 8-100 URL-safe characters.")
    return key


def request_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def lookup(session: Session, user_id: str, scope: str, key: str | None, req_hash: str | None) -> str | None:
    if not key:
        return None
    row = session.scalar(select(IdempotencyKey).where(
        IdempotencyKey.user_id == user_id, IdempotencyKey.scope == scope, IdempotencyKey.key == key))
    if row is None:
        return None
    if req_hash and row.request_hash and row.request_hash != req_hash:
        raise AppError(ErrorCode.IDEMPOTENCY_CONFLICT)
    return row.resource_id


def store(session: Session, user_id: str, scope: str, key: str | None, req_hash: str | None,
          resource_type: str, resource_id: str) -> None:
    if not key:
        return
    session.add(IdempotencyKey(user_id=user_id, scope=scope, key=key, request_hash=req_hash,
                               resource_type=resource_type, resource_id=resource_id))
