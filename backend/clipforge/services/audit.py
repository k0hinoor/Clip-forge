"""Audit logs for privileged / security-relevant actions (TRD §40)."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from clipforge.db.models import AuditLog


def audit(
    session: Session,
    action: str,
    *,
    actor_user_id: str | None,
    target_type: str | None = None,
    target_id: str | None = None,
    ip_address: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    session.add(AuditLog(
        actor_user_id=actor_user_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        ip_address=ip_address,
        metadata_json=metadata or None,
    ))
