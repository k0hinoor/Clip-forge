"""Billing provider abstraction (TRD §35).

Billing is fully decoupled from processing: providers only create/update
``subscriptions`` and ``payments`` rows; the EntitlementService reads them.
Local mode uses ``NullBillingProvider`` (no payment provider, zero cost).
"""

from __future__ import annotations

from typing import Any, Protocol

from sqlalchemy.orm import Session

from clipforge.core.config import Settings
from clipforge.core.errors import AppError, ErrorCode
from clipforge.db.models import User


class BillingProvider(Protocol):
    name: str
    enabled: bool

    def create_checkout(self, session: Session, user: User, plan: str) -> dict[str, Any]: ...

    def handle_webhook(self, session: Session, payload: bytes, headers: dict[str, str]) -> dict[str, Any]: ...

    def cancel_subscription(self, session: Session, user: User) -> None: ...


class NullBillingProvider:
    name = "none"
    enabled = False

    def create_checkout(self, session: Session, user: User, plan: str) -> dict[str, Any]:
        raise AppError(ErrorCode.NOT_IMPLEMENTED, "Billing is not enabled on this server.")

    def handle_webhook(self, session: Session, payload: bytes, headers: dict[str, str]) -> dict[str, Any]:
        raise AppError(ErrorCode.NOT_IMPLEMENTED, "Billing is not enabled on this server.")

    def cancel_subscription(self, session: Session, user: User) -> None:
        return None


def get_billing_provider(settings: Settings) -> BillingProvider:
    # Future providers (e.g. Stripe, Paddle, LemonSqueezy) plug in here without
    # touching worker or pipeline code.
    return NullBillingProvider()
