"""Time helpers.

All timestamps are stored as *naive UTC* datetimes so SQLite and PostgreSQL
behave identically; API serialisation appends ``Z``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def to_naive_utc(value: datetime) -> datetime:
    if value.tzinfo is not None:
        value = value.astimezone(UTC).replace(tzinfo=None)
    return value


def iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return to_naive_utc(value).isoformat(timespec="milliseconds") + "Z"


def day_start(now: datetime | None = None) -> date:
    return (now or utcnow()).date()


def month_start(now: datetime | None = None) -> date:
    d = (now or utcnow()).date()
    return d.replace(day=1)


def hours_from_now(hours: float) -> datetime:
    return utcnow() + timedelta(hours=hours)
