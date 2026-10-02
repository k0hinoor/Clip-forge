"""Entitlements, quotas and usage metering (TRD §34, §35, §43, §55).

The processing pipeline only ever asks ``can_process(user, duration)`` and
receives an ``Entitlement``. It never knows whether the answer came from a
free plan, a subscription, credits, or an admin override.

Usage is recorded *after* a job completes. In-flight jobs are counted as
reserved minutes so concurrent submissions cannot exceed a quota, without
permanently deducting usage before the job was accepted and processed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from clipforge.core.config import Settings
from clipforge.core.errors import AppError, ErrorCode
from clipforge.core.presets import plans_config
from clipforge.core.states import IN_FLIGHT_STATES
from clipforge.core.timeutil import day_start, month_start, utcnow
from clipforge.db.models import Job, Subscription, UsageCounter, User


@dataclass(frozen=True)
class Plan:
    name: str
    daily_source_minutes: float | None
    monthly_source_minutes: float | None
    concurrent_jobs: int
    max_duration_minutes: float | None
    max_clips_per_job: int
    priority: int
    batch_enabled: bool

    @classmethod
    def from_config(cls, name: str, cfg: dict[str, Any]) -> Plan:
        return cls(
            name=name,
            daily_source_minutes=cfg.get("daily_source_minutes"),
            monthly_source_minutes=cfg.get("monthly_source_minutes"),
            concurrent_jobs=int(cfg.get("concurrent_jobs") or 1),
            max_duration_minutes=cfg.get("max_duration_minutes"),
            max_clips_per_job=int(cfg.get("max_clips_per_job") or 10),
            priority=int(cfg.get("priority") or 0),
            batch_enabled=bool(cfg.get("batch_enabled", False)),
        )


@dataclass(frozen=True)
class Entitlement:
    """What the processing engine sees (TRD §55)."""

    plan: str
    processing_minutes_remaining: float | None  # None = unlimited
    daily_minutes_remaining: float | None
    monthly_minutes_remaining: float | None
    concurrent_jobs: int
    active_jobs: int
    max_duration_seconds: float | None
    max_clips_per_job: int
    priority: int
    batch_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Decision:
    allowed: bool
    code: ErrorCode | None = None
    message: str | None = None

    def raise_if_denied(self) -> None:
        if not self.allowed:
            raise AppError(self.code or ErrorCode.QUOTA_EXCEEDED, self.message)


class UsageService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _counter(self, user_id: str, period: str, start: date) -> UsageCounter:
        row = self.session.scalar(select(UsageCounter).where(
            UsageCounter.user_id == user_id, UsageCounter.period == period, UsageCounter.period_start == start))
        if row is None:
            row = UsageCounter(user_id=user_id, period=period, period_start=start, source_seconds=0.0,
                               jobs_completed=0, clips_rendered=0)
            self.session.add(row)
            self.session.flush()
        return row

    def record_job_completed(self, job: Job, clips_rendered: int) -> None:
        """Called inside the job-completion transaction. Idempotent per job."""
        if job.usage_recorded:
            return
        seconds = float(job.source_seconds or 0.0)
        now = utcnow()
        for period, start in (("day", day_start(now)), ("month", month_start(now))):
            c = self._counter(job.user_id, period, start)
            c.source_seconds = (c.source_seconds or 0.0) + seconds
            c.jobs_completed = (c.jobs_completed or 0) + 1
            c.clips_rendered = (c.clips_rendered or 0) + clips_rendered
        job.usage_recorded = True

    def record_rerender(self, user_id: str) -> None:
        now = utcnow()
        for period, start in (("day", day_start(now)), ("month", month_start(now))):
            c = self._counter(user_id, period, start)
            c.clips_rendered = (c.clips_rendered or 0) + 1

    def used_seconds(self, user_id: str, period: str) -> float:
        start = day_start() if period == "day" else month_start()
        row = self.session.scalar(select(UsageCounter).where(
            UsageCounter.user_id == user_id, UsageCounter.period == period, UsageCounter.period_start == start))
        return float(row.source_seconds) if row else 0.0

    def reserved_seconds(self, user_id: str, exclude_job_id: str | None = None) -> float:
        q = select(func.coalesce(func.sum(Job.source_seconds), 0.0)).where(
            Job.user_id == user_id, Job.status.in_([s.value for s in IN_FLIGHT_STATES]),
            Job.deleted_at.is_(None))
        if exclude_job_id:
            q = q.where(Job.id != exclude_job_id)
        return float(self.session.scalar(q) or 0.0)

    def active_jobs(self, user_id: str, exclude_job_id: str | None = None) -> int:
        q = select(func.count()).select_from(Job).where(
            Job.user_id == user_id, Job.status.in_([s.value for s in IN_FLIGHT_STATES]),
            Job.deleted_at.is_(None))
        if exclude_job_id:
            q = q.where(Job.id != exclude_job_id)
        return int(self.session.scalar(q) or 0)

    def summary(self, user_id: str) -> dict[str, Any]:
        return {
            "today_source_minutes": round(self.used_seconds(user_id, "day") / 60, 2),
            "month_source_minutes": round(self.used_seconds(user_id, "month") / 60, 2),
            "reserved_source_minutes": round(self.reserved_seconds(user_id) / 60, 2),
            "active_jobs": self.active_jobs(user_id),
        }


class EntitlementService:
    def __init__(self, session: Session, settings: Settings) -> None:
        self.session = session
        self.settings = settings
        self.usage = UsageService(session)

    def plans(self) -> dict[str, Plan]:
        return {name: Plan.from_config(name, cfg) for name, cfg in plans_config(self.settings).items()}

    def plan_for(self, user: User) -> Plan:
        plans = self.plans()
        name = user.plan or self.settings.DEFAULT_PLAN
        sub = self.session.scalar(select(Subscription).where(
            Subscription.user_id == user.id, Subscription.status == "active"
        ).order_by(Subscription.created_at.desc()))
        if sub and sub.plan in plans and (sub.current_period_end is None or sub.current_period_end > utcnow()):
            name = sub.plan
        plan = plans.get(name) or plans.get(self.settings.DEFAULT_PLAN) or next(iter(plans.values()))
        override = user.quota_override or {}
        if override:
            merged = {**asdict(plan), **{k: v for k, v in override.items() if k in asdict(plan)}}
            merged.pop("name", None)
            plan = Plan(name=plan.name, **merged)
        return plan

    def entitlement(self, user: User, *, exclude_job_id: str | None = None) -> Entitlement:
        plan = self.plan_for(user)
        reserved = self.usage.reserved_seconds(user.id, exclude_job_id) / 60
        daily_left = monthly_left = None
        if plan.daily_source_minutes is not None:
            daily_left = max(0.0, plan.daily_source_minutes - self.usage.used_seconds(user.id, "day") / 60 - reserved)
        if plan.monthly_source_minutes is not None:
            monthly_left = max(0.0, plan.monthly_source_minutes
                               - self.usage.used_seconds(user.id, "month") / 60 - reserved)
        remaining_candidates = [v for v in (daily_left, monthly_left) if v is not None]
        return Entitlement(
            plan=plan.name,
            processing_minutes_remaining=round(min(remaining_candidates), 2) if remaining_candidates else None,
            daily_minutes_remaining=None if daily_left is None else round(daily_left, 2),
            monthly_minutes_remaining=None if monthly_left is None else round(monthly_left, 2),
            concurrent_jobs=plan.concurrent_jobs,
            active_jobs=self.usage.active_jobs(user.id, exclude_job_id),
            max_duration_seconds=None if plan.max_duration_minutes is None else plan.max_duration_minutes * 60,
            max_clips_per_job=min(plan.max_clips_per_job, self.settings.MAX_CLIPS_PER_JOB),
            priority=plan.priority,
            batch_enabled=plan.batch_enabled,
        )

    def can_process(self, user: User, duration_seconds: float, *, exclude_job_id: str | None = None) -> Decision:
        ent = self.entitlement(user, exclude_job_id=exclude_job_id)
        if duration_seconds > self.settings.MAX_VIDEO_SECONDS:
            return Decision(False, ErrorCode.MEDIA_TOO_LONG)
        if ent.max_duration_seconds is not None and duration_seconds > ent.max_duration_seconds:
            return Decision(False, ErrorCode.MEDIA_TOO_LONG,
                            f"Your plan allows videos up to {int(ent.max_duration_seconds // 60)} minutes.")
        if ent.active_jobs >= ent.concurrent_jobs:
            return Decision(False, ErrorCode.QUOTA_EXCEEDED,
                            f"Your plan allows {ent.concurrent_jobs} concurrent job(s). "
                            "Wait for a running job to finish.")
        needed = duration_seconds / 60
        if ent.processing_minutes_remaining is not None and needed > ent.processing_minutes_remaining + 1e-6:
            return Decision(False, ErrorCode.QUOTA_EXCEEDED,
                            f"This video needs {needed:.1f} processing minutes but you have "
                            f"{ent.processing_minutes_remaining:.1f} left in this period.")
        return Decision(True)
