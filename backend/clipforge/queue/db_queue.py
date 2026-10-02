"""Database-backed queue with leases — no extra infrastructure for local mode.

Claims use a conditional UPDATE (compare-and-set on the lease columns) so they
are safe with multiple worker processes on SQLite and PostgreSQL alike.
"""

from __future__ import annotations

import time
from datetime import timedelta

from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import Session, sessionmaker

from clipforge.core.logging import get_logger
from clipforge.core.states import ACTIVE_STATES, JobStatus, RenderStatus
from clipforge.core.timeutil import utcnow
from clipforge.db.models import Job, JobEvent, Render
from clipforge.db.session import session_scope

log = get_logger(__name__)


class DatabaseQueue:
    name = "database"

    def __init__(self, session_factory: sessionmaker[Session], poll_interval: float = 2.0) -> None:
        self.session_factory = session_factory
        self.poll_interval = poll_interval

    # ----------------------------------------------------------- enqueueing
    def enqueue(self, job_id: str) -> None:  # job row is the queue entry
        return None

    def enqueue_render(self, render_id: str) -> None:
        return None

    # -------------------------------------------------------------- claiming
    def claim_job(self, worker_id: str, lease_seconds: int) -> str | None:
        now = utcnow()
        with session_scope(self.session_factory) as s:
            ids = s.scalars(
                select(Job.id)
                .where(
                    Job.status == JobStatus.QUEUED.value,
                    Job.deleted_at.is_(None),
                    Job.cancel_requested.is_(False),
                    or_(Job.next_attempt_at.is_(None), Job.next_attempt_at <= now),
                    or_(Job.lock_expires_at.is_(None), Job.lock_expires_at < now),
                )
                .order_by(Job.priority.desc(), Job.created_at.asc())
                .limit(10)
            ).all()
            for job_id in ids:
                res = s.execute(
                    update(Job)
                    .where(
                        Job.id == job_id,
                        Job.status == JobStatus.QUEUED.value,
                        or_(Job.lock_expires_at.is_(None), Job.lock_expires_at < now),
                    )
                    .values(lock_owner=worker_id, lock_expires_at=now + timedelta(seconds=lease_seconds))
                    .execution_options(synchronize_session=False)
                )
                if res.rowcount == 1:
                    return job_id
        return None

    def claim_render(self, worker_id: str, lease_seconds: int) -> str | None:
        now = utcnow()
        with session_scope(self.session_factory) as s:
            ids = s.scalars(
                select(Render.id)
                .where(
                    Render.status == RenderStatus.QUEUED.value,
                    Render.source == "user",
                    or_(Render.lock_expires_at.is_(None), Render.lock_expires_at < now),
                )
                .order_by(Render.created_at.asc())
                .limit(10)
            ).all()
            for render_id in ids:
                res = s.execute(
                    update(Render)
                    .where(
                        Render.id == render_id,
                        Render.status == RenderStatus.QUEUED.value,
                        or_(Render.lock_expires_at.is_(None), Render.lock_expires_at < now),
                    )
                    .values(lock_owner=worker_id, lock_expires_at=now + timedelta(seconds=lease_seconds))
                    .execution_options(synchronize_session=False)
                )
                if res.rowcount == 1:
                    return render_id
        return None

    # ---------------------------------------------------------------- leases
    def renew_job(self, job_id: str, worker_id: str, lease_seconds: int) -> bool:
        with session_scope(self.session_factory) as s:
            res = s.execute(
                update(Job)
                .where(Job.id == job_id, Job.lock_owner == worker_id)
                .values(lock_expires_at=utcnow() + timedelta(seconds=lease_seconds))
                .execution_options(synchronize_session=False)
            )
            return res.rowcount == 1

    def renew_render(self, render_id: str, worker_id: str, lease_seconds: int) -> bool:
        with session_scope(self.session_factory) as s:
            res = s.execute(
                update(Render)
                .where(Render.id == render_id, Render.lock_owner == worker_id)
                .values(lock_expires_at=utcnow() + timedelta(seconds=lease_seconds))
                .execution_options(synchronize_session=False)
            )
            return res.rowcount == 1

    def release_job(self, job_id: str, worker_id: str) -> None:
        with session_scope(self.session_factory) as s:
            s.execute(
                update(Job)
                .where(Job.id == job_id, Job.lock_owner == worker_id)
                .values(lock_owner=None, lock_expires_at=None)
                .execution_options(synchronize_session=False)
            )

    def release_render(self, render_id: str, worker_id: str) -> None:
        with session_scope(self.session_factory) as s:
            s.execute(
                update(Render)
                .where(Render.id == render_id, Render.lock_owner == worker_id)
                .values(lock_owner=None, lock_expires_at=None)
                .execution_options(synchronize_session=False)
            )

    # -------------------------------------------------------------- recovery
    def recover_abandoned(self) -> dict[str, int]:
        """TRD §45: find active jobs whose worker lease expired and re-queue them.

        The job keeps ``current_stage`` and its artifact manifest, so the next
        worker resumes from the last valid stage instead of starting over.
        """
        now = utcnow()
        recovered_jobs = 0
        recovered_renders = 0
        with session_scope(self.session_factory) as s:
            stale = s.scalars(
                select(Job).where(
                    Job.status.in_([st.value for st in ACTIVE_STATES]),
                    or_(Job.lock_expires_at.is_(None), Job.lock_expires_at < now),
                )
            ).all()
            for job in stale:
                previous = job.status
                job.status = JobStatus.QUEUED.value
                job.lock_owner = None
                job.lock_expires_at = None
                job.next_attempt_at = now
                s.add(JobEvent(job_id=job.id, event_type="job.recovered", stage=job.current_stage,
                               status=JobStatus.QUEUED.value, progress=job.progress,
                               message=f"Recovered after worker interruption during {previous}"))
                recovered_jobs += 1
            stale_renders = s.scalars(
                select(Render).where(
                    Render.status.in_([RenderStatus.RENDERING.value, RenderStatus.QA.value]),
                    Render.source == "user",
                    or_(Render.lock_expires_at.is_(None), Render.lock_expires_at < now),
                )
            ).all()
            for render in stale_renders:
                render.status = RenderStatus.QUEUED.value
                render.lock_owner = None
                render.lock_expires_at = None
                recovered_renders += 1
        if recovered_jobs or recovered_renders:
            log.warning("recovered abandoned work",
                        extra={"jobs": recovered_jobs, "renders": recovered_renders})
        return {"jobs": recovered_jobs, "renders": recovered_renders}

    def queue_length(self) -> int:
        with session_scope(self.session_factory) as s:
            return int(s.scalar(select(func.count()).select_from(Job).where(
                Job.status == JobStatus.QUEUED.value, Job.deleted_at.is_(None))) or 0)

    def wait_for_work(self, timeout: float) -> None:
        time.sleep(min(timeout, self.poll_interval))
