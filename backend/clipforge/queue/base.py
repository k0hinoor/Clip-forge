"""Queue provider interface (TRD §5, §44, §52).

The database owns job state; a queue provider only decides *which* worker
gets the next job and maintains leases (``lock_owner`` / ``lock_expires_at``).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class QueueProvider(Protocol):
    name: str

    def enqueue(self, job_id: str) -> None:
        """Signal that a job is ready (the job row itself is already persisted)."""

    def enqueue_render(self, render_id: str) -> None: ...

    def claim_job(self, worker_id: str, lease_seconds: int) -> str | None:
        """Atomically lease the next runnable job for ``worker_id``."""

    def claim_render(self, worker_id: str, lease_seconds: int) -> str | None: ...

    def renew_job(self, job_id: str, worker_id: str, lease_seconds: int) -> bool: ...

    def renew_render(self, render_id: str, worker_id: str, lease_seconds: int) -> bool: ...

    def release_job(self, job_id: str, worker_id: str) -> None: ...

    def release_render(self, render_id: str, worker_id: str) -> None: ...

    def recover_abandoned(self) -> dict[str, int]:
        """Re-queue work whose lease expired (worker crash)."""

    def queue_length(self) -> int: ...

    def wait_for_work(self, timeout: float) -> None:
        """Block until work may be available or ``timeout`` elapses."""
