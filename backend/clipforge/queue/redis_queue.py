"""Optional Redis wake-up layer on top of the database queue (TRD §5, Phase 7).

The database remains the source of truth for job state and leases; Redis is
only used to wake idle workers immediately instead of polling. Requires the
``redis`` package (``pip install clipforge[redis]``).
"""

from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker

from clipforge.core.logging import get_logger
from clipforge.queue.db_queue import DatabaseQueue

log = get_logger(__name__)
_CHANNEL = "clipforge:work"


class RedisNotifyingQueue(DatabaseQueue):
    name = "redis"

    def __init__(self, session_factory: sessionmaker[Session], redis_url: str, poll_interval: float = 2.0):
        super().__init__(session_factory, poll_interval)
        try:
            import redis  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Redis queue requires the 'redis' package") from exc
        self.redis = redis.Redis.from_url(redis_url)

    def _notify(self) -> None:
        try:
            self.redis.lpush(_CHANNEL, b"1")
            self.redis.ltrim(_CHANNEL, 0, 1000)
        except Exception:  # pragma: no cover - redis outage must not break job creation
            log.warning("redis notify failed", exc_info=True)

    def enqueue(self, job_id: str) -> None:
        self._notify()

    def enqueue_render(self, render_id: str) -> None:
        self._notify()

    def wait_for_work(self, timeout: float) -> None:
        try:
            self.redis.brpop(_CHANNEL, timeout=max(1, int(min(timeout, self.poll_interval * 5))))
        except Exception:  # pragma: no cover
            super().wait_for_work(timeout)
