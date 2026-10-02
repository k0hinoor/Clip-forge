"""Queue providers."""

from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker

from clipforge.core.config import Settings
from clipforge.queue.base import QueueProvider
from clipforge.queue.db_queue import DatabaseQueue


def get_queue(settings: Settings, session_factory: sessionmaker[Session]) -> QueueProvider:
    redis_url = settings.REDIS_URL
    if redis_url:
        from clipforge.queue.redis_queue import RedisNotifyingQueue

        return RedisNotifyingQueue(session_factory, redis_url, settings.WORKER_POLL_INTERVAL_SECONDS)
    return DatabaseQueue(session_factory, settings.WORKER_POLL_INTERVAL_SECONDS)


__all__ = ["DatabaseQueue", "QueueProvider", "get_queue"]
