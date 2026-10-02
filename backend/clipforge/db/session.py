"""Engine/session management. SQLite uses WAL mode (TRD §4)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from clipforge.core.config import Settings

_engines: dict[str, Engine] = {}


def _sqlite_pragmas(dbapi_conn, _record) -> None:  # pragma: no cover - trivial
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA busy_timeout=15000")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.close()


def create_db_engine(settings: Settings) -> Engine:
    url = settings.DATABASE_URL
    if url in _engines:
        return _engines[url]
    kwargs: dict = {"echo": settings.DATABASE_ECHO, "future": True}
    if url.startswith("sqlite"):
        db_path = url.split("///", 1)[-1]
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    else:
        kwargs.update(pool_pre_ping=True, pool_size=10, max_overflow=20)
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        event.listen(engine, "connect", _sqlite_pragmas)
    _engines[url] = engine
    return engine


def dispose_engines() -> None:
    for engine in _engines.values():
        engine.dispose()
    _engines.clear()


def make_session_factory(settings: Settings) -> sessionmaker[Session]:
    return sessionmaker(bind=create_db_engine(settings), expire_on_commit=False, autoflush=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on error."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
