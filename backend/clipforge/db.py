"""SQLite persistence.

Single-file database (WAL mode) holding projects, transcripts, discovered
candidates, clips, the job queue, the asset library and user settings. The
schema is additive-migratable: ``init_db()`` creates what is missing and adds
new columns to existing tables, so users never lose their projects on upgrade.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    event,
    select,
    text as sql_text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from .config import Env, ensure_dirs

# --------------------------------------------------------------------------- #
# Naming helpers
# --------------------------------------------------------------------------- #


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def slugify(value: str, *, max_length: int = 60) -> str:
    """Filesystem-safe slug: no traversal, no reserved characters, no surprises."""
    cleaned = []
    for char in (value or "").lower():
        if char.isalnum():
            cleaned.append(char)
        elif char in " -_":
            cleaned.append("-")
    slug = "".join(cleaned)
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug.strip("-.") or "untitled"
    return slug[:max_length].strip("-.") or "untitled"


# --------------------------------------------------------------------------- #
# ORM base
# --------------------------------------------------------------------------- #


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("prj"))
    slug: Mapped[str] = mapped_column(String(80), default="untitled")
    title: Mapped[str] = mapped_column(String(400), default="Untitled project")
    channel: Mapped[str] = mapped_column(String(200), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    source_url: Mapped[str] = mapped_column(Text, default="")
    source_type: Mapped[str] = mapped_column(String(20), default="youtube")  # youtube | url | upload
    source_id: Mapped[str] = mapped_column(String(80), default="")            # youtube id / link hash
    thumbnail_url: Mapped[str] = mapped_column(Text, default="")
    thumbnail_path: Mapped[str] = mapped_column(Text, default="")
    duration: Mapped[float] = mapped_column(Float, default=0.0)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    fps: Mapped[float] = mapped_column(Float, default=0.0)

    status: Mapped[str] = mapped_column(String(20), default="draft")   # draft|queued|running|ready|failed|cancelled
    stage: Mapped[str] = mapped_column(String(40), default="")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    status_message: Mapped[str] = mapped_column(Text, default="")

    language: Mapped[str] = mapped_column(String(20), default="")
    language_secondary: Mapped[str] = mapped_column(String(20), default="")
    language_mode: Mapped[str] = mapped_column(String(20), default="")  # monolingual|mixed
    language_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    speakers: Mapped[int] = mapped_column(Integer, default=0)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    segment_count: Mapped[int] = mapped_column(Integer, default=0)
    candidate_count: Mapped[int] = mapped_column(Integer, default=0)
    clip_count: Mapped[int] = mapped_column(Integer, default=0)

    settings_json: Mapped[str] = mapped_column(Text, default="{}")   # analysis options for this project
    media_json: Mapped[str] = mapped_column(Text, default="{}")      # ffprobe summary
    paths_json: Mapped[str] = mapped_column(Text, default="{}")      # local file layout
    stats_json: Mapped[str] = mapped_column(Text, default="{}")      # timing/analysis telemetry
    error_code: Mapped[str] = mapped_column(String(60), default="")
    error_message: Mapped[str] = mapped_column(Text, default="")
    error_hint: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    analyzed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    clips: Mapped[list["Clip"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    segments: Mapped[list["TranscriptSegment"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    candidates: Mapped[list["Candidate"]] = relationship(back_populates="project", cascade="all, delete-orphan")

    # ------------------------------------------------------------- helpers
    @property
    def settings(self) -> dict[str, Any]:
        try:
            return json.loads(self.settings_json or "{}")
        except json.JSONDecodeError:
            return {}

    @property
    def paths(self) -> dict[str, str]:
        try:
            return json.loads(self.paths_json or "{}")
        except json.JSONDecodeError:
            return {}

    @property
    def media(self) -> dict[str, Any]:
        try:
            return json.loads(self.media_json or "{}")
        except json.JSONDecodeError:
            return {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "slug": self.slug,
            "title": self.title,
            "channel": self.channel,
            "description": self.description,
            "source_url": self.source_url,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "thumbnail_url": self.thumbnail_url,
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 3),
            "status_message": self.status_message,
            "language": self.language,
            "language_secondary": self.language_secondary,
            "language_mode": self.language_mode,
            "language_confidence": self.language_confidence,
            "speakers": self.speakers,
            "word_count": self.word_count,
            "segment_count": self.segment_count,
            "candidate_count": self.candidate_count,
            "clip_count": self.clip_count,
            "settings": self.settings,
            "media": self.media,
            "paths": self.paths,
            "stats": json.loads(self.stats_json or "{}"),
            "error": (
                {"code": self.error_code, "message": self.error_message, "hint": self.error_hint}
                if self.error_code
                else None
            ),
            "created_at": iso(self.created_at),
            "updated_at": iso(self.updated_at),
            "analyzed_at": iso(self.analyzed_at),
        }


class TranscriptSegment(Base):
    """One utterance/sentence, with its word-level timings embedded as JSON."""

    __tablename__ = "transcript_segments"
    __table_args__ = (Index("ix_segments_project_start", "project_id", "start"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("seg"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    idx: Mapped[int] = mapped_column(Integer, default=0)
    start: Mapped[float] = mapped_column(Float, default=0.0)
    end: Mapped[float] = mapped_column(Float, default=0.0)
    text: Mapped[str] = mapped_column(Text, default="")
    speaker: Mapped[str] = mapped_column(String(40), default="SPEAKER_01")
    language: Mapped[str] = mapped_column(String(12), default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    words_json: Mapped[str] = mapped_column(Text, default="[]")

    project: Mapped[Project] = relationship(back_populates="segments")

    @property
    def words(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.words_json or "[]")
        except json.JSONDecodeError:
            return []

    def to_dict(self, *, with_words: bool = True) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "index": self.idx,
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "speaker": self.speaker,
            "language": self.language,
            "confidence": round(self.confidence, 3),
            "word_count": len(self.words),
        }
        if with_words:
            payload["words"] = self.words
        return payload


class Candidate(Base):
    """A candidate moment found during analysis (may or may not become a clip).

    Keeping rejected and duplicate candidates in the database is deliberate: the
    UI shows what was discovered and why it was dropped, and it makes the
    analysis auditable instead of a black box.
    """

    __tablename__ = "candidates"
    __table_args__ = (Index("ix_candidates_project_score", "project_id", "score"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("cand"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    idx: Mapped[int] = mapped_column(Integer, default=0)
    start: Mapped[float] = mapped_column(Float, default=0.0)
    end: Mapped[float] = mapped_column(Float, default=0.0)
    duration: Mapped[float] = mapped_column(Float, default=0.0)
    title: Mapped[str] = mapped_column(String(300), default="")
    hook: Mapped[str] = mapped_column(Text, default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    category: Mapped[str] = mapped_column(String(60), default="other")
    reason: Mapped[str] = mapped_column(Text, default="")
    score: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    factors_json: Mapped[str] = mapped_column(Text, default="{}")
    penalties_json: Mapped[str] = mapped_column(Text, default="{}")
    highlights_json: Mapped[str] = mapped_column(Text, default="[]")
    transcript_text: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(20), default="heuristic")  # heuristic|llm|merged
    status: Mapped[str] = mapped_column(String(20), default="candidate")  # candidate|kept|duplicate|rejected
    duplicate_of: Mapped[str] = mapped_column(String(40), default="")
    boundary_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    project: Mapped[Project] = relationship(back_populates="candidates")

    def to_dict(self) -> dict[str, Any]:
        def load(raw: str, fallback: Any) -> Any:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return fallback

        return {
            "id": self.id,
            "index": self.idx,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 2),
            "title": self.title,
            "hook": self.hook,
            "summary": self.summary,
            "category": self.category,
            "reason": self.reason,
            "score": round(self.score, 1),
            "confidence": round(self.confidence, 3),
            "factors": load(self.factors_json, {}),
            "penalties": load(self.penalties_json, {}),
            "why": load(self.highlights_json, []),
            "source": self.source,
            "status": self.status,
            "duplicate_of": self.duplicate_of,
            "boundary": load(self.boundary_json, {}),
        }


class Clip(Base):
    """A clip the user can preview, edit, render and export."""

    __tablename__ = "clips"
    __table_args__ = (Index("ix_clips_project_start", "project_id", "start"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("clip"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    candidate_id: Mapped[str] = mapped_column(String(40), default="")
    index: Mapped[int] = mapped_column(Integer, default=0)

    title: Mapped[str] = mapped_column(String(300), default="")
    hook: Mapped[str] = mapped_column(Text, default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    category: Mapped[str] = mapped_column(String(60), default="other")
    score: Mapped[float] = mapped_column(Float, default=0.0)
    why_json: Mapped[str] = mapped_column(Text, default="[]")
    factors_json: Mapped[str] = mapped_column(Text, default="{}")

    start: Mapped[float] = mapped_column(Float, default=0.0)
    end: Mapped[float] = mapped_column(Float, default=0.0)
    duration: Mapped[float] = mapped_column(Float, default=0.0)
    words_json: Mapped[str] = mapped_column(Text, default="[]")   # captions source of truth
    segments_json: Mapped[str] = mapped_column(Text, default="[]")
    trim_json: Mapped[str] = mapped_column(Text, default="{}")     # silence removal + speed edits applied
    layout_json: Mapped[str] = mapped_column(Text, default="{}")   # composition plan used for the render

    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending|queued|rendering|rendered|failed|cancelled
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    stage: Mapped[str] = mapped_column(String(40), default="")
    file_path: Mapped[str] = mapped_column(Text, default="")
    preview_path: Mapped[str] = mapped_column(Text, default="")
    thumb_path: Mapped[str] = mapped_column(Text, default="")
    subtitle_path: Mapped[str] = mapped_column(Text, default="")
    timeline_json: Mapped[str] = mapped_column(Text, default="{}")
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    fps: Mapped[int] = mapped_column(Integer, default=0)
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    render_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    error_code: Mapped[str] = mapped_column(String(60), default="")
    error_message: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    rendered_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    project: Mapped[Project] = relationship(back_populates="clips")

    @property
    def why(self) -> list[str]:
        try:
            return json.loads(self.why_json or "[]")
        except json.JSONDecodeError:
            return []

    def to_dict(self, *, include_words: bool = False) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "project_id": self.project_id,
            "index": self.index,
            "title": self.title,
            "hook": self.hook,
            "summary": self.summary,
            "category": self.category,
            "category_label": _category_label(self.category),
            "score": round(self.score, 1),
            "why": self.why,
            "factors": json.loads(self.factors_json or "{}"),
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 2),
            "status": self.status,
            "progress": round(self.progress, 3),
            "stage": self.stage,
            "has_render": bool(self.file_path),
            "has_preview": bool(self.preview_path),
            "has_thumbnail": bool(self.thumb_path),
            "file_size": self.file_size,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "render_seconds": round(self.render_seconds, 2),
            "layout": json.loads(self.layout_json or "{}"),
            "trim": json.loads(self.trim_json or "{}"),
            "error": {"code": self.error_code, "message": self.error_message} if self.error_code else None,
            "created_at": iso(self.created_at),
            "rendered_at": iso(self.rendered_at),
            "preview_url": f"/api/clips/{self.id}/preview" if self.preview_path else None,
            "render_url": f"/api/clips/{self.id}/file" if self.file_path else None,
            "thumbnail_url": f"/api/clips/{self.id}/thumbnail" if self.thumb_path else None,
            "subtitle_url": f"/api/clips/{self.id}/subtitles" if self.subtitle_path else None,
        }
        if include_words:
            try:
                payload["words"] = json.loads(self.words_json or "[]")
            except json.JSONDecodeError:
                payload["words"] = []
        return payload


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_status_created", "status", "created_at"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("job"))
    project_id: Mapped[str] = mapped_column(String(40), default="", index=True)
    clip_id: Mapped[str] = mapped_column(String(40), default="", index=True)
    kind: Mapped[str] = mapped_column(String(30), default="analyze")  # analyze|render_clip|render_all|import
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued|running|succeeded|failed|cancelled
    priority: Mapped[int] = mapped_column(Integer, default=5)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    stage: Mapped[str] = mapped_column(String(40), default="")
    message: Mapped[str] = mapped_column(Text, default="")
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    result_json: Mapped[str] = mapped_column(Text, default="{}")
    log_json: Mapped[str] = mapped_column(Text, default="[]")
    error_code: Mapped[str] = mapped_column(String(60), default="")
    error_message: Mapped[str] = mapped_column(Text, default="")
    error_hint: Mapped[str] = mapped_column(Text, default="")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    worker: Mapped[str] = mapped_column(String(60), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    def payload(self) -> dict[str, Any]:
        try:
            return json.loads(self.payload_json or "{}")
        except json.JSONDecodeError:
            return {}

    def stage_log(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.log_json or "[]")
        except json.JSONDecodeError:
            return []

    def to_dict(self, *, with_log: bool = True) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "project_id": self.project_id,
            "clip_id": self.clip_id,
            "kind": self.kind,
            "status": self.status,
            "priority": self.priority,
            "progress": round(self.progress, 4),
            "stage": self.stage,
            "message": self.message,
            "attempts": self.attempts,
            "worker": self.worker,
            "cancel_requested": bool(self.cancel_requested),
            "created_at": iso(self.created_at),
            "started_at": iso(self.started_at),
            "finished_at": iso(self.finished_at),
            "duration_seconds": duration_between(self.started_at, self.finished_at),
            "error": (
                {"code": self.error_code, "message": self.error_message, "hint": self.error_hint}
                if self.error_code
                else None
            ),
        }
        if with_log:
            payload["stages"] = self.stage_log()
        return payload


class Asset(Base):
    """User-imported gameplay / B-roll / music."""

    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("asset"))
    kind: Mapped[str] = mapped_column(String(20), default="gameplay")  # gameplay|broll|music
    category: Mapped[str] = mapped_column(String(60), default="general")
    name: Mapped[str] = mapped_column(String(200), default="")
    filename: Mapped[str] = mapped_column(String(260), default="")
    path: Mapped[str] = mapped_column(Text, default="")
    duration: Mapped[float] = mapped_column(Float, default=0.0)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    tags_json: Mapped[str] = mapped_column(Text, default="[]")
    favorite: Mapped[bool] = mapped_column(Boolean, default=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "category": self.category,
            "name": self.name,
            "filename": self.filename,
            "path": self.path,
            "duration": round(self.duration, 2),
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "size_bytes": self.size_bytes,
            "tags": json.loads(self.tags_json or "[]"),
            "favorite": bool(self.favorite),
            "enabled": bool(self.enabled),
            "created_at": iso(self.created_at),
            "stream_url": f"/api/assets/{self.id}/file",
        }


class Template(Base):
    """Saved render preset (caption theme + layout + audio recipe)."""

    __tablename__ = "templates"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("tpl"))
    name: Mapped[str] = mapped_column(String(120), default="Untitled template")
    description: Mapped[str] = mapped_column(Text, default="")
    builtin: Mapped[bool] = mapped_column(Boolean, default=False)
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "builtin": bool(self.builtin),
            "config": json.loads(self.config_json or "{}"),
            "created_at": iso(self.created_at),
        }


class KeyValue(Base):
    __tablename__ = "kv"

    key: Mapped[str] = mapped_column(String(80), primary_key=True)
    value_json: Mapped[str] = mapped_column(Text, default="{}")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


# --------------------------------------------------------------------------- #
# Engine / sessions
# --------------------------------------------------------------------------- #

_ENGINE: Engine | None = None
_SESSION_FACTORY: sessionmaker[Session] | None = None


def _category_label(key: str) -> str:
    from .constants import category_label  # constants is a leaf module

    return category_label(key or "")


def iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def duration_between(start: datetime | None, end: datetime | None) -> float | None:
    if not start or not end:
        return None
    delta = (end - start).total_seconds()
    return round(delta, 2)


def engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        ensure_dirs()
        url = f"sqlite:///{Env.DB_PATH}"
        _ENGINE = create_engine(
            url,
            echo=False,
            future=True,
            connect_args={"check_same_thread": False, "timeout": 30},
        )

        @event.listens_for(_ENGINE, "connect")
        def _pragmas(dbapi_connection, _record) -> None:  # pragma: no cover - driver hook
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

    return _ENGINE


def session_factory() -> sessionmaker[Session]:
    global _SESSION_FACTORY
    if _SESSION_FACTORY is None:
        _SESSION_FACTORY = sessionmaker(bind=engine(), expire_on_commit=False, future=True)
    return _SESSION_FACTORY


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on error."""
    session = session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def db() -> Iterator[Session]:
    """FastAPI dependency."""
    session = session_factory()()
    try:
        yield session
    finally:
        session.close()


# --------------------------------------------------------------------------- #
# Schema management
# --------------------------------------------------------------------------- #

_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    # table -> {column: DDL type}. Used to upgrade existing local databases.
    "projects": {
        "stats_json": "TEXT DEFAULT '{}'",
        "candidate_count": "INTEGER DEFAULT 0",
        "language_mode": "TEXT DEFAULT ''",
        "language_secondary": "TEXT DEFAULT ''",
    },
    "clips": {"timeline_json": "TEXT DEFAULT '{}'", "preview_path": "TEXT DEFAULT ''"},
    "jobs": {"worker": "TEXT DEFAULT ''"},
}


def init_db() -> None:
    """Create the database, then apply additive upgrades."""
    ensure_dirs()
    Base.metadata.create_all(engine())
    _apply_additive_migrations()
    seed_builtin_templates()


def _apply_additive_migrations() -> None:
    with engine().begin() as connection:
        for table, columns in _ADDED_COLUMNS.items():
            existing = {
                row[1] for row in connection.execute(sql_text(f'PRAGMA table_info("{table}")')).fetchall()
            }
            if not existing:
                continue  # table does not exist yet; create_all will handle it
            for column, ddl in columns.items():
                if column not in existing:
                    connection.execute(sql_text(f'ALTER TABLE "{table}" ADD COLUMN {column} {ddl}'))


# --------------------------------------------------------------------------- #
# Key/value helpers (settings store)
# --------------------------------------------------------------------------- #


def kv_get(key: str, default: Any = None) -> Any:
    with session_scope() as session:
        row = session.get(KeyValue, key)
        if row is None:
            return default
        try:
            return json.loads(row.value_json)
        except json.JSONDecodeError:
            return default


def kv_set(key: str, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False)
    with session_scope() as session:
        row = session.get(KeyValue, key)
        if row is None:
            session.add(KeyValue(key=key, value_json=payload))
        else:
            row.value_json = payload
            row.updated_at = utcnow()


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #

BUILTIN_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "tpl_bold_creator",
        "name": "Bold Creator",
        "description": "Big uppercase word-by-word captions, split-screen gameplay, punch-ins on emphasis.",
        "config": {
            "aspect_ratio": "9:16",
            "layout": "split",
            "split_ratio": 65,
            "caption": {"preset": "bold_creator", "font_size": 68, "highlight_color": "#FFD400", "animation": "word_by_word", "max_words_per_line": 3},
            "gameplay_enabled": True,
            "auto_zoom": True,
        },
    },
    {
        "id": "tpl_cinematic",
        "name": "Cinematic",
        "description": "Soft serif captions, blurred background behind the speaker, gentle zoom.",
        "config": {
            "aspect_ratio": "9:16",
            "layout": "blur",
            "caption": {"preset": "cinematic", "font_size": 54, "position": "lower-middle", "animation": "pop", "max_words_per_line": 5},
            "auto_zoom": True,
            "gameplay_enabled": False,
        },
    },
    {
        "id": "tpl_karaoke",
        "name": "Karaoke",
        "description": "Line-by-line karaoke highlighting synced to word timings.",
        "config": {
            "aspect_ratio": "9:16",
            "layout": "podcast",
            "caption": {"preset": "karaoke", "animation": "karaoke", "max_words_per_line": 4},
            "gameplay_enabled": False,
        },
    },
    {
        "id": "tpl_documentary",
        "name": "Documentary",
        "description": "Lower-third subtitles, minimal movement, no gameplay.",
        "config": {
            "aspect_ratio": "16:9",
            "layout": "cinematic",
            "caption": {"preset": "documentary", "font_size": 44, "max_words_per_line": 6, "uppercase": False},
            "gameplay_enabled": False,
            "auto_zoom": False,
        },
    },
    {
        "id": "tpl_podcast_split",
        "name": "Podcast Split",
        "description": "60/40 split with satisfying gameplay and highlight captions.",
        "config": {
            "aspect_ratio": "9:16",
            "layout": "split",
            "split_ratio": 60,
            "caption": {"preset": "highlight", "animation": "word_by_word"},
            "gameplay_enabled": True,
            "gameplay_mode": "auto",
        },
    },
    {
        "id": "tpl_minimal",
        "name": "Minimal",
        "description": "Clean single-line captions, full-frame speaker, no effects.",
        "config": {
            "aspect_ratio": "9:16",
            "layout": "podcast",
            "caption": {"preset": "minimal", "font_size": 48, "animation": "none", "max_words_per_line": 6, "max_lines": 1},
            "auto_zoom": False,
            "gameplay_enabled": False,
        },
    },
]


def seed_builtin_templates() -> None:
    with session_scope() as session:
        existing = {row[0] for row in session.execute(select(Template.id)).fetchall()}
        for template in BUILTIN_TEMPLATES:
            if template["id"] in existing:
                row = session.get(Template, template["id"])
                if row is not None:
                    row.name = template["name"]
                    row.description = template["description"]
                    row.config_json = json.dumps(template["config"])
                    row.builtin = True
                continue
            session.add(
                Template(
                    id=template["id"],
                    name=template["name"],
                    description=template["description"],
                    builtin=True,
                    config_json=json.dumps(template["config"]),
                )
            )


# --------------------------------------------------------------------------- #
# Log helper for pipeline stages (also mirrored to file logs)
# --------------------------------------------------------------------------- #


def append_job_log(session: Session, job: Job, stage: str, message: str, *, progress: float | None = None) -> None:
    entries = job.stage_log()
    entries.append({"stage": stage, "message": message, "at": time.time()})
    job.log_json = json.dumps(entries[-80:])
    job.stage = stage
    job.message = message
    if progress is not None:
        job.progress = max(0.0, min(1.0, progress))
    session.add(job)


__all__ = [
    "Asset",
    "Base",
    "Candidate",
    "Clip",
    "Job",
    "KeyValue",
    "Project",
    "Template",
    "TranscriptSegment",
    "append_job_log",
    "db",
    "engine",
    "init_db",
    "iso",
    "kv_get",
    "kv_set",
    "new_id",
    "session_factory",
    "session_scope",
    "slugify",
    "utcnow",
]
