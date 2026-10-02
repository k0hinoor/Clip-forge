"""SQLAlchemy ORM models (TRD §27, §28).

Portable across SQLite and PostgreSQL: UUIDs are stored as 36-char strings,
JSON uses the generic JSON type and timestamps are naive UTC.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from clipforge.core.timeutil import utcnow

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)


# --------------------------------------------------------------------- users
class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str | None] = mapped_column(String(255))
    display_name: Mapped[str | None] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(String(20), default="user")  # user | admin
    plan: Mapped[str] = mapped_column(String(40), default="free")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    quota_override: Mapped[dict[str, Any] | None] = mapped_column(JSON)  # admin override
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime)

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


class UserSession(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime)
    rotated_from_id: Mapped[str | None] = mapped_column(String(36))
    user_agent: Mapped[str | None] = mapped_column(String(255))
    ip_address: Mapped[str | None] = mapped_column(String(64))

    user: Mapped[User] = relationship(lazy="joined")


class Project(TimestampMixin, Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200), default="Default")
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


# -------------------------------------------------------------------- assets
class Asset(TimestampMixin, Base):
    __tablename__ = "assets"
    __table_args__ = (Index("ix_assets_user_checksum", "user_id", "checksum_sha256"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id", ondelete="SET NULL"))
    kind: Mapped[str] = mapped_column(String(20), default="source")
    status: Mapped[str] = mapped_column(String(20), default="UPLOADING", index=True)
    original_filename: Mapped[str] = mapped_column(String(255))  # sanitised; display only
    extension: Mapped[str] = mapped_column(String(10))
    storage_key: Mapped[str] = mapped_column(String(512))  # internal; never exposed
    expected_size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    bytes_received: Mapped[int] = mapped_column(BigInteger, default=0)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64))
    mime_type: Mapped[str | None] = mapped_column(String(100))
    container: Mapped[str | None] = mapped_column(String(100))
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    fps: Mapped[float | None] = mapped_column(Float)
    video_codec: Mapped[str | None] = mapped_column(String(50))
    audio_codec: Mapped[str | None] = mapped_column(String(50))
    has_audio: Mapped[bool | None] = mapped_column(Boolean)
    has_video: Mapped[bool | None] = mapped_column(Boolean)
    media_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)  # immutable probe output
    error_code: Mapped[str | None] = mapped_column(String(50))
    upload_expires_at: Mapped[datetime | None] = mapped_column(DateTime)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, index=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime)


# ---------------------------------------------------------------------- jobs
class Job(TimestampMixin, Base):
    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_status_next_attempt", "status", "next_attempt_at"),
        Index("ix_jobs_user_status", "user_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id", ondelete="SET NULL"))
    source_asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(32), default="QUEUED")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    current_stage: Mapped[str | None] = mapped_column(String(50))
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    error_code: Mapped[str | None] = mapped_column(String(50))
    error_message: Mapped[str | None] = mapped_column(String(500))
    diagnostic_id: Mapped[str | None] = mapped_column(String(32))
    retryable: Mapped[bool | None] = mapped_column(Boolean)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    lock_owner: Mapped[str | None] = mapped_column(String(100))
    lock_expires_at: Mapped[datetime | None] = mapped_column(DateTime)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    source_seconds: Mapped[float | None] = mapped_column(Float)
    usage_recorded: Mapped[bool] = mapped_column(Boolean, default=False)
    pipeline_version: Mapped[str] = mapped_column(String(20), default="1.0.0")
    versions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    manifest: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime)

    source_asset: Mapped[Asset] = relationship(lazy="joined")


class JobEvent(Base):
    __tablename__ = "job_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    event_type: Mapped[str] = mapped_column(String(50))
    stage: Mapped[str | None] = mapped_column(String(50))
    status: Mapped[str | None] = mapped_column(String(32))
    progress: Mapped[int | None] = mapped_column(Integer)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    message: Mapped[str | None] = mapped_column(String(500))
    error_code: Mapped[str | None] = mapped_column(String(50))
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)


# --------------------------------------------------------------- transcripts
class Transcript(Base):
    __tablename__ = "transcripts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), unique=True)
    language: Mapped[str | None] = mapped_column(String(16))
    provider: Mapped[str] = mapped_column(String(50))
    model: Mapped[str] = mapped_column(String(50))
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text, default="")
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    storage_key: Mapped[str | None] = mapped_column(String(512))
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    segments: Mapped[list[TranscriptSegment]] = relationship(
        order_by="TranscriptSegment.idx", cascade="all, delete-orphan", lazy="selectin"
    )


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transcript_id: Mapped[str] = mapped_column(ForeignKey("transcripts.id", ondelete="CASCADE"), index=True)
    idx: Mapped[int] = mapped_column(Integer)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    words: Mapped[list[Any]] = mapped_column(JSON, default=list)
    speaker_id: Mapped[str | None] = mapped_column(String(32))
    topic_id: Mapped[int | None] = mapped_column(Integer)
    avg_logprob: Mapped[float | None] = mapped_column(Float)
    no_speech_prob: Mapped[float | None] = mapped_column(Float)


# ---------------------------------------------------------------- candidates
class Candidate(Base):
    __tablename__ = "candidates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    score: Mapped[float] = mapped_column(Float)
    reasons: Mapped[list[Any]] = mapped_column(JSON, default=list)
    selected: Mapped[bool] = mapped_column(Boolean, default=False)
    topic_id: Mapped[int | None] = mapped_column(Integer)
    scoring_version: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    features: Mapped[CandidateFeatures | None] = relationship(
        cascade="all, delete-orphan", uselist=False, lazy="selectin"
    )


class CandidateFeatures(Base):
    __tablename__ = "candidate_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    candidate_id: Mapped[str] = mapped_column(ForeignKey("candidates.id", ondelete="CASCADE"), unique=True)
    features: Mapped[dict[str, Any]] = mapped_column(JSON)  # raw feature values (0..1)
    weights: Mapped[dict[str, Any]] = mapped_column(JSON)
    raw: Mapped[dict[str, Any] | None] = mapped_column(JSON)  # raw signals behind features
    scoring_version: Mapped[str] = mapped_column(String(20))


# --------------------------------------------------------------------- clips
class Clip(TimestampMixin, Base):
    __tablename__ = "clips"
    __table_args__ = (Index("ix_clips_job_rank", "job_id", "rank"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    candidate_id: Mapped[str | None] = mapped_column(ForeignKey("candidates.id", ondelete="SET NULL"))
    rank: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), default="PENDING")
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    slug: Mapped[str] = mapped_column(String(60))
    title: Mapped[str | None] = mapped_column(String(200))
    hook: Mapped[str | None] = mapped_column(String(300))
    description: Mapped[str | None] = mapped_column(Text)
    keywords: Mapped[list[Any]] = mapped_column(JSON, default=list)
    reasoning: Mapped[str | None] = mapped_column(Text)
    transcript_excerpt: Mapped[str | None] = mapped_column(Text)
    score: Mapped[float | None] = mapped_column(Float)
    reasons: Mapped[list[Any]] = mapped_column(JSON, default=list)
    aspect_ratio: Mapped[str] = mapped_column(String(10), default="9:16")
    caption_preset: Mapped[str] = mapped_column(String(30), default="clean")
    captions_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    framing_mode: Mapped[str] = mapped_column(String(20), default="auto")
    crop: Mapped[dict[str, Any] | None] = mapped_column(JSON)  # reproducible crop trajectory
    captions: Mapped[dict[str, Any] | None] = mapped_column(JSON)  # {"preset":..., "items": timeline}
    visual: Mapped[dict[str, Any] | None] = mapped_column(JSON)  # sampled frame analysis
    current_render_id: Mapped[str | None] = mapped_column(String(36))
    metadata_source: Mapped[str | None] = mapped_column(String(50))  # llm provider used
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, index=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime)


class RenderProfile(TimestampMixin, Base):
    __tablename__ = "render_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(50), unique=True)  # e.g. vertical_v1
    aspect_ratio: Mapped[str] = mapped_column(String(10))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    fps: Mapped[float | None] = mapped_column(Float)  # None = source fps
    video_codec: Mapped[str] = mapped_column(String(20), default="h264")
    audio_codec: Mapped[str] = mapped_column(String(20), default="aac")
    crf: Mapped[int] = mapped_column(Integer, default=20)
    preset: Mapped[str] = mapped_column(String(20), default="veryfast")
    audio_bitrate_kbps: Mapped[int] = mapped_column(Integer, default=160)
    version: Mapped[str] = mapped_column(String(20), default="1")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Render(Base):
    __tablename__ = "renders"
    __table_args__ = (Index("ix_renders_status", "status"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    clip_id: Mapped[str] = mapped_column(ForeignKey("clips.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    render_profile_id: Mapped[str | None] = mapped_column(ForeignKey("render_profiles.id", ondelete="SET NULL"))
    render_profile_name: Mapped[str] = mapped_column(String(50))
    render_profile_version: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="QUEUED")
    source: Mapped[str] = mapped_column(String(20), default="pipeline")  # pipeline | user
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    storage_key: Mapped[str | None] = mapped_column(String(512))
    thumbnail_key: Mapped[str | None] = mapped_column(String(512))
    subtitles_key: Mapped[str | None] = mapped_column(String(512))
    filename: Mapped[str | None] = mapped_column(String(255))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64))
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # full reproducible params
    qa: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error_code: Mapped[str | None] = mapped_column(String(50))
    error_message: Mapped[str | None] = mapped_column(String(500))
    lock_owner: Mapped[str | None] = mapped_column(String(100))
    lock_expires_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime)


# ------------------------------------------------------------ usage/billing
class UsageCounter(Base):
    __tablename__ = "usage_counters"
    __table_args__ = (UniqueConstraint("user_id", "period", "period_start", name="uq_usage_period"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    period: Mapped[str] = mapped_column(String(10))  # day | month
    period_start: Mapped[date] = mapped_column(Date)
    source_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    jobs_completed: Mapped[int] = mapped_column(Integer, default=0)
    clips_rendered: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class Subscription(TimestampMixin, Base):
    __tablename__ = "subscriptions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    plan: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(20))  # active | cancelled | past_due
    provider: Mapped[str] = mapped_column(String(40))
    provider_ref: Mapped[str | None] = mapped_column(String(200))
    current_period_end: Mapped[datetime | None] = mapped_column(DateTime)


class Payment(Base):
    __tablename__ = "payments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    amount_cents: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(3))
    provider: Mapped[str] = mapped_column(String(40))
    provider_ref: Mapped[str | None] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AffiliateEvent(Base):
    __tablename__ = "affiliate_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    placement: Mapped[str] = mapped_column(String(100))
    partner: Mapped[str] = mapped_column(String(100))
    event: Mapped[str] = mapped_column(String(20))  # impression | click
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# ------------------------------------------------------------- operations
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor_user_id: Mapped[str | None] = mapped_column(String(36), index=True)
    action: Mapped[str] = mapped_column(String(100))
    target_type: Mapped[str | None] = mapped_column(String(50))
    target_id: Mapped[str | None] = mapped_column(String(36))
    ip_address: Mapped[str | None] = mapped_column(String(64))
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class SystemSetting(Base):
    __tablename__ = "system_settings"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class IdempotencyKey(Base):
    __tablename__ = "idempotency_keys"
    __table_args__ = (UniqueConstraint("user_id", "scope", "key", name="uq_idempotency_scope_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    scope: Mapped[str] = mapped_column(String(50))
    key: Mapped[str] = mapped_column(String(100))
    request_hash: Mapped[str | None] = mapped_column(String(64))
    resource_type: Mapped[str] = mapped_column(String(50))
    resource_id: Mapped[str] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class WorkerHeartbeat(Base):
    __tablename__ = "worker_heartbeats"

    worker_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    hostname: Mapped[str] = mapped_column(String(255))
    pid: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20))  # idle | busy | stopping
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    active_jobs: Mapped[list[Any]] = mapped_column(JSON, default=list)
    capabilities: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class AnalyticsEvent(Base):
    __tablename__ = "analytics_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str | None] = mapped_column(String(36), index=True)
    job_id: Mapped[str | None] = mapped_column(String(36))
    clip_id: Mapped[str | None] = mapped_column(String(36))
    properties: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
