"""Pydantic request/response models for the public API (TRD §28)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def iso(dt: datetime | None) -> str | None:
    return dt.isoformat() + "Z" if dt else None


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- auth
class RegisterRequest(Model):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)
    display_name: str | None = Field(default=None, max_length=120)


class LoginRequest(Model):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)


class ChangePasswordRequest(Model):
    current_password: str = Field(max_length=256)
    new_password: str = Field(max_length=256)


class UserOut(BaseModel):
    id: str
    email: str
    display_name: str | None
    role: str
    plan: str
    created_at: str | None


class AuthResponse(BaseModel):
    user: UserOut
    token: str | None = None
    csrf_token: str | None = None
    expires_at: str | None = None


# ---------------------------------------------------------------- uploads
class UploadInitRequest(Model):
    filename: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(gt=0)
    project_id: str | None = None


class UploadCompleteRequest(Model):
    checksum_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")


class AssetOut(BaseModel):
    id: str
    status: str
    filename: str
    size_bytes: int | None
    bytes_received: int | None
    duration_seconds: float | None
    width: int | None
    height: int | None
    fps: float | None
    video_codec: str | None
    audio_codec: str | None
    has_audio: bool | None
    checksum_sha256: str | None
    error_code: str | None
    duplicate_of: str | None = None
    created_at: str | None
    expires_at: str | None


# ---------------------------------------------------------------- jobs
class JobSettingsIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_platform: str | None = Field(default=None, max_length=40)
    aspect_ratio: Literal["9:16", "16:9", "1:1"] | None = None
    num_clips: int | None = Field(default=None, ge=1, le=50)
    caption_preset: str | None = Field(default=None, max_length=40)
    captions_enabled: bool | None = None
    framing_mode: Literal["auto", "face", "center", "fit"] | None = None
    min_clip_seconds: float | None = Field(default=None, ge=1, le=600)
    max_clip_seconds: float | None = Field(default=None, ge=1, le=600)
    whisper_model: str | None = Field(default=None, max_length=40)
    language: str | None = Field(default=None, max_length=12)


class JobCreateRequest(Model):
    upload_id: str = Field(min_length=1, max_length=36)
    settings: JobSettingsIn = Field(default_factory=JobSettingsIn)


class JobOut(BaseModel):
    id: str
    status: str
    progress: int
    current_stage: str | None
    upload_id: str
    settings: dict[str, Any]
    attempt: int
    max_attempts: int
    error: dict[str, Any] | None
    cancel_requested: bool
    source_seconds: float | None
    versions: dict[str, Any] | None
    clip_count: int | None = None
    created_at: str | None
    started_at: str | None
    completed_at: str | None
    next_attempt_at: str | None


class JobListOut(BaseModel):
    items: list[JobOut]
    total: int
    limit: int
    offset: int


class JobEventOut(BaseModel):
    id: int
    event_type: str
    stage: str | None
    status: str | None
    progress: int | None
    timestamp: str | None
    duration_ms: int | None
    message: str | None
    error_code: str | None
    metadata: dict[str, Any] | None


# ---------------------------------------------------------------- clips
class RenderOut(BaseModel):
    id: str
    status: str
    source: str
    profile: str | None
    width: int | None
    height: int | None
    duration_seconds: float | None
    size_bytes: int | None
    filename: str | None
    error_code: str | None
    error_message: str | None
    created_at: str | None
    completed_at: str | None


class ClipOut(BaseModel):
    id: str
    job_id: str
    rank: int
    status: str
    start: float
    end: float
    duration: float
    title: str | None
    hook: str | None
    description: str | None
    keywords: list[str]
    reasoning: str | None
    transcript_excerpt: str | None
    score: float | None
    reasons: list[str]
    features: dict[str, float] | None = None
    aspect_ratio: str
    caption_preset: str
    captions_enabled: bool
    framing_mode: str
    framing: dict[str, Any] | None
    metadata_source: str | None
    render: RenderOut | None
    pending_render: RenderOut | None
    download_url: str | None
    thumbnail_url: str | None
    subtitles_url: str | None
    expires_at: str | None
    created_at: str | None
    updated_at: str | None


class ClipPatchRequest(Model):
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, gt=0)
    title: str | None = Field(default=None, max_length=200)
    hook: str | None = Field(default=None, max_length=300)
    description: str | None = Field(default=None, max_length=2000)
    aspect_ratio: Literal["9:16", "16:9", "1:1"] | None = None
    caption_preset: str | None = Field(default=None, max_length=40)
    captions_enabled: bool | None = None
    framing_mode: Literal["auto", "face", "center", "fit"] | None = None
    render: bool = True  # auto re-render on render-affecting changes


class ClipPatchResponse(BaseModel):
    clip: ClipOut
    render_required: bool
    render_queued: bool


# ---------------------------------------------------------------- me/admin
class UserSettingsPatch(Model):
    default_aspect_ratio: Literal["9:16", "16:9", "1:1"] | None = None
    default_caption_preset: str | None = Field(default=None, max_length=40)
    default_num_clips: int | None = Field(default=None, ge=1, le=50)
    default_framing_mode: Literal["auto", "face", "center", "fit"] | None = None
    default_target_platform: str | None = Field(default=None, max_length=40)
    captions_enabled: bool | None = None
    analytics_opt_out: bool | None = None
    display_name: str | None = Field(default=None, max_length=120)

    @field_validator("display_name")
    @classmethod
    def _strip(cls, v: str | None) -> str | None:
        return v.strip() if v else v


class AdminUserPatch(Model):
    role: Literal["user", "admin"] | None = None
    plan: str | None = Field(default=None, max_length=40)
    is_active: bool | None = None
    quota_override: dict[str, Any] | None = None
