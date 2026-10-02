"""Application configuration.

Every deployment-specific value comes from environment variables (or a ``.env``
file) so that the same code runs in local (zero-cost) and cloud mode
(TRD §36, §37). Nothing in here should ever be a secret default for production.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Lists accept JSON *or* comma-separated env values (parsed by the validator below).
StrList = Annotated[list[str], NoDecode]

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = BACKEND_ROOT / "config"

INSECURE_DEFAULT_SECRET = "dev-insecure-secret-change-me"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ app
    APP_ENV: Literal["development", "test", "production"] = "development"
    DEPLOYMENT_MODE: Literal["local", "cloud"] = "local"
    SERVICE_NAME: str = "api"
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = True
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    METRICS_TOKEN: str = ""  # when set, /metrics requires "Authorization: Bearer <token>"
    SSE_POLL_SECONDS: float = 1.0
    SSE_RETRY_MS: int = 3000
    MAX_REQUEST_BODY_BYTES: int = 1024 * 1024  # JSON bodies (uploads are exempt)
    API_PREFIX: str = "/api/v1"
    PUBLIC_BASE_URL: str = ""  # used for absolute signed URLs; relative when empty
    CORS_ORIGINS: StrList = Field(default_factory=lambda: ["http://localhost:3000"])
    TRUST_PROXY_HEADERS: bool = False
    AUTO_MIGRATE: bool = True

    # ------------------------------------------------------------- database
    DATABASE_URL: str = ""  # default derived from STORAGE_ROOT (SQLite)
    DATABASE_ECHO: bool = False
    REDIS_URL: str = ""  # optional: wakes workers instantly instead of DB polling

    # -------------------------------------------------------------- storage
    STORAGE_BACKEND: Literal["local", "s3"] = "local"
    STORAGE_ROOT: Path = BACKEND_ROOT / "data"
    S3_BUCKET: str = ""
    S3_ENDPOINT_URL: str = ""
    S3_REGION: str = "us-east-1"
    S3_ACCESS_KEY_ID: str = ""
    S3_SECRET_ACCESS_KEY: str = ""
    SIGNED_URL_TTL_SECONDS: int = 3600

    # --------------------------------------------------------------- limits
    MAX_UPLOAD_BYTES: int = 2 * 1024**3
    MAX_VIDEO_SECONDS: int = 3 * 3600
    MIN_VIDEO_SECONDS: float = 5.0
    MAX_VIDEO_WIDTH: int = 7680
    MAX_VIDEO_HEIGHT: int = 7680
    ALLOWED_EXTENSIONS: StrList = Field(
        default_factory=lambda: [".mp4", ".mov", ".mkv", ".webm", ".m4v"]
    )
    UPLOAD_CHUNK_MAX_BYTES: int = 64 * 1024**2
    MAX_CLIPS_PER_JOB: int = 20
    DEFAULT_CLIPS_PER_JOB: int = 5

    # Clip duration constraints (TRD §13)
    MIN_CLIP_SECONDS: float = 10.0
    TARGET_MIN_SECONDS: float = 20.0
    TARGET_MAX_SECONDS: float = 60.0
    MAX_CLIP_SECONDS: float = 90.0

    # --------------------------------------------------------------- worker
    WORKER_ID: str = ""
    WORKER_CONCURRENCY: int = 1
    WORKER_POLL_INTERVAL_SECONDS: float = 2.0
    WORKER_LEASE_SECONDS: int = 120
    WORKER_HEARTBEAT_SECONDS: int = 15
    WORKER_OFFLINE_AFTER_SECONDS: int = 90
    MAX_CONCURRENT_FFMPEG: int = 2
    MAX_CONCURRENT_TRANSCRIPTIONS: int = 1
    MAX_JOB_SECONDS: int = 6 * 3600
    MAX_JOB_ATTEMPTS: int = 3
    RETRY_BACKOFF_BASE_SECONDS: int = 30
    FFMPEG_TIMEOUT_SECONDS: int = 3600
    FFPROBE_TIMEOUT_SECONDS: int = 30
    MIN_FREE_DISK_BYTES: int = 2 * 1024**3
    MIN_FREE_RAM_BYTES: int = 512 * 1024**2
    MAX_WORK_DISK_BYTES: int = 50 * 1024**3
    MAX_GPU_MEMORY_FRACTION: float = 0.9

    # ---------------------------------------------------------- AI / models
    TRANSCRIPTION_PROVIDER: Literal["faster_whisper", "fake"] = "faster_whisper"
    WHISPER_MODEL: str = "auto"  # auto | tiny | base | small | medium | large-v3
    WHISPER_DEVICE: Literal["auto", "cpu", "cuda"] = "auto"
    WHISPER_COMPUTE_TYPE: str = "auto"
    WHISPER_MODEL_DIR: str = ""
    WHISPER_LANGUAGE: str = ""  # empty -> auto detect
    WHISPER_BEAM_SIZE: int = 5
    WARMUP_MODELS: bool = False
    FAKE_TRANSCRIPT_PATH: str = ""

    LLM_PROVIDER: Literal["heuristic", "ollama", "transformers", "openai_compatible", "none"] = (
        "heuristic"
    )
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3.2:3b"
    LLM_TIMEOUT_SECONDS: float = 60.0
    REMOTE_LLM_BASE_URL: str = ""
    REMOTE_LLM_API_KEY: str = ""
    ALLOW_REMOTE_PROVIDERS: bool = False  # TRD §53: never send data remotely unless allowed

    EMBEDDINGS_PROVIDER: Literal["lexical", "sentence_transformers"] = "lexical"
    EMBEDDINGS_MODEL: str = "all-MiniLM-L6-v2"

    VISION_PROVIDER: Literal["auto", "opencv", "ffmpeg", "none"] = "auto"
    VISION_SAMPLE_FPS: float = 2.0
    VISION_FRAME_WIDTH: int = 320

    FFMPEG_PATH: str = "ffmpeg"
    FFPROBE_PATH: str = "ffprobe"
    CAPTION_FONTS_DIR: str = ""

    SCORING_CONFIG_FILE: Path = CONFIG_DIR / "scoring.yaml"
    PLANS_CONFIG_FILE: Path = CONFIG_DIR / "plans.yaml"
    RENDER_PROFILES_FILE: Path = CONFIG_DIR / "render_profiles.yaml"
    CAPTION_PRESETS_FILE: Path = CONFIG_DIR / "caption_presets.yaml"

    # ------------------------------------------------------------ retention
    RETENTION_HOURS: int = 24  # source video retention after successful processing
    OUTPUT_RETENTION_HOURS: int = 168
    FAILED_SOURCE_RETENTION_HOURS: int = 72
    ABANDONED_UPLOAD_HOURS: int = 6
    CLEANUP_INTERVAL_SECONDS: int = 600
    DELETE_INTERMEDIATES_ON_COMPLETE: bool = True

    # ---------------------------------------------------------------- auth
    AUTH_ENABLED: bool = True  # local mode may disable (single implicit user)
    ALLOW_REGISTRATION: bool = True
    JWT_SECRET: str = INSECURE_DEFAULT_SECRET  # HMAC secret: signed URLs / tokens
    SESSION_TTL_HOURS: int = 24 * 14
    SESSION_COOKIE_NAME: str = "cf_session"
    CSRF_COOKIE_NAME: str = "cf_csrf"
    SESSION_COOKIE_SECURE: bool = False
    SESSION_COOKIE_SAMESITE: Literal["lax", "strict", "none"] = "lax"
    ADMIN_EMAILS: StrList = Field(default_factory=list)
    LOCAL_USER_EMAIL: str = "local@clipforge.local"
    PASSWORD_MIN_LENGTH: int = 8

    # ---------------------------------------------------------- rate limits
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_AUTH_PER_MINUTE: int = 10
    RATE_LIMIT_UPLOADS_PER_HOUR: int = 60
    RATE_LIMIT_DEFAULT_PER_MINUTE: int = 600

    # -------------------------------------------------------------- quotas
    DEFAULT_PLAN: str = ""  # empty -> "unlimited" in local mode, "free" in cloud
    FREE_DAILY_MINUTES: int | None = None  # override for plans.yaml free.daily_source_minutes

    # ------------------------------------------------------------- billing
    BILLING_PROVIDER: Literal["none"] = "none"

    # -------------------------------------------------------------- alerts
    ALERT_DISK_PERCENT: float = 90.0
    ALERT_QUEUE_LENGTH: int = 50
    ALERT_FAILED_RENDERS: int = 5

    # ------------------------------------------------------------------------
    @field_validator("CORS_ORIGINS", "ALLOWED_EXTENSIONS", "ADMIN_EMAILS", mode="before")
    @classmethod
    def _split_list(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            if value.startswith("["):
                return json.loads(value)
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    @field_validator("FREE_DAILY_MINUTES", mode="before")
    @classmethod
    def _empty_int(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _derive(self) -> Settings:
        self.STORAGE_ROOT = Path(self.STORAGE_ROOT).expanduser().resolve()
        if not self.DATABASE_URL:
            self.DATABASE_URL = f"sqlite:///{self.STORAGE_ROOT / 'clipforge.db'}"
        self.ALLOWED_EXTENSIONS = [e.lower() if e.startswith(".") else f".{e.lower()}"
                                   for e in self.ALLOWED_EXTENSIONS]
        self.ADMIN_EMAILS = [e.lower() for e in self.ADMIN_EMAILS]
        if not self.DEFAULT_PLAN:
            self.DEFAULT_PLAN = "unlimited" if self.DEPLOYMENT_MODE == "local" else "free"
        if self.APP_ENV == "production":
            if self.JWT_SECRET == INSECURE_DEFAULT_SECRET or len(self.JWT_SECRET) < 32:
                raise ValueError("JWT_SECRET must be set to a random value of >= 32 chars in production")
            if self.DEPLOYMENT_MODE == "cloud" and not self.AUTH_ENABLED:
                raise ValueError("AUTH_ENABLED cannot be false in cloud mode")
        if not (self.MIN_CLIP_SECONDS <= self.TARGET_MIN_SECONDS <= self.TARGET_MAX_SECONDS
                <= self.MAX_CLIP_SECONDS):
            raise ValueError("Clip duration constraints must satisfy MIN<=TARGET_MIN<=TARGET_MAX<=MAX")
        return self

    # ------------------------------------------------------------------------
    @property
    def is_sqlite(self) -> bool:
        return self.DATABASE_URL.startswith("sqlite")

    @property
    def auth_required(self) -> bool:
        # Cloud deployments always require authentication.
        return self.AUTH_ENABLED or self.DEPLOYMENT_MODE == "cloud"

    def ensure_directories(self) -> None:
        # The work directory is always local (even with S3 storage) because
        # FFmpeg/whisper operate on local files.
        for sub in ("uploads", "work", "outputs", "thumbnails", "transcripts", "logs"):
            (self.STORAGE_ROOT / sub).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
