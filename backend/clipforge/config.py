"""Configuration for CLIPFORGE AI.

Two layers:

1. *Environment* (``clipforge.config.ENV`` / :class:`Env`) - process level knobs
   that need to be known before the database exists (paths, host, port, log
   level). Read once at import time.
2. *User settings* (:class:`AppSettings`) - everything the Settings page can
   change at runtime. Persisted as JSON in SQLite so the user's preferences
   survive restarts, and hot-reloaded by the worker between jobs.

Nothing here contains secrets and nothing is sent to the frontend that is not
also safe to display.
"""

from __future__ import annotations

import json
import os
import platform
import threading
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .errors import invalid_input

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent


def _default_data_dir() -> Path:
    """Where projects/logs/db live.

    ``CLIPFORGE_DATA_DIR`` wins; otherwise a ``data/`` folder next to the repo so
    a portable install stays self-contained.
    """
    env = os.environ.get("CLIPFORGE_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (REPO_ROOT / "data").resolve()


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def env_flag(name: str, default: bool) -> bool:
    """Boolean environment variable (``1/true/yes/on`` vs ``0/false/no/off``)."""
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


class Env:
    """Process-level environment configuration (read once)."""

    DATA_DIR: Path = _default_data_dir()
    LOG_DIR: Path = Path(os.environ.get("CLIPFORGE_LOG_DIR", "")).expanduser().resolve() if os.environ.get("CLIPFORGE_LOG_DIR") else DATA_DIR / "logs"
    DB_PATH: Path = Path(os.environ.get("CLIPFORGE_DB", "")).expanduser().resolve() if os.environ.get("CLIPFORGE_DB") else DATA_DIR / "clipforge.db"
    HOST: str = os.environ.get("CLIPFORGE_HOST", "127.0.0.1")  # local-only by default
    PORT: int = int(os.environ.get("CLIPFORGE_PORT", "8317"))
    WORKERS: int = max(0, int(os.environ.get("CLIPFORGE_WORKERS", "1")))
    EMBED_WORKER: bool = os.environ.get("CLIPFORGE_EMBED_WORKER", "true").lower() not in {"0", "false", "no"}
    FRONTEND_DIST: Path | None = (
        Path(os.environ["CLIPFORGE_FRONTEND_DIST"]).expanduser().resolve()
        if os.environ.get("CLIPFORGE_FRONTEND_DIST")
        else None
    )
    LOG_LEVEL: str = os.environ.get("CLIPFORGE_LOG_LEVEL", "INFO").upper()
    OPEN_BROWSER: bool = os.environ.get("CLIPFORGE_OPEN_BROWSER", "true").lower() not in {"0", "false", "no"}
    IS_WINDOWS = platform.system() == "Windows"

    @classmethod
    def relative_paths(cls) -> dict[str, Path]:
        return {"data_dir": cls.DATA_DIR, "db_path": cls.DB_PATH, "log_dir": cls.LOG_DIR}

    @classmethod
    def local_paths_allowed(cls) -> bool:
        """Whether API callers may point the server at its own filesystem/desktop.

        Importing a server-side path, opening a folder in the file manager and
        similar conveniences only make sense when the browser and the backend
        share a machine. They default to *on* for a loopback host (the desktop
        app) and *off* when the API listens publicly; set
        ``CLIPFORGE_ALLOW_LOCAL_PATHS`` to override either way.
        """
        return env_flag("CLIPFORGE_ALLOW_LOCAL_PATHS", cls.HOST in LOOPBACK_HOSTS)


def ensure_dirs() -> None:
    for path in (
        Env.DATA_DIR,
        Env.LOG_DIR,
        Env.DATA_DIR / "projects",
        Env.DATA_DIR / "cache",
        Env.DATA_DIR / "exports",
        Env.DATA_DIR / "downloads",
    ):
        path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Runtime user settings
# --------------------------------------------------------------------------- #

AspectRatio = Literal["9:16", "1:1", "16:9"]
CaptionPosition = Literal["top", "middle", "lower-middle", "bottom"]
GameplayMode = Literal["off", "random", "auto", "category"]
LayoutMode = Literal["split", "podcast", "broll", "gameplay", "cinematic", "blur"]
HardwareAccel = Literal["auto", "nvenc", "qsv", "amf", "videotoolbox", "none"]
ClipMode = Literal["best", "balanced", "max"]


class CaptionTheme(BaseModel):
    """Full caption look-and-feel (drives the ASS style that ffmpeg burns in)."""

    preset: str = "bold_creator"
    font: str = "Inter ExtraBold"
    fallback_fonts: list[str] = Field(default_factory=lambda: ["Arial Black", "Arial", "DejaVu Sans"])
    font_size: int = 68
    weight: int = 900
    position: CaptionPosition = "lower-middle"
    margin_v: int = 320
    uppercase: bool = True
    primary_color: str = "#FFFFFF"
    highlight_color: str = "#FFD400"
    secondary_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline_width: int = 5
    shadow: int = 2
    background_enabled: bool = False
    background_color: str = "#000000"
    background_opacity: int = 45
    animation: Literal["none", "pop", "karaoke", "word_by_word", "slide_up"] = "word_by_word"
    max_words_per_line: int = 3
    max_lines: int = 2
    safe_margin_pct: int = 8
    emphasis_scale: float = 115.0
    emphasis_words: list[str] = Field(default_factory=list)

    def resolved_font(self) -> str:
        return self.font or "Arial"


class AppSettings(BaseModel):
    """Everything the Settings page can change (persisted in SQLite)."""

    # ------------------------------------------------------------ general
    concurrency: int = 1
    max_concurrent_renders: int = 1
    auto_start_worker: bool = True
    telemetry: bool = False  # always local; kept explicit so nothing phones home

    # ------------------------------------------------------------ ai / llm
    llm_enabled: bool = True
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:4b"
    ollama_timeout_seconds: int = 180
    ollama_temperature: float = 0.2
    ollama_num_ctx: int = 8192
    llm_max_workers: int = 3
    llm_chunk_chars: int = 9000
    llm_max_chunks: int = 40
    vision_enabled: bool = False
    vision_model: str = ""

    # ---------------------------------------------------- transcription
    whisper_model: str = "small"
    whisper_device: Literal["auto", "cpu", "cuda"] = "auto"
    whisper_compute_type: Literal["auto", "int8", "int8_float16", "float16", "float32"] = "auto"
    whisper_beam_size: int = 5
    whisper_batch_size: int = 8
    whisper_vad: bool = True
    whisper_vad_min_silence_ms: int = 400
    whisper_condition_on_previous_text: bool = False
    whisper_initial_prompt: str = ""
    whisper_cache_dir: str = ""  # empty = default HuggingFace cache
    word_alignment: Literal["auto", "whisperx", "faster_whisper", "interpolate"] = "auto"
    diarization: Literal["auto", "whisperx", "energy", "off"] = "energy"
    max_speakers: int = 6
    min_word_confidence: float = 0.35
    language_hint: str = ""  # "" = auto detect
    keep_source_audio: bool = False

    # ------------------------------------------------------------ video
    ffmpeg_path: str = ""   # empty = auto-discover
    ffprobe_path: str = ""
    hw_accel: HardwareAccel = "auto"

    # -------------------------------------------------------- downloads
    cookies_path: str = ""      # Netscape cookies.txt for age-restricted/member videos
    proxy: str = ""             # only used for the download step
    max_download_height: int = 1080
    prefer_mp4: bool = True
    max_source_hours: float = 6.0
    download_concurrency: int = 4
    output_fps: int = 30
    allow_60fps: bool = True
    render_preset: Literal["ultrafast", "veryfast", "fast", "medium", "slow"] = "medium"
    crf: int = 19
    video_bitrate_kbps: int = 0  # 0 = CRF quality mode
    audio_bitrate_kbps: int = 192

    # --------------------------------------------------- clip defaults
    clip_mode: ClipMode = "balanced"
    target_clip_seconds: float = 60.0
    min_clip_seconds: float = 35.0
    max_clip_seconds: float = 75.0
    min_score: float = 70.0
    max_clips: int = 0  # 0 = unlimited
    aspect_ratio: AspectRatio = "9:16"
    output_width: int = 1080
    output_height: int = 1920
    smart_reframe: bool = True
    auto_zoom: bool = True
    speaker_tracking: bool = True
    remove_silence: bool = True
    silence_threshold_db: float = -32.0
    silence_min_duration: float = 0.35
    silence_max_cut: float = 1.6
    keep_natural_pauses: float = 0.18

    # ------------------------------------------------------ captions
    captions_enabled: bool = True
    translate_captions: bool = False
    translation_language: str = ""
    caption: CaptionTheme = Field(default_factory=CaptionTheme)

    # ------------------------------------------------------ gameplay
    gameplay_enabled: bool = True
    gameplay_mode: GameplayMode = "auto"
    gameplay_category: str = "satisfying"
    layout: LayoutMode = "split"
    split_ratio: int = 65  # podcast share of the screen
    gameplay_volume: float = 0.08
    gameplay_random: bool = True
    gameplay_pace: Literal["calm", "medium", "high"] = "medium"
    broll_enabled: bool = False
    broll_category: str = "broll"

    # ---------------------------------------------------------- audio
    normalize_loudness: bool = True
    target_lufs: float = -14.0
    true_peak_db: float = -1.5
    noise_reduction: bool = False
    voice_boost: bool = True
    voice_gain_db: float = 2.0
    music_enabled: bool = False
    music_mood: str = "cinematic"
    music_volume: float = 0.12
    ducking: bool = True
    ducking_db: float = -12.0

    # ---------------------------------------------------------- export
    export_dir: str = ""      # empty = <data>/exports
    export_filename_template: str = "{project_slug}_{index:02d}_{title_slug}"
    auto_open_folder: bool = False

    # --------------------------------------------------------- storage
    keep_source_video: bool = True
    cache_transcripts: bool = True
    cache_downloads: bool = True
    cleanup_days: int = 14
    max_cache_gb: float = 50.0

    # --------------------------------------------------------- uploads
    uploads_enabled: bool = True
    max_upload_gb: float = 8.0

    model_config = {"extra": "forbid"}

    # ------------------------------------------------------------------ #
    @field_validator("concurrency", "max_concurrent_renders", "llm_max_workers")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be at least 1")
        return min(value, 16)

    @field_validator("min_score")
    @classmethod
    def _score_range(cls, value: float) -> float:
        if not 0 <= value <= 100:
            raise ValueError("must be between 0 and 100")
        return value

    @field_validator("output_fps")
    @classmethod
    def _fps(cls, value: int) -> int:
        if value not in {24, 25, 30, 50, 60}:
            raise ValueError("must be one of 24, 25, 30, 50, 60")
        return value

    @field_validator("export_filename_template")
    @classmethod
    def _filename_template(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            return "{project_slug}_{index:02d}_{title_slug}"
        try:
            value.format(project_slug="project", index=1, title_slug="title")
        except (KeyError, IndexError, ValueError, AttributeError) as exc:
            raise ValueError("use only the {project_slug}, {index} (e.g. {index:02d}) and {title_slug} placeholders") from exc
        return value

    @model_validator(mode="after")
    def _sync_size_to_ratio(self) -> "AppSettings":
        """Keep the pixel size on the chosen aspect ratio.

        ``aspect_ratio`` is what the user picks; ``output_width``/``output_height``
        are the pixels. Without this, switching 9:16 -> 1:1 would keep 1080x1920
        and the ratio would silently never change.
        """
        preset_width, preset_height = ASPECT_PRESETS[self.aspect_ratio]
        width, height = int(self.output_width or 0), int(self.output_height or 0)
        if width and height and abs(width / height - preset_width / preset_height) > 0.02:
            self.output_width, self.output_height = preset_width, preset_height
        return self

    # ------------------------------------------------------------- helpers
    def resolved_export_dir(self) -> Path:
        return Path(self.export_dir).expanduser().resolve() if self.export_dir else Env.DATA_DIR / "exports"

    def aspect_dims(self) -> tuple[int, int]:
        """Output size: the custom width/height when they match the chosen aspect ratio.

        ``aspect_ratio`` is the preset the user picks; ``output_width``/``output_height``
        are the actual pixels (e.g. 540x960 for a fast draft). They are honoured only
        when they agree with the ratio, so a stale size can never distort the frame.
        """
        preset_width, preset_height = ASPECT_PRESETS[self.aspect_ratio]
        width = int(self.output_width or 0)
        height = int(self.output_height or 0)
        if width >= 240 and height >= 240:
            target = preset_width / preset_height
            if abs(width / height - target) <= 0.02:
                return width - width % 2, height - height % 2
        return preset_width, preset_height

    def clip_limits(self) -> tuple[float, float, float]:
        """Return ``(min, target, max)`` seconds with sane ordering enforced."""
        low = float(self.min_clip_seconds)
        target = float(self.target_clip_seconds)
        high = float(self.max_clip_seconds)
        low = max(8.0, low)
        high = max(low + 2.0, high)
        target = min(max(target, low), high)
        return low, target, high

    def to_public_dict(self) -> dict[str, Any]:
        return json.loads(self.model_dump_json())


ASPECT_PRESETS: dict[str, tuple[int, int]] = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
}

SECTION_FIELDS: dict[str, tuple[str, ...]] = {
    "general": ("concurrency", "max_concurrent_renders", "auto_start_worker", "telemetry"),
    "ai": (
        "llm_enabled", "ollama_base_url", "ollama_model", "ollama_timeout_seconds",
        "ollama_temperature", "ollama_num_ctx", "llm_max_workers", "llm_chunk_chars",
        "llm_max_chunks", "vision_enabled", "vision_model",
    ),
    "transcription": (
        "whisper_model", "whisper_device", "whisper_compute_type", "whisper_beam_size",
        "whisper_batch_size", "whisper_vad", "whisper_vad_min_silence_ms",
        "whisper_condition_on_previous_text", "whisper_initial_prompt", "word_alignment",
        "whisper_cache_dir",
        "diarization", "max_speakers", "min_word_confidence", "language_hint",
    ),
    "video": (
        "ffmpeg_path", "ffprobe_path", "hw_accel", "output_fps", "allow_60fps",
        "cookies_path", "proxy", "max_download_height", "prefer_mp4",
        "max_source_hours", "download_concurrency",
        "render_preset", "crf", "video_bitrate_kbps", "audio_bitrate_kbps",
        "clip_mode", "target_clip_seconds", "min_clip_seconds", "max_clip_seconds",
        "min_score", "max_clips", "aspect_ratio", "output_width", "output_height",
        "smart_reframe", "auto_zoom", "speaker_tracking", "remove_silence",
        "silence_threshold_db", "silence_min_duration", "silence_max_cut",
        "keep_natural_pauses",
    ),
    "captions": ("captions_enabled", "translate_captions", "translation_language", "caption"),
    "gameplay": (
        "gameplay_enabled", "gameplay_mode", "gameplay_category", "layout", "split_ratio",
        "gameplay_volume", "gameplay_random", "gameplay_pace", "broll_enabled", "broll_category",
    ),
    "audio": (
        "normalize_loudness", "target_lufs", "true_peak_db", "noise_reduction", "voice_boost",
        "voice_gain_db", "music_enabled", "music_mood", "music_volume", "ducking", "ducking_db",
    ),
    "export": (
        "export_dir", "export_filename_template", "auto_open_folder", "keep_source_audio",
    ),
    "storage": (
        "keep_source_video", "cache_transcripts", "cache_downloads", "cleanup_days",
        "max_cache_gb", "uploads_enabled", "max_upload_gb",
    ),
}

SECRET_FREE = True  # nothing in AppSettings is sensitive; safe to echo to the UI

# Settings that point at the server's own filesystem or binaries. They can only
# be changed when local-path features are allowed (see Env.local_paths_allowed):
# on a public deployment they would let any visitor write files anywhere or swap
# the ffmpeg binary.
SERVER_PATH_FIELDS: tuple[str, ...] = ("export_dir", "ffmpeg_path", "ffprobe_path", "cookies_path", "whisper_cache_dir")


def merge_settings_patch(current: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Apply a partial settings patch to a settings dict.

    Top-level keys replace their old value, but the nested ``caption`` theme is
    *merged*: sending ``{"caption": {"font_size": 70}}`` changes the size and keeps
    every other caption choice. When the patch names a caption ``preset``, that
    preset's look is applied first and any explicit caption keys in the same
    patch win - so templates and per-project options can be partial.
    """
    merged = {**current, **{key: value for key, value in patch.items() if key != "caption"}}
    if "caption" not in patch:
        return merged
    caption_patch = patch["caption"]
    if not isinstance(caption_patch, dict):
        merged["caption"] = caption_patch  # validation reports the bad type
        return merged
    caption = dict(current.get("caption") or {})
    preset = caption_patch.get("preset")
    if preset:
        from .constants import CAPTION_PRESETS  # local: constants is a leaf module

        definition = CAPTION_PRESETS.get(str(preset))
        if definition:
            caption.update(dict(definition.get("theme") or {}))
    caption.update(caption_patch)
    merged["caption"] = caption
    return merged


def describe_validation_error(exc: ValidationError, limit: int = 4) -> str:
    """``field: problem`` pairs instead of pydantic's multi-line dump."""
    parts = []
    for error in exc.errors()[:limit]:
        location = ".".join(str(part) for part in error.get("loc", ()) if part != "__root__") or "settings"
        message = str(error.get("msg", "invalid value")).removeprefix("Value error, ")
        parts.append(f"{location}: {message}")
    return "; ".join(parts) or "invalid value"


def load_settings_leniently(raw: dict[str, Any]) -> AppSettings:
    """Validate stored settings, dropping only the entries that no longer fit.

    Settings saved by another version may contain renamed or out-of-range
    fields. Rather than silently resetting *everything* to defaults, unknown
    keys are ignored and invalid values fall back to their default one by one.
    """
    defaults = AppSettings().model_dump()
    known = {key: value for key, value in raw.items() if key in AppSettings.model_fields}
    candidate = merge_settings_patch(defaults, known)
    for _attempt in range(len(known) + 1):
        try:
            return AppSettings.model_validate(candidate)
        except ValidationError as exc:
            broken = {str(error["loc"][0]) for error in exc.errors() if error.get("loc")}
            if not broken or not broken & set(candidate):
                break
            for key in broken:
                candidate[key] = defaults.get(key)
    return AppSettings()


class SettingsStore:
    """Thread-safe settings holder with SQLite persistence and change listeners."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._settings = AppSettings()
        self._listeners: list[Callable[[AppSettings], None]] = []

    # ------------------------------------------------------------- access
    @property
    def current(self) -> AppSettings:
        with self._lock:
            return self._settings

    def on_change(self, listener: Callable[[AppSettings], None]) -> None:
        with self._lock:
            self._listeners.append(listener)

    # ---------------------------------------------------------- mutation
    def update(self, patch: dict[str, Any]) -> AppSettings:
        """Apply a partial update (unknown keys are rejected, ``caption`` is merged)."""
        if not isinstance(patch, dict):
            raise invalid_input("Settings payload must be a JSON object.")
        unknown = sorted(key for key in patch if key not in AppSettings.model_fields)
        if unknown:
            raise invalid_input(
                f"Unknown setting{'s' if len(unknown) > 1 else ''}: {', '.join(unknown)}.",
                "GET /api/settings/schema lists every setting this version understands.",
            )
        with self._lock:
            merged = merge_settings_patch(self._settings.model_dump(), patch)
            try:
                new_settings = AppSettings.model_validate(merged)
            except ValidationError as exc:
                raise invalid_input(f"Invalid setting - {describe_validation_error(exc)}", "Check the value ranges in Settings.") from exc
            self._settings = new_settings
            listeners = list(self._listeners)
        self.persist()
        for listener in listeners:
            try:
                listener(new_settings)
            except Exception:  # a bad listener must never break a settings update
                pass
        return new_settings

    def reset(self) -> AppSettings:
        return self.update(AppSettings().model_dump())

    # --------------------------------------------------------- persistence
    def persist(self) -> None:
        from .db import kv_set  # local import to avoid a cycle

        kv_set("settings", self._settings.to_public_dict())

    def load(self) -> AppSettings:
        from .db import kv_get

        raw = kv_get("settings")
        if isinstance(raw, dict):
            loaded = load_settings_leniently(raw)
            with self._lock:
                self._settings = loaded
        return self._settings

    # ------------------------------------------------------------ helpers
    def sections(self) -> dict[str, dict[str, Any]]:
        data = self._settings.to_public_dict()
        out: dict[str, dict[str, Any]] = {}
        claimed: set[str] = set()
        for name, fields in SECTION_FIELDS.items():
            out[name] = {f: data[f] for f in fields if f in data}
            claimed.update(fields)
        out["advanced"] = {k: v for k, v in data.items() if k not in claimed}
        return out


_STORE: SettingsStore | None = None
_STORE_LOCK = threading.Lock()


def settings_store() -> SettingsStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = SettingsStore()
        return _STORE


def get_settings() -> AppSettings:
    """Convenience accessor used throughout the codebase."""
    return settings_store().current


def export_dir() -> Path:
    path = get_settings().resolved_export_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def projects_dir() -> Path:
    path = Env.DATA_DIR / "projects"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    path = Env.DATA_DIR / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path
