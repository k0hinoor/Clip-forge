"""Request/response models for the REST API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..media.download import classify_url


def caption_theme_patch(preset: str) -> dict[str, Any]:
    """Settings patch selecting a caption preset (merged over the user's caption theme)."""
    from ..constants import CAPTION_PRESETS

    if not preset or preset not in CAPTION_PRESETS:
        return {}
    return {"caption": {"preset": preset}}


def local_media_path(text: str) -> Path | None:
    """``text`` as an existing local media file, when such imports are allowed."""
    from ..config import Env

    if not text or "://" in text or not Env.local_paths_allowed():
        return None
    try:
        candidate = Path(text).expanduser()
        return candidate if candidate.is_file() else None
    except (OSError, ValueError):
        return None


class ClipOptions(BaseModel):
    """Optional per-project overrides for the clip search (Create screen).

    Only the options a client actually sends are stored on the project; every
    other setting keeps following the global Settings. The defaults below are
    documentation for API users, not values forced onto new projects.
    """

    clip_mode: Literal["best", "balanced", "max"] = "balanced"
    target_clip_seconds: float = Field(60.0, ge=10, le=180)
    min_clip_seconds: float = Field(35.0, ge=5, le=120)
    max_clip_seconds: float = Field(75.0, ge=15, le=300)
    min_score: float = Field(70.0, ge=0, le=100)
    max_clips: int = Field(0, ge=0, le=500, description="0 = unlimited")
    aspect_ratio: Literal["9:16", "1:1", "16:9"] = "9:16"
    caption_preset: str = "bold_creator"  # see constants.CAPTION_PRESETS
    layout: Literal["split", "podcast", "broll", "gameplay", "cinematic", "blur"] = "split"
    split_ratio: int = Field(65, ge=30, le=80)
    gameplay_enabled: bool = True
    broll_enabled: bool = False
    remove_silence: bool = True
    auto_zoom: bool = True
    speaker_tracking: bool = True
    captions_enabled: bool = True
    translate_captions: bool = False
    language_hint: str = ""
    llm_enabled: bool = True

    @model_validator(mode="after")
    def _max_after_min(self) -> "ClipOptions":
        if {"min_clip_seconds", "max_clip_seconds"} <= self.model_fields_set and self.max_clip_seconds <= self.min_clip_seconds:
            raise ValueError("maximum duration must be greater than the minimum")
        return self

    def to_settings_patch(self) -> dict[str, Any]:
        """The explicitly chosen options as an AppSettings patch."""
        chosen = self.model_dump(include=self.model_fields_set - {"caption_preset"})
        if "caption_preset" in self.model_fields_set:
            chosen.update(caption_theme_patch(self.caption_preset))
        return chosen


class ProjectCreate(BaseModel):
    url: str = Field("", description="Any video link: YouTube, a direct .mp4, or a site yt-dlp supports")
    title: str = ""
    source_type: str = Field("", description="youtube | url | upload (inferred from the link when empty)")
    options: ClipOptions = Field(default_factory=ClipOptions)
    analyze: bool = True
    priority: int = 1

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            return ""  # uploads create projects without a URL
        if local_media_path(text) is not None:
            return text  # desktop app: a file on this machine is imported
        classify_url(text)  # raises a friendly error for anything unusable
        return text


class AnalyzeRequest(BaseModel):
    project_id: str = ""
    url: str = ""
    title: str = ""
    options: ClipOptions | None = None
    priority: int = Field(1, ge=0, le=10)
    force: bool = False


class ProjectAnalyzeRequest(BaseModel):
    """Body of ``POST /api/projects/{id}/analyze`` (everything optional)."""

    options: ClipOptions | None = None
    priority: int = Field(1, ge=0, le=10)
    force: bool = Field(False, description="Re-extract audio and ignore cached transcripts")


class RenderAllRequest(BaseModel):
    """Body of ``POST /api/projects/{id}/render-all``."""

    clip_ids: list[str] = Field(default_factory=list, description="Empty = every clip that still needs a render")
    export: bool = False
    force: bool = Field(False, description="Also re-render clips that are already rendered")


class ClipUpdate(BaseModel):
    title: str | None = None
    start: float | None = Field(None, ge=0)
    end: float | None = Field(None, ge=0)
    category: str | None = None
    hook: str | None = None
    summary: str | None = None
    layout: Literal["split", "podcast", "broll", "gameplay", "cinematic", "blur"] | None = None
    split_ratio: int | None = Field(None, ge=30, le=80)
    caption: dict[str, Any] | None = None
    caption_preset: str | None = None
    captions_enabled: bool | None = None
    remove_silence: bool | None = None
    auto_zoom: bool | None = None
    gameplay_enabled: bool | None = None
    gameplay_asset_id: str | None = Field(None, description='"" = pick one automatically')
    broll_asset_id: str | None = Field(None, description='"" = pick one automatically')
    music_asset_id: str | None = Field(None, description='"" = pick one automatically')
    music_enabled: bool | None = None
    aspect_ratio: Literal["9:16", "1:1", "16:9"] | None = None


class RenderRequest(BaseModel):
    clip_ids: list[str] = Field(default_factory=list)
    export: bool = False
    overrides: dict[str, Any] = Field(default_factory=dict)
    priority: int = 6


class SettingsUpdate(BaseModel):
    model_config = {"extra": "allow"}


class AssetUpdate(BaseModel):
    enabled: bool | None = None
    favorite: bool | None = None
    category: str | None = None


class TemplateCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str = ""
    config: dict[str, Any] = Field(default_factory=dict)


class AssetImportRequest(BaseModel):
    model_config = {"populate_by_name": True}

    path: str
    kind: Literal["gameplay", "broll", "music"] = "gameplay"
    category: str = "general"
    name: str = ""
    copy_file: bool = Field(True, alias="copy", serialization_alias="copy")


class CaptionPreviewRequest(BaseModel):
    clip_id: str = Field("", description="Preview this clip's words; empty = sample text")
    preset: str | None = None
    theme: dict[str, Any] | None = None


class OllamaTestRequest(BaseModel):
    base_url: str = ""
    model: str = ""


__all__ = [
    "AnalyzeRequest",
    "AssetImportRequest",
    "AssetUpdate",
    "CaptionPreviewRequest",
    "ClipOptions",
    "ClipUpdate",
    "OllamaTestRequest",
    "ProjectAnalyzeRequest",
    "ProjectCreate",
    "RenderAllRequest",
    "RenderRequest",
    "SettingsUpdate",
    "TemplateCreate",
    "caption_theme_patch",
    "local_media_path",
]
