"""Request/response models for the REST API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ..media.download import classify_url


def caption_theme_patch(preset: str) -> dict[str, Any]:
    """Map a caption preset name onto the caption theme it defines."""
    from ..constants import CAPTION_PRESETS
    from ..media.captions import preset_theme
    from ..config import CaptionTheme

    if not preset or preset not in CAPTION_PRESETS:
        return {}
    return {"caption": preset_theme(preset, CaptionTheme()).model_dump()}


class ClipOptions(BaseModel):
    """Optional per-project overrides for the clip search (Create screen)."""

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
    translate_captions: bool = False
    language_hint: str = ""
    llm_enabled: bool = True

    @field_validator("max_clip_seconds")
    @classmethod
    def _max_after_min(cls, value: float, info) -> float:
        minimum = info.data.get("min_clip_seconds", 5.0)
        if value <= minimum:
            raise ValueError("maximum duration must be greater than the minimum")
        return value

    def to_settings_patch(self) -> dict[str, Any]:
        return {
            "clip_mode": self.clip_mode,
            "target_clip_seconds": self.target_clip_seconds,
            "min_clip_seconds": self.min_clip_seconds,
            "max_clip_seconds": self.max_clip_seconds,
            "min_score": self.min_score,
            "max_clips": self.max_clips,
            "aspect_ratio": self.aspect_ratio,
            "layout": self.layout,
            "split_ratio": self.split_ratio,
            "gameplay_enabled": self.gameplay_enabled,
            "broll_enabled": self.broll_enabled,
            "remove_silence": self.remove_silence,
            "auto_zoom": self.auto_zoom,
            "speaker_tracking": self.speaker_tracking,
            "translate_captions": self.translate_captions,
            "language_hint": self.language_hint,
            "llm_enabled": self.llm_enabled,
            **caption_theme_patch(self.caption_preset),
        }


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
        classify_url(text)  # raises a friendly error for anything unusable
        return text


class AnalyzeRequest(BaseModel):
    project_id: str = ""
    url: str = ""
    options: ClipOptions | None = None
    priority: int = 1
    force: bool = False


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
    gameplay_asset_id: str | None = None
    music_asset_id: str | None = None
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
    clip_id: str
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
    "ProjectCreate",
    "RenderRequest",
    "SettingsUpdate",
    "TemplateCreate",
]
