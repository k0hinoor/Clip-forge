"""Loaders for configuration-driven presets (scoring, plans, renders, captions)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from clipforge.core.config import Settings


@lru_cache(maxsize=16)
def _load_yaml(path: str, mtime: float) -> dict[str, Any]:  # mtime busts the cache on edits
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping")
    return data


def load_yaml(path: Path) -> dict[str, Any]:
    p = Path(path)
    return _load_yaml(str(p), p.stat().st_mtime)


def scoring_config(settings: Settings) -> dict[str, Any]:
    return load_yaml(settings.SCORING_CONFIG_FILE)


def plans_config(settings: Settings) -> dict[str, dict[str, Any]]:
    plans = dict(load_yaml(settings.PLANS_CONFIG_FILE).get("plans", {}))
    if settings.FREE_DAILY_MINUTES is not None and "free" in plans:
        plans["free"] = {**plans["free"], "daily_source_minutes": settings.FREE_DAILY_MINUTES}
    return plans


def render_profiles_config(settings: Settings) -> dict[str, dict[str, Any]]:
    return dict(load_yaml(settings.RENDER_PROFILES_FILE).get("profiles", {}))


def caption_presets(settings: Settings) -> dict[str, dict[str, Any]]:
    return dict(load_yaml(settings.CAPTION_PRESETS_FILE).get("presets", {}))


ASPECT_RATIOS = ("9:16", "16:9", "1:1")
FRAMING_MODES = ("auto", "face", "center", "fit")
