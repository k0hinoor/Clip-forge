"""Settings, capability schema and storage tests."""

from __future__ import annotations

import pytest


def test_defaults_match_the_product_spec():
    from clipforge.config import AppSettings

    # Deliberately *not* the live store: other tests change settings, and this
    # test is about what a fresh install ships with.
    settings = AppSettings()
    assert settings.aspect_ratio == "9:16"
    assert (settings.output_width, settings.output_height) == (1080, 1920)
    assert settings.output_fps == 30
    assert (settings.min_clip_seconds, settings.target_clip_seconds, settings.max_clip_seconds) == (35.0, 60.0, 75.0)
    assert settings.min_score == 70.0
    assert settings.max_clips == 0  # no arbitrary cap
    assert settings.remove_silence is True
    assert settings.silence_min_duration == 0.35
    assert settings.split_ratio == 65
    assert settings.caption.preset in {"minimal", "cinematic", "bold_creator", "karaoke", "highlight", "documentary"}
    assert settings.captions_enabled  # one caption switch, not two that can disagree
    assert settings.translate_captions is False  # never translate unless asked
    assert 8.0 <= settings.target_lufs * -1 <= 16.0


def test_nine_settings_sections_are_exposed():
    from clipforge.config import SECTION_FIELDS

    expected = {"general", "ai", "transcription", "video", "captions", "gameplay", "audio", "export", "storage"}
    assert set(SECTION_FIELDS) == expected
    for section, fields in SECTION_FIELDS.items():
        assert fields, f"{section} has no fields"


def test_update_persists_and_validates():
    from clipforge.config import get_settings, settings_store

    store = settings_store()
    updated = store.update({"min_score": 82.5, "target_clip_seconds": 58, "whisper_model": "medium"})
    assert updated.min_score == 82.5
    assert updated.target_clip_seconds == 58
    assert get_settings().whisper_model == "medium"

    with pytest.raises(Exception):
        store.update({"min_score": 250})

    with pytest.raises(Exception):
        store.update({"output_fps": -5})

    store.reset()
    assert get_settings().min_score == 70.0


def test_caption_presets_all_build_a_theme():
    from clipforge.constants import CAPTION_PRESETS
    from clipforge.media.captions import preset_theme

    assert set(CAPTION_PRESETS) == {"minimal", "cinematic", "bold_creator", "karaoke", "highlight", "documentary"}
    for name in CAPTION_PRESETS:
        theme = preset_theme(name)
        assert theme.font
        assert theme.font_size > 0
        assert theme.primary_color.startswith("#")


def test_clip_limits_helper(settings):
    low, target, high = settings.clip_limits()
    assert low <= target <= high


def test_aspect_dims_setting_roundtrip(settings):
    from clipforge.config import ASPECT_PRESETS

    assert "9:16" in ASPECT_PRESETS
    width, height = ASPECT_PRESETS["9:16"]
    assert (width, height) == (1080, 1920)
