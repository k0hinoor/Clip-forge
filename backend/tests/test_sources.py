"""Any-link ingestion and the caption settings that actually reach the encoder.

Two things broke in the wild and are pinned here:

* CLIPFORGE only accepted YouTube links - a plain ``.mp4`` URL, or a link to any
  other site, was rejected at the API boundary;
* the clip editor's caption controls were decorative: the preset it sent was not
  a known field, and the per-clip flags it stored were never read back at render
  time, so captions could not be restyled or switched off.
"""

from __future__ import annotations

import pytest

from clipforge.errors import ClipForgeError
from clipforge.media.download import classify_url


# --------------------------------------------------------------------------- #
# Links
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("url", "kind", "source_id"),
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "youtube", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "youtube", "dQw4w9WgXcQ"),
        ("youtube.com/shorts/dQw4w9WgXcQ", "youtube", "dQw4w9WgXcQ"),
    ],
)
def test_youtube_links_resolve_to_a_video_id(url, kind, source_id):
    source = classify_url(url)
    assert source.kind == kind
    assert source.source_id == source_id
    assert source.url == f"https://www.youtube.com/watch?v={source_id}"


@pytest.mark.parametrize(
    ("url", "suffix", "filename"),
    [
        ("https://cdn.example.com/talks/episode-3.mp4", ".mp4", "episode-3"),
        ("https://cdn.example.com/clip.webm?token=abc", ".webm", "clip"),
        ("https://cdn.example.com/stream/master.m3u8", ".m3u8", "master"),
        ("https://cdn.example.com/get?file=My%20Talk.mp4", ".mp4", "My_Talk"),
    ],
)
def test_direct_media_links_are_recognised(url, suffix, filename):
    source = classify_url(url)
    assert source.kind == "direct"
    assert source.suffix == suffix
    assert source.filename == filename
    assert source.url == url  # untouched: the CDN may need every query parameter


@pytest.mark.parametrize(
    "url",
    [
        "https://vimeo.com/76979871",
        "https://x.com/someone/status/1234567890",
        "https://www.dailymotion.com/video/x8t0h0z",
        "https://example.com/watch/abc",
    ],
)
def test_any_other_site_goes_to_the_yt_dlp_extractor(url):
    source = classify_url(url)
    assert source.kind == "web"
    assert source.source_id  # stable id so downloads can be cached
    assert source.filename


@pytest.mark.parametrize("url", ["", "   ", "ftp://example.com/a.mp4", "javascript:alert(1)", "hello"])
def test_unusable_links_are_rejected_with_a_friendly_error(url):
    with pytest.raises(ClipForgeError) as excinfo:
        classify_url(url)
    assert excinfo.value.code in {"invalid_url", "invalid_input"}
    assert excinfo.value.hint  # every rejection tells the user what to do next


def test_the_api_accepts_a_non_youtube_link():
    from clipforge.api.schemas import ProjectCreate

    payload = ProjectCreate(url="https://cdn.example.com/episode.mp4")
    assert payload.url == "https://cdn.example.com/episode.mp4"

    with pytest.raises(ClipForgeError):
        ProjectCreate(url="ftp://example.com/episode.mp4")


# --------------------------------------------------------------------------- #
# Captions
# --------------------------------------------------------------------------- #


def test_the_create_screen_caption_style_reaches_the_settings():
    from clipforge.api.schemas import ClipOptions

    patch = ClipOptions(caption_preset="karaoke").to_settings_patch()
    assert patch["caption"]["preset"] == "karaoke"
    assert patch["caption"]["animation"] == "karaoke"


def test_a_clip_can_be_given_a_caption_style_by_name():
    from clipforge.api.schemas import ClipUpdate

    update = ClipUpdate(caption_preset="minimal")
    assert update.caption_preset == "minimal"


def test_per_clip_caption_choices_change_the_render_settings():
    from clipforge.config import AppSettings
    from clipforge.pipeline.edit import clip_settings

    base = AppSettings()
    layout = {
        "captions_enabled": False,
        "aspect_ratio": "1:1",
        "remove_silence": False,
        "caption": {"font_size": 90, "preset": "cinematic"},
    }

    effective = clip_settings(base, layout)
    assert effective.captions_enabled is False
    assert effective.aspect_ratio == "1:1"
    assert effective.aspect_dims() == (1080, 1080)
    assert effective.remove_silence is False
    assert effective.caption.font_size == 90
    assert effective.caption.preset == "cinematic"

    # A plan with nothing stored keeps the project settings untouched.
    assert clip_settings(base, {"layout": "split"}).model_dump() == base.model_dump()


def test_changing_the_aspect_ratio_moves_the_pixel_size():
    from clipforge.config import AppSettings

    settings = AppSettings.model_validate({**AppSettings().model_dump(), "aspect_ratio": "16:9"})
    assert settings.aspect_dims() == (1920, 1080)


def test_rendering_without_libass_degrades_instead_of_failing(tmp_path, monkeypatch):
    """No libass must still produce a clip - and say why it has no captions."""
    from clipforge.media import compose
    from clipforge.media.compose import RenderSpec
    from clipforge.media.captions import CaptionWord, plan_captions
    from clipforge.media.timeline import Segment, Timeline
    from clipforge.config import CaptionTheme
    from clipforge.system import FfmpegInfo

    monkeypatch.setattr(compose, "ffmpeg_info", lambda: FfmpegInfo(available=True, ffmpeg="ffmpeg"))

    spec = RenderSpec(
        source_path=tmp_path / "source.mp4",
        output_path=tmp_path / "out.mp4",
        timeline=Timeline(segments=[Segment(0.0, 10.0, 0.0, 10.0)], speed=1.0, notes=[],
                          source_start=0.0, source_end=10.0, source_duration=10.0),
        captions=plan_captions([CaptionWord("hello", 0.2, 0.8)], theme=CaptionTheme()),
        subtitle_path=tmp_path / "captions.ass",
        source_fps=30.0,
    )

    graph, _, _, _ = compose.build_filter_graph(spec)
    assert "ass='" not in graph  # no subtitle filter in the chain
    assert any("libass" in note for note in spec.notes)
