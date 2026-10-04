"""Render graph + edit-timeline round-trip checks.

Cheap, high-signal guards around ``clipforge.media.compose``: the ffmpeg filter
graph is built from real plans and inspected as text, and the timeline is pushed
through JSON exactly like the database does it. Encoding a real file is covered
by the ``e2e`` module.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipforge.config import AppSettings
from clipforge.media.compose import (
    MAX_ZOOM,
    RenderSpec,
    ZoomPoint,
    build_filter_graph,
    graph_summary,
    zoom_expression,
    zoom_filter,
)
from clipforge.media.framing import plan_for_layout
from clipforge.media.timeline import Segment, Timeline


def _timeline(*, removed: bool = True) -> Timeline:
    """A 20 s source trimmed to two keeps (5 s + 15 s) with 2.5 s of silence cut."""
    timeline = Timeline(
        segments=[Segment(100.0, 105.0, 0.0, 5.0), Segment(112.0, 127.0, 5.0, 20.0)],
        speed=1.0,
        notes=["Removed 2.5s of silence in 2 cuts (threshold 0.35s)."],
        source_start=100.0,
        source_end=127.0,
        source_duration=27.0,
    )
    if removed:
        timeline.removed = [(105.0, 106.0), (108.0, 109.5)]
    return timeline


def _spec(**overrides) -> RenderSpec:
    spec = RenderSpec(
        source_path=__file__,
        output_path=__file__,
        timeline=_timeline(),
        width=540,
        height=960,
        source_fps=30.0,
        settings=AppSettings(),
    )
    for key, value in overrides.items():
        setattr(spec, key, value)
    return spec


def test_timeline_round_trips_removed_silence():
    """A reloaded plan must still know how much silence it cut (used to report 0.0s)."""
    payload = json.loads(json.dumps(_timeline().to_dict()))
    restored = Timeline(
        segments=[
            Segment(seg["src_start"], seg["src_end"], seg["out_start"], seg["out_end"])
            for seg in payload["segments"]
        ],
        removed=[(pair[0], pair[1]) for pair in payload["removed"]],
        speed=payload["speed"],
    )

    assert _timeline().removed_seconds == pytest.approx(2.5)
    assert restored.removed_seconds == pytest.approx(2.5)
    assert restored.removed == [(105.0, 106.0), (108.0, 109.5)]

    # and the loader the render path uses rebuilds it from the stored JSON
    from clipforge.pipeline.edit import load_timeline

    class Clip:
        trim_json = json.dumps(payload)
        start, end = 100.0, 127.0

    loaded = load_timeline(Clip(), AppSettings())
    assert loaded.removed_seconds == pytest.approx(2.5)
    lines = graph_summary(_spec(timeline=loaded))
    assert any("2.5s removed" in line for line in lines), lines


def test_render_summary_reports_the_plan():
    lines = graph_summary(_spec())
    assert any(line.startswith("Output: 540x960") for line in lines), lines
    assert any("2.5s removed" in line for line in lines), lines
    assert any("Duration: 20.0s" in line for line in lines), lines


def test_zoom_expression_is_empty_without_points():
    assert zoom_expression([]) == ""


def test_zoom_expression_uses_supported_ffmpeg_primitives():
    """ffmpeg has no ``smoothstep``: the curve has to come from its primitives.

    A past regression shipped ``smoothstep()`` inside the filter graph and every
    render died with rc=234, so the expression is asserted, not merely exercised.
    """
    points = [
        ZoomPoint(time=6.0, strength=1.0, reason="hook"),
        ZoomPoint(time=21.0, strength=0.8, reason="reveal"),
    ]
    expression = zoom_expression(points, duration=40.0)

    assert expression.startswith("1.0000+min(0.14,"), expression
    assert "smoothstep" not in expression
    assert "between(in_time," in expression
    assert expression.count("clip(") >= 4  # rise + fall ramps for both punch-ins

    graph = zoom_filter(expression, panel_width=540, panel_height=960, fps=25)
    assert graph.startswith("zoompan=")
    assert "s=540x960" in graph
    assert ":fps=25" in graph
    assert "d=1" in graph
    # x/y are in input pixels, so the window must be centred through iw/zoom.
    assert "iw/2-(iw/zoom)/2" in graph


def test_zoom_drops_points_on_the_edges_and_too_close_together():
    points = [
        ZoomPoint(time=0.1, strength=1.0, reason="too early"),
        ZoomPoint(time=39.7, strength=1.0, reason="too late"),
        ZoomPoint(time=6.0, strength=1.0, reason="keep me"),
        ZoomPoint(time=6.2, strength=1.0, reason="too close"),
    ]
    expression = zoom_expression(points, duration=40.0)

    assert expression.count("between(in_time,") == 3, "one punch-in survives: three phase terms"
    assert "between(in_time,6.000" in expression and "5.550" in expression, "the rise starts 0.45s early"
    assert "6.200" not in expression


def test_zoom_amplitude_is_bounded():
    loud = zoom_expression([ZoomPoint(time=5.0, strength=9.0, reason="")], duration=30.0)
    quiet = zoom_expression([ZoomPoint(time=5.0, strength=0.0, reason="")], duration=30.0)

    assert "min(0.14," in loud, loud
    assert f"{MAX_ZOOM * 1.5:.4f}*" in loud, "an extreme strength must clamp at 1.5x"
    assert f"{MAX_ZOOM * 0.3:.4f}*" in quiet, "a weak moment still gets a visible push"


def test_punch_ins_run_on_the_normalised_output_cadence():
    """zoompan stamps one frame per input frame on its own clock.

    Its input is the stream already normalised to the output rate, so it must use
    that rate - a 25 fps source zoomed on a 25 fps clock after normalisation to
    30 fps came out 20% longer than its audio.
    """
    spec = _spec(layout="podcast", zoom_points=[ZoomPoint(time=4.0, strength=1.0, reason="hook")], source_fps=25.0)
    graph, video_label, _audio, _inputs = build_filter_graph(spec)

    assert f"fps={spec.fps}[vstd]" in graph
    assert f":fps={spec.fps}," in graph, "zoompan must run at the normalised output rate"
    assert ":fps=25" not in graph
    assert video_label == "vout"


def test_split_panel_keeps_the_speaker_aspect_ratio():
    """A crop cut for the full 9:16 frame is re-cut for the shorter split panel."""
    full_frame = plan_for_layout("split", 1920, 1080, 540, 960, [], duration=10.0, smart=False, tracking=False)
    spec = _spec(layout="split", crop_plan=full_frame, gameplay_path=Path("/tmp/gameplay.mp4"), split_ratio=60)
    graph, _video, _audio, _inputs = build_filter_graph(spec)

    panel_height = int(round(960 * 60 / 100.0 / 2) * 2)
    expected_crop_width = int(round(1080 * 540 / panel_height))
    expected_crop_width -= expected_crop_width % 2
    assert f"crop=w={expected_crop_width}:h=1080" in graph, graph[:400]
    assert f"scale=540:{panel_height}:force_original_aspect_ratio=increase" in graph
    assert f"crop=540:{panel_height}" in graph


def test_music_ducking_splits_the_voice_pad():
    """The voice feeds both amix and the side-chain, so it must be split first."""
    settings = AppSettings(music_enabled=True, ducking=True, ducking_db=-12.0)
    spec = _spec(layout="podcast", music_path=Path("/tmp/music.mp3"), settings=settings)
    graph, _video, _audio, _inputs = build_filter_graph(spec)

    assert "asplit=2[voice][voicekey]" in graph
    assert "[mus][voicekey]sidechaincompress" in graph
    assert graph.count("[voice]") == 2, "defined once by asplit, consumed once by amix"
    assert "ratio=9.0" in graph


def test_filter_graph_terminates_every_layer():
    spec = _spec(
        layout="podcast",
        crop_plan=plan_for_layout("podcast", 1920, 1080, 540, 960, [], duration=10.0, smart=False, tracking=False),
        zoom_points=[ZoomPoint(time=4.0, strength=1.0, reason="hook")],
    )

    graph, video_label, audio_label, input_args = build_filter_graph(spec)

    assert video_label == "vout" and audio_label == "aout"
    assert input_args.count("-i") == 1, "one source, one input"
    assert "-ss" in input_args and "-t" in input_args
    assert "zoompan=" in graph, "punch-ins must reach the filter graph"
    assert graph.rstrip().endswith("[vout]") is False, graph[-80:]
    assert "[vout]" in graph and "[aout]" in graph
    # captions and the format tail both hang off the framed label - never a bare pad
    assert "[vframed]format=yuv420p[vout]" in graph


def test_blur_layout_splits_the_pad_before_reading_it_twice():
    """ffmpeg refuses a label consumed twice: the background/frame split is mandatory."""
    spec = _spec(layout="blur")
    graph, _video, _audio, _inputs = build_filter_graph(spec)

    assert "[vstd]split=2[vbg][vfg]" in graph, "the pad is produced once and split immediately"
    # the pad is named exactly twice: produced by fps, consumed by split=2 only
    assert graph.count("[vstd]") == 2, graph
    assert graph.count("[vbg]") == 2 and graph.count("[vfg]") == 2
    assert "[bg][fg]overlay=(W-w)/2:(H-h)/2[vframed]" in graph


def test_split_layout_falls_back_to_one_panel_without_an_asset():
    spec = _spec(layout="gameplay", gameplay_path=None)
    graph, _video, _audio, _inputs = build_filter_graph(spec)

    assert "vstack" not in graph
    assert any("full-frame speaker cut" in note for note in spec.notes), spec.notes


def test_split_layout_stacks_speaker_and_gameplay(tone_media):
    """With a real asset the panels are laid out at the ratio the user picked."""
    for ratio, speaker_height in ((50, 480), (65, 624), (70, 672)):
        spec = _spec(
            layout="split",
            split_ratio=ratio,
            gameplay_path=tone_media,
            settings=AppSettings(split_ratio=ratio),
        )
        graph, video_label, audio_label, input_args = build_filter_graph(spec)

        assert f"scale=540:{speaker_height}" in graph, graph
        assert "[pod][asset]vstack=inputs=2[vframed]" in graph
        assert "-stream_loop" in input_args, "the asset loops under the speaker"
        assert video_label == "vout" and audio_label == "aout"


def test_gameplay_layout_overlays_the_speaker_on_the_asset(tone_media):
    """``gameplay`` is the picture-in-picture layout: full-screen asset + speaker inset."""
    spec = _spec(layout="gameplay", split_ratio=65, gameplay_path=tone_media)
    graph, video_label, _audio, input_args = build_filter_graph(spec)

    assert "[asset][podsmall]overlay=(W-w)/2:(H-h)/2[vframed]" in graph, graph
    assert "scale=432:" in graph, "the speaker inset is 80% of its panel"
    assert "-stream_loop" in input_args
    assert video_label == "vout"


def test_every_layout_renders_something(tone_media):
    for layout in ("podcast", "cinematic", "blur", "split", "gameplay", "broll"):
        spec = _spec(layout=layout, gameplay_path=tone_media, broll_path=tone_media)
        graph, video_label, audio_label, _inputs = build_filter_graph(spec)

        assert video_label == "vout" and audio_label == "aout"
        assert "[vout]" in graph and "[aout]" in graph
        assert graph.count("[vout]") == 1, "one label, one producer"
