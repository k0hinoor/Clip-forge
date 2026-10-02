from __future__ import annotations

from clipforge.core.config import Settings
from clipforge.core.presets import caption_presets
from clipforge.media.captions import build_caption_timeline, render_ass, wrap_lines
from clipforge.media.framing import cover_crop, plan_framing, smooth_positions
from clipforge.scoring.segmentation import FlatWord


def words(text: str, start: float = 0.0, step: float = 0.4) -> list[FlatWord]:
    return [FlatWord(start=start + i * step, end=start + i * step + step * 0.9, word=w, segment_idx=0)
            for i, w in enumerate(text.split())]


def test_wrap_lines_respects_limits():
    lines = wrap_lines(["quick", "brown", "fox", "jumps", "over"], 14, 2)
    assert lines == ["quick brown", "fox jumps over"]
    assert wrap_lines(["supercalifragilistic"], 14, 2) == ["supercalifragilistic"]  # never splits a word


def test_caption_timeline_is_relative_and_bounded():
    preset = caption_presets(Settings())["bold"]
    ws = words("one two three four five six seven eight nine ten eleven twelve", start=10.0)
    timeline = build_caption_timeline(ws, 10.0, 14.0, preset)
    assert timeline
    assert timeline[0]["start"] >= 0.0
    assert all(c["end"] <= 4.0 + 1e-6 for c in timeline)
    assert all(c["end"] > c["start"] for c in timeline)
    for a, b in zip(timeline, timeline[1:], strict=False):
        assert a["end"] <= b["start"] + 1e-6


def test_ass_escapes_override_tags_and_fits_width():
    preset = caption_presets(Settings())["creator"]
    tl = build_caption_timeline(words("hello {\\pos(1,1)} world"), 0, 3, preset)
    ass = render_ass(tl, preset, 1080, 1920)
    assert "\\pos(1,1)" not in ass
    style = next(line for line in ass.splitlines() if line.startswith("Style:"))
    size = int(style.split(",")[2])
    # 14 bold uppercase chars must fit inside a 1080px frame with margins
    assert size * 14 * 0.74 <= 1080 * 0.88


def test_all_presets_render():
    presets = caption_presets(Settings())
    assert {"clean", "bold", "creator", "minimal", "high_contrast"} <= set(presets)
    for preset in presets.values():
        tl = build_caption_timeline(words("a quick brown fox jumps"), 0, 3, preset)
        assert "[Events]" in render_ass(tl, preset, 1080, 1920)


def test_cover_crop():
    assert cover_crop(1920, 1080, "9:16") == (606, 1080) or cover_crop(1920, 1080, "9:16")[1] == 1080
    w, h = cover_crop(1920, 1080, "1:1")
    assert w == h == 1080


def test_framing_auto_modes():
    common = dict(src_w=1920, src_h=1080, out_w=1080, out_h=1920, aspect="9:16", duration=10.0)
    # no visual data -> centre crop
    assert plan_framing(mode="auto", frames=None, **common)["mode"] == "center"
    # faces in most frames -> face tracking
    frames = [{"t": t * 0.5, "face_boxes": [[0.65, 0.3, 0.1, 0.18]]} for t in range(20)]
    plan = plan_framing(mode="auto", frames=frames, **common)
    assert plan["mode"] == "face" and plan["face_ratio"] >= 0.9
    # no faces at all -> fit with blurred background
    empty = [{"t": t * 0.5, "face_boxes": []} for t in range(20)]
    assert plan_framing(mode="auto", frames=empty, **common)["mode"] == "fit"
    # same aspect ratio -> no crop needed
    same = dict(common, out_w=1920, out_h=1080, aspect="16:9")
    assert plan_framing(mode="auto", frames=frames, **same)["mode"] == "center"


def test_smoothing_limits_speed():
    samples = [(i * 0.5, 0.2 if i < 5 else 0.8) for i in range(12)]
    out = smooth_positions(samples, crop_size=0.3, max_speed=0.6, dead_zone=0.08)
    for (t0, x0), (t1, x1) in zip(out, out[1:], strict=False):
        assert abs(x1 - x0) <= 0.6 * 0.3 * (t1 - t0) + 1e-6
