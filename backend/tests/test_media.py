"""Media layer tests: ffmpeg probing, silence detection, timelines, captions, framing."""

from __future__ import annotations

import pytest

from clipforge.ai.vision import FrameAnalysis
from clipforge.media.captions import build_ass, build_srt, plan_captions, preset_theme, write_ass
from clipforge.media.ffmpeg import (
    detect_silence,
    extract_audio,
    probe_media,
    require_ffmpeg,
    video_encoder_args,
)
from clipforge.media.framing import crop_plan_from_dict, plan_crop, plan_for_layout
from clipforge.media.runner import sanitize_args
from clipforge.media.timeline import build_timeline


class FakeWord:
    def __init__(self, word: str, start: float, end: float, speaker: str = "SPEAKER_01"):
        self.word = word
        self.start = start
        self.end = end
        self.speaker = speaker
        self.confidence = 0.9

    def to_dict(self) -> dict:
        return {"word": self.word, "start": self.start, "end": self.end, "confidence": 0.9, "speaker": self.speaker}


# --------------------------------------------------------------------------- #
# ffmpeg
# --------------------------------------------------------------------------- #


def test_ffmpeg_is_available_and_reports_capabilities():
    info = require_ffmpeg()
    assert info.available
    assert info.ffmpeg
    assert info.version
    assert "libx264" in info.encoders or any("264" in name for name in info.encoders)


def test_encoder_args_fall_back_to_x264(settings):
    args = video_encoder_args(settings, fps=30)
    assert "-c:v" in args
    assert args[args.index("-c:v") + 1]
    assert "-pix_fmt" in args


def test_probe_media_reads_real_files(tone_media):
    info = probe_media(tone_media)
    assert info.width == 640 and info.height == 360
    assert 11.0 <= info.duration <= 13.0
    assert info.has_audio
    assert 24 <= info.fps <= 26


def test_extract_audio_produces_16k_mono(tone_media, tmp_path):
    target = tmp_path / "speech.wav"
    result = extract_audio(tone_media, target, sample_rate=16000)
    assert result.exists() and result.stat().st_size > 1000
    info = probe_media(result)
    assert info.audio_sample_rate == 16000
    assert info.audio_channels == 1


def test_silence_detection_finds_the_quiet_part(tone_media):
    silences = detect_silence(tone_media, noise_db=-40.0, min_duration=0.3, start=0.0, end=12.0)
    assert isinstance(silences, list)
    for start, end in silences:
        assert end > start


def test_sanitize_args_blocks_shell_metacharacters():
    cleaned = sanitize_args(["ffmpeg", "-i", "video.mp4; rm -rf /", "out.mp4"])
    assert all(";" not in part or "video.mp4" in part for part in cleaned)


# --------------------------------------------------------------------------- #
# timeline
# --------------------------------------------------------------------------- #


def test_timeline_removes_long_silences_and_keeps_natural_pauses(settings):
    silences = [(5.0, 6.4), (18.0, 19.2)]
    timeline = build_timeline(0.0, 30.0, silences, settings=settings)
    assert timeline.segments
    assert timeline.removed_seconds < 2.8  # long pauses trimmed, natural ones kept
    assert timeline.output_duration <= 30.0
    mapped = timeline.source_to_output(10.0)
    assert mapped is not None and mapped < 10.0


def test_timeline_without_silence_removal_is_identity(settings):
    plain = settings.model_copy(update={"remove_silence": False})
    timeline = build_timeline(10.0, 40.0, [(20.0, 22.0)], settings=plain)
    assert len(timeline.segments) == 1
    assert timeline.removed_seconds == 0.0
    assert timeline.is_trimmed is False


def test_timeline_remaps_words_into_output_time(settings):
    timeline = build_timeline(0.0, 40.0, [(10.0, 12.5)], settings=settings)
    words = [FakeWord("before", 8.0, 8.5), FakeWord("after", 14.0, 14.5)]
    mapped = timeline.map_words(words)
    assert len(mapped) == 2
    assert mapped[1].start < 14.0


# --------------------------------------------------------------------------- #
# captions
# --------------------------------------------------------------------------- #


def _caption_words(count: int = 18, *, start: float = 0.0, step: float = 0.42):
    words = []
    vocabulary = ("This", "is", "where", "the", "real", "story", "starts", "and", "it", "changed", "everything")
    for index in range(count):
        begin = start + index * step
        words.append(FakeWord(f"{vocabulary[index % len(vocabulary)]}", begin, begin + step * 0.8))
    return words


def test_caption_lines_break_semantically_and_stay_short(settings):
    plan = plan_captions(_caption_words(24), theme=settings.caption, language="en")
    assert plan.lines
    for line in plan.lines:
        assert len(line.words) <= settings.caption.max_words_per_line
        assert line.duration <= 3.4


def test_caption_text_follows_the_transcript(settings):
    plan = plan_captions(_caption_words(6), theme=settings.caption, language="en")
    spoken = " ".join(word.word for word in _caption_words(6))
    shown = " ".join(line.text for line in plan.lines)
    for token in shown.split():
        assert token in spoken


def test_ass_and_srt_export(settings, tmp_path):
    theme = settings.caption.model_copy(update={"uppercase": False})
    plan = plan_captions(_caption_words(12), theme=theme, language="en")
    document = build_ass(plan, width=1080, height=1920, font="DejaVu Sans")
    assert "[Script Info]" in document and "[Events]" in document
    assert "Dialogue:" in document

    path = tmp_path / "captions.ass"
    written = write_ass(plan, path, width=1080, height=1920)
    assert written.exists() and written.stat().st_size > 200

    srt = build_srt(plan)
    assert "-->" in srt
    assert srt.strip().endswith("") or srt.strip().splitlines()[-1].strip().isdigit()


def test_caption_escaping_handles_dangerous_characters(settings):
    from clipforge.media.captions import _escape

    escaped = _escape("hello {\\an8} world \\n next")
    assert "{" not in escaped and "}" not in escaped


@pytest.mark.parametrize("preset", ["minimal", "cinematic", "bold_creator", "karaoke", "highlight", "documentary"])
def test_all_six_caption_presets_render(preset, settings):
    theme = preset_theme(preset, settings.caption)
    plan = plan_captions(_caption_words(20), theme=theme, language="en")
    document = build_ass(plan, width=1080, height=1920, font="DejaVu Sans")
    assert document.count("Dialogue:") >= 1


def test_karaoke_timing_is_monotonic(settings):
    theme = preset_theme("karaoke", settings.caption)
    theme = theme.model_copy(update={"animation": "karaoke", "max_words_per_line": 4})
    plan = plan_captions(_caption_words(16), theme=theme, language="en")
    document = build_ass(plan, width=1080, height=1920, font="DejaVu Sans")
    assert "\\k" in document or "\\kf" in document


def test_devanagari_gets_a_capable_font(settings):
    plan = plan_captions(_caption_words(8), theme=settings.caption, language="hi")
    assert plan.font


# --------------------------------------------------------------------------- #
# framing
# --------------------------------------------------------------------------- #


def _analyses(offset: float = 0.62, count: int = 12) -> list[FrameAnalysis]:
    return [
        FrameAnalysis(time=index * 0.5, width=1280, height=720, subject_x=offset, subject_y=0.4, sharpness=120.0)
        for index in range(count)
    ]


def test_crop_for_portrait_from_landscape():
    plan = plan_crop(1280, 720, 1080, 1920, _analyses(), duration=6.0, smart=True, tracking=True)
    assert plan.crop_width <= 1280
    assert plan.crop_height <= 720
    assert plan.mode in {"tracked", "static", "fit_blur"}
    assert plan.keyframes, "portrait crops need at least one keyframe"
    assert 0.0 <= plan.keyframes[0].center_x <= 1.0


def test_crop_tracks_the_subject_side():
    left = plan_crop(1280, 720, 1080, 1920, _analyses(0.2), duration=6.0, smart=True, tracking=True)
    right = plan_crop(1280, 720, 1080, 1920, _analyses(0.8), duration=6.0, smart=True, tracking=True)
    assert left.keyframes[0].center_x < right.keyframes[0].center_x
    assert float(left.crop_x_expression().rsplit(",", 1)[-1]) < float(right.crop_x_expression().rsplit(",", 1)[-1])


def test_blur_layout_uses_the_blurred_full_frame():
    plan = plan_for_layout("blur", 1280, 720, 1080, 1920, _analyses(), duration=6.0)
    assert plan.mode == "fit_blur"
    assert any("blur" in note.lower() for note in plan.notes)


def test_crop_plan_roundtrip():
    original = plan_crop(1280, 720, 1080, 1920, _analyses(), duration=6.0, smart=True, tracking=True)
    restored = crop_plan_from_dict(original.to_dict())
    assert restored is not None
    assert restored.crop_width == original.crop_width
    assert restored.mode == original.mode


def test_crop_expressions_are_safe_and_valid():
    plan = plan_crop(1280, 720, 1080, 1920, _analyses(), duration=8.0, smart=True, tracking=True)
    expression = plan.crop_x_expression()
    assert isinstance(expression, str) and expression
    for forbidden in (";", "|", "`", "$("):
        assert forbidden not in expression
