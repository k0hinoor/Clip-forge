"""Local AI components that must work offline: diarization, transcripts, language ID."""

from __future__ import annotations

import json

import pytest

from clipforge.ai import diarize as diarize_mod


# --------------------------------------------------------------------------- #
# Diarization (pure numpy - no downloads, always runnable)
# --------------------------------------------------------------------------- #


def test_diarization_separates_two_voices(speech_wav):
    from clipforge.ai.transcribe import Utterance, Word

    utterances = [
        Utterance(text="first voice", start=0.2, end=1.9, words=[Word("first", 0.2, 0.9), Word("voice", 1.0, 1.9)]),
        Utterance(text="second voice", start=3.1, end=5.8, words=[Word("second", 3.1, 4.2), Word("voice", 4.3, 5.8)]),
    ]
    grouped, report = diarize_mod.diarize(speech_wav.path, utterances, max_speakers=3, mode="energy")
    assert report["speakers"] >= 1
    assert len(grouped) == len(utterances)
    labels = {utterance.speaker for utterance in grouped}
    assert all(label.startswith("SPEAKER_") for label in labels)
    assert grouped[0].start <= grouped[1].start


def test_diarization_can_be_disabled(speech_wav):
    from clipforge.ai.transcribe import Utterance, Word

    utterances = [Utterance(text="hello", start=0.2, end=1.5, words=[Word("hello", 0.2, 1.5)])]
    grouped, report = diarize_mod.diarize(speech_wav.path, utterances, mode="off")
    assert grouped[0].speaker == utterances[0].speaker
    assert report.get("mode") == "off"


def test_speaker_turn_map_is_monotonic(speech_wav):
    from clipforge.ai.transcribe import Utterance, Word

    utterances = [
        Utterance(text="a", start=0.1, end=1.0, speaker="SPEAKER_01", words=[Word("a", 0.1, 1.0)]),
        Utterance(text="b", start=1.2, end=2.4, speaker="SPEAKER_02", words=[Word("b", 1.2, 2.4)]),
        Utterance(text="c", start=3.2, end=4.6, speaker="SPEAKER_01", words=[Word("c", 3.2, 4.6)]),
    ]
    turns = diarize_mod.speaker_turn_map(utterances)
    assert len(turns) == 3
    for turn in turns:
        assert turn["end"] > turn["start"]


# --------------------------------------------------------------------------- #
# Transcript files (SRT / VTT / json3 / txt)
# --------------------------------------------------------------------------- #

SRT_SAMPLE = """1
00:00:01,000 --> 00:00:04,000
Nobody talks about what happens after you become successful.

2
00:00:04,200 --> 00:00:08,000
Everyone thinks success is going to solve their problems.
"""

VTT_SAMPLE = """WEBVTT

00:00:01.000 --> 00:00:03.500
I charged twelve thousand rupees.

00:00:03.600 --> 00:00:06.000
For four months of my life.
"""


def test_srt_parsing_produces_monotonic_word_timings(tmp_path):
    from clipforge.media.download import parse_subtitle_captions

    path = tmp_path / "captions.srt"
    path.write_text(SRT_SAMPLE)
    words = parse_subtitle_captions(path)
    assert len(words) == 18
    assert words[0]["word"] == "Nobody"
    assert words[0]["start"] == pytest.approx(1.0, abs=0.05)
    assert words[-1]["end"] == pytest.approx(7.8, abs=0.4)
    for previous, following in zip(words, words[1:]):
        assert previous["end"] <= following["start"] + 0.001, (previous, following)
        assert previous["start"] <= previous["end"]


def test_vtt_parsing_works_too(tmp_path):
    from clipforge.media.download import parse_subtitle_captions

    path = tmp_path / "captions.vtt"
    path.write_text(VTT_SAMPLE)
    words = parse_subtitle_captions(path)
    assert "twelve" in [word["word"] for word in words]
    assert words[0]["start"] == pytest.approx(1.0, abs=0.05)


def test_generated_srt_round_trips(settings, tmp_path):
    """Whatever the caption engine writes must be readable again."""
    from clipforge.media.captions import build_srt, plan_captions
    from clipforge.media.download import parse_subtitle_captions

    class _Word:
        def __init__(self, word, start, end):
            self.word, self.start, self.end, self.confidence, self.speaker = word, start, end, 0.9, "S1"

    words = [_Word(f"word{index}", index * 0.5, index * 0.5 + 0.4) for index in range(12)]
    plan = plan_captions(words, theme=settings.caption, language="en")
    path = tmp_path / "roundtrip.srt"
    path.write_text(build_srt(plan), encoding="utf-8")
    parsed = parse_subtitle_captions(path)
    assert len(parsed) == 12
    assert parsed[0]["word"] == "word0"


def test_transcript_file_discovery_prefers_json3(tmp_path):
    from clipforge.media.download import transcript_files

    (tmp_path / "notes.txt").write_text("[00:01] hello world this is a transcript line")
    (tmp_path / "caps.srt").write_text(SRT_SAMPLE)
    (tmp_path / "auto.json3").write_text(
        json.dumps({"events": [{"tStartMs": 1000, "segs": [{"utf8": "hello"}, {"utf8": "there"}]}]})
    )
    found = transcript_files(tmp_path)
    assert [path.suffix for path in found][0] == ".json3"


def test_json3_caption_parser_handles_word_level_events(tmp_path):
    from clipforge.media.download import parse_json3_captions

    payload = {
        "events": [
            {"tStartMs": 1000, "dDurationMs": 2000, "segs": [{"utf8": "Hello "}, {"utf8": "world", "acAsrConf": 0.9}]},
            {"tStartMs": 3200, "dDurationMs": 1500, "segs": [{"utf8": "again"}]},
        ]
    }
    path = tmp_path / "captions.json3"
    path.write_text(json.dumps(payload))
    words = parse_json3_captions(path)
    assert [word["word"] for word in words] == ["Hello", "world", "again"]
    assert words[0]["start"] == pytest.approx(1.0, abs=0.05)
    assert words[-1]["start"] == pytest.approx(3.2, abs=0.05)


# --------------------------------------------------------------------------- #
# Speech recognition (only when a model is already on disk)
# --------------------------------------------------------------------------- #


def _cached_whisper_model(model: str = "tiny") -> bool:
    """True when faster-whisper already has the model locally (no download needed)."""
    import os

    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:  # noqa: BLE001
        HF_HUB_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    from pathlib import Path

    for root in {Path(HF_HUB_CACHE), Path(os.environ.get("CLIPFORGE_DATA_DIR", "")) / "cache"}:
        if root.exists() and any(root.glob(f"**/models--*faster-whisper-{model}")):
            return True
    return False


@pytest.mark.asr
def test_transcribe_reports_a_friendly_error_without_a_model(monkeypatch, speech_wav):
    """With no model available the error must be actionable, never a traceback."""
    import clipforge.ai.transcribe as transcribe_mod
    from clipforge.errors import ClipForgeError

    if _cached_whisper_model("tiny"):
        pytest.skip("a cached tiny model exists - the failure path cannot be exercised offline")

    class _Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("connection to huggingface was closed")

    monkeypatch.setattr(transcribe_mod, "_MODELS", {}, raising=False)
    monkeypatch.setattr(transcribe_mod, "ai_stack", lambda: {"faster_whisper": True, "transcription_available": True})

    settings = transcribe_mod.get_settings().model_copy(update={"whisper_model": "tiny", "whisper_device": "cpu"})
    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", _Boom)
    with pytest.raises(ClipForgeError) as error:
        transcribe_mod.transcribe(speech_wav.path, settings=settings)
    assert error.value.message
    assert "model" in error.value.hint.lower() or "settings" in error.value.hint.lower()


@pytest.mark.asr
def test_transcribe_with_a_cached_model(speech_wav):
    """Real ASR on the synthetic audio (skipped unless a model is already cached)."""
    if not _cached_whisper_model("tiny"):
        pytest.skip("no cached faster-whisper model: run a transcription once with internet access")

    from clipforge.ai.transcribe import transcribe
    from clipforge.config import get_settings

    settings = get_settings().model_copy(update={"whisper_model": "tiny", "whisper_device": "cpu", "whisper_compute_type": "int8"})
    transcript = transcribe(speech_wav.path, settings=settings)
    assert transcript.engine
    assert transcript.utterances
    for utterance in transcript.utterances:
        assert utterance.end >= utterance.start
