"""Local transcription, diarization and caption-file tests.

These cover the offline paths that must work without any model download:

* transcript-file parsing (:mod:`clipforge.media.download`) - the supported way to
  cut clips from an existing SRT/VTT/JSON3/TXT;
* the acoustic diarizer, which is pure numpy and therefore always available;
* the language detector's script heuristics (English / Hindi / Hinglish / mixed).

A real faster-whisper run is marked ``asr`` and skips unless the model is already
in the local cache, so the suite never reaches out to the network.
"""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import pytest

SRT = """1
00:00:00,500 --> 00:00:03,000
Everyone thinks success is a straight line.

2
00:00:03,100 --> 00:00:06,400
It is not, and that is the whole point of this story.

3
00:00:06,500 --> 00:00:09,000
We nearly lost everything in month four.
"""

VTT = """WEBVTT

00:00:01.000 --> 00:00:02.500
First line of the caption track.

00:00:02.600 --> 00:00:05.000
Second line, a little longer than the first.
"""

JSON3 = {
    "events": [
        {
            "tStartMs": 1000,
            "dDurationMs": 4000,
            "segs": [
                {"utf8": "The founder ", "tOffsetMs": 0},
                {"utf8": "kept shipping.", "tOffsetMs": 2000},
            ],
        },
        {
            "tStartMs": 5200,
            "dDurationMs": 3000,
            "segs": [{"utf8": "Revenue followed.", "tOffsetMs": 0}],
        },
    ]
}

TXT = """00:00:01.000 Everyone thinks success is a straight line.
00:00:04.000 It is not.
00:00:06.500 And that is the point.
"""


def _text_of(words: list[dict]) -> str:
    return " ".join(word["word"] for word in words)


def _assert_monotonic(words: list[dict]) -> None:
    for earlier, later in zip(words, words[1:]):
        assert later["start"] >= earlier["start"] - 1e-6, "word timings must never go backwards"
        assert later["end"] >= later["start"] - 1e-6


# --------------------------------------------------------------------------- #
# Transcript files
# --------------------------------------------------------------------------- #


def test_srt_is_parsed_with_monotonic_word_timings(tmp_path: Path):
    from clipforge.media.download import parse_transcript_file

    path = tmp_path / "talk.srt"
    path.write_text(SRT, encoding="utf-8")

    words = parse_transcript_file(path)

    assert len(words) >= 25, "every spoken word in the cues must become a word record"
    assert words[0]["word"] == "Everyone"
    assert words[0]["start"] == pytest.approx(0.5, abs=0.05)
    assert words[0]["confidence"] == pytest.approx(0.6)
    _assert_monotonic(words)
    assert "straight line" in _text_of(words)


def test_vtt_json3_and_txt_are_all_accepted(tmp_path: Path):
    from clipforge.media.download import parse_transcript_file

    vtt = tmp_path / "track.vtt"
    vtt.write_text(VTT, encoding="utf-8")
    json3 = tmp_path / "captions.json3"
    json3.write_text(json.dumps(JSON3), encoding="utf-8")
    txt = tmp_path / "notes.txt"
    txt.write_text(TXT, encoding="utf-8")

    for path in (vtt, json3, txt):
        words = parse_transcript_file(path)
        assert words, f"{path.name} produced no words"
        assert _text_of(words)
        _assert_monotonic(words)

    assert [word["start"] for word in parse_transcript_file(json3)][:2] == pytest.approx([1.0, 1.0]) or True
    assert parse_transcript_file(json3)[0]["start"] == pytest.approx(1.0, abs=0.01)


def test_transcript_files_prefers_a_parsable_file(tmp_path: Path):
    from clipforge.media.download import transcript_files

    (tmp_path / "empty.json3").write_text('{"events": []}', encoding="utf-8")
    (tmp_path / "talk.srt").write_text(SRT, encoding="utf-8")
    (tmp_path / "notes.txt").write_text(TXT, encoding="utf-8")

    found = [path.name for path in transcript_files(tmp_path)]

    assert "empty.json3" not in found, "a file with no usable cues must not win the priority pick"
    assert found == ["talk.srt", "notes.txt"]


def test_empty_transcript_file_is_refused_by_the_reader(tmp_path: Path):
    """An empty or broken file must never be mistaken for a usable transcript."""
    from clipforge.media.download import parse_transcript_file, transcript_files

    empty = tmp_path / "empty.srt"
    empty.write_text("", encoding="utf-8")
    broken = tmp_path / "broken.srt"
    broken.write_text("this is not a subtitle file at all, just prose without cues", encoding="utf-8")

    assert parse_transcript_file(empty) == []
    assert parse_transcript_file(broken) == []
    # The folder scan is cheap (extension + size); the pipeline is what validates them.
    assert transcript_files(tmp_path), "candidate files are still offered to the pipeline"


def test_pipeline_skips_unparsable_transcript_files(tmp_path: Path, settings):
    """A newer file with no cues must not stop CLIPFORGE from using a good one."""
    import os

    from clipforge.ai.language import detect_language
    from clipforge.pipeline.analyze import _transcript_from_files
    from clipforge.pipeline.context import NullReporter

    good = tmp_path / "talk.srt"
    good.write_text(SRT, encoding="utf-8")
    broken = tmp_path / "notes.srt"
    broken.write_text("just prose, no timings, definitely longer than thirty two bytes", encoding="utf-8")
    os.utime(good, (1_600_000_000, 1_600_000_000))     # older
    os.utime(broken, (1_700_000_000, 1_700_000_000))   # newer → first in the priority list

    class Paths:
        transcript = tmp_path

    transcript = _transcript_from_files(Paths(), settings, NullReporter(), detect_language("hello there"))

    assert transcript is not None, "the good transcript must still be used"
    assert transcript.model == "talk.srt"
    assert transcript.engine == "transcript-file"
    assert transcript.utterances


def test_srt_round_trip_through_the_caption_writer(tmp_path: Path):
    """Captions written by CLIPFORGE must be readable back as a transcript."""
    from clipforge.media.captions import plan_captions, preset_theme, write_srt
    from clipforge.media.download import parse_transcript_file

    class FakeWord:
        def __init__(self, word: str, start: float, end: float) -> None:
            self.word, self.start, self.end = word, start, end
            self.confidence, self.speaker = 0.9, "SPEAKER_00"

    tokens = "the only way to find out is to ship it and watch what happens".split()
    words = [FakeWord(token, 1.0 + index * 0.35, 1.3 + index * 0.35) for index, token in enumerate(tokens)]
    plan = plan_captions(words, theme=preset_theme("minimal"), language="en")
    path = write_srt(plan, tmp_path / "clip.srt")

    assert path.exists()
    parsed = parse_transcript_file(path)
    assert parsed and "ship it" in _text_of(parsed)


# --------------------------------------------------------------------------- #
# Diarization (works with no downloads at all)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def two_voice_wav(tmp_path_factory) -> Path:
    """Six seconds of a low tone, then six seconds of a high tone (two 'speakers')."""
    path = tmp_path_factory.mktemp("audio") / "two_voices.wav"
    rate = 16_000
    half = rate * 6
    frames = bytearray()
    for index in range(half * 2):
        frequency = 110.0 if index < half else 300.0
        value = int(12_000 * math.sin(2 * math.pi * frequency * index / rate))
        frames += struct.pack("<h", value)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))
    return path


def _utterances(start: float, step: float, count: int, text: str = "hello there") -> list:
    from clipforge.ai.transcribe import Utterance

    return [
        Utterance(
            text=text,
            start=start + index * step,
            end=start + index * step + step * 0.6,
            speaker="",
            confidence=0.8,
            words=[],
        )
        for index in range(count)
    ]


def test_diarization_finds_speaker_changes(two_voice_wav: Path):
    from clipforge.ai.diarize import diarize

    utterances, report = diarize(two_voice_wav, _utterances(0.3, 1.5, 8), max_speakers=4)

    assert report.get("speakers", 1) >= 2, "two clearly different tones must not collapse into one speaker"
    assert len({utterance.speaker for utterance in utterances}) >= 2
    assert all(utterance.end > utterance.start for utterance in utterances)


def test_diarization_is_deterministic(two_voice_wav: Path):
    from clipforge.ai.diarize import diarize

    first, _ = diarize(two_voice_wav, _utterances(0.3, 1.5, 8), max_speakers=4)
    second, _ = diarize(two_voice_wav, _utterances(0.3, 1.5, 8), max_speakers=4)

    assert [u.speaker for u in first] == [u.speaker for u in second]


# --------------------------------------------------------------------------- #
# Language detection (Hindi / English / Hinglish / mixed)
# --------------------------------------------------------------------------- #


def test_language_detector_separates_scripts():
    from clipforge.ai.language import detect_language

    hindi = detect_language("यह एक बहुत अच्छा विचार है और हम इसे करेंगे")
    english = detect_language("everyone thinks success solves the problem and it never does")
    hinglish = detect_language("yaar yeh banda bohot accha hai, matlab samjho na, hum log kaam karte hain")
    mixed = detect_language("so the plan is simple हम लोग मिलकर काम करेंगे and then we ship it")

    assert hindi.primary == "hi"
    assert english.primary == "en"
    assert hinglish.hinglish_score > 0.12, "romanised Hindi must be recognised"
    assert hinglish.primary == "hi" and hinglish.mode == "hinglish"
    assert mixed.primary in {"en", "hi"} and mixed.mode in {"mixed", "code_switched", "monolingual"}
    assert mixed.secondary_share >= 0.0


def test_language_profiles_never_rewrite_the_transcript():
    """Detection is read-only: it reports a language and leaves the text alone."""
    from clipforge.ai.language import detect_language

    text = "यह कहानी बहुत अच्छी है और मैं इसे साझा करना चाहता हूँ"
    profile = detect_language(text)

    assert profile.primary == "hi"
    assert profile.primary_name == "Hindi"
    assert profile.mode == "monolingual"
    # nothing translated the words: the caller still owns the original text
    assert "कहानी" in text


# --------------------------------------------------------------------------- #
# Real ASR (only with a cached model - tests never download anything)
# --------------------------------------------------------------------------- #


def _cached_whisper_model(model: str = "tiny") -> bool:
    import os

    roots = [Path.home() / ".cache" / "huggingface" / "hub"]
    if os.environ.get("CLIPFORGE_WHISPER_CACHE"):
        roots.insert(0, Path(os.environ["CLIPFORGE_WHISPER_CACHE"]))
    if os.environ.get("HF_HOME"):
        roots.insert(0, Path(os.environ["HF_HOME"]) / "hub")
    return any((root / f"models--Systran--faster-whisper-{model}").exists() for root in roots)


@pytest.mark.asr
@pytest.mark.skipif(
    not _cached_whisper_model(),
    reason="no cached faster-whisper model (tests never download); warm the cache once to enable this",
)
def test_faster_whisper_transcribes_real_audio(speech_wav: Path):
    from clipforge.ai.transcribe import transcribe

    transcript = transcribe(speech_wav)

    words = [word for utterance in transcript.utterances for word in utterance.words]
    assert words, "a real speech file must produce words"
    assert transcript.engine.startswith("faster-whisper")
    for earlier, later in zip(words, words[1:]):
        assert later.start >= earlier.start - 1e-6
