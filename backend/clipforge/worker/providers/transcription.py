"""Transcription providers (TRD §11, §52).

``FasterWhisperProvider`` is the default local provider. ``FakeTranscriptionProvider``
exists for tests/CI and must be explicitly selected (``TRANSCRIPTION_PROVIDER=fake``).
"""

from __future__ import annotations

import json
import threading
import wave
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from clipforge.core.errors import AppError, ErrorCode
from clipforge.core.logging import get_logger

log = get_logger(__name__)

ProgressFn = Callable[[float], None]
CancelFn = Callable[[], bool]


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float | None = None


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    speaker_id: str | None = None


@dataclass
class TranscriptResult:
    language: str | None
    duration: float
    segments: list[Segment]
    provider: str
    model: str
    device: str = "cpu"
    language_probability: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the TRD §11 transcript schema."""
        return {
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "provider": self.provider,
            "model": self.model,
            "device": self.device,
            "segments": [asdict(s) for s in self.segments],
        }

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()


class TranscriptionProvider(Protocol):
    name: str

    def is_available(self) -> bool: ...

    def transcribe(self, audio_path: Path, *, model: str, device: str, compute_type: str,
                   language: str | None, progress: ProgressFn | None = None,
                   cancel_check: CancelFn | None = None) -> TranscriptResult: ...


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate() or 16000)


# ------------------------------------------------------------ faster-whisper
class FasterWhisperProvider:
    name = "faster_whisper"

    def __init__(self, model_dir: str = "", beam_size: int = 5) -> None:
        self.model_dir = model_dir or None
        self.beam_size = beam_size
        self._models: dict[tuple[str, str, str], Any] = {}
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    def load(self, model: str, device: str, compute_type: str) -> Any:
        key = (model, device, compute_type)
        with self._lock:
            if key not in self._models:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:
                    raise AppError(ErrorCode.MODEL_UNAVAILABLE, retryable=False,
                                   internal="faster-whisper is not installed (pip install 'clipforge[ai]')") from exc
                try:
                    self._models[key] = WhisperModel(model, device=device, compute_type=compute_type,
                                                     download_root=self.model_dir)
                except Exception as exc:  # model download / CUDA init failures
                    raise _map_model_error(exc) from exc
            return self._models[key]

    def unload_all(self) -> None:
        with self._lock:
            self._models.clear()

    def transcribe(self, audio_path: Path, *, model: str, device: str, compute_type: str,
                   language: str | None, progress: ProgressFn | None = None,
                   cancel_check: CancelFn | None = None) -> TranscriptResult:
        whisper = self.load(model, device, compute_type)
        try:
            seg_iter, info = whisper.transcribe(
                str(audio_path), language=language or None, beam_size=self.beam_size,
                word_timestamps=True, vad_filter=True, condition_on_previous_text=False,
            )
            total = float(getattr(info, "duration", 0.0) or wav_duration(audio_path))
            segments: list[Segment] = []
            for i, seg in enumerate(seg_iter):  # generator: transcription happens lazily here
                if cancel_check and cancel_check():
                    raise AppError(ErrorCode.JOB_CANCELLED)
                words = [Word(start=round(w.start, 3), end=round(w.end, 3), word=w.word.strip(),
                              probability=round(getattr(w, "probability", 0.0) or 0.0, 4))
                         for w in (seg.words or []) if w.word.strip()]
                segments.append(Segment(id=i, start=round(seg.start, 3), end=round(seg.end, 3),
                                        text=seg.text.strip(), words=words,
                                        avg_logprob=getattr(seg, "avg_logprob", None),
                                        no_speech_prob=getattr(seg, "no_speech_prob", None)))
                if progress and total:
                    progress(min(seg.end / total, 1.0))
        except AppError:
            raise
        except Exception as exc:
            raise _map_model_error(exc, default=ErrorCode.TRANSCRIPTION_FAILED) from exc
        return TranscriptResult(language=getattr(info, "language", None), duration=total, segments=segments,
                                provider=self.name, model=model, device=device,
                                language_probability=getattr(info, "language_probability", None))


def _map_model_error(exc: Exception, default: ErrorCode = ErrorCode.MODEL_UNAVAILABLE) -> AppError:
    msg = str(exc).lower()
    if "out of memory" in msg or "cuda_error_out_of_memory" in msg or "cublas" in msg and "alloc" in msg:
        return AppError(ErrorCode.INSUFFICIENT_MEMORY, internal=str(exc)[:500])
    if isinstance(exc, MemoryError):
        return AppError(ErrorCode.INSUFFICIENT_MEMORY, internal="MemoryError")
    if any(s in msg for s in ("cuda", "cudnn", "no such file", "not found", "download", "connection",
                              "repository", "unable to open")):
        return AppError(ErrorCode.MODEL_UNAVAILABLE, internal=str(exc)[:500])
    return AppError(default, internal=str(exc)[:500])


# ---------------------------------------------------------------------- fake
_FAKE_SCRIPT = [
    "Why do most people fail at learning something new?",
    "Here's the truth nobody tells you.",
    "It's not about talent, it's about the system you build around practice.",
    "When I started making videos, I was terrible.",
    "My first hundred uploads got almost no views, and honestly it was painful.",
    "Then I changed one simple thing.",
    "I started reviewing every video for ten minutes the next morning.",
    "That small habit showed me the same three mistakes again and again.",
    "So the lesson is simple: feedback beats motivation every single time.",
    "Now let's talk about money, because this part surprises everyone.",
    "Most creators think you need a million followers to earn a living.",
    "The data says otherwise.",
    "Creators with ten thousand loyal viewers often earn more than huge accounts.",
    "The reason is trust, and trust converts.",
    "That's why I always tell people to build a small audience that really cares.",
    "Okay, so what should you actually do tomorrow?",
    "First, pick one topic you can talk about for a year.",
    "Second, publish on a schedule you can actually keep.",
    "Third, study your best video and make it again, but better.",
    "And that's the whole framework, it is boring, and it works.",
]


class FakeTranscriptionProvider:
    """Deterministic transcript spread across the audio duration (tests/CI only)."""

    name = "fake"

    def __init__(self, script_path: str = "") -> None:
        self.script_path = script_path

    def is_available(self) -> bool:
        return True

    def _script(self) -> list[str]:
        if self.script_path and Path(self.script_path).exists():
            data = Path(self.script_path).read_text(encoding="utf-8")
            try:
                return [str(s) for s in json.loads(data)]
            except json.JSONDecodeError:
                return [line.strip() for line in data.splitlines() if line.strip()]
        return _FAKE_SCRIPT

    def transcribe(self, audio_path: Path, *, model: str, device: str, compute_type: str,
                   language: str | None, progress: ProgressFn | None = None,
                   cancel_check: CancelFn | None = None) -> TranscriptResult:
        duration = wav_duration(audio_path)
        segments: list[Segment] = []
        t = 0.6
        wps = 2.8
        i = 0
        script = self._script()
        while t < duration - 1.0:
            if cancel_check and cancel_check():
                raise AppError(ErrorCode.JOB_CANCELLED)
            sentence = script[i % len(script)]
            tokens = sentence.split()
            words: list[Word] = []
            for tok in tokens:
                dur = max(0.18, len(tok) / 12.0 + 0.12) / (wps / 2.8)
                if t + dur > duration - 0.2:
                    break
                words.append(Word(start=round(t, 3), end=round(t + dur, 3), word=tok, probability=0.95))
                t += dur + 0.06
            if not words:
                break
            segments.append(Segment(id=len(segments), start=words[0].start, end=words[-1].end,
                                    text=" ".join(w.word for w in words), words=words,
                                    avg_logprob=-0.25, no_speech_prob=0.02))
            t += 0.45 if (i % 5) else 1.4  # sentence pause, longer pause every 5 sentences
            i += 1
            if progress:
                progress(min(t / duration, 1.0))
        return TranscriptResult(language=language or "en", duration=duration, segments=segments,
                                provider=self.name, model="fake", device="cpu", language_probability=1.0)
