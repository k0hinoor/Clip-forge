"""Speech-to-text.

Primary engine: **faster-whisper** (CTranslate2) - real local transcription with
word-level timestamps, multilingual support and CUDA acceleration when the
machine has an NVIDIA GPU. Optional **WhisperX** alignment improves timestamp
accuracy when installed.

Fallback order when faster-whisper is missing:

1. WhisperX (if installed), else
2. YouTube caption track downloaded with the video (**only** when the user has
   not disabled it), else
3. a clear, actionable error - CLIPFORGE never invents a transcript.
"""

from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ..config import AppSettings, get_settings
from ..errors import ClipForgeError, ErrorCode
from ..logging_setup import get_logger
from ..system import ai_stack, fit_whisper_model, hardware

log = get_logger("clipforge.ai")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Word:
    word: str
    start: float
    end: float
    confidence: float = 0.0
    speaker: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "confidence": round(self.confidence, 3),
            "speaker": self.speaker,
        }


@dataclass
class Utterance:
    text: str
    start: float
    end: float
    speaker: str = "SPEAKER_01"
    confidence: float = 0.0
    words: list[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Transcript:
    language: str
    language_confidence: float
    utterances: list[Utterance]
    engine: str
    model: str
    duration: float
    language_probabilities: dict[str, float] = field(default_factory=dict)
    diarized: bool = False
    speaker_count: int = 0
    warnings: list[str] = field(default_factory=list)
    translated: bool = False
    source_language: str = ""
    source: str = "whisper"
    filename: str = ""
    timing_granularity: str = "word"  # word for ASR/alignment; cue for subtitle sources

    @property
    def words(self) -> list[Word]:
        out: list[Word] = []
        for utterance in self.utterances:
            out.extend(utterance.words)
        return out

    @property
    def text(self) -> str:
        return " ".join(utterance.text.strip() for utterance in self.utterances if utterance.text.strip())

    @property
    def word_count(self) -> int:
        timed_words = sum(len(utterance.words) for utterance in self.utterances)
        if timed_words:
            return timed_words
        # Subtitle-only input has text words but intentionally no per-word timings.
        return sum(len((utterance.text or "").split()) for utterance in self.utterances)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Transcript":
        """Inverse of :meth:`to_dict` (used by the transcript cache)."""
        utterances = [
            Utterance(
                text=str(item.get("text", "")),
                start=float(item.get("start", 0.0)),
                end=float(item.get("end", 0.0)),
                speaker=str(item.get("speaker") or "SPEAKER_01"),
                confidence=float(item.get("confidence", 0.0)),
                words=[
                    Word(
                        word=str(word.get("word", "")),
                        start=float(word.get("start", 0.0)),
                        end=float(word.get("end", 0.0)),
                        confidence=float(word.get("confidence", 0.0)),
                        speaker=str(word.get("speaker", "")),
                    )
                    for word in item.get("words") or []
                ],
            )
            for item in payload.get("utterances") or []
        ]
        engine = str(payload.get("engine", ""))
        if "source" in payload:
            source = str(payload.get("source") or "whisper")
        elif "caption" in engine.lower():
            source = "source_captions"
        elif "transcript-file" in engine.lower():
            source = "uploaded_legacy"
        else:
            source = "whisper"
        timing_granularity = str(payload.get("timing_granularity") or ("word" if any(item.words for item in utterances) else "cue"))
        return cls(
            language=str(payload.get("language", "en")),
            language_confidence=float(payload.get("language_confidence", 0.0)),
            utterances=utterances,
            engine=engine,
            model=str(payload.get("model", "")),
            duration=float(payload.get("duration", 0.0)),
            language_probabilities=dict(payload.get("language_probabilities") or {}),
            diarized=bool(payload.get("diarized", False)),
            speaker_count=int(payload.get("speaker_count", 0)),
            warnings=list(payload.get("warnings") or []),
            translated=bool(payload.get("translated", False)),
            source_language=str(payload.get("source_language", "")),
            source=source,
            filename=str(payload.get("filename", "")),
            timing_granularity=timing_granularity,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "language_confidence": round(self.language_confidence, 3),
            "language_probabilities": {k: round(v, 4) for k, v in list(self.language_probabilities.items())[:8]},
            "engine": self.engine,
            "model": self.model,
            "duration": round(self.duration, 3),
            "word_count": self.word_count,
            "speaker_count": self.speaker_count,
            "diarized": self.diarized,
            "warnings": self.warnings,
            "translated": self.translated,
            "source_language": self.source_language,
            "source": self.source,
            "filename": self.filename,
            "timing_granularity": self.timing_granularity,
            "utterances": [
                {
                    "text": utterance.text,
                    "start": round(utterance.start, 3),
                    "end": round(utterance.end, 3),
                    "speaker": utterance.speaker,
                    "confidence": round(utterance.confidence, 3),
                    "words": [word.to_dict() for word in utterance.words],
                }
                for utterance in self.utterances
            ],
        }


# --------------------------------------------------------------------------- #
# Model cache (loading a Whisper model takes seconds - keep it warm)
# --------------------------------------------------------------------------- #

_MODELS: dict[tuple[str, str, str], Any] = {}
_MODEL_LOCK = threading.Lock()


def _device_and_compute(settings: AppSettings) -> tuple[str, str]:
    report = hardware()
    has_cuda = bool(report.get("gpu", {}).get("cuda"))
    device = settings.whisper_device
    if device == "auto":
        device = "cuda" if has_cuda else "cpu"
    if device == "cuda" and not has_cuda:
        log.warning("CUDA requested but no NVIDIA GPU detected; using CPU")
        device = "cpu"

    compute = settings.whisper_compute_type
    if compute == "auto":
        if device == "cuda":
            compute = "float16"
        else:
            compute = "int8"  # fastest on CPU and accurate enough for captions
    if device == "cpu" and compute in {"float16", "int8_float16"}:
        compute = "int8"
    return device, compute


def _load_faster_whisper(settings: AppSettings) -> tuple[Any, str, str, str]:
    """Load (or reuse) a faster-whisper model: ``(model, device, compute, model_name)``.

    The model is downgraded when the configured one cannot fit in memory (see
    :func:`clipforge.system.fit_whisper_model`); the reason is logged.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        raise ClipForgeError(
            code=ErrorCode.TRANSCRIPTION_UNAVAILABLE,
            message="Local speech-to-text is not installed.",
            hint='Install it with:  pip install -r requirements-ai.txt   (installs faster-whisper)',
            status_code=503,
        ) from exc

    device, compute = _device_and_compute(settings)
    model_name = settings.whisper_model
    if device == "cpu":
        model_name, note = fit_whisper_model(settings.whisper_model)
        if note:
            log.warning(note)
    key = (model_name, device, compute)
    with _MODEL_LOCK:
        cached = _MODELS.get(key)
        if cached is not None:
            return cached, device, compute, model_name

    model_dir = Path(settings.whisper_cache_dir) if getattr(settings, "whisper_cache_dir", "") else None
    log.info("loading faster-whisper model=%s device=%s compute=%s", model_name, device, compute)
    started = time.time()
    try:
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute,
            download_root=str(model_dir) if model_dir else None,
            cpu_threads=0,
            num_workers=1,
        )
    except Exception as exc:  # noqa: BLE001
        message = str(exc).lower()
        if "cuda" in message or "cublas" in message or "cudnn" in message:
            log.warning("CUDA init failed (%s); retrying on CPU", exc)
            device, compute = "cpu", "int8"
            model_name, note = fit_whisper_model(settings.whisper_model)
            if note:
                log.warning(note)
            key = (model_name, device, compute)
            try:
                model = WhisperModel(
                    model_name,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(model_dir) if model_dir else None,
                    num_workers=1,
                )
            except Exception as exc2:  # noqa: BLE001
                raise ClipForgeError(
                    code=ErrorCode.WHISPER_FAILED,
                    message="The speech recognition model could not be loaded.",
                    hint="Try a smaller model in Settings -> Transcription (tiny/base/small), or reinstall faster-whisper.",
                    detail=str(exc2),
                    status_code=500,
                ) from exc2
        else:
            raise ClipForgeError(
                code=ErrorCode.WHISPER_FAILED,
                message=f"The Whisper model '{model_name}' could not be loaded.",
                hint="Check the model name (tiny, base, small, medium, large-v3) and your internet connection for the first download.",
                detail=str(exc),
                status_code=500,
            ) from exc

    with _MODEL_LOCK:
        _MODELS[key] = model
    log.info("faster-whisper %s ready in %.1fs", model_name, time.time() - started)
    return model, device, compute, model_name


def unload_models() -> None:
    with _MODEL_LOCK:
        _MODELS.clear()
    gc.collect()


# --------------------------------------------------------------------------- #
# faster-whisper transcription
# --------------------------------------------------------------------------- #


def transcribe_faster_whisper(
    audio_path: str | Path,
    *,
    settings: AppSettings | None = None,
    progress: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    language: str | None = None,
) -> Transcript:
    settings = settings or get_settings()
    model, device, compute, model_name = _load_faster_whisper(settings)
    audio = str(audio_path)

    kwargs: dict[str, Any] = {
        "beam_size": max(1, settings.whisper_beam_size),
        "word_timestamps": True,
        "vad_filter": settings.whisper_vad,
        "condition_on_previous_text": settings.whisper_condition_on_previous_text,
        "temperature": 0.0,
        "language": language or settings.language_hint or None,
        # Never translate unless the user switched translation on.
        "task": "translate" if settings.translate_captions else "transcribe",
    }
    if settings.whisper_vad:
        kwargs["vad_parameters"] = {
            "min_silence_duration_ms": settings.whisper_vad_min_silence_ms,
            "speech_pad_ms": 120,
        }
    if settings.whisper_initial_prompt:
        kwargs["initial_prompt"] = settings.whisper_initial_prompt
    if settings.whisper_batch_size > 1:
        kwargs["batch_size"] = settings.whisper_batch_size

    log.info("transcribing %s (device=%s compute=%s)", Path(audio).name, device, compute)
    started = time.time()
    try:
        segment_iter, info = model.transcribe(audio, **kwargs)
    except TypeError:
        # Older faster-whisper without batch_size support.
        kwargs.pop("batch_size", None)
        segment_iter, info = model.transcribe(audio, **kwargs)

    duration = float(getattr(info, "duration", 0.0) or 0.0)
    if not duration:
        try:
            from ..media.ffmpeg import probe_media

            duration = probe_media(audio).duration
        except Exception:
            duration = 0.0

    utterances: list[Utterance] = []
    words: list[Word] = []
    for index, segment in enumerate(segment_iter):
        if should_cancel and should_cancel():
            raise ClipForgeError(code=ErrorCode.CANCELLED, message="Transcription cancelled.", status_code=409)
        segment_words: list[Word] = []
        for word in getattr(segment, "words", None) or []:
            text = (word.word or "").strip()
            if not text:
                continue
            confidence = float(getattr(word, "probability", 0.0) or 0.0)
            segment_words.append(
                Word(
                    word=text,
                    start=float(word.start or 0.0),
                    end=float(word.end or word.start or 0.0),
                    confidence=confidence,
                )
            )
        text = (segment.text or "").strip()
        if not text and not segment_words:
            continue
        segment_words = _repair_word_times(segment_words, float(segment.start), float(segment.end))
        words.extend(segment_words)
        utterances.append(
            Utterance(
                text=text,
                start=float(segment.start or 0.0),
                end=float(segment.end or 0.0),
                confidence=_mean([word.confidence for word in segment_words]) if segment_words else 0.0,
                words=segment_words,
            )
        )
        if progress and duration:
            progress(min(float(segment.end) / duration, 1.0), f"transcribed {_clock(segment.end)} / {_clock(duration)}")

    if not utterances:
        raise ClipForgeError(
            code=ErrorCode.NO_SPEECH,
            message="No speech was detected in this audio.",
            hint="The video may be music-only, silent, or in a language the current model cannot hear. Try a larger model.",
            status_code=422,
        )

    probabilities = dict(getattr(info, "all_language_probs", None) or {})
    detected = getattr(info, "language", "") or "en"
    confidence = float(getattr(info, "language_probability", 0.0) or 0.0)
    log.info("transcribed %s (%.0fs of audio) in %.1fs", Path(audio).name, duration, time.time() - started)

    warnings = [] if confidence >= 0.6 else ["Language detection confidence is low; captions may need review."]
    if model_name != settings.whisper_model:
        warnings.append(fit_whisper_model(settings.whisper_model)[1] or f"Used the '{model_name}' model.")
    transcript = Transcript(
        language=detected,
        language_confidence=confidence,
        utterances=utterances,
        engine=f"faster-whisper ({device}/{compute})",
        model=model_name,
        duration=duration or (utterances[-1].end if utterances else 0.0),
        language_probabilities=probabilities,
        warnings=warnings,
    )
    return _mark_translated(transcript) if settings.translate_captions else transcript


def _mark_translated(transcript: Transcript) -> Transcript:
    """Label a Whisper ``task="translate"`` result: the text is English now."""
    spoken = transcript.language or "unknown"
    transcript.source_language = spoken
    transcript.translated = spoken != "en"
    transcript.language = "en"
    if transcript.translated:
        transcript.engine = f"{transcript.engine} - translated"
        transcript.warnings.append(
            f"Speech in '{spoken}' was translated to English by Whisper; word timings in translated captions are approximate."
        )
    return transcript


def _repair_word_times(words: list[Word], segment_start: float, segment_end: float) -> list[Word]:
    """Make word timings monotonic and inside their segment."""
    if not words:
        return words
    fixed: list[Word] = []
    cursor = segment_start
    for index, word in enumerate(words):
        start = max(word.start, cursor, segment_start)
        end = max(word.end, start + 0.02)
        if index == len(words) - 1 and segment_end > start:
            end = min(max(end, start + 0.02), max(segment_end, start + 0.02))
        cursor = end
        fixed.append(Word(word=word.word, start=start, end=end, confidence=word.confidence, speaker=word.speaker))
    return fixed


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _clock(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


# --------------------------------------------------------------------------- #
# WhisperX (optional, better alignment / diarization)
# --------------------------------------------------------------------------- #


def whisperx_available() -> bool:
    return bool(ai_stack().get("whisperx"))


def transcribe_whisperx(
    audio_path: str | Path,
    *,
    settings: AppSettings | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> Transcript:
    settings = settings or get_settings()
    try:
        import whisperx  # type: ignore
    except ImportError as exc:
        raise ClipForgeError(
            code=ErrorCode.TRANSCRIPTION_UNAVAILABLE,
            message="WhisperX is not installed.",
            hint="pip install whisperx  (adds forced alignment and diarization)",
            status_code=503,
        ) from exc

    device, compute = _device_and_compute(settings)
    device = "cuda" if device == "cuda" else "cpu"
    if progress:
        progress(0.05, "loading WhisperX")
    model = whisperx.load_model(settings.whisper_model, device, compute_type="float16" if device == "cuda" else "int8")
    audio = whisperx.load_audio(str(audio_path))
    if progress:
        progress(0.15, "transcribing")
    task = "translate" if settings.translate_captions else "transcribe"
    try:
        result = model.transcribe(audio, batch_size=max(1, settings.whisper_batch_size), language=settings.language_hint or None, task=task)
    except TypeError:  # older WhisperX without the task argument
        if task == "translate":
            raise ClipForgeError(
                code=ErrorCode.WHISPER_FAILED,
                message="This WhisperX version cannot translate.",
                hint="Set Word alignment to 'auto' so faster-whisper handles translation, or update WhisperX.",
                status_code=500,
            ) from None
        result = model.transcribe(audio, batch_size=max(1, settings.whisper_batch_size), language=settings.language_hint or None)
    language = result.get("language", "en")

    utterances: list[Utterance] = []
    for segment in result.get("segments", []):
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        utterances.append(
            Utterance(
                text=text,
                start=float(segment.get("start") or 0.0),
                end=float(segment.get("end") or 0.0),
                confidence=float(segment.get("avg_logprob") and min(1.0, max(0.0, 1 + segment["avg_logprob"])) or 0.5),
            )
        )

    # Forced alignment matches text against the spoken audio, which is
    # meaningless for a translation - translated captions keep segment timings.
    if task == "transcribe":
        try:
            if progress:
                progress(0.55, "aligning word timestamps")
            align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
            aligned = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)
            utterances = _utterances_from_whisperx(aligned.get("segments", []), utterances)
        except Exception as exc:  # noqa: BLE001 - alignment is best-effort
            log.warning("WhisperX alignment failed (%s); falling back to segment timings", exc)

    if progress:
        progress(0.9, "finalising transcript")
    transcript = Transcript(
        language=language,
        language_confidence=0.9,
        utterances=utterances,
        engine=f"whisperx ({device})",
        model=settings.whisper_model,
        duration=utterances[-1].end if utterances else 0.0,
    )
    return _mark_translated(transcript) if task == "translate" else transcript


def _utterances_from_whisperx(segments: list[dict[str, Any]], fallback: list[Utterance]) -> list[Utterance]:
    out: list[Utterance] = []
    for index, segment in enumerate(segments):
        words = [
            Word(
                word=str(word.get("word", "")).strip(),
                start=float(word.get("start") or 0.0),
                end=float(word.get("end") or word.get("start") or 0.0),
                confidence=float(word.get("score") or 0.0),
            )
            for word in segment.get("words", [])
            if str(word.get("word", "")).strip()
        ]
        text = (segment.get("text") or "").strip()
        if not text and not words:
            continue
        out.append(
            Utterance(
                text=text or " ".join(word.word for word in words),
                start=float(segment.get("start") or 0.0),
                end=float(segment.get("end") or 0.0),
                confidence=_mean([word.confidence for word in words]) if words else 0.0,
                words=words,
            )
        )
    return out or fallback


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def transcribe(
    audio_path: str | Path,
    *,
    settings: AppSettings | None = None,
    progress: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Transcript:
    """Transcribe audio using the best locally available engine."""
    settings = settings or get_settings()
    stack = ai_stack()

    # WhisperX's forced alignment cannot help a translation, so faster-whisper
    # handles translated runs whenever it is installed.
    prefer_whisperx = settings.word_alignment == "whisperx" and not (settings.translate_captions and stack.get("faster_whisper"))
    if prefer_whisperx and stack.get("whisperx"):
        try:
            return transcribe_whisperx(audio_path, settings=settings, progress=progress)
        except ClipForgeError as exc:
            log.warning("WhisperX path failed (%s); falling back to faster-whisper", exc.message)

    if stack.get("faster_whisper"):
        return transcribe_faster_whisper(audio_path, settings=settings, progress=progress, should_cancel=should_cancel)

    if settings.word_alignment in {"auto", "whisperx"} and stack.get("whisperx"):
        return transcribe_whisperx(audio_path, settings=settings, progress=progress)

    raise ClipForgeError(
        code=ErrorCode.TRANSCRIPTION_UNAVAILABLE,
        message="No local speech-to-text engine is installed.",
        hint=(
            "Install the AI extras:  pip install -r requirements-ai.txt   "
            "(faster-whisper). CLIPFORGE can also fall back to YouTube captions, "
            "but only if the video has a caption track."
        ),
        status_code=503,
    )


def transcription_available() -> bool:
    return bool(ai_stack().get("transcription_available"))


__all__ = [
    "Transcript",
    "Utterance",
    "Word",
    "transcribe",
    "transcribe_faster_whisper",
    "transcribe_whisperx",
    "transcription_available",
    "unload_models",
    "whisperx_available",
]
