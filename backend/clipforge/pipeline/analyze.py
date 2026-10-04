"""The analysis pipeline: YouTube URL → ranked, editable clips.

Stage order (each one reports real progress and can be cancelled):

    metadata → download → audio → language → transcribe → diarize →
    segment → discover → score → dedupe → validate → boundaries → prepare

Nothing here is simulated: every stage either produces artefacts from the actual
media (ffmpeg, Whisper, acoustic clustering, statistic scoring) or fails loudly
with an actionable message.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ..ai import candidates as discover_mod
from ..ai import language as language_mod
from ..ai import scoring as scoring_mod
from ..ai.diarize import diarize
from ..ai.features import TranscriptIndex
from ..ai.llm import LLMStatus, discover_moments
from ..ai.llm import status as llm_status
from ..ai.segment import build_blocks, build_sentences, sentence_map
from ..ai.transcribe import Transcript, Utterance, transcribe, transcription_available
from ..config import AppSettings, get_settings
from ..constants import progress_for
from ..db import Candidate, Clip, Project, session_scope, utcnow
from ..errors import ClipForgeError, ErrorCode
from ..logging_setup import get_logger
from ..media.download import (
    download_captions,
    download_video,
    fetch_metadata,
    parse_json3_captions,
)
from ..media.ffmpeg import extract_audio, probe_media
from ..system import ai_stack, ensure_disk_space, ffmpeg_info, hardware
from .context import (
    NullReporter,
    ProgressReporter,
    ProjectPaths,
    cached_source_path,
    link_or_copy,
    load_sentences,
    save_transcript,
    write_analysis_files,
    write_transcript_files,
)

log = get_logger("clipforge.worker")

PREPARE_LIMIT_DEFAULT = 60


@dataclass
class AnalysisOutcome:
    project_id: str
    candidate_count: int
    clip_count: int
    language: str
    speakers: int
    duration: float
    llm_used: bool
    llm_note: str = ""
    stats: dict[str, Any] = None  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "candidates": self.candidate_count,
            "clips": self.clip_count,
            "language": self.language,
            "speakers": self.speakers,
            "duration": self.duration,
            "llm_used": self.llm_used,
            "llm_note": self.llm_note,
            "stats": self.stats or {},
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def project_settings(project: Project) -> AppSettings:
    """Merge global settings with the per-project overrides chosen on Create."""
    base = get_settings().model_dump()
    overrides = {key: value for key, value in project.settings.items() if key in base}
    if overrides:
        base = {**base, **overrides}
    return AppSettings.model_validate(base)


def _update_project(project_id: str, **fields: Any) -> None:
    with session_scope() as session:
        project = session.get(Project, project_id)
        if project is None:
            return
        for key, value in fields.items():
            if key.endswith("_json") and not isinstance(value, str):
                value = json.dumps(value, default=str)
            setattr(project, key, value)


def _check_cancel(report: ProgressReporter) -> None:
    if report.cancelled():
        raise ClipForgeError(code=ErrorCode.CANCELLED, message="Analysis cancelled.", status_code=409)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #


def analyze(
    project_id: str,
    *,
    report: ProgressReporter | None = None,
    force: bool = False,
) -> AnalysisOutcome:
    """Run the whole analysis for an existing project row."""
    report = report or NullReporter()
    started = time.time()

    with session_scope() as session:
        project = session.get(Project, project_id)
        if project is None:
            raise ClipForgeError(code=ErrorCode.NOT_FOUND, message="Project not found.", status_code=404)
        snapshot = project.to_dict()
        source_url = project.source_url
        source_type = project.source_type
        title = project.title

    settings = project_settings_from_snapshot(snapshot)
    paths = ProjectPaths.for_snapshot(snapshot).ensure()
    if not force:
        ensure_disk_space(2.0)

    stats: dict[str, Any] = {"started_at": time.time(), "stages": {}}

    def mark(stage: str, seconds: float, **extra: Any) -> None:
        stats["stages"][stage] = {"seconds": round(seconds, 2), **extra}

    # ------------------------------------------------------------ 1. metadata
    report.stage("metadata", "reading video metadata", fraction=0.05)
    t0 = time.time()
    media_info = None
    metadata: dict[str, Any] = {}
    if source_type == "youtube":
        meta = fetch_metadata(source_url)
        metadata = meta.to_dict()
        _update_project(
            project_id,
            title=meta.title,
            channel=meta.channel,
            duration=meta.duration,
            thumbnail_url=meta.thumbnail_url,
            source_id=meta.video_id,
            description=(meta.description or "")[:4000],
        )
        paths.metadata.mkdir(parents=True, exist_ok=True)
        (paths.metadata / "source.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=1), encoding="utf-8")
        title = meta.title
    else:
        existing = paths.source
        local = next((path for path in existing.iterdir() if path.is_file()), None) if existing.exists() else None
        if local is None:
            raise ClipForgeError(
                code=ErrorCode.NOT_FOUND,
                message="The uploaded file for this project is missing.",
                hint="Delete the project and upload the file again.",
                status_code=404,
            )
        media_info = probe_media(local)
        metadata = {"title": title, "duration": media_info.duration, "uploader": "local file"}
        _update_project(project_id, duration=media_info.duration)
    mark("metadata", time.time() - t0)

    # ------------------------------------------------------------ 2. download
    report.stage("download", "acquiring the source video", fraction=0.02)
    t0 = time.time()
    media_path = _find_existing_media(paths)
    if media_path is None and source_type == "youtube":
        if media_info is None:
            media_info = None
        media_path = _download_source(project_id, source_url, paths, settings, report, metadata)
    if media_path is None:
        raise ClipForgeError(
            code=ErrorCode.DOWNLOAD_FAILED,
            message="No source media is available for this project.",
            hint="Re-run the analysis; if it keeps failing, check logs/render.log.",
            status_code=500,
        )
    media_info = media_info or probe_media(media_path)
    if not media_info.has_video:
        raise ClipForgeError(
            code=ErrorCode.UNSUPPORTED_FORMAT,
            message="The source has no video stream.",
            hint="CLIPFORGE needs video to render vertical clips. Audio-only sources are not supported yet.",
            status_code=422,
        )
    if not media_info.has_audio:
        raise ClipForgeError(
            code=ErrorCode.NO_AUDIO,
            message="The source has no audio stream, so there is nothing to transcribe.",
            hint="Try a different video - CLIPFORGE clips are built around speech.",
            status_code=422,
        )
    _update_project(
        project_id,
        duration=media_info.duration,
        width=media_info.width,
        height=media_info.height,
        fps=media_info.fps,
        media_json=media_info.to_dict(),
        paths_json=paths.to_dict(),
    )
    mark("download", time.time() - t0, bytes=media_info.filesize)
    report.sub(1.0, f"source ready · {media_info.width}x{media_info.height} · {media_info.duration / 60:.1f} min")

    # --------------------------------------------------------------- 3. audio
    report.stage("audio", "extracting audio for speech recognition", fraction=0.05)
    t0 = time.time()
    audio_path = paths.audio / "speech_16k.wav"
    if audio_path.exists() and audio_path.stat().st_size > 1024 and not force:
        report.sub(1.0, "reusing the cached audio track")
        report.log("audio track already extracted - reusing it")
    else:
        extract_audio(media_path, audio_path, sample_rate=16000, mono=True)
    mark("audio", time.time() - t0)
    audio_seconds = probe_media(audio_path).duration

    # ------------------------------------------------------------ 4. language
    report.stage("language", "detecting language", fraction=0.1)
    t0 = time.time()
    language_profile = _detect_language(media_path, audio_path, paths, settings, metadata, report)
    mark("language", time.time() - t0, language=language_profile.primary, mode=language_profile.mode)

    # ---------------------------------------------------------- 5. transcribe
    transcript = _transcribe(media_path, audio_path, paths, settings, metadata, report, language_profile)
    report.stage("transcribe", "transcript complete", fraction=1.0)
    report.sub(1.0, f"{transcript.word_count} words · {transcript.language} · {transcript.engine}")

    # ------------------------------------------------------------ 6. diarize
    report.stage("diarize", "working out who is speaking", fraction=0.1)
    t0 = time.time()
    utterances, diarization = diarize(
        audio_path,
        transcript.utterances,
        max_speakers=settings.max_speakers,
        mode=settings.diarization,
        progress=lambda fraction, message: report.sub(fraction, message),
    )
    _check_cancel(report)
    mark("diarize", time.time() - t0, **diarization)

    # ------------------------------------------------------------- 7. segment
    report.stage("segment", "splitting the transcript into thoughts and topics", fraction=0.2)
    t0 = time.time()
    sentences = build_sentences(utterances)
    blocks = build_blocks(sentences)
    index = TranscriptIndex.build(sentences, blocks, transcript.duration or media_info.duration)
    save_transcript(
        project_id,
        utterances,
        language=transcript.language,
        engine=transcript.engine,
        model=transcript.model,
    )
    write_transcript_files(paths, sentences, language=transcript.language, engine=transcript.engine, model=transcript.model)
    write_analysis_files(paths, "segmentation", {"blocks": [block.to_dict() for block in blocks], "sentences": sentence_map(sentences)})
    mark("segment", time.time() - t0, sentences=len(sentences), blocks=len(blocks))
    report.sub(1.0, f"{len(sentences)} sentences in {len(blocks)} topic blocks")

    min_seconds, target_seconds, max_seconds = settings.clip_limits()

    # ------------------------------------------------------------ 8. discover
    report.stage("discover", "finding every valuable moment", fraction=0.0)
    t0 = time.time()
    llm_result: LLMStatus | None = None
    llm_seeds: list[dict[str, Any]] = []
    if settings.llm_enabled:
        try:
            llm_result = llm_status(settings)
        except Exception as exc:  # noqa: BLE001 - LLM is optional
            log.warning("LLM status check failed: %s", exc)
            llm_result = None

    analyse_progress = lambda fraction, message: report.sub(fraction, message)  # noqa: E731
    candidates = discover_mod.generate_candidates(
        index,
        min_seconds=min_seconds,
        target_seconds=target_seconds,
        max_seconds=max_seconds,
        progress=analyse_progress,
        should_cancel=report.cancelled,
    )
    heuristic_count = len(candidates)
    report.sub(0.75, f"{heuristic_count} candidate moments from the transcript analysis")

    llm_note = ""
    if llm_result and llm_result.available:
        report.sub(0.78, f"asking {settings.ollama_model} for its own read of the transcript")
        try:
            llm_seeds = discover_moments(
                sentences,
                language=language_profile.describe(),
                speakers=diarization.get("speakers", 1),
                duration=index.duration,
                settings=settings,
                progress=lambda fraction, message: report.sub(0.78 + 0.18 * fraction, message),
                should_cancel=report.cancelled,
            )
        except ClipForgeError as exc:
            llm_note = f"Local model unavailable: {exc.message}"
            log.warning("LLM discovery failed: %s", exc.message)
        if llm_seeds:
            candidates = discover_mod.merge_llm_seeds(index, candidates, llm_seeds, min_seconds, max_seconds, target_seconds)
    else:
        llm_note = (
            "Analytical mode: Ollama was not used "
            + (f"({llm_result.error})" if llm_result and llm_result.error else "(disabled in Settings)")
        )
        report.log(llm_note)

    mark("discover", time.time() - t0, heuristic=heuristic_count, llm=len(llm_seeds), merged=len(candidates))
    report.sub(1.0, f"{len(candidates)} candidate moments discovered")

    # --------------------------------------------------------------- 9. score
    report.stage("score", "scoring every candidate", fraction=0.1)
    t0 = time.time()

    def on_phase(key: str, message: str) -> None:
        report.stage(key, message, fraction=0.1)
        report.sub(0.1, message)

    result = scoring_mod.finalize_candidates(
        index,
        candidates,
        min_seconds=min_seconds,
        target_seconds=target_seconds,
        max_seconds=max_seconds,
        min_score=settings.min_score,
        mode=settings.clip_mode,
        max_clips=settings.max_clips,
        optimize=True,
        on_phase=on_phase,
    )
    selected = result["selected"]
    report.stage("boundaries", "boundaries optimised", fraction=1.0)
    report.sub(1.0, f"{len(selected)} clips passed the quality gate")
    mark("score", time.time() - t0, **result["report"])

    # ---------------------------------------------------------- 10. persist
    report.stage("prepare", "preparing clips, captions and framing", fraction=0.05)
    t0 = time.time()
    stored_candidates = sorted(result["kept"] + result["duplicates"], key=lambda item: -item.score)[: discover_mod.MAX_STORED_CANDIDATES]
    _persist_candidates(project_id, stored_candidates, selected)
    clip_ids = _persist_clips(project_id, selected, paths, index, settings, language_profile)
    mark("prepare", time.time() - t0, candidates=len(stored_candidates), clips=len(clip_ids))

    # ---------------------------------------------------------- 11. finalise
    write_analysis_files(
        paths,
        "candidates",
        {
            "discovered": len(candidates),
            "kept": len(result["kept"]),
            "duplicates": len(result["duplicates"]),
            "selected": len(selected),
            "report": result["report"],
            "top": [candidate.to_dict() for candidate in selected[:50]],
        },
    )
    stats.update(
        {
            "finished_at": time.time(),
            "total_seconds": round(time.time() - started, 2),
            "heuristic_candidates": heuristic_count,
            "llm_candidates": len(llm_seeds),
            "selected_clips": len(selected),
            "audio_seconds": round(audio_seconds, 2),
            "diarization": diarization,
            "llm": llm_result.to_dict() if llm_result else None,
            "llm_note": llm_note,
        }
    )

    with session_scope() as session:
        project = session.get(Project, project_id)
        if project is not None:
            try:
                existing_stats = json.loads(project.stats_json or "{}")
            except (TypeError, ValueError):
                existing_stats = {}
            existing_stats["analysis"] = stats
            project.stats_json = json.dumps(existing_stats, default=str)
            project.status = "ready"
            project.stage = "ready"
            project.progress = 1.0
            project.status_message = f"{len(selected)} clips ready"
            project.language = language_profile.primary
            project.language_secondary = language_profile.secondary
            project.language_mode = language_profile.mode
            project.language_confidence = language_profile.confidence
            project.speakers = int(diarization.get("speakers", 1) or 1)
            project.candidate_count = len(candidates)
            project.clip_count = len(selected)
            project.analyzed_at = utcnow()
            project.error_code = ""
            project.error_message = ""
            project.error_hint = ""

    log.info(
        "analysis finished for %s: %d candidates -> %d clips in %.1fs",
        project_id,
        len(candidates),
        len(selected),
        time.time() - started,
    )
    return AnalysisOutcome(
        project_id=project_id,
        candidate_count=len(candidates),
        clip_count=len(selected),
        language=language_profile.primary,
        speakers=int(diarization.get("speakers", 1) or 1),
        duration=index.duration,
        llm_used=bool(llm_seeds),
        llm_note=llm_note,
        stats=stats,
    )


# --------------------------------------------------------------------------- #
# Stage helpers
# --------------------------------------------------------------------------- #


def project_settings_from_snapshot(snapshot: dict[str, Any]) -> AppSettings:
    base = get_settings().model_dump()
    overrides = {key: value for key, value in (snapshot.get("settings") or {}).items() if key in base}
    return AppSettings.model_validate({**base, **overrides})


def _find_existing_media(paths: ProjectPaths) -> Path | None:
    extensions = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi"}
    if not paths.source.exists():
        return None
    candidates = [
        path for path in paths.source.iterdir()
        if path.is_file() and path.suffix.lower() in extensions and path.stat().st_size > 4096
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda path: -path.stat().st_size)
    return candidates[0]


def _download_source(
    project_id: str,
    url: str,
    paths: ProjectPaths,
    settings: AppSettings,
    report: ProgressReporter,
    metadata: dict[str, Any],
) -> Path:
    video_id = metadata.get("video_id") or "source"
    cache_target = cached_source_path(str(video_id))
    if settings.cache_downloads and cache_target.exists() and cache_target.stat().st_size > 4096:
        report.sub(1.0, "reusing the cached download for this video")
        report.log(f"using cached source {cache_target.name}")
        media_path = link_or_copy(cache_target, paths.source / cache_target.name)
        return media_path

    def on_progress(fraction: float, message: str) -> None:
        report.sub(fraction, message)

    try:
        downloaded, snapshot = download_video(url, paths.source, progress=on_progress, should_cancel=report.cancelled)
    except ClipForgeError as exc:
        if exc.code == ErrorCode.CANCELLED:
            raise
        raise

    if settings.cache_downloads and downloaded.exists():
        try:
            if not cache_target.exists():
                link_or_copy(downloaded, cache_target)
        except OSError as exc:
            log.debug("could not cache the download: %s", exc)
    return downloaded


def _detect_language(
    media_path: Path,
    audio_path: Path,
    paths: ProjectPaths,
    settings: AppSettings,
    metadata: dict[str, Any],
    report: ProgressReporter,
) -> language_mod.LanguageProfile:
    """Fast language probe on the first minutes, then script/lexical analysis."""
    probe_seconds = 150.0
    whisper_language = ""
    confidence = 0.0
    probabilities: dict[str, float] = {}
    sample_text = ""

    # A transcript the user supplied is both faster and more accurate than a
    # 150-second ASR probe, so it wins here too.
    from ..media.download import parse_transcript_file, transcript_files

    provided = transcript_files(paths.transcript)
    if provided:
        words = parse_transcript_file(provided[0])
        sample_text = " ".join(item["word"] for item in words[:1200])
        report.sub(0.6, f"language read from {provided[0].name}")
        report.log(f"language detection used the provided transcript ({provided[0].name})")

    if not sample_text and transcription_available():
        try:
            from ..media.ffmpeg import extract_audio

            sample_path = paths.audio / "language_probe.wav"
            if not sample_path.exists():
                extract_audio(media_path, sample_path, sample_rate=16000, mono=True, duration=probe_seconds)
            report.sub(0.3, "listening to the opening minutes")
            sample_settings = settings.model_copy(update={"whisper_beam_size": 1, "whisper_vad": True})
            transcript = transcribe(sample_path, settings=sample_settings)
            whisper_language = transcript.language
            confidence = transcript.language_confidence
            probabilities = transcript.language_probabilities
            sample_text = transcript.text
            report.sub(0.8, f"language detected: {language_mod.language_name(whisper_language)}")
        except ClipForgeError as exc:
            report.log(f"language probe failed ({exc.message}); continuing with automatic detection during transcription")
    elif not sample_text:
        report.log("Local speech-to-text is not installed - trying YouTube captions instead")

    profile = language_mod.detect_language(
        sample_text,
        whisper_language=whisper_language or str(metadata.get("language") or ""),
        whisper_confidence=confidence,
        probabilities=probabilities,
    )
    if settings.language_hint:
        profile.primary = settings.language_hint.split("-")[0]
        profile.primary_name = language_mod.language_name(profile.primary)
        profile.notes.append(f"Language forced to {profile.primary_name} in Settings -> Transcription.")
    write_analysis_files(paths, "language", profile.to_dict())
    return profile


def _transcribe(
    media_path: Path,
    audio_path: Path,
    paths: ProjectPaths,
    settings: AppSettings,
    metadata: dict[str, Any],
    report: ProgressReporter,
    profile: language_mod.LanguageProfile,
) -> Transcript:
    report.stage("transcribe", "running local speech recognition", fraction=0.0)

    # 1. A transcript the user supplied for this project wins over everything:
    #    they already have captions, so re-recognising would only lose accuracy.
    provided = _transcript_from_files(paths, settings, report, profile)
    if provided is not None:
        return provided

    if transcription_available():
        try:
            transcript = transcribe(
                audio_path,
                settings=settings,
                progress=lambda fraction, message: report.sub(fraction, message),
                should_cancel=report.cancelled,
            )
            (paths.transcript / "engine.json").write_text(
                json.dumps({"engine": transcript.engine, "model": transcript.model, "language": transcript.language}, indent=1),
                encoding="utf-8",
            )
            return transcript
        except ClipForgeError as exc:
            if exc.code == ErrorCode.CANCELLED:
                raise
            report.log(f"local transcription failed: {exc.message}")
            log.warning("transcription failed, trying caption fallback: %s", exc.message)

    return _transcribe_from_captions(paths, settings, metadata, report, profile)


def _transcript_from_files(
    paths: ProjectPaths,
    settings: AppSettings,
    report: ProgressReporter,
    profile: language_mod.LanguageProfile,
) -> Transcript | None:
    """Build a transcript from a caption/transcript file the user provided.

    Supported: ``.json3`` (YouTube), ``.srt``, ``.vtt`` and ``.txt`` with
    ``[hh:mm:ss]`` stamps, dropped in the project's ``transcript/`` folder or
    uploaded through the API. Word-level timings inside a subtitle cue are
    estimated (and reported as such) - accuracy is lower than local ASR, which is
    why this only happens when the user supplies the file.
    """
    from ..media.download import parse_transcript_file, transcript_files

    candidates = transcript_files(paths.transcript)
    if not candidates:
        return None

    source = candidates[0]
    words: list[dict[str, Any]] = []
    for candidate in candidates:
        # A file can pass the extension/size filter and still hold no cues, so work
        # down the priority list until one of them actually yields words.
        parsed = parse_transcript_file(candidate)
        if parsed:
            source, words = candidate, parsed
            break
        report.log(f"{candidate.name} contained no readable timings - trying the next transcript file")
        log.warning("provided transcript %s could not be parsed", candidate)

    if not words:
        report.log("none of the supplied transcript files contained readable timings - falling back to speech recognition")
        return None

    report.sub(0.05, f"using the transcript file you provided ({source.name})")

    from ..ai.transcribe import Word

    utterances: list[Utterance] = []
    buffer: list[Word] = []
    for item in words:
        buffer.append(Word(word=item["word"], start=item["start"], end=item["end"], confidence=item.get("confidence", 0.6)))
        if len(buffer) >= 14 or (buffer and item["end"] - buffer[0].start > 7.0):
            utterances.append(_utterance_from_words(buffer))
            buffer = []
    if buffer:
        utterances.append(_utterance_from_words(buffer))

    duration = utterances[-1].end if utterances else 0.0
    report.sub(0.9, f"{len(words)} words loaded from {source.name}")
    report.log(f"transcript supplied by the user: {source.name} ({len(words)} words, {duration / 60:.1f} min)")
    return Transcript(
        language=profile.primary or "en",
        language_confidence=profile.confidence,
        utterances=utterances,
        engine="transcript-file",
        model=source.name,
        duration=duration,
        diarized=False,
        speaker_count=1,
        warnings=[f"Transcript read from {source.name}; word timings inside each cue are estimated."],
    )


def _transcribe_from_captions(
    paths: ProjectPaths,
    settings: AppSettings,
    metadata: dict[str, Any],
    report: ProgressReporter,
    profile: language_mod.LanguageProfile,
) -> Transcript:
    """Fallback: use the source's own caption track (clearly labelled as external)."""
    url = metadata.get("url") or metadata.get("webpage_url") or ""
    caption_file: Path | None = None
    if url:
        report.sub(0.2, "no local ASR engine - downloading the video's caption track")
        try:
            caption_file = download_captions(url, paths.transcript)
        except ClipForgeError as exc:
            report.log(f"caption download failed: {exc.message}")

    if caption_file is None:
        stack = ai_stack()
        raise ClipForgeError(
            code=ErrorCode.TRANSCRIPTION_FAILED,
            message="CLIPFORGE could not transcribe this video.",
            hint=(
                "No local speech-to-text engine is installed and no caption track was available. "
                'Install the AI extras with:  pip install -r requirements-ai.txt   (faster-whisper)'
                if not stack.get("transcription_available")
                else "Transcription failed - check logs/ai.log for the underlying error."
            ),
            status_code=503,
        )

    words = parse_json3_captions(caption_file)
    if not words:
        raise ClipForgeError(
            code=ErrorCode.NO_SPEECH,
            message="The downloaded caption track was empty or unreadable.",
            hint="Install faster-whisper for real local transcription instead of relying on captions.",
            status_code=422,
        )

    from ..ai.transcribe import Word

    utterances: list[Utterance] = []
    buffer: list[Word] = []
    for item in words:
        buffer.append(
            Word(word=item["word"], start=item["start"], end=item["end"], confidence=item.get("confidence", 0.6))
        )
        if len(buffer) >= 12 or (buffer and item["end"] - buffer[0].start > 6.0):
            utterances.append(_utterance_from_words(buffer))
            buffer = []
    if buffer:
        utterances.append(_utterance_from_words(buffer))

    report.log("transcript built from the video's own caption track (external captions, word timings estimated)")
    return Transcript(
        language=profile.primary or "en",
        language_confidence=profile.confidence,
        utterances=utterances,
        engine="youtube-captions (fallback)",
        model="caption-track",
        duration=utterances[-1].end if utterances else 0.0,
        diarized=False,
        speaker_count=1,
        warnings=["Captions came from the video's caption track because no local ASR engine is installed."],
    )


def _utterance_from_words(words: Sequence[Any]) -> Utterance:
    text = " ".join(word.word for word in words).strip()
    confidence = sum(word.confidence for word in words) / max(len(words), 1)
    return Utterance(
        text=text,
        start=words[0].start,
        end=words[-1].end,
        speaker="SPEAKER_01",
        confidence=confidence,
        words=list(words),
    )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def _persist_candidates(project_id: str, candidates: Sequence[Any], selected: Sequence[Any]) -> None:
    from ..db import Candidate as CandidateRow

    selected_keys = {f"{candidate.start_index}:{candidate.end_index}" for candidate in selected}
    with session_scope() as session:
        session.query(CandidateRow).filter(CandidateRow.project_id == project_id).delete()
        for order, candidate in enumerate(candidates):
            session.add(
                CandidateRow(
                    project_id=project_id,
                    idx=order,
                    start=candidate.start,
                    end=candidate.end,
                    duration=candidate.duration,
                    title=candidate.title[:300],
                    hook=candidate.hook,
                    summary=candidate.summary,
                    category=candidate.category,
                    reason=candidate.reason,
                    score=candidate.score,
                    confidence=candidate.confidence,
                    factors_json=json.dumps(candidate.features.factors),
                    penalties_json=json.dumps(candidate.features.penalties),
                    highlights_json=json.dumps(candidate.why),
                    transcript_text=(candidate.features.meta.get("text") or candidate.summary or "")[:8000],
                    source=candidate.source,
                    status="selected" if candidate.key() in selected_keys else candidate.status,
                    duplicate_of=candidate.duplicate_of,
                    boundary_json=json.dumps(
                        {
                            "sentence_start": candidate.start_index,
                            "sentence_end": candidate.end_index,
                            "notes": candidate.features.warnings[:6],
                        }
                    ),
                )
            )


def _persist_clips(
    project_id: str,
    selected: Sequence[Any],
    paths: ProjectPaths,
    index: TranscriptIndex,
    settings: AppSettings,
    profile: language_mod.LanguageProfile,
) -> list[str]:
    """Create clip rows with their caption/edit plan pre-computed."""
    from ..pipeline.edit import build_clip_plan

    clip_ids: list[str] = []
    with session_scope() as session:
        session.query(Clip).filter(Clip.project_id == project_id).delete()

    for order, candidate in enumerate(selected, start=1):
        words = [
            word
            for position in range(candidate.start_index, candidate.end_index)
            for word in index.sentences[position].words
        ]
        plan = build_clip_plan(
            project_id=project_id,
            index=order,
            candidate=candidate,
            words=words,
            settings=settings,
            language=profile,
            paths=paths,
            source_path=_find_existing_media(paths),
            features=candidate.features,
            sentences=[index.sentences[position] for position in range(candidate.start_index, candidate.end_index)],
        )
        with session_scope() as session:
            row = Clip(
                project_id=project_id,
                index=order,
                title=candidate.title[:300],
                hook=candidate.hook,
                summary=candidate.summary,
                category=candidate.category,
                score=candidate.score,
                why_json=json.dumps(candidate.why),
                factors_json=json.dumps(candidate.features.factors),
                start=candidate.start,
                end=candidate.end,
                duration=candidate.duration,
                words_json=json.dumps([word.to_dict() for word in words]),
                segments_json=json.dumps(
                    {
                        "sentence_start": candidate.start_index,
                        "sentence_end": candidate.end_index,
                        "sentences": [
                            index.sentences[position].to_dict()
                            for position in range(candidate.start_index, candidate.end_index)
                        ],
                    }
                ),
                trim_json=json.dumps(plan.get("timeline", {})),
                layout_json=json.dumps(plan),
                status="pending",
            )
            session.add(row)
            session.flush()
            clip_ids.append(row.id)
    return clip_ids


__all__ = ["AnalysisOutcome", "analyze", "project_settings"]
