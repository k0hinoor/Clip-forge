"""The analysis pipeline: YouTube URL → ranked, editable clips.

Stage order (each one reports real progress and can be cancelled):

    metadata → download → audio → language → transcribe → diarize →
    segment → discover → score → dedupe → validate → boundaries → prepare

Nothing here is simulated: every stage either produces artefacts from the actual
media (ffmpeg, Whisper, acoustic clustering, statistic scoring) or fails loudly
with an actionable message.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic import ValidationError

from ..ai import candidates as discover_mod
from ..ai import language as language_mod
from ..ai import scoring as scoring_mod
from ..ai.diarize import diarize
from ..ai.features import TranscriptIndex
from ..ai.llm import LLMStatus, discover_moments
from ..ai.llm import status as llm_status
from ..ai.segment import build_blocks, build_sentences, sentence_map
from ..ai.transcribe import Transcript, Utterance, transcribe, transcription_available
from ..config import AppSettings, describe_validation_error, get_settings, merge_settings_patch
from ..constants import progress_for
from ..db import Candidate, Clip, Project, session_scope, utcnow
from ..errors import ClipForgeError, ErrorCode
from ..logging_setup import get_logger
from ..media.download import (
    classify_url,
    download_captions,
    download_video,
    fetch_metadata,
    parse_transcript_segments,
    transcript_files,
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
    return project_settings_from_snapshot({"settings": project.settings})


def _update_project(project_id: str, **fields: Any) -> None:
    with session_scope() as session:
        project = session.get(Project, project_id)
        if project is None:
            return
        for key, value in fields.items():
            if key.endswith("_json") and not isinstance(value, str):
                value = json.dumps(value, default=str)
            setattr(project, key, value)


def _update_analysis_snapshot(project_id: str, **fields: Any) -> None:
    """Persist live discovery/scoring diagnostics without replacing other stats."""
    with session_scope() as session:
        project = session.get(Project, project_id)
        if project is None:
            return
        try:
            stats = json.loads(project.stats_json or "{}")
        except (TypeError, ValueError):
            stats = {}
        analysis = stats.setdefault("analysis", {})
        diagnostics = analysis.setdefault("diagnostics", {})
        incoming = fields.pop("diagnostics", {})
        if isinstance(incoming, dict):
            diagnostics.update(incoming)
        analysis.update(fields)
        project.stats_json = json.dumps(stats, default=str)


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
        transcript_source = project.transcript_source
        transcript_preference = project.transcript_preference
        transcript_filename = project.transcript_filename

    selected_transcript_source = transcript_preference or transcript_source
    selected_transcript_filename = transcript_filename if selected_transcript_source.startswith("uploaded_") else ""
    settings = project_settings_from_snapshot(snapshot)
    paths = ProjectPaths.for_snapshot(snapshot).ensure()
    if not force:
        ensure_disk_space(2.0)

    stats: dict[str, Any] = {"started_at": time.time(), "stages": {}}
    _update_analysis_snapshot(
        project_id,
        diagnostics={
            "discovered": 0,
            "scored": 0,
            "scoring_errors": 0,
            "threshold_pass": 0,
            "context_rejections": 0,
            "overlap_rejections": 0,
            "final_accepted": 0,
            "queued": 0,
            "rendered": 0,
            "threshold": 0.0 if settings.debug_mode else settings.min_score,
            "debug_mode": settings.debug_mode,
        },
    )

    def mark(stage: str, seconds: float, **extra: Any) -> None:
        stats["stages"][stage] = {"seconds": round(seconds, 2), **extra}

    # ------------------------------------------------------------ 1. metadata
    report.stage("metadata", "reading video metadata", fraction=0.05)
    t0 = time.time()
    media_info = None
    metadata: dict[str, Any] = {}
    if source_type in {"youtube", "url"}:
        source_metadata_path = paths.metadata / "source.json"
        cached_metadata = None
        try:
            cached_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            cached_metadata = None
        if isinstance(cached_metadata, dict) and cached_metadata.get("url") == source_url and cached_metadata.get("title"):
            metadata = cached_metadata
            title = str(metadata.get("title") or title)
            _update_project(
                project_id,
                title=title,
                channel=str(metadata.get("channel") or ""),
                duration=float(metadata.get("duration") or 0.0),
                thumbnail_url=str(metadata.get("thumbnail_url") or ""),
                source_id=str(metadata.get("video_id") or ""),
                description=str(metadata.get("description") or "")[:4000],
            )
            report.sub(1.0, "reusing cached video metadata")
        else:
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
            source_metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=1), encoding="utf-8")
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
    if media_path is None and source_type in {"youtube", "url"}:
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
    if media_info.duration > settings.max_source_hours * 3600:
        # Direct links only reveal their length once the file is on disk.
        raise ClipForgeError(
            code=ErrorCode.VIDEO_TOO_LONG,
            message=f"That video is {media_info.duration / 3600:.1f} hours long; the limit is {settings.max_source_hours:g} hours.",
            hint="Raise the limit in Settings -> Video, or pick a shorter video.",
            status_code=422,
        )
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
    if audio_path.exists() and audio_path.stat().st_size > 1024:
        report.sub(1.0, "reusing the cached audio track")
        report.log("audio track already extracted - reusing it")
    else:
        extract_audio(media_path, audio_path, sample_rate=16000, mono=True)
    mark("audio", time.time() - t0)
    audio_seconds = probe_media(audio_path).duration

    # ------------------------------------------------------------ 4. language
    report.stage("language", "detecting language", fraction=0.1)
    t0 = time.time()
    language_profile = _detect_language(
        media_path, audio_path, paths, settings, metadata, report,
        transcript_source=selected_transcript_source,
        transcript_filename=selected_transcript_filename,
    )
    mark("language", time.time() - t0, language=language_profile.primary, mode=language_profile.mode)

    # ---------------------------------------------------------- 5. transcribe
    transcript = _transcribe(
        media_path, audio_path, paths, settings, metadata, report, language_profile,
        preferred_source=selected_transcript_source,
        preferred_filename=selected_transcript_filename,
        force=force,
    )
    transcript_source = transcript.source
    transcript_filename = transcript.filename
    _update_project(
        project_id,
        transcript_source=transcript_source,
        transcript_preference=transcript.source,
        transcript_filename=transcript_filename,
        transcript_timing=transcript.timing_granularity,
    )
    for warning in transcript.warnings:
        report.log(warning)
    if transcript.translated:
        language_profile = _translated_profile(language_profile, transcript)
        write_analysis_files(paths, "language", language_profile.to_dict())
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
    sentences = build_sentences(
        utterances,
        preserve_cue_boundaries=transcript.timing_granularity == "cue",
    )
    blocks = build_blocks(sentences)
    index = TranscriptIndex.build(sentences, blocks, transcript.duration or media_info.duration)
    save_transcript(
        project_id,
        utterances,
        language=transcript.language,
        engine=transcript.engine,
        model=transcript.model,
        source=transcript.source,
        filename=transcript.filename,
        timing_granularity=transcript.timing_granularity,
    )
    write_transcript_files(
        paths,
        sentences,
        language=transcript.language,
        engine=transcript.engine,
        model=transcript.model,
        source=transcript.source,
        filename=transcript.filename,
        timing_granularity=transcript.timing_granularity,
    )
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
    discovery_errors: list[str] = []

    def on_discovery_error(start_index: int, end_index: int, exc: Exception) -> None:
        discovery_errors.append(f"{start_index}:{end_index} {type(exc).__name__}: {exc}")
        if len(discovery_errors) <= 8:
            report.log(f"candidate scoring failed for sentence range {start_index}:{end_index}: {type(exc).__name__}: {exc}")

    candidates = discover_mod.generate_candidates(
        index,
        min_seconds=min_seconds,
        target_seconds=target_seconds,
        max_seconds=max_seconds,
        progress=analyse_progress,
        should_cancel=report.cancelled,
        on_error=on_discovery_error,
    )
    heuristic_count = len(candidates)
    report.sub(0.75, f"{heuristic_count} candidate moments from the transcript analysis")
    _persist_candidates(project_id, candidates, [])
    _update_project(project_id, candidate_count=len(candidates))
    _update_analysis_snapshot(
        project_id,
        diagnostics={
            "discovered": len(candidates),
            "scored": 0,
            "scoring_errors": len(discovery_errors),
        },
        heuristic_candidates=heuristic_count,
    )

    llm_note = ""
    llm_diagnostics: dict[str, int] = {}
    llm_merge_errors: list[str] = []
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
                diagnostics=llm_diagnostics,
            )
        except ClipForgeError as exc:
            llm_note = f"Local model unavailable: {exc.message}"
            log.warning("LLM discovery failed: %s", exc.message)
            report.log(llm_note)
        if llm_seeds:
            def on_llm_merge_error(code: str, message: str) -> None:
                llm_merge_errors.append(code)
                report.log(message)

            candidates = discover_mod.merge_llm_seeds(
                index,
                candidates,
                llm_seeds,
                min_seconds,
                max_seconds,
                target_seconds,
                on_error=on_llm_merge_error,
            )
        if not llm_seeds and sum(llm_diagnostics.get(key, 0) for key in ("chunks_failed", "invalid_payloads", "invalid_moments", "invalid_scores")):
            llm_note = "Ollama returned invalid or unscorable moments; see the analysis log for raw-output diagnostics."
            report.log(llm_note)
    else:
        llm_note = (
            "Analytical mode: Ollama was not used "
            + (f"({llm_result.error})" if llm_result and llm_result.error else "(disabled in Settings)")
        )
        report.log(llm_note)

    if llm_seeds:
        _persist_candidates(project_id, candidates, [])
    _update_project(project_id, candidate_count=len(candidates))
    _update_analysis_snapshot(
        project_id,
        diagnostics={
            "discovered": len(candidates),
            "scoring_errors": len(discovery_errors) + len(llm_merge_errors),
            "llm_parse_errors": sum(llm_diagnostics.get(key, 0) for key in ("invalid_payloads", "invalid_moments", "invalid_scores")),
        },
        heuristic_candidates=heuristic_count,
        llm_candidates=len(llm_seeds),
        llm_diagnostics=llm_diagnostics,
    )
    mark(
        "discover",
        time.time() - t0,
        heuristic=heuristic_count,
        llm=len(llm_seeds),
        merged=len(candidates),
        scoring_errors=len(discovery_errors) + len(llm_merge_errors),
        llm_diagnostics=llm_diagnostics,
    )
    report.sub(1.0, f"{len(candidates)} candidate moments discovered · {len(discovery_errors) + len(llm_merge_errors)} scoring errors")

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
        debug_mode=settings.debug_mode,
        on_phase=on_phase,
    )
    selected = result["selected"]
    score_report = result["report"]
    total_scoring_errors = len(discovery_errors) + len(llm_merge_errors) + int(score_report.get("scoring_errors", 0))
    report.stage("boundaries", "boundaries optimised", fraction=1.0)
    report.sub(1.0, f"{len(selected)} clips accepted · {total_scoring_errors} scoring errors")
    report.log(
        "analysis counts: "
        f"discovered={len(candidates)}, scored={score_report.get('scored', 0)}, "
        f"threshold_pass={score_report.get('threshold_pass', 0)}, "
        f"context_rejections={score_report.get('rejected_context', 0)}, "
        f"overlap_rejections={score_report.get('overlap_rejections', 0)}, "
        f"final_accepted={len(selected)}, threshold={score_report.get('min_score', settings.min_score):.1f}"
    )
    if not selected:
        reason_counts = score_report.get("rejection_reasons") or {}
        top_reason = max(reason_counts, key=reason_counts.get) if reason_counts else "no_candidate_passed_validation"
        report.log(
            f"zero accepted clips: candidates={len(candidates)}, average_score={score_report.get('average_score', 0):.1f}, "
            f"highest_score={score_report.get('highest_score', 0):.1f}, "
            f"threshold={score_report.get('min_score', settings.min_score):.1f}, top_rejection={top_reason}"
        )
    _update_analysis_snapshot(
        project_id,
        diagnostics={
            "discovered": len(candidates),
            "scored": int(score_report.get("scored", 0)),
            "scoring_errors": total_scoring_errors,
            "threshold_pass": int(score_report.get("threshold_pass", 0)),
            "threshold": float(score_report.get("min_score", settings.min_score)),
            "context_rejections": int(score_report.get("rejected_context", 0)),
            "overlap_rejections": int(score_report.get("overlap_rejections", 0)),
            "final_accepted": len(selected),
            "average_score": float(score_report.get("average_score", 0.0)),
            "highest_score": float(score_report.get("highest_score", 0.0)),
            "top_rejection_reason": top_reason if not selected else "",
            "debug_mode": settings.debug_mode,
        },
        scoring=score_report,
    )
    mark("score", time.time() - t0, **score_report, total_scoring_errors=total_scoring_errors)

    # ---------------------------------------------------------- 10. persist
    report.stage("prepare", "preparing clips, captions and framing", fraction=0.05)
    t0 = time.time()
    _persist_candidates(project_id, candidates, selected)
    stored_candidate_count = min(len(candidates), discover_mod.MAX_STORED_CANDIDATES)
    clip_ids = _persist_clips(project_id, selected, paths, index, settings, language_profile)
    _update_project(project_id, candidate_count=len(candidates), clip_count=len(clip_ids))
    _update_analysis_snapshot(
        project_id,
        diagnostics={"final_accepted": len(clip_ids), "queued": 0, "rendered": 0},
        persisted_candidates=stored_candidate_count,
        accepted_clip_records=len(clip_ids),
        render_jobs_queued=0,
    )
    mark("prepare", time.time() - t0, candidates=stored_candidate_count, clips=len(clip_ids), render_jobs_queued=0)

    # ---------------------------------------------------------- 11. finalise
    write_analysis_files(
        paths,
        "candidates",
        {
            "discovered": len(candidates),
            "kept": len(result["kept"]),
            "duplicates": len(result["duplicates"]),
            "selected": len(selected),
            "diagnostics": {
                "discovered": len(candidates),
                "scored": int(score_report.get("scored", 0)),
                "scoring_errors": total_scoring_errors,
                "threshold_pass": int(score_report.get("threshold_pass", 0)),
                "context_rejections": int(score_report.get("rejected_context", 0)),
                "overlap_rejections": int(score_report.get("overlap_rejections", 0)),
                "final_accepted": len(selected),
                "threshold": float(score_report.get("min_score", settings.min_score)),
                "average_score": float(score_report.get("average_score", 0.0)),
                "highest_score": float(score_report.get("highest_score", 0.0)),
            },
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
            "diagnostics": {
                "discovered": len(candidates),
                "scored": int(score_report.get("scored", 0)),
                "scoring_errors": total_scoring_errors,
                "threshold_pass": int(score_report.get("threshold_pass", 0)),
                "context_rejections": int(score_report.get("rejected_context", 0)),
                "overlap_rejections": int(score_report.get("overlap_rejections", 0)),
                "final_accepted": len(selected),
                "queued": 0,
                "rendered": 0,
                "threshold": float(score_report.get("min_score", settings.min_score)),
                "average_score": float(score_report.get("average_score", 0.0)),
                "highest_score": float(score_report.get("highest_score", 0.0)),
            },
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
            project.status_message = f"{len(selected)} accepted clips ready to render"
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
    """Global settings with a project's stored overrides applied on top.

    The caption theme is merged rather than replaced, so a project that only
    chose a caption preset keeps the rest of the user's caption settings. An
    override that no longer validates (e.g. saved by an older version) is
    skipped instead of breaking the analysis or the render.
    """
    base = get_settings().model_dump()
    overrides = {key: value for key, value in (snapshot.get("settings") or {}).items() if key in base}
    try:
        return AppSettings.model_validate(merge_settings_patch(base, overrides))
    except ValidationError as exc:
        log.warning("ignoring invalid project overrides (%s)", describe_validation_error(exc))
        broken = {str(error["loc"][0]) for error in exc.errors() if error.get("loc")}
        usable = {key: value for key, value in overrides.items() if key not in broken}
        try:
            return AppSettings.model_validate(merge_settings_patch(base, usable))
        except ValidationError:
            return get_settings()


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
        try:
            os.utime(cache_target)  # recently used: storage cleanup expires by age
        except OSError:
            pass
        return link_or_copy(cache_target, paths.source / cache_target.name)

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


def _provided_transcript_file(
    paths: ProjectPaths,
    *,
    preferred_source: str = "",
    preferred_filename: str = "",
    allow_legacy: bool = False,
) -> Path | None:
    """Resolve only a user-supplied transcript, never a downloaded caption track."""
    if preferred_source == "whisper" or preferred_source == "source_captions":
        return None
    extension = Path(preferred_filename).suffix.lower()
    if extension in {".srt", ".vtt", ".json3", ".txt"}:
        target = paths.transcript / f"provided{extension}"
        if target.is_file():
            return target
        if preferred_source.startswith("uploaded_"):
            raise ClipForgeError(
                code=ErrorCode.TRANSCRIPTION_FAILED,
                message="The uploaded transcript is missing from this project.",
                hint="Attach the subtitle file again; CLIPFORGE will not silently replace it with speech recognition.",
                status_code=409,
            )
    provided = [
        path for path in transcript_files(paths.transcript)
        if path.stem.lower().startswith("provided")
    ]
    if provided:
        return provided[0]
    if allow_legacy and not preferred_source:
        # Backwards compatibility for projects created before transcript source
        # metadata existed. Explicitly named downloaded caption tracks are excluded.
        legacy = [
            path for path in transcript_files(paths.transcript)
            if not path.stem.lower().startswith(("captions", "auto", "source"))
        ]
        return legacy[0] if legacy else None
    return None


def _language_cache_key(
    audio_path: Path,
    settings: AppSettings,
    *,
    transcript_source: str,
    transcript_filename: str,
    transcript_path: Path | None,
) -> dict[str, Any]:
    """Fingerprint the exact transcript/audio and language settings for reuse."""
    transcript_hash = ""
    if transcript_path is not None:
        try:
            transcript_hash = hashlib.sha256(transcript_path.read_bytes()).hexdigest()
        except OSError:
            transcript_hash = "missing"
    try:
        audio_stat = audio_path.stat()
        audio_fingerprint = {"name": audio_path.name, "size": audio_stat.st_size, "mtime_ns": audio_stat.st_mtime_ns}
    except OSError:
        audio_fingerprint = {"name": audio_path.name, "size": 0, "mtime_ns": 0}
    return {
        "source": transcript_source,
        "filename": transcript_filename,
        "transcript_sha256": transcript_hash,
        "audio": audio_fingerprint,
        "whisper_model": settings.whisper_model,
        "language_hint": settings.language_hint,
    }


def _language_profile_from_dict(payload: dict[str, Any]) -> language_mod.LanguageProfile:
    return language_mod.LanguageProfile(
        primary=str(payload.get("primary") or "en"),
        primary_name=str(payload.get("primary_name") or "English"),
        secondary=str(payload.get("secondary") or ""),
        secondary_name=str(payload.get("secondary_name") or ""),
        mode=str(payload.get("mode") or "monolingual"),
        confidence=float(payload.get("confidence") or 0.0),
        whistle_language=str(payload.get("whisper_language") or ""),
        probabilities=dict(payload.get("probabilities") or {}),
        script=dict(payload.get("script") or {}),
        hinglish_score=float(payload.get("hinglish_score") or 0.0),
        secondary_share=float(payload.get("secondary_share") or 0.0),
        notes=list(payload.get("notes") or []),
    )


def _detect_language(
    media_path: Path,
    audio_path: Path,
    paths: ProjectPaths,
    settings: AppSettings,
    metadata: dict[str, Any],
    report: ProgressReporter,
    *,
    transcript_source: str = "",
    transcript_filename: str = "",
) -> language_mod.LanguageProfile:
    """Read language from an authoritative user transcript before probing audio."""
    allow_legacy = not transcript_source
    probe_seconds = 150.0
    whisper_language = ""
    confidence = 0.0
    probabilities: dict[str, float] = {}
    sample_text = ""

    provided = _provided_transcript_file(
        paths,
        preferred_source=transcript_source,
        preferred_filename=transcript_filename,
        allow_legacy=allow_legacy,
    )
    cache_source = transcript_source or (f"uploaded_{provided.suffix.lower().lstrip('.')}" if provided else "whisper")
    cache_filename = transcript_filename or (provided.name if provided else "")
    cache_key = _language_cache_key(
        audio_path,
        settings,
        transcript_source=cache_source,
        transcript_filename=cache_filename,
        transcript_path=provided,
    )
    language_cache = paths.analysis / "language.json"
    cacheable_language = not (not transcript_source and provided is not None and not provided.stem.lower().startswith("provided"))
    try:
        cached_language = json.loads(language_cache.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        cached_language = None
    if cacheable_language and isinstance(cached_language, dict) and cached_language.get("_cache_key") == cache_key:
        profile_payload = {key: value for key, value in cached_language.items() if key != "_cache_key"}
        report.sub(1.0, "reusing cached language analysis")
        report.log("language detection reused the cached result for the same transcript/audio and settings")
        return _language_profile_from_dict(profile_payload)

    if provided is not None:
        candidates = [provided]
        if not transcript_source and allow_legacy and not provided.stem.lower().startswith("provided"):
            candidates = [
                path for path in transcript_files(paths.transcript)
                if not path.stem.lower().startswith(("captions", "auto", "source", "provided"))
            ]
        for candidate in candidates:
            cues = parse_transcript_segments(candidate)
            if cues:
                provided = candidate
                sample_text = " ".join(item["text"] for item in cues[:1200])
                break
        if sample_text:
            report.sub(0.6, f"language read from uploaded transcript ({transcript_filename or provided.name})")
            report.log(f"language detection used the uploaded transcript ({transcript_filename or provided.name}); Whisper probe skipped")
        elif transcript_source.startswith("uploaded_"):
            raise ClipForgeError(
                code=ErrorCode.INVALID_INPUT,
                message="The uploaded transcript contains no valid timestamped cues.",
                hint="Check the SRT/VTT timestamps and attach the file again.",
                status_code=422,
            )

    if not sample_text and transcription_available():
        try:
            from ..media.ffmpeg import extract_audio

            sample_path = paths.audio / "language_probe.wav"
            if not sample_path.exists():
                extract_audio(media_path, sample_path, sample_rate=16000, mono=True, duration=probe_seconds)
            report.sub(0.3, "listening to the opening minutes")
            # The probe must hear the spoken language, never a translation.
            sample_settings = settings.model_copy(update={"whisper_beam_size": 1, "whisper_vad": True, "translate_captions": False})
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
    language_payload = profile.to_dict()
    if cacheable_language:
        language_payload["_cache_key"] = cache_key
    write_analysis_files(paths, "language", language_payload)
    return profile


def _transcribe(
    media_path: Path,
    audio_path: Path,
    paths: ProjectPaths,
    settings: AppSettings,
    metadata: dict[str, Any],
    report: ProgressReporter,
    profile: language_mod.LanguageProfile,
    *,
    preferred_source: str = "",
    preferred_filename: str = "",
    force: bool = False,
) -> Transcript:
    supplied_path = _provided_transcript_file(
        paths,
        preferred_source=preferred_source,
        preferred_filename=preferred_filename,
        allow_legacy=not preferred_source,
    )
    user_source = preferred_source.startswith("uploaded_") or supplied_path is not None
    report.stage(
        "transcribe",
        "loading the uploaded transcript (Whisper skipped)" if user_source else "running Whisper speech recognition",
        fraction=0.0,
    )

    provided = _transcript_from_files(
        paths,
        settings,
        report,
        profile,
        preferred_source=preferred_source,
        preferred_filename=preferred_filename,
        allow_legacy=not preferred_source,
    )
    if provided is not None:
        return provided

    if transcription_available():
        cache_key = _transcript_cache_key(media_path, settings)
        if settings.cache_transcripts and not force:
            cached = _load_cached_transcript(paths, cache_key)
            if cached is not None:
                report.sub(1.0, f"reusing the cached Whisper transcript ({cached.word_count} words)")
                report.log("speech recognition skipped: the cached transcript matches this source and these settings")
                return cached
        try:
            transcript = transcribe(
                audio_path,
                settings=settings,
                progress=lambda fraction, message: report.sub(fraction, message),
                should_cancel=report.cancelled,
            )
            transcript.source = "whisper"
            transcript.timing_granularity = "word" if any(utterance.words for utterance in transcript.utterances) else "cue"
            (paths.transcript / "engine.json").write_text(
                json.dumps({"engine": transcript.engine, "model": transcript.model, "language": transcript.language, "source": transcript.source}, indent=1),
                encoding="utf-8",
            )
            if settings.cache_transcripts:
                _store_cached_transcript(paths, cache_key, transcript)
            return transcript
        except ClipForgeError as exc:
            if exc.code == ErrorCode.CANCELLED:
                raise
            report.log(f"local transcription failed: {exc.message}")
            log.warning("transcription failed, trying caption fallback: %s", exc.message)

    return _transcribe_from_captions(paths, settings, metadata, report, profile)


TRANSCRIPT_CACHE_FILE = "asr_cache.json"


def _transcript_cache_key(media_path: Path, settings: AppSettings) -> dict[str, Any]:
    """Everything that changes what speech recognition would produce."""
    try:
        size = media_path.stat().st_size
    except OSError:
        size = 0
    return {
        "source": media_path.name,
        "size": size,
        "model": settings.whisper_model,
        "language_hint": settings.language_hint,
        "translate": settings.translate_captions,
        "alignment": settings.word_alignment,
        "vad": settings.whisper_vad,
        "beam": settings.whisper_beam_size,
        "prompt": settings.whisper_initial_prompt,
    }


def _load_cached_transcript(paths: ProjectPaths, key: dict[str, Any]) -> Transcript | None:
    cache_file = paths.transcript / TRANSCRIPT_CACHE_FILE
    if not cache_file.exists():
        return None
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        if payload.get("key") != key:
            return None
        cached_transcript = payload["transcript"]
        engine = str(cached_transcript.get("engine", "")).lower()
        if "transcript-file" in engine or ("caption" in engine and "timing_granularity" not in cached_transcript):
            log.info("ignoring legacy transcript cache with estimated word timestamps; reload the original cue file or transcribe again")
            return None
        return Transcript.from_dict(cached_transcript)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning("ignoring unreadable transcript cache %s: %s", cache_file, exc)
        return None


def _store_cached_transcript(paths: ProjectPaths, key: dict[str, Any], transcript: Transcript) -> None:
    try:
        paths.transcript.mkdir(parents=True, exist_ok=True)
        (paths.transcript / TRANSCRIPT_CACHE_FILE).write_text(
            json.dumps({"key": key, "transcript": transcript.to_dict()}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:  # caching is an optimisation
        log.warning("could not cache the transcript: %s", exc)


def _translated_profile(profile: language_mod.LanguageProfile, transcript: Transcript) -> language_mod.LanguageProfile:
    """The captions are English now; remember what was actually spoken."""
    spoken = transcript.source_language or profile.primary
    spoken_name = language_mod.language_name(spoken)
    return language_mod.LanguageProfile(
        primary="en",
        primary_name="English",
        secondary=spoken,
        secondary_name=spoken_name,
        mode="translated",
        confidence=profile.confidence,
        whistle_language=profile.whistle_language,
        probabilities=profile.probabilities,
        notes=[*profile.notes, f"Speech in {spoken_name} was translated to English for the captions."],
    )


def _transcript_from_files(
    paths: ProjectPaths,
    settings: AppSettings,
    report: ProgressReporter,
    profile: language_mod.LanguageProfile,
    *,
    preferred_source: str = "",
    preferred_filename: str = "",
    allow_legacy: bool = False,
) -> Transcript | None:
    """Load an authoritative user transcript as cue-level timed text."""
    source = _provided_transcript_file(
        paths,
        preferred_source=preferred_source,
        preferred_filename=preferred_filename,
        allow_legacy=allow_legacy,
    )
    if source is None:
        return None

    strict_user_file = preferred_source.startswith("uploaded_") or source.stem.lower().startswith("provided")
    candidates = [source]
    if allow_legacy and not preferred_source and not strict_user_file:
        candidates = [
            path for path in transcript_files(paths.transcript)
            if not path.stem.lower().startswith(("captions", "auto", "source", "provided"))
        ]
    cues: list[dict[str, Any]] = []
    for candidate in candidates:
        parsed = parse_transcript_segments(candidate)
        if parsed:
            source, cues = candidate, parsed
            break
        report.log(f"{candidate.name} did not contain readable transcript cues")
    if not cues:
        if strict_user_file:
            raise ClipForgeError(
                code=ErrorCode.INVALID_INPUT,
                message=f"The uploaded transcript {preferred_filename or source.name} contains no valid timed cues.",
                hint="Check that the file contains valid timestamps and text, then attach it again.",
                status_code=422,
            )
        return None

    source_code = preferred_source or f"uploaded_{source.suffix.lower().lstrip('.') or 'txt'}"
    original_filename = preferred_filename or source.name
    utterances = [
        Utterance(
            text=cue["text"],
            start=float(cue["start"]),
            end=float(cue["end"]),
            speaker="SPEAKER_01",
            confidence=1.0,
            words=[],
        )
        for cue in cues
    ]
    duration = max((utterance.end for utterance in utterances), default=0.0)
    word_count = sum(len(utterance.text.split()) for utterance in utterances)
    report.sub(0.9, f"{len(cues)} transcript cues loaded from {original_filename}; Whisper skipped")
    report.log(
        f"authoritative user transcript: {original_filename} ({len(cues)} cues, {word_count} words, "
        f"{duration / 60:.1f} min); cue timestamps preserved and no word timings fabricated"
    )
    return Transcript(
        language=profile.primary or "en",
        language_confidence=profile.confidence,
        utterances=utterances,
        engine="uploaded transcript",
        model=original_filename,
        duration=duration,
        diarized=False,
        speaker_count=1,
        warnings=["Uploaded transcript used as authoritative cue-level text; no per-word timing was invented."],
        source=source_code,
        filename=original_filename,
        timing_granularity="cue",
    )


def _transcribe_from_captions(
    paths: ProjectPaths,
    settings: AppSettings,
    metadata: dict[str, Any],
    report: ProgressReporter,
    profile: language_mod.LanguageProfile,
) -> Transcript:
    """Fallback: use the source's own caption track, clearly distinct from Whisper."""
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

    cues = parse_transcript_segments(caption_file)
    if not cues:
        raise ClipForgeError(
            code=ErrorCode.NO_SPEECH,
            message="The downloaded caption track was empty or unreadable.",
            hint="Install faster-whisper for real local transcription instead of relying on captions.",
            status_code=422,
        )

    utterances = [
        Utterance(
            text=cue["text"],
            start=float(cue["start"]),
            end=float(cue["end"]),
            speaker="SPEAKER_01",
            confidence=0.8,
            words=[],
        )
        for cue in cues
    ]
    word_count = sum(len(utterance.text.split()) for utterance in utterances)
    report.log(
        f"transcript built from source caption track {caption_file.name} ({len(cues)} cues); "
        "caption cue timings preserved, no word timings fabricated"
    )
    return Transcript(
        language=profile.primary or "en",
        language_confidence=profile.confidence,
        utterances=utterances,
        engine="source caption track",
        model=caption_file.name,
        duration=max((utterance.end for utterance in utterances), default=0.0),
        diarized=False,
        speaker_count=1,
        warnings=["These are the video's downloaded caption cues, not a Whisper transcript."],
        source="source_captions",
        filename=caption_file.name,
        timing_granularity="cue",
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
    """Persist discovered, scored, rejected and selected candidates separately from clips."""
    from ..db import Candidate as CandidateRow

    selected_keys = {f"{candidate.start_index}:{candidate.end_index}" for candidate in selected}
    ordered = sorted(candidates, key=lambda item: (-item.score, item.start))
    if len(ordered) > discover_mod.MAX_STORED_CANDIDATES:
        selected_candidates = [item for item in ordered if item.key() in selected_keys]
        selected_ids = {id(item) for item in selected_candidates}
        remaining = [item for item in ordered if id(item) not in selected_ids]
        ordered = (selected_candidates + remaining)[: discover_mod.MAX_STORED_CANDIDATES]

    with session_scope() as session:
        session.query(CandidateRow).filter(CandidateRow.project_id == project_id).delete()
        for order, candidate in enumerate(ordered):
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
                    rejection_code=candidate.rejection_code,
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
