"""TRANSCRIBE stage (TRD §11, §54) with GPU→CPU and model-size fallback."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import delete, select

from clipforge.core.errors import AppError, ErrorCode
from clipforge.core.logging import get_logger
from clipforge.db.models import Job, Transcript, TranscriptSegment
from clipforge.db.session import session_scope
from clipforge.services.events import record_job_event
from clipforge.storage import Keys
from clipforge.worker.pipeline.base import PipelineContext, Stage
from clipforge.worker.providers.transcription import TranscriptResult

log = get_logger(__name__)
FALLBACK_CODES = {ErrorCode.MODEL_UNAVAILABLE, ErrorCode.INSUFFICIENT_MEMORY}


class TranscribeStage(Stage):
    name = "transcribe"

    def execute(self, ctx: PipelineContext) -> dict[str, Any]:
        audio = ctx.manifest.artifact_path("audio")
        if not ctx.media.get("has_audio") or audio is None:
            return {"language": None, "duration": ctx.media.get("duration"), "provider": "none", "model": "none",
                    "device": "none", "segments": []}
        provider = ctx.models.transcriber
        attempts = ctx.models.transcription_plan(ctx.job_settings.get("whisper_model"))
        language = ctx.job_settings.get("language") or ctx.settings.WHISPER_LANGUAGE or None
        last: AppError | None = None
        for i, attempt in enumerate(attempts):
            ctx.check_cancel()
            try:
                with ctx.guard.transcription_slots:
                    result: TranscriptResult = provider.transcribe(
                        audio, model=attempt.model, device=attempt.device, compute_type=attempt.compute_type,
                        language=language, progress=lambda f: ctx.report(f * 0.98), cancel_check=ctx.is_cancelled)
                data = result.to_dict()
                data["compute_type"] = attempt.compute_type
                return data
            except AppError as err:
                if err.code not in FALLBACK_CODES:
                    raise
                if err.code == ErrorCode.MODEL_UNAVAILABLE and err.retryable is False:
                    raise  # provider not installed: smaller models will not help
                last = err
                nxt = attempts[i + 1] if i + 1 < len(attempts) else None
                log.warning("transcription model fallback", extra={"model": attempt.model, "device": attempt.device,
                                                                   "error_code": err.code.value})
                with session_scope(ctx.session_factory) as s:
                    job = s.get(Job, ctx.job_id)
                    record_job_event(s, job, "model.fallback", stage=self.name, error_code=err.code.value,
                                     message=(f"{attempt.model} on {attempt.device} unavailable; trying "
                                              f"{nxt.model} on {nxt.device}") if nxt else "No smaller model left")
        raise last or AppError(ErrorCode.MODEL_UNAVAILABLE)

    def persist(self, ctx: PipelineContext, output: dict[str, Any]) -> list[str]:
        ctx.transcript = output
        p = ctx.save_json("transcript.json", output)
        ctx.manifest.add_artifact("transcript", p, stage=self.name)
        key = Keys.transcript(ctx.job_id)
        ctx.storage.put_bytes(key, json.dumps(output).encode("utf-8"))
        text = " ".join(seg["text"].strip() for seg in output["segments"]).strip()
        with session_scope(ctx.session_factory) as s:
            existing = s.scalar(select(Transcript).where(Transcript.job_id == ctx.job_id))
            if existing is not None:
                s.execute(delete(TranscriptSegment).where(TranscriptSegment.transcript_id == existing.id))
                s.delete(existing)
                s.flush()
            t = Transcript(job_id=ctx.job_id, language=output.get("language"), provider=output["provider"],
                           model=output["model"], duration_seconds=output.get("duration"), text=text,
                           word_count=sum(len(seg.get("words") or []) for seg in output["segments"]),
                           storage_key=key, metadata_json={"device": output.get("device"),
                                                           "compute_type": output.get("compute_type"),
                                                           "language_probability": output.get("language_probability")})
            s.add(t)
            s.flush()
            for seg in output["segments"]:
                s.add(TranscriptSegment(transcript_id=t.id, idx=seg["id"], start=seg["start"], end=seg["end"],
                                        text=seg["text"], words=seg.get("words") or [],
                                        speaker_id=seg.get("speaker_id"), avg_logprob=seg.get("avg_logprob"),
                                        no_speech_prob=seg.get("no_speech_prob")))
            job = s.get(Job, ctx.job_id)
            job.versions = {**(job.versions or {}), "transcription_provider": output["provider"],
                            "whisper_model": output["model"], "whisper_device": output.get("device")}
        ctx.versions.update(whisper_model=output["model"])
        return ["transcript"]

    def validate_output(self, ctx: PipelineContext, output: dict[str, Any]) -> None:
        for seg in output["segments"]:
            if seg["end"] < seg["start"]:
                raise AppError(ErrorCode.TRANSCRIPTION_FAILED, internal="segment end before start")

    def load(self, ctx: PipelineContext) -> None:
        ctx.transcript = ctx.load_json("transcript.json")
