"""FINALIZE stage: publish results, record usage, start retention timers (PRD §14, §19)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from clipforge.core.states import ClipStatus, JobStatus
from clipforge.core.timeutil import hours_from_now, utcnow
from clipforge.db.models import Asset, Clip, Job
from clipforge.db.session import session_scope
from clipforge.services.analytics import track
from clipforge.services.events import record_job_event
from clipforge.services.jobs import transition
from clipforge.services.quotas import UsageService
from clipforge.worker.pipeline.base import PipelineContext, Stage

INTERMEDIATES = ("audio.wav", "renders")


class FinalizeStage(Stage):
    name = "finalize"

    def execute(self, ctx: PipelineContext) -> dict[str, Any]:
        with session_scope(ctx.session_factory) as s:
            ready = s.scalar(select(func.count()).select_from(Clip).where(
                Clip.job_id == ctx.job_id, Clip.status == ClipStatus.READY.value)) or 0
        return {"clips_ready": int(ready)}

    def persist(self, ctx: PipelineContext, output: dict[str, Any]) -> list[str]:
        ctx.manifest.mark_stage(self.name, [], 0)
        ctx.manifest.data["completed_at"] = utcnow().isoformat() + "Z"
        ctx.manifest.save()
        # Single transaction: completion + usage + retention (TRD §43).
        with session_scope(ctx.session_factory) as s:
            job = s.get(Job, ctx.job_id)
            job.progress = 100
            transition(s, job, JobStatus.COMPLETED, stage=self.name,
                       message=f"Completed with {output['clips_ready']} clip(s)")
            job.completed_at = utcnow()
            job.error_code = job.error_message = job.diagnostic_id = None
            job.manifest = ctx.manifest.data
            UsageService(s).record_job_completed(job, output["clips_ready"])
            asset = s.get(Asset, ctx.asset_id)
            if asset is not None:
                asset.expires_at = hours_from_now(ctx.settings.RETENTION_HOURS)
            processing = (job.completed_at - job.started_at).total_seconds() if job.started_at else None
            track(s, "job_completed", user_id=job.user_id, job_id=job.id,
                  properties={"clips": output["clips_ready"], "source_seconds": job.source_seconds,
                              "processing_seconds": processing})
            record_job_event(s, job, "job.completed", stage=self.name, metadata={"clips": output["clips_ready"]})
        return []

    def cleanup(self, ctx: PipelineContext) -> None:
        if not ctx.settings.DELETE_INTERMEDIATES_ON_COMPLETE:
            return
        import shutil

        for name in INTERMEDIATES:
            p = ctx.work_dir / name
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
        cached_source = [p for p in ctx.work_dir.glob("source.*")]
        for p in cached_source:
            p.unlink(missing_ok=True)
