"""CAPTION_LAYOUT stage (TRD §19): caption timelines for every planned clip."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from clipforge.db.models import Clip
from clipforge.db.session import session_scope
from clipforge.worker.pipeline.base import PipelineContext, Stage
from clipforge.worker.pipeline.rendering import ClipRenderer


class CaptionStage(Stage):
    name = "caption_layout"

    def execute(self, ctx: PipelineContext) -> dict[str, Any]:
        renderer = ClipRenderer(ctx.settings, ctx.storage, ctx.session_factory, ctx.models.vision)
        out: list[dict[str, Any]] = []
        with session_scope(ctx.session_factory) as s:
            clips = s.scalars(select(Clip).where(Clip.job_id == ctx.job_id, Clip.deleted_at.is_(None))
                              .order_by(Clip.rank)).all()
            for clip in clips:
                items = renderer.ensure_captions(clip, ctx.words) if clip.captions_enabled else []
                out.append({"clip_id": clip.id, "captions": len(items), "preset": clip.caption_preset})
        return {"clips": out}

    def persist(self, ctx: PipelineContext, output: dict[str, Any]) -> list[str]:
        p = ctx.save_json("captions.json", output)
        ctx.manifest.add_artifact("captions", p, stage=self.name)
        return ["captions"]
