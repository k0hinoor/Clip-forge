"""VISUAL_ANALYSIS stage (TRD §17, §18): sampled-frame analysis + crop trajectories."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from clipforge.core.presets import scoring_config
from clipforge.db.models import Candidate, Clip
from clipforge.db.session import session_scope
from clipforge.scoring.engine import ScoringEngine
from clipforge.services.clips import profile_for_aspect
from clipforge.worker.pipeline.base import PipelineContext, Stage
from clipforge.worker.pipeline.rendering import ClipRenderer, RenderInputs, profile_dict
from clipforge.worker.providers.vision import visual_quality


class VisualAnalysisStage(Stage):
    name = "visual_analysis"

    def execute(self, ctx: PipelineContext) -> dict[str, Any]:
        renderer = ClipRenderer(ctx.settings, ctx.storage, ctx.session_factory, ctx.models.vision,
                                cancel_check=ctx.is_cancelled)
        inputs = RenderInputs(source_path=ctx.source_path, media=ctx.media, words=ctx.words, work_dir=ctx.work_dir)
        engine = ScoringEngine(scoring_config(ctx.settings))
        summary: list[dict[str, Any]] = []
        with session_scope(ctx.session_factory) as s:
            clips = s.scalars(select(Clip).where(Clip.job_id == ctx.job_id, Clip.deleted_at.is_(None))
                              .order_by(Clip.rank)).all()
            for i, clip in enumerate(clips):
                ctx.check_cancel()
                profile = profile_dict(profile_for_aspect(s, clip.aspect_ratio))
                plan = renderer.ensure_framing(clip, inputs, profile)
                frames = (clip.visual or {}).get("frames", [])
                vq = visual_quality(frames, plan["mode"])
                # Refine the stored score with the visual signal (raw features kept).
                cand = s.get(Candidate, clip.candidate_id) if clip.candidate_id else None
                if cand is not None and cand.features is not None:
                    feats = {**cand.features.features, "visual_quality": vq}
                    cand.features.features = feats
                    cand.score = engine.score(feats)
                    clip.score = cand.score
                summary.append({"clip_id": clip.id, "mode": plan["mode"], "face_ratio": plan["face_ratio"],
                                "frames": len(frames), "visual_quality": vq})
                ctx.report((i + 1) / max(len(clips), 1))
        return {"provider": ctx.models.vision.name, "clips": summary}

    def persist(self, ctx: PipelineContext, output: dict[str, Any]) -> list[str]:
        p = ctx.save_json("visual.json", output)
        ctx.manifest.add_artifact("visual", p, stage=self.name)
        return ["visual"]
