"""CANDIDATE_GENERATION stage (TRD §13)."""

from __future__ import annotations

from typing import Any

from clipforge.core.presets import scoring_config
from clipforge.scoring.candidates import CandidateWindow, generate_candidates, time_based_candidates
from clipforge.worker.pipeline.base import PipelineContext, Stage


def clip_constraints(ctx: PipelineContext) -> dict[str, float]:
    s = ctx.settings
    return {
        "min_seconds": s.MIN_CLIP_SECONDS,
        "target_min": float(ctx.job_settings.get("min_clip_seconds") or s.TARGET_MIN_SECONDS),
        "target_max": float(ctx.job_settings.get("max_clip_seconds") or s.TARGET_MAX_SECONDS),
        "max_seconds": s.MAX_CLIP_SECONDS,
    }


class CandidateGenerationStage(Stage):
    name = "candidate_generation"

    def execute(self, ctx: PipelineContext) -> list[CandidateWindow]:
        cfg = scoring_config(ctx.settings).get("selection", {})
        c = clip_constraints(ctx)
        cands: list[CandidateWindow] = []
        if ctx.sentences:
            cands = generate_candidates(ctx.sentences, min_seconds=c["min_seconds"], target_min=c["target_min"],
                                        target_max=c["target_max"], max_seconds=c["max_seconds"],
                                        ends_per_start=int(cfg.get("ends_per_start", 4)),
                                        max_candidates=int(cfg.get("max_candidates", 3000)))
        if not cands:
            # No usable speech: fall back to evenly spaced windows (graceful degradation).
            duration = float(ctx.media.get("duration") or 0)
            length = min(max((c["target_min"] + c["target_max"]) / 2, c["min_seconds"]), duration)
            cands = time_based_candidates(duration, length=length)
        ctx.candidates = cands
        return cands

    def persist(self, ctx: PipelineContext, output: list[CandidateWindow]) -> list[str]:
        p = ctx.save_json("candidates.json", [c.to_dict() for c in output])
        ctx.manifest.add_artifact("candidates", p, stage=self.name)
        return ["candidates"]

    def load(self, ctx: PipelineContext) -> None:
        fields = CandidateWindow.__dataclass_fields__
        ctx.candidates = [CandidateWindow(**{k: v for k, v in d.items() if k in fields})
                          for d in ctx.load_json("candidates.json")]


def to_dict_list(cands: list[CandidateWindow]) -> list[dict[str, Any]]:
    return [c.to_dict() for c in cands]
