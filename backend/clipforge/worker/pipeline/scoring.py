"""CANDIDATE_SCORING (TRD §14, §15) and BOUNDARY_OPTIMIZATION (PRD §8) stages."""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, select

from clipforge.core.errors import AppError, ErrorCode
from clipforge.core.presets import scoring_config
from clipforge.core.states import ClipStatus
from clipforge.db.models import Candidate, CandidateFeatures, Clip, Job, Render, new_id
from clipforge.db.session import session_scope
from clipforge.scoring.boundaries import optimize_boundary
from clipforge.scoring.candidates import CandidateWindow
from clipforge.scoring.engine import ScoredCandidate, ScoringEngine, select_top
from clipforge.scoring.features import FeatureContext, compute_features
from clipforge.scoring.text import TfIdf
from clipforge.security.files import slugify
from clipforge.worker.pipeline.base import PipelineContext, Stage
from clipforge.worker.pipeline.candidates import clip_constraints
from clipforge.worker.providers.llm import HeuristicProvider
from clipforge.worker.providers.metadata import MetadataGenerator


def _serialize(c: ScoredCandidate, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"window": c.window.to_dict(), "features": c.features, "raw": c.raw, "score": c.score,
            "reasons": c.reasons, "metadata": metadata}


class CandidateScoringStage(Stage):
    name = "candidate_scoring"

    def execute(self, ctx: PipelineContext) -> dict[str, Any]:
        engine = ScoringEngine(scoring_config(ctx.settings))
        tfidf = TfIdf([s.text for s in ctx.sentences]) if ctx.sentences else None
        c = clip_constraints(ctx)
        fctx = FeatureContext(sentences=ctx.sentences, tfidf=tfidf, audio_levels=ctx.audio_levels,
                              min_seconds=c["min_seconds"], target_min=c["target_min"],
                              target_max=c["target_max"], max_seconds=c["max_seconds"])
        scored: list[ScoredCandidate] = []
        total = max(len(ctx.candidates), 1)
        for i, window in enumerate(ctx.candidates):
            if i % 200 == 0:
                ctx.check_cancel()
                ctx.report(0.6 * i / total)
            features, raw = compute_features(window, fctx)
            scored.append(engine.apply(ScoredCandidate(window=window, features=features, raw=raw)))

        k = int(ctx.job_settings.get("num_clips") or ctx.settings.DEFAULT_CLIPS_PER_JOB)
        generator = MetadataGenerator(ctx.models.llm)
        language = (ctx.transcript or {}).get("language")
        metadata: dict[int, dict[str, Any]] = {}

        def meta_for(cand: ScoredCandidate) -> dict[str, Any]:
            key = id(cand)
            if key not in metadata:
                metadata[key] = generator.generate(cand.window.text, duration=cand.window.duration,
                                                   reasons=cand.reasons, tfidf=tfidf, language=language).to_dict()
            return metadata[key]

        if engine.weights.get("llm_interest", 0) > 0 and not isinstance(ctx.models.llm, HeuristicProvider):
            # Semantic re-ranking of a shortlist by the local LLM (excerpts only).
            pool = select_top(scored, k * 3, engine.max_overlap_iou)
            for j, cand in enumerate(pool):
                ctx.check_cancel()
                interest = meta_for(cand).get("interest")
                if interest is not None:
                    cand.features["llm_interest"] = round(float(interest), 4)
                    engine.apply(cand)
                ctx.report(0.6 + 0.3 * (j + 1) / max(len(pool), 1))

        selected = select_top(scored, k, engine.max_overlap_iou)
        for j, cand in enumerate(selected):
            ctx.check_cancel()
            meta_for(cand)
            ctx.report(0.9 + 0.1 * (j + 1) / max(len(selected), 1))
        top = sorted(scored, key=lambda x: -x.score)[: engine.persist_top_n]
        return {
            "scoring_version": engine.version,
            "weights": engine.weights,
            "candidate_count": len(scored),
            "selected": [_serialize(cand, metadata.get(id(cand))) for cand in selected],
            "top": [_serialize(cand) for cand in top],
        }

    def persist(self, ctx: PipelineContext, output: dict[str, Any]) -> list[str]:
        ctx.selected = output["selected"]
        p = ctx.save_json("scored.json", output)
        ctx.manifest.add_artifact("scored", p, stage=self.name)
        version = output["scoring_version"]
        selected_spans = {(s["window"]["start"], s["window"]["end"]) for s in output["selected"]}
        with session_scope(ctx.session_factory) as s:
            s.execute(delete(Candidate).where(Candidate.job_id == ctx.job_id))
            seen: set[tuple[float, float]] = set()
            rank = 0
            for item in [*output["selected"], *output["top"]]:
                span = (item["window"]["start"], item["window"]["end"])
                if span in seen:
                    continue
                seen.add(span)
                rank += 1
                cand = Candidate(id=new_id(), job_id=ctx.job_id, rank=rank, start=span[0], end=span[1],
                                 text=item["window"]["text"], score=item["score"], reasons=item["reasons"],
                                 selected=span in selected_spans,
                                 topic_id=(item["window"].get("topic_ids") or [None])[0],
                                 scoring_version=version)
                cand.features = CandidateFeatures(features=item["features"], weights=output["weights"],
                                                  raw=item["raw"], scoring_version=version)
                s.add(cand)
                item["candidate_id"] = cand.id
            job = s.get(Job, ctx.job_id)
            job.versions = {**(job.versions or {}), "scoring_version": version,
                            **ctx.models.versions()}
        p = ctx.save_json("scored.json", output)  # now includes candidate ids
        ctx.manifest.add_artifact("scored", p, stage=self.name)
        return ["scored"]

    def load(self, ctx: PipelineContext) -> None:
        ctx.selected = ctx.load_json("scored.json")["selected"]


class BoundaryOptimizationStage(Stage):
    name = "boundary_optimization"

    def execute(self, ctx: PipelineContext) -> list[dict[str, Any]]:
        duration = float(ctx.media.get("duration") or 0)
        plan = []
        for rank, item in enumerate(ctx.selected, start=1):
            fields = CandidateWindow.__dataclass_fields__
            window = CandidateWindow(**{k: v for k, v in item["window"].items() if k in fields})
            b = optimize_boundary(window, ctx.sentences, ctx.words, source_duration=duration,
                                  min_seconds=ctx.settings.MIN_CLIP_SECONDS, max_seconds=ctx.settings.MAX_CLIP_SECONDS)
            excerpt = " ".join(w.word for w in ctx.words if w.start >= b.start - 0.05 and w.end <= b.end + 0.05)
            meta = item.get("metadata") or {}
            plan.append({"rank": rank, "start": b.start, "end": b.end, "candidate_id": item.get("candidate_id"),
                         "score": item["score"], "reasons": item["reasons"], "excerpt": excerpt,
                         "title": meta.get("title"), "hook": meta.get("hook"),
                         "description": meta.get("description"), "keywords": meta.get("keywords") or [],
                         "reasoning": meta.get("reasoning"), "metadata_source": meta.get("source")})
        return plan

    def persist(self, ctx: PipelineContext, output: list[dict[str, Any]]) -> list[str]:
        js = ctx.job_settings
        with session_scope(ctx.session_factory) as s:
            old_ids = s.scalars(select(Clip.id).where(Clip.job_id == ctx.job_id)).all()
            if old_ids:  # re-running this stage replaces previously planned clips
                s.execute(delete(Render).where(Render.clip_id.in_(old_ids)))
                s.execute(delete(Clip).where(Clip.id.in_(old_ids)))
            for item in output:
                clip = Clip(user_id=ctx.user_id, job_id=ctx.job_id, candidate_id=item["candidate_id"],
                            rank=item["rank"], status=ClipStatus.PENDING.value, start=item["start"],
                            end=item["end"], slug=slugify(item["title"] or f"clip-{item['rank']}"),
                            title=item["title"], hook=item["hook"], description=item["description"],
                            keywords=item["keywords"], reasoning=item["reasoning"],
                            transcript_excerpt=item["excerpt"], score=item["score"], reasons=item["reasons"],
                            aspect_ratio=js.get("aspect_ratio", "9:16"), caption_preset=js.get("caption_preset", "clean"),
                            captions_enabled=bool(js.get("captions_enabled", True)) and bool(item["excerpt"]),
                            framing_mode=js.get("framing_mode", "auto"), metadata_source=item["metadata_source"])
                s.add(clip)
                s.flush()
                item["clip_id"] = clip.id
        p = ctx.save_json("clips_plan.json", output)
        ctx.manifest.add_artifact("clips_plan", p, stage=self.name)
        return ["clips_plan"]

    def validate_output(self, ctx: PipelineContext, output: list[dict[str, Any]]) -> None:
        for item in output:
            if not 0 < item["end"] - item["start"] <= ctx.settings.MAX_CLIP_SECONDS + 1:
                raise AppError(ErrorCode.UNKNOWN_ERROR, internal=f"invalid boundary {item}")
