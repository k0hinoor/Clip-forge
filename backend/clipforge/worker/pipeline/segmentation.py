"""SEMANTIC_SEGMENT stage (TRD §12)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from clipforge.db.models import Transcript
from clipforge.db.session import session_scope
from clipforge.scoring.segmentation import Sentence, assign_topics, build_sentences, flatten_words
from clipforge.worker.pipeline.base import PipelineContext, Stage
from clipforge.worker.providers.embeddings import build_embedder


class SegmentStage(Stage):
    name = "segment"

    def execute(self, ctx: PipelineContext) -> dict[str, Any]:
        words = flatten_words(ctx.transcript or {"segments": []})
        sentences = build_sentences(words)
        ctx.report(0.4)
        sentences = assign_topics(sentences, embed=build_embedder(ctx.settings))
        ctx.words, ctx.sentences = words, sentences
        return {"sentences": [s.to_dict() for s in sentences],
                "topics": len({s.topic_id for s in sentences}) if sentences else 0}

    def persist(self, ctx: PipelineContext, output: dict[str, Any]) -> list[str]:
        p = ctx.save_json("segments.json", output)
        ctx.manifest.add_artifact("segments", p, stage=self.name)
        # Mirror topic ids onto transcript segments (by midpoint) for the transcript API.
        with session_scope(ctx.session_factory) as s:
            t = s.scalar(select(Transcript).where(Transcript.job_id == ctx.job_id))
            if t is not None and ctx.sentences:
                for seg in t.segments:
                    mid = (seg.start + seg.end) / 2
                    best = min(ctx.sentences, key=lambda x: 0 if x.start <= mid <= x.end else
                               min(abs(x.start - mid), abs(x.end - mid)))
                    seg.topic_id = best.topic_id
        return ["segments"]

    def load(self, ctx: PipelineContext) -> None:
        ctx.words = flatten_words(ctx.transcript or {"segments": []})
        data = ctx.load_json("segments.json")
        fields = Sentence.__dataclass_fields__
        ctx.sentences = [Sentence(**{k: v for k, v in d.items() if k in fields}) for d in data["sentences"]]
