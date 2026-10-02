"""Configurable, deterministic scoring + selection (TRD §14; PRD §7).

Scores are 0-100 and describe how well a candidate matches measurable content
signals — they are *not* a virality prediction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from clipforge.scoring.candidates import CandidateWindow, iou

REASON_LABELS = {
    "hook": "Strong opening hook",
    "completeness": "Complete thought",
    "emotional_interest": "Emotional / high-energy moment",
    "information_value": "High information density",
    "narrative_completeness": "Clear story arc or conclusion",
    "caption_suitability": "Good pacing for captions",
    "length_fit": "Ideal short-form length",
}
END_REASON = "Clear ending"


@dataclass
class ScoredCandidate:
    window: CandidateWindow
    features: dict[str, float]
    raw: dict[str, Any]
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def span(self) -> tuple[float, float]:
        return (self.window.start, self.window.end)


class ScoringEngine:
    def __init__(self, config: dict[str, Any]) -> None:
        self.version = str(config.get("version", "1.0"))
        weights = {k: float(v) for k, v in (config.get("weights") or {}).items()}
        if not weights or sum(weights.values()) <= 0:
            raise ValueError("Scoring weights must be configured and sum to > 0")
        if any(w < 0 for w in weights.values()):
            raise ValueError("Scoring weights must be non-negative")
        self.weights = weights
        self.reason_thresholds = {k: float(v) for k, v in (config.get("reasons") or {}).items()}
        sel = config.get("selection") or {}
        self.max_overlap_iou = float(sel.get("max_overlap_iou", 0.25))
        self.persist_top_n = int(sel.get("persist_top_n", 50))
        self.max_candidates = int(sel.get("max_candidates", 3000))
        self.ends_per_start = int(sel.get("ends_per_start", 4))

    def score(self, features: dict[str, float]) -> float:
        total_w = sum(self.weights.values())
        value = sum(w * float(features.get(name, 0.0)) for name, w in self.weights.items())
        return round(100.0 * value / total_w, 2)

    def reasons(self, features: dict[str, float], raw: dict[str, Any] | None = None) -> list[str]:
        if raw and raw.get("synthetic"):
            return ["No speech detected — evenly spaced segment"]
        out = []
        ranked = sorted(self.reason_thresholds.items(), key=lambda kv: -(features.get(kv[0], 0) - kv[1]))
        for name, threshold in ranked:
            if features.get(name, 0.0) >= threshold and name in REASON_LABELS:
                out.append(REASON_LABELS[name])
        if raw and raw.get("end_clean") and END_REASON not in out:
            out.append(END_REASON)
        return out[:5] or ["Best available segment for this video"]

    def apply(self, cand: ScoredCandidate) -> ScoredCandidate:
        cand.score = self.score(cand.features)
        cand.reasons = self.reasons(cand.features, cand.raw)
        return cand


def select_top(cands: Sequence[ScoredCandidate], k: int, max_iou: float) -> list[ScoredCandidate]:
    """Greedy non-maximum suppression: best scores first, limited overlap."""
    ordered = sorted(cands, key=lambda c: (-c.score, c.window.start, c.window.end))
    picked: list[ScoredCandidate] = []
    for c in ordered:
        if all(iou(c.span, p.span) <= max_iou and not _contains(c.span, p.span) for p in picked):
            picked.append(c)
            if len(picked) >= k:
                break
    return picked


def _contains(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return (a[0] <= b[0] and a[1] >= b[1]) or (b[0] <= a[0] and b[1] >= a[1])
