"""Candidate window generation (TRD §13).

Windows always start at a sentence start and end at a sentence end, so clips
never begin or end mid-sentence. Overlapping windows are generated within the
duration constraints; scoring + non-maximum suppression picks the final set.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from clipforge.scoring.segmentation import Sentence


@dataclass
class CandidateWindow:
    start_idx: int
    end_idx: int  # inclusive sentence index
    start: float
    end: float
    text: str
    topic_ids: list[int] = field(default_factory=list)
    synthetic: bool = False  # time-based fallback (no speech)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["duration"] = round(self.duration, 3)
        return d


def generate_candidates(
    sentences: Sequence[Sentence],
    *,
    min_seconds: float,
    target_min: float,
    target_max: float,
    max_seconds: float,
    ends_per_start: int = 4,
    max_candidates: int = 3000,
) -> list[CandidateWindow]:
    n = len(sentences)
    target_mid = (target_min + target_max) / 2
    per_start: list[list[CandidateWindow]] = []
    for i in range(n):
        options: list[tuple[tuple, int]] = []
        for j in range(i, n):
            dur = sentences[j].end - sentences[i].start
            if dur > max_seconds:
                break
            if dur < min_seconds:
                continue
            in_target = target_min <= dur <= target_max
            crosses_topic = sentences[j].topic_id != sentences[i].topic_id
            priority = (not in_target, not sentences[j].terminal, crosses_topic, abs(dur - target_mid))
            options.append((priority, j))
        options.sort()
        chosen = sorted(j for _, j in options[:ends_per_start])
        per_start.append([
            CandidateWindow(
                start_idx=i, end_idx=j, start=sentences[i].start, end=sentences[j].end,
                text=" ".join(s.text for s in sentences[i:j + 1]),
                topic_ids=sorted({s.topic_id for s in sentences[i:j + 1]}),
            )
            for j in chosen
        ])
    total = sum(len(c) for c in per_start)
    if total > max_candidates:
        stride = math.ceil(total / max_candidates)
        per_start = per_start[::stride]
    return [c for group in per_start for c in group]


def time_based_candidates(duration: float, *, length: float, max_count: int = 50) -> list[CandidateWindow]:
    """Fallback for videos without detectable speech: evenly spaced windows."""
    if duration <= 0:
        return []
    length = min(length, duration)
    count = max(1, min(max_count, int(duration // length)))
    step = (duration - length) / max(count - 1, 1) if count > 1 else 0.0
    return [CandidateWindow(start_idx=-1, end_idx=-1, start=round(k * step, 3), end=round(k * step + length, 3),
                            text="", synthetic=True) for k in range(count)]


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0
