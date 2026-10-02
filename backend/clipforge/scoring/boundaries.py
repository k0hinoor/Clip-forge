"""Clip boundary optimisation (PRD §8).

Snaps to sentence/word boundaries, adds natural padding into surrounding
silence, and enforces min/max duration without cutting mid-word.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from clipforge.scoring.candidates import CandidateWindow
from clipforge.scoring.segmentation import FlatWord, Sentence

PAD_BEFORE = 0.25
PAD_AFTER = 0.45


@dataclass(frozen=True)
class Boundary:
    start: float
    end: float
    start_idx: int
    end_idx: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def optimize_boundary(
    cand: CandidateWindow,
    sentences: Sequence[Sentence],
    words: Sequence[FlatWord],
    *,
    source_duration: float,
    min_seconds: float,
    max_seconds: float,
) -> Boundary:
    if cand.synthetic or not sentences:
        start = max(0.0, min(cand.start, source_duration - min_seconds))
        end = min(source_duration, max(cand.end, start + min_seconds))
        return Boundary(round(start, 3), round(min(end, start + max_seconds), 3), -1, -1)

    i, j = cand.start_idx, cand.end_idx
    n = len(sentences)

    def span(a: int, b: int) -> float:
        return sentences[b].end - sentences[a].start

    # Too long: drop trailing sentences first (keep the hook), then leading.
    while span(i, j) > max_seconds and j > i and span(i, j - 1) >= min_seconds:
        j -= 1
    while span(i, j) > max_seconds and j > i and span(i + 1, j) >= min_seconds:
        i += 1
    # Too short: extend forward, then backward, without exceeding max.
    while span(i, j) < min_seconds and j + 1 < n and span(i, j + 1) <= max_seconds:
        j += 1
    while span(i, j) < min_seconds and i > 0 and span(i - 1, j) <= max_seconds:
        i -= 1

    first, last = sentences[i], sentences[j]
    start = first.start - min(PAD_BEFORE, max(first.pause_before, 0.0) / 2)
    end = last.end + min(PAD_AFTER, max(min(last.pause_after, 10.0), 0.0) / 2)

    if end - start > max_seconds:
        # A single very long sentence: cut at the last word ending before the limit.
        limit = start + max_seconds
        word_ends = [w.end for w in words[first.word_start:last.word_end + 1] if w.end <= limit]
        end = max(word_ends) if word_ends else limit
    if end - start < min_seconds:
        deficit = min_seconds - (end - start)
        start -= deficit / 2
        end += deficit / 2
    start = max(0.0, start)
    end = min(source_duration, end)
    if end - start < min_seconds:  # clipped by the source edges
        if start <= 0.0:
            end = min(source_duration, start + min_seconds)
        else:
            start = max(0.0, end - min_seconds)
    return Boundary(round(start, 3), round(end, 3), i, j)
