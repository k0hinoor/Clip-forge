"""Candidate discovery.

The generator is deliberately *exhaustive before it is selective*:

1. every sentence boundary in the transcript is treated as a possible clip start;
2. for each start, every sentence boundary inside the allowed duration window
   (``min_clip_seconds`` .. ``max_clip_seconds``) is a possible end;
3. each resulting range is turned into a real feature vector
   (:mod:`clipforge.ai.features`) and scored;
4. the best-scoring end per start becomes a candidate moment, and other strong
   ends of the same start are kept as separate candidates;
5. optional LLM seeds (from :mod:`clipforge.ai.llm`) are merged in, with the
   score being the average of the analytical and the model judgement - so a
   model hallucination cannot create a high-scoring clip on its own.

For a two-hour podcast this evaluates tens of thousands of ranges and yields
hundreds of candidate moments, which are then deduplicated and filtered in
:mod:`clipforge.ai.scoring`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..constants import CATEGORIES, coerce_category
from ..logging_setup import get_logger
from .features import CandidateFeatures, TranscriptIndex, compute_features, evidence_only, top_human_evidence
from .segment import Sentence
from .textutil import ellipsize, normalize_word, sentence_case, words_only

log = get_logger("clipforge.ai")

# How many alternative end points to evaluate per start sentence.
ENDS_PER_START = 14
# Hard ceiling on stored candidates (keeps the SQLite file and the UI sane).
MAX_STORED_CANDIDATES = 500


@dataclass
class Candidate:
    start: float
    end: float
    start_index: int
    end_index: int
    features: CandidateFeatures
    score: float = 0.0
    title: str = ""
    hook: str = ""
    summary: str = ""
    category: str = "other"
    category_confidence: float = 0.0
    reason: str = ""
    why: list[str] = field(default_factory=list)
    source: str = "heuristic"
    confidence: float = 0.0
    context_before: str = ""
    context_after: str = ""
    status: str = "candidate"
    duplicate_of: str = ""
    llm_note: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def key(self) -> str:
        return f"{self.start_index}:{self.end_index}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 2),
            "score": round(self.score, 1),
            "title": self.title,
            "hook": self.hook,
            "summary": self.summary,
            "category": self.category,
            "category_label": CATEGORIES.get(self.category, self.category),
            "reason": self.reason,
            "why": self.why,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "status": self.status,
            "duplicate_of": self.duplicate_of,
            "features": self.features.to_dict(),
            "meta": self.features.meta,
            "sentence_range": [self.start_index, self.end_index],
        }


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def generate_candidates(
    index: TranscriptIndex,
    *,
    min_seconds: float = 35.0,
    target_seconds: float = 60.0,
    max_seconds: float = 75.0,
    llm_seeds: Sequence[dict[str, Any]] | None = None,
    progress=None,
    should_cancel=None,
) -> list[Candidate]:
    """Discover every plausible short-form moment in the transcript."""
    sentences = index.sentences
    if not sentences:
        return []

    log.info("candidate discovery over %d sentences (%.0fs of speech)", len(sentences), index.duration)
    ends_per_start = _build_end_windows(index, min_seconds, max_seconds)

    best_per_start: dict[int, Candidate] = {}
    extra: list[Candidate] = []
    evaluations = 0

    for start_index, end_options in ends_per_start.items():
        if should_cancel and should_cancel():
            break
        scored: list[tuple[float, int, CandidateFeatures]] = []
        for end_index in end_options:
            features = compute_features(
                index,
                start_index,
                end_index,
                target_seconds=target_seconds,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
            )
            evaluations += 1
            scored.append((features.weighted_score, end_index, features))
        if not scored:
            continue
        scored.sort(key=lambda item: -item[0])
        top_score, top_end, top_features = scored[0]
        best_per_start[start_index] = _build_candidate(index, start_index, top_end, top_features, top_score, target_seconds)

        # Keep genuinely different endings of the same start (score within 6 points
        # but at least 8 seconds apart), so the user can choose a longer version.
        for score, end_index, features in scored[1:6]:
            if score < top_score - 0.06:
                break
            if abs(end_index - top_end) >= 3 and all(abs(end_index - other.end_index) >= 3 for other in extra[-3:]):
                alternate = _build_candidate(index, start_index, end_index, features, score, target_seconds)
                alternate.title = f"{alternate.title} (extended)" if end_index > top_end else alternate.title
                extra.append(alternate)

        if progress and len(best_per_start) % 25 == 0:
            fraction = (start_index + 1) / max(len(sentences), 1)
            progress(fraction, f"analysed {start_index + 1}/{len(sentences)} moments · {len(best_per_start) + len(extra)} candidates")

    candidates = list(best_per_start.values()) + extra
    if llm_seeds:
        candidates = merge_llm_seeds(index, candidates, llm_seeds, min_seconds, max_seconds, target_seconds)

    candidates.sort(key=lambda candidate: -candidate.score)
    log.info("discovery: %d evaluations -> %d candidates (best %.1f)", evaluations, len(candidates), candidates[0].score if candidates else 0.0)
    return candidates


def _build_end_windows(index: TranscriptIndex, min_seconds: float, max_seconds: float) -> dict[int, list[int]]:
    """For every start sentence, the valid end sentences inside the duration window."""
    starts = index.sentences
    total = len(starts)
    windows: dict[int, list[int]] = {}
    j_min = 1
    for i in range(total):
        start_time = starts[i].start
        if j_min < i + 1:
            j_min = i + 1
        while j_min <= total and (starts[min(j_min, total) - 1].end - start_time) < min_seconds:
            j_min += 1
        if j_min > total:
            break
        options: list[int] = []
        j = j_min
        while j <= total and (starts[j - 1].end - start_time) <= max_seconds:
            options.append(j)
            j += 1
        if not options:
            continue
        # Sample the window: always include the extremes plus evenly spaced endings.
        if len(options) > ENDS_PER_START:
            step = (len(options) - 1) / (ENDS_PER_START - 1)
            sampled = sorted({options[round(index * step)] for index in range(ENDS_PER_START)})
        else:
            sampled = options
        windows[i] = sampled
    return windows


def _build_candidate(
    index: TranscriptIndex,
    start_index: int,
    end_index: int,
    features: CandidateFeatures,
    score: float,
    target_seconds: float,
) -> Candidate:
    sentences = index.sentences[start_index:end_index]
    start = sentences[0].start
    end = sentences[-1].end
    duration = max(0.0, end - start)

    title, hook = _title_and_hook(sentences, features)
    summary = _summarize(sentences)
    category, category_confidence = classify_category(features, " ".join(sentence.text for sentence in sentences))
    why = top_human_evidence(features, limit=9)
    reason = _reason_sentence(features, duration, category)

    confidence = _confidence(features, duration, target_seconds, score)
    context_before = index.range_text(max(0, start_index - 2), start_index)
    context_after = index.range_text(end_index, min(len(index.sentences), end_index + 2))

    return Candidate(
        start=start,
        end=end,
        start_index=start_index,
        end_index=end_index,
        features=features,
        score=round(score * 100, 1),
        title=title,
        hook=hook,
        summary=summary,
        category=category,
        category_confidence=category_confidence,
        reason=reason,
        why=why,
        source="heuristic",
        confidence=confidence,
        context_before=context_before,
        context_after=context_after,
    )


# --------------------------------------------------------------------------- #
# Presentation: title, hook, summary, category
# --------------------------------------------------------------------------- #

_PUNCT = re.compile(r"[\"'“”‘’]")


def _title_and_hook(sentences: Sequence[Sentence], features: CandidateFeatures) -> tuple[str, str]:
    """Title = the most quotable line; hook = the actual opening line."""
    hook = _clean(sentences[0].text)
    candidates: list[tuple[float, str]] = []
    for sentence in sentences:
        tokens = words_only(sentence.text)
        if not (4 <= len(tokens) <= 20):
            continue
        text = _clean(sentence.text)
        lowered = text.lower()
        score = 0.4
        if any(marker in lowered for marker in ("never", "always", "everyone", "nobody", "the truth", "most people", "you should", "the reason", "the problem")):
            score += 0.35
        if sentence.text.strip().endswith("."):
            score += 0.05
        if text.endswith("?"):
            score -= 0.1
        score += 0.1 * sentence.features.get("emotion", 0.0)
        candidates.append((score, text))
    if candidates:
        candidates.sort(key=lambda item: -item[0])
        title = candidates[0][1]
    else:
        title = hook
    title = ellipsize(_PUNCT.sub("", title).strip(), 78).rstrip(" .,!?")
    if title and title[0].islower():
        title = title[0].upper() + title[1:]
    return title or "Untitled moment", ellipsize(_PUNCT.sub("", hook).strip(), 240)


def _summarize(sentences: Sequence[Sentence]) -> str:
    if not sentences:
        return ""
    chosen: list[str] = []
    if len(sentences) <= 3:
        chosen = [sentence.text for sentence in sentences]
    else:
        indexes = _spread_indexes(len(sentences), 3)
        chosen = [sentences[index].text for index in indexes]
    return ellipsize(" ".join(_clean(text) for text in chosen), 400)


def _spread_indexes(length: int, count: int) -> list[int]:
    if length <= count:
        return list(range(length))
    step = (length - 1) / (count - 1)
    return sorted({round(index * step) for index in range(count)})


def _clean(text: str) -> str:
    return " ".join((text or "").split())


def classify_category(features: CandidateFeatures, text: str) -> tuple[str, float]:
    """Rule-based category classification driven by computed features."""
    factors = features.factors
    lowered = text.lower()
    scores: dict[str, float] = {
        "story": factors.get("story", 0.0) * 1.15 + (0.25 if features.meta.get("speaker_turns", 0) == 0 else 0.0),
        "emotional": factors.get("emotion", 0.0) * 1.1,
        "funny": factors.get("humour", 0.0) * 1.35,
        "controversial": factors.get("controversy", 0.0) * 1.3,
        "advice": factors.get("practical", 0.0) * 1.25,
        "lesson": factors.get("insight", 0.0) * 1.15 + factors.get("payoff", 0.0) * 0.4,
        "information": factors.get("insight", 0.0) * 0.9 + (0.15 if features.meta.get("numbers") else 0.0),
        "surprising": factors.get("unexpectedness", 0.0) * 1.2,
        "curiosity": factors.get("curiosity", 0.0) * 1.05,
        "inspirational": factors.get("emotion", 0.0) * 0.5 + factors.get("shareability", 0.0) * 0.6,
        "quote": factors.get("shareability", 0.0) * 1.05,
        "argument": factors.get("controversy", 0.0) * 0.8 + factors.get("engagement", 0.0) * 0.4,
        "revelation": factors.get("unexpectedness", 0.0) * 0.9 + factors.get("curiosity", 0.0) * 0.5,
        "experience": factors.get("story", 0.0) * 0.85,
        "qa": 0.35 if features.meta.get("questions", 0) >= 2 else 0.0,
        "opinion": factors.get("controversy", 0.0) * 0.7 + factors.get("insight", 0.0) * 0.4,
        "punchline": factors.get("humour", 0.0) * 0.9 + factors.get("payoff", 0.0) * 0.5,
        "hook": factors.get("hook", 0.0) * 0.85,
        "conclusion": factors.get("payoff", 0.0) * 0.95,
        "story_mini": factors.get("story", 0.0) * 0.9,
    }
    if "?" in text and features.meta.get("questions", 0) >= 2:
        scores["qa"] = max(scores["qa"], 0.5)
    if any(word in lowered for word in ("joke", "funny", "laugh")):
        scores["funny"] = max(scores["funny"], 0.75)
    if any(word in lowered for word in ("i learned", "lesson", "mistake i made")):
        scores["lesson"] = max(scores["lesson"], 0.7)

    category = max(scores.items(), key=lambda item: item[1])
    if category[1] < 0.28:
        return "other", 0.3
    total = sum(scores.values()) or 1.0
    return coerce_category(category[0]), round(min(category[1] / total * 2.2, 1.0), 3)


def _reason_sentence(features: CandidateFeatures, duration: float, category: str) -> str:
    """One-line human explanation, built from the strongest computed signals."""
    positives = sorted(
        ((name, value) for name, value in features.factors.items() if value >= 0.45),
        key=lambda item: -item[1],
    )[:3]
    parts = [f"{name} {value:.2f}" for name, value in positives]
    penalties = sorted(
        ((name, value) for name, value in features.penalties.items() if value >= 0.4),
        key=lambda item: -item[1],
    )[:2]
    text = f"{CATEGORIES.get(category, category)} · {duration:.0f}s · signals: " + ", ".join(parts)
    if penalties:
        text += " · caveats: " + ", ".join(f"{name} {value:.2f}" for name, value in penalties)
    return text


def _confidence(features: CandidateFeatures, duration: float, target_seconds: float, score: float) -> float:
    """How trustworthy the score is (length fit, ASR confidence, penalty load)."""
    length_fit = 1.0 - min(abs(duration - target_seconds) / max(target_seconds, 1.0), 1.0)
    asr = float(features.meta.get("mean_word_confidence", 0.7) or 0.7)
    penalty_load = sum(features.penalties.values()) / max(len(features.penalties), 1)
    confidence = 0.45 * score + 0.25 * length_fit + 0.2 * asr + 0.10 * (1.0 - penalty_load)
    return round(max(0.0, min(1.0, confidence)), 3)


# --------------------------------------------------------------------------- #
# LLM seed merging
# --------------------------------------------------------------------------- #


def merge_llm_seeds(
    index: TranscriptIndex,
    candidates: list[Candidate],
    seeds: Sequence[dict[str, Any]],
    min_seconds: float,
    max_seconds: float,
    target_seconds: float,
) -> list[Candidate]:
    """Blend model-suggested moments with the analytical candidate pool."""
    by_range = {candidate.key(): candidate for candidate in candidates}
    added = 0

    for seed in seeds:
        start = _as_float(seed.get("start"))
        end = _as_float(seed.get("end"))
        if start is None or end is None or end <= start:
            continue
        start_index, end_index = _snap_to_sentences(index, start, end)
        if end_index <= start_index:
            continue
        duration = index.range_duration(start_index, end_index)
        if duration < min_seconds * 0.8 or duration > max_seconds * 1.3:
            continue

        features = compute_features(
            index, start_index, end_index,
            target_seconds=target_seconds, min_seconds=min_seconds, max_seconds=max_seconds,
        )
        llm_score = _as_float(seed.get("score")) or 0.0
        if llm_score > 1.0:
            llm_score /= 100.0
        blended = 0.55 * features.weighted_score + 0.45 * llm_score

        key = f"{start_index}:{end_index}"
        existing = by_range.get(key)
        if existing is not None:
            # Same range found by both paths: trust it more, average the scores.
            existing.score = round(0.5 * (existing.score + blended * 100), 1)
            existing.source = "merged"
            existing.llm_note = (seed.get("reason") or "")[:400]
            if seed.get("category"):
                existing.category = coerce_category(str(seed["category"]))
            if seed.get("title"):
                existing.title = ellipsize(str(seed["title"]), 78)
            if seed.get("hook"):
                existing.hook = ellipsize(str(seed["hook"]), 240)
            existing.why = top_human_evidence(features, limit=9)
            continue

        candidate = _build_candidate(index, start_index, end_index, features, blended, target_seconds)
        candidate.source = "llm"
        candidate.llm_note = (seed.get("reason") or "")
        if seed.get("title"):
            candidate.title = ellipsize(str(seed["title"]), 78)
        if seed.get("hook"):
            candidate.hook = ellipsize(str(seed["hook"]), 240)
        if seed.get("category"):
            candidate.category = coerce_category(str(seed["category"]))
        if seed.get("summary"):
            candidate.summary = ellipsize(str(seed["summary"]), 400)
        if seed.get("why"):
            extras = [str(item) for item in seed["why"]][:4]
            candidate.why = list(dict.fromkeys(candidate.why + extras))
        by_range[key] = candidate
        added += 1

    if added:
        log.info("merged %d LLM-suggested moments into the candidate pool", added)
    return list(by_range.values())


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _snap_to_sentences(index: TranscriptIndex, start: float, end: float) -> tuple[int, int]:
    """Convert absolute times into a sentence range that covers them."""
    sentences = index.sentences
    start_index = 0
    for position, sentence in enumerate(sentences):
        if sentence.end > start:
            start_index = position
            break
    else:
        return len(sentences), len(sentences)

    end_index = len(sentences)
    for position in range(start_index, len(sentences)):
        if sentences[position].start >= end:
            end_index = position
            break
    if end_index <= start_index:
        end_index = min(len(sentences), start_index + 1)
    return start_index, end_index


# --------------------------------------------------------------------------- #
# Filtering / ranking helpers
# --------------------------------------------------------------------------- #


def select_by_mode(
    candidates: Sequence[Candidate],
    *,
    min_score: float = 70.0,
    mode: str = "balanced",
    max_clips: int = 0,
) -> list[Candidate]:
    """Turn a filtered candidate pool into the ranked clip list.

    * ``best``     - only clips at or above the score threshold, ranked by score.
    * ``balanced`` - threshold * 0.92, so near-misses stay visible but ranked lower.
    * ``max``      - every candidate that clears the hard quality gate
      (score >= threshold * 0.85 and no disqualifying penalty).
    """
    if mode == "max":
        floor = min_score * 0.85
        eligible = [
            candidate
            for candidate in candidates
            if candidate.score >= floor
            and candidate.features.penalties.get("context_dependency", 0) < 0.75
            and candidate.features.penalties.get("low_confidence", 0) < 0.6
        ]
    elif mode == "best":
        eligible = [candidate for candidate in candidates if candidate.score >= min_score]
    else:  # balanced
        eligible = [candidate for candidate in candidates if candidate.score >= min_score * 0.92]
        eligible.sort(key=lambda candidate: -candidate.score)
        threshold_pass = [candidate for candidate in eligible if candidate.score >= min_score]
        near_misses = [candidate for candidate in eligible if candidate.score < min_score]
        eligible = threshold_pass + near_misses[: max(3, len(threshold_pass) // 3)]

    eligible.sort(key=lambda candidate: -candidate.score)
    if max_clips and max_clips > 0:
        eligible = eligible[:max_clips]
    return eligible


def discovery_stats(index: TranscriptIndex, candidates: Sequence[Candidate]) -> dict[str, Any]:
    if not candidates:
        return {"candidates": 0}
    scores = sorted((candidate.score for candidate in candidates), reverse=True)
    categories: dict[str, int] = {}
    for candidate in candidates:
        categories[candidate.category] = categories.get(candidate.category, 0) + 1
    return {
        "candidates": len(candidates),
        "duration_covered": round(sum(candidate.duration for candidate in candidates), 1),
        "sentences": len(index.sentences),
        "best_score": round(scores[0], 1),
        "median_score": round(scores[len(scores) // 2], 1),
        "above_70": sum(1 for score in scores if score >= 70),
        "above_80": sum(1 for score in scores if score >= 80),
        "above_90": sum(1 for score in scores if score >= 90),
        "categories": dict(sorted(categories.items(), key=lambda item: -item[1])),
        "speech_rate_wpm": index.speech_rate_wpm,
    }


__all__ = [
    "Candidate",
    "MAX_STORED_CANDIDATES",
    "classify_category",
    "discovery_stats",
    "generate_candidates",
    "merge_llm_seeds",
    "select_by_mode",
]
