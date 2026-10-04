"""Scoring, deduplication, context validation and boundary optimisation.

This module turns the raw candidate pool from :mod:`clipforge.ai.candidates`
into the ranked clip list:

* **context validation** - drop moments that only make sense after watching the
  preceding 40 minutes;
* **duplicate detection** - temporal overlap *and* semantic similarity, so
  ``00:42:10-00:43:12``, ``00:42:18-00:43:15`` and ``00:42:24-00:43:17`` collapse
  into one clip while genuinely different moments from the same topic survive;
* **boundary optimisation** - snap to sentence boundaries, restore the setup
  sentence when a clip opens mid-thought, trim trailing topic drift, and refuse
  to end immediately after the hook;
* **re-scoring** - features are recomputed on the *final* boundaries, so the
  number shown in the UI describes the clip that will actually render.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..constants import category_label
from ..logging_setup import get_logger
from .candidates import Candidate, select_by_mode
from .features import TranscriptIndex, compute_features, top_human_evidence
from .textutil import cosine, tfidf_matrix

log = get_logger("clipforge.ai")

OVERLAP_IOU_THRESHOLD = 0.55
CONTAINMENT_THRESHOLD = 0.78
SEMANTIC_DUPLICATE_THRESHOLD = 0.66
MAX_CONTEXT_DEPENDENCY = 0.62
MIN_STANDALONE = 0.30


# --------------------------------------------------------------------------- #
# Temporal overlap helpers
# --------------------------------------------------------------------------- #


def temporal_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if intersection <= 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union > 0 else 0.0


def containment(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """How much of the shorter clip is inside the longer one."""
    shorter = min(a_end - a_start, b_end - b_start)
    if shorter <= 0:
        return 0.0
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    return intersection / shorter


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_context(candidate: Candidate, *, max_context_dependency: float = MAX_CONTEXT_DEPENDENCY) -> tuple[bool, str]:
    """Reject clips that cannot stand alone."""
    penalties = candidate.features.penalties
    if penalties.get("context_dependency", 0.0) >= max_context_dependency:
        return False, "Needs earlier context to make sense"
    if candidate.features.factors.get("standalone", 0.0) < MIN_STANDALONE:
        return False, "Does not stand alone as a complete thought"
    if penalties.get("low_confidence", 0.0) >= 0.7:
        return False, "Transcript confidence is too low to caption reliably"
    return True, ""


def deduplicate(
    candidates: Sequence[Candidate],
    *,
    iou_threshold: float = OVERLAP_IOU_THRESHOLD,
    semantic_threshold: float = SEMANTIC_DUPLICATE_THRESHOLD,
) -> tuple[list[Candidate], list[Candidate]]:
    """Split the pool into kept candidates and duplicates of a better clip."""
    ordered = sorted(candidates, key=lambda candidate: (-candidate.score, candidate.start))
    if not ordered:
        return [], []

    texts = [candidate.title + ". " + candidate.summary for candidate in ordered]
    vectors, _ = tfidf_matrix(texts)

    kept: list[Candidate] = []
    duplicates: list[Candidate] = []

    for index, candidate in enumerate(ordered):
        duplicate_of: Candidate | None = None
        similarity_note = ""
        for kept_index, existing in enumerate(kept):
            position = ordered.index(existing)
            iou = temporal_iou(candidate.start, candidate.end, existing.start, existing.end)
            inside = containment(candidate.start, candidate.end, existing.start, existing.end)
            similarity = cosine(vectors[index], vectors[position])

            if iou >= iou_threshold or inside >= CONTAINMENT_THRESHOLD:
                # Same moment. Unless they are semantically quite different
                # (two distinct points inside one long take), it is a duplicate.
                if similarity >= semantic_threshold or iou >= 0.8:
                    duplicate_of = existing
                    similarity_note = f"{int(iou * 100)}% temporal overlap, {int(similarity * 100)}% semantic similarity"
                    break
                hook_gap = abs(candidate.start_index - existing.start_index)
                if hook_gap <= 6:
                    duplicate_of = existing
                    similarity_note = f"same moment ({int(iou * 100)}% overlap, hook {hook_gap} sentences apart)"
                    break
            elif similarity >= 0.86 and abs(candidate.start - existing.start) < 90:
                duplicate_of = existing
                similarity_note = f"{int(similarity * 100)}% semantic similarity within 90 seconds"
                break

        if duplicate_of is None:
            candidate.status = "kept"
            candidate.duplicate_of = ""
            kept.append(candidate)
        else:
            candidate.status = "duplicate"
            candidate.duplicate_of = f"{duplicate_of.start_index}:{duplicate_of.end_index}"
            candidate.reason = (candidate.reason + f" · duplicate of {format_timestamp(duplicate_of.start)} ({similarity_note})").strip(" ·")
            duplicates.append(candidate)

    log.info("dedupe: %d candidates -> %d kept, %d duplicates", len(candidates), len(kept), len(duplicates))
    return kept, duplicates


# --------------------------------------------------------------------------- #
# Boundary optimisation
# --------------------------------------------------------------------------- #
# Selection guards
# --------------------------------------------------------------------------- #


def enforce_non_overlap(
    candidates: Sequence[Candidate],
    *,
    max_overlap_ratio: float = 0.35,
    protected_top: int = 0,
) -> tuple[list[Candidate], list[Candidate]]:
    """Greedy score-ordered filter: no two selected clips may share a moment.

    A clip survives when it overlaps an already accepted clip by less than
    ``max_overlap_ratio`` of the shorter clip - so back-to-back clips that merely
    touch are fine, while the same joke told twice is not.
    """
    accepted: list[Candidate] = []
    dropped: list[Candidate] = []
    ordered = sorted(candidates, key=lambda candidate: -candidate.score)
    for position, candidate in enumerate(ordered):
        clash = False
        for existing in accepted:
            overlap = min(candidate.end, existing.end) - max(candidate.start, existing.start)
            if overlap <= 0:
                continue
            shorter = min(candidate.duration, existing.duration) or 1.0
            if overlap / shorter > max_overlap_ratio:
                clash = True
                if position >= protected_top:
                    candidate.reason = (
                        candidate.reason + f" · trimmed out: overlaps {format_timestamp(existing.start)}"
                    ).strip(" ·")
                    candidate.status = "duplicate"
                break
        if clash and position >= protected_top:
            dropped.append(candidate)
        else:
            accepted.append(candidate)
    return accepted, dropped


@dataclass
class BoundaryResult:
    start_index: int
    end_index: int
    notes: list[str] = field(default_factory=list)
    changed: bool = False


def optimize_bounds(
    index: TranscriptIndex,
    start_index: int,
    end_index: int,
    *,
    min_seconds: float,
    max_seconds: float,
    target_seconds: float,
) -> BoundaryResult:
    """Move the edges to natural thought boundaries without changing the moment."""
    notes: list[str] = []
    stats = index.stats
    original = (start_index, end_index)

    def duration(first: int, last: int) -> float:
        return index.range_duration(first, last)

    # ------------------------------------------------ extend backwards for setup
    for _ in range(3):
        if start_index <= 0:
            break
        first = stats[start_index]
        previous = stats[start_index - 1]
        needs_setup = (
            first.starts_deictic
            or first.starts_conjunction
            or first.token_count < 4
            or (previous.questions and not first.questions)
            or (previous.duration < 9 and first.starts_deictic)
        )
        if not needs_setup:
            break
        if duration(start_index - 1, end_index) > max_seconds * 1.02:
            break
        start_index -= 1
        notes.append("Extended the start back to include the sentence that sets it up")

    # ------------------------------------------------------- trim trailing drift
    while end_index - start_index > 2 and duration(start_index, end_index) > max_seconds:
        end_index -= 1
        notes.append("Trimmed a trailing sentence to respect the maximum duration")

    if end_index - start_index > 2:
        last = stats[end_index - 1]
        previous = stats[end_index - 2]
        trailing_topic_shift = previous.text and last.text and index.sentences[end_index - 1].block != index.sentences[end_index - 2].block
        pointless_tail = last.token_count <= 12 and duration(start_index, end_index - 1) >= min_seconds
        if (not last.ends_terminal and pointless_tail) or (trailing_topic_shift and pointless_tail):
            end_index -= 1
            notes.append("Ended on the previous complete thought instead of a trailing tangent")

    # --------------------------------------------------- grow if it is too short
    while duration(start_index, end_index) < min_seconds and end_index < len(index.sentences):
        end_index += 1
        notes.append("Extended the end to reach the minimum duration")

    # -------------------------------------------- prefer an ending with a period
    if end_index - start_index >= 2 and not stats[end_index - 1].ends_terminal:
        for candidate_end in range(end_index - 1, start_index + 1, -1):
            if stats[candidate_end - 1].ends_terminal and duration(start_index, candidate_end) >= min_seconds:
                if candidate_end != end_index:
                    notes.append("Snapped the end to the last complete sentence")
                end_index = candidate_end
                break

    if end_index <= start_index:
        end_index = min(len(index.sentences), start_index + 1)

    # --------------------------------------------------------- final duration fit
    while duration(start_index, end_index) > max_seconds and end_index - start_index > 1:
        end_index -= 1
    while duration(start_index, end_index) < min_seconds and start_index > 0 and (end_index - start_index) < 3:
        start_index -= 1

    result = BoundaryResult(start_index=start_index, end_index=end_index, notes=notes, changed=(start_index, end_index) != original)
    if result.changed:
        log.debug(
            "bounds %s -> %s (%.1fs -> %.1fs)",
            original,
            (start_index, end_index),
            index.range_duration(*original),
            duration(start_index, end_index),
        )
    return result


def rescore(index: TranscriptIndex, candidate: Candidate, *, min_seconds: float, max_seconds: float, target_seconds: float) -> Candidate:
    """Recompute features and score on the final boundaries."""
    features = compute_features(
        index,
        candidate.start_index,
        candidate.end_index,
        target_seconds=target_seconds,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
    )
    candidate.features = features
    candidate.score = round(features.weighted_score * 100, 1)
    candidate.why = top_human_evidence(features, limit=9)
    candidate.start = index.sentences[candidate.start_index].start
    candidate.end = index.sentences[candidate.end_index - 1].end
    return candidate


# --------------------------------------------------------------------------- #
# Full pipeline step
# --------------------------------------------------------------------------- #


def finalize_candidates(
    index: TranscriptIndex,
    candidates: Sequence[Candidate],
    *,
    min_seconds: float,
    target_seconds: float,
    max_seconds: float,
    min_score: float,
    mode: str = "balanced",
    max_clips: int = 0,
    optimize: bool = True,
    on_phase: Any = None,
) -> dict[str, Any]:
    """Validate → optimise boundaries → re-score → dedupe → rank → select.

    ``on_phase(phase_key, message)`` is called before each sub-phase so the UI can
    show real progress through scoring, boundary work, dedupe and selection.
    """

    def phase(key: str, message: str) -> None:
        if on_phase is not None:
            try:
                on_phase(key, message)
            except Exception:  # noqa: BLE001 - reporting must never break analysis
                pass

    phase("score", f"scoring {len(candidates)} candidates against 14 quality signals")
    report: dict[str, Any] = {
        "input": len(candidates),
        "rejected_context": 0,
        "rejected_other": 0,
        "boundaries_changed": 0,
        "duplicates": 0,
    }

    # 1. context validation on the raw pool (only obvious rejects, so that
    #    boundary optimisation still gets a chance to fix marginal ones).
    usable: list[Candidate] = []
    for candidate in candidates:
        if candidate.features.penalties.get("context_dependency", 0.0) >= 0.85:
            candidate.status = "rejected"
            candidate.reason = (candidate.reason + " · rejected: cannot stand alone").strip(" ·")
            report["rejected_context"] += 1
            continue
        usable.append(candidate)

    # 2. boundary optimisation + re-score.
    phase("boundaries", f"optimising clip boundaries for {len(usable)} candidates")
    for candidate in usable:
        if optimize:
            bounds = optimize_bounds(
                index,
                candidate.start_index,
                candidate.end_index,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
                target_seconds=target_seconds,
            )
            if bounds.changed:
                report["boundaries_changed"] += 1
                candidate.start_index, candidate.end_index = bounds.start_index, bounds.end_index
        rescore(index, candidate, min_seconds=min_seconds, max_seconds=max_seconds, target_seconds=target_seconds)

    # 3. hard validation on final boundaries.
    phase("validate", "checking that every clip stands on its own")
    valid: list[Candidate] = []
    for candidate in usable:
        ok, why = validate_context(candidate)
        if not ok:
            candidate.status = "rejected"
            candidate.reason = (candidate.reason + f" · rejected: {why}").strip(" ·")
            report["rejected_other"] += 1
            continue
        valid.append(candidate)

    # 4. deduplicate (twice: once on the pool, once again after optimisation may
    #    have widened overlaps).
    phase("dedupe", f"removing overlaps and near-duplicates from {len(valid)} candidates")
    kept, duplicates = deduplicate(valid)
    if duplicates:
        kept, duplicates_again = deduplicate(kept)
        duplicates.extend(duplicates_again)
    report["duplicates"] = len(duplicates)

    # 5. rank + select according to the user's mode and threshold.
    phase("rank", f"ranking clips (mode: {mode}, minimum score {min_score:.0f})")
    selection_pool = kept if mode != "max" else kept + duplicates
    selected = select_by_mode(selection_pool, min_score=min_score, mode=mode, max_clips=max_clips)
    # Final safety net: two clips may never sit on the same moment, even in "max"
    # mode where near-duplicates are allowed back into the pool.
    selected, overlap_dropped = enforce_non_overlap(selected)
    report["dropped_overlap"] = len(overlap_dropped)
    # How many otherwise-usable moments were skipped purely because of the score
    # threshold - the UI shows this so a low clip count never looks like a bug.
    report["below_threshold"] = sum(1 for candidate in selection_pool if candidate.score < min_score)
    for candidate in selected:
        candidate.status = "selected"

    selected.sort(key=lambda candidate: -candidate.score)
    report.update(
        {
            "usable": len(usable),
            "valid": len(valid),
            "kept": len(kept),
            "selected": len(selected),
            "best_score": round(selected[0].score, 1) if selected else 0.0,
            "min_score": min_score,
            "mode": mode,
        }
    )
    log.info(
        "finalize: %d in -> %d selected (%d duplicates, %d rejected)",
        report["input"],
        report["selected"],
        report["duplicates"],
        report["rejected_context"] + report["rejected_other"],
    )
    return {"kept": kept, "duplicates": duplicates, "selected": selected, "report": report}


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def clip_summary_line(candidate: Candidate) -> str:
    return (
        f"[{format_timestamp(candidate.start)}] {candidate.score:.0f}/100 "
        f"{category_label(candidate.category)} · {candidate.title}"
    )


def ranking_report(candidates: Sequence[Candidate], limit: int = 12) -> list[dict[str, Any]]:
    return [
        {
            "rank": index + 1,
            "title": candidate.title,
            "score": round(candidate.score, 1),
            "category": candidate.category,
            "start": round(candidate.start, 2),
            "duration": round(candidate.duration, 1),
            "why": candidate.why[:4],
        }
        for index, candidate in enumerate(sorted(candidates, key=lambda item: -item.score)[:limit])
    ]


__all__ = [
    "BoundaryResult",
    "clip_summary_line",
    "containment",
    "deduplicate",
    "finalize_candidates",
    "format_timestamp",
    "optimize_bounds",
    "ranking_report",
    "rescore",
    "temporal_iou",
    "validate_context",
]
