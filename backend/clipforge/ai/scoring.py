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
        rejection_code = ""
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
                    rejection_code = "overlap_duplicate"
                    similarity_note = f"{int(iou * 100)}% temporal overlap, {int(similarity * 100)}% semantic similarity"
                    break
                hook_gap = abs(candidate.start_index - existing.start_index)
                if hook_gap <= 6:
                    duplicate_of = existing
                    rejection_code = "overlap_duplicate"
                    similarity_note = f"same moment ({int(iou * 100)}% overlap, hook {hook_gap} sentences apart)"
                    break
            elif similarity >= semantic_threshold and abs(candidate.start - existing.start) < 90:
                duplicate_of = existing
                rejection_code = "semantic_duplicate"
                similarity_note = f"{int(similarity * 100)}% semantic similarity within 90 seconds"
                break

        if duplicate_of is None:
            candidate.status = "kept"
            candidate.rejection_code = ""
            candidate.duplicate_of = ""
            kept.append(candidate)
        else:
            candidate.status = "duplicate"
            candidate.rejection_code = rejection_code
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
    debug_mode: bool = False,
    on_phase: Any = None,
) -> dict[str, Any]:
    """Validate, re-score, deduplicate and select candidate clips.

    Scores are always 0-100. Debug mode removes score/context/semantic gates,
    but keeps timestamp validity, boundary safety and non-overlap checks.
    """

    def phase(key: str, message: str) -> None:
        if on_phase is not None:
            try:
                on_phase(key, message)
            except Exception:  # noqa: BLE001 - reporting must never break analysis
                pass

    candidate_list = list(candidates)
    effective_threshold = 0.0 if debug_mode else float(min_score)
    phase("score", f"scoring {len(candidate_list)} candidates against 14 quality signals")
    report: dict[str, Any] = {
        "input": len(candidate_list),
        "discovered": len(candidate_list),
        "scored": 0,
        "scoring_errors": 0,
        "rejected_timestamp": 0,
        "rejected_context": 0,
        "rejected_other": 0,
        "boundaries_changed": 0,
        "duplicates": 0,
        "overlap_rejections": 0,
        "threshold_pass": 0,
        "debug_mode": debug_mode,
        "configured_min_score": float(min_score),
        "min_score": effective_threshold,
    }

    def basic_invalid(candidate: Candidate) -> str:
        if candidate.start_index < 0 or candidate.end_index <= candidate.start_index or candidate.end_index > len(index.sentences):
            return "Candidate sentence range is invalid"
        if candidate.start < 0 or candidate.end <= candidate.start or candidate.end > index.duration + 1.0:
            return "Candidate timestamps are invalid or outside the transcript"
        return ""

    # 1. Basic timestamp/range validity is never disabled, even for diagnostics.
    usable: list[Candidate] = []
    for candidate in candidate_list:
        invalid = basic_invalid(candidate)
        if invalid:
            candidate.status = "rejected"
            candidate.rejection_code = "invalid_timestamp"
            candidate.reason = (candidate.reason + f" · rejected: {invalid}").strip(" ·")
            report["rejected_timestamp"] += 1
            continue
        if not debug_mode and candidate.features.penalties.get("context_dependency", 0.0) >= 0.85:
            candidate.status = "rejected"
            candidate.rejection_code = "context_dependency"
            candidate.reason = (candidate.reason + " · rejected: cannot stand alone").strip(" ·")
            report["rejected_context"] += 1
            continue
        usable.append(candidate)

    # 2. Boundary optimisation + re-score. Failures stay explicit; they are not
    #    converted to a synthetic score of zero.
    phase("boundaries", f"optimising clip boundaries for {len(usable)} candidates")
    successfully_scored: list[Candidate] = []
    for candidate in usable:
        try:
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
                    candidate.start_index, candidate.end_index = bounds.start_index, bounds.end_index
                    report["boundaries_changed"] += 1
            invalid = basic_invalid(candidate)
            if invalid:
                raise ValueError(invalid)
            rescore(index, candidate, min_seconds=min_seconds, max_seconds=max_seconds, target_seconds=target_seconds)
            if not (0.0 <= float(candidate.score) <= 100.0):
                raise ValueError(f"score outside 0-100 range: {candidate.score!r}")
        except Exception as exc:  # noqa: BLE001 - preserve and report individual scoring errors
            candidate.status = "scoring_error"
            candidate.rejection_code = "scoring_error"
            candidate.reason = (candidate.reason + f" · scoring error: {type(exc).__name__}: {exc}").strip(" ·")
            report["scoring_errors"] += 1
            log.exception("could not score candidate %.3f-%.3f", candidate.start, candidate.end)
            continue
        report["scored"] += 1
        successfully_scored.append(candidate)

    # 3. Standalone/context validation is optional only in debug mode.
    phase("validate", "checking that every clip stands on its own" if not debug_mode else "debug mode: context rejection disabled")
    valid: list[Candidate] = []
    for candidate in successfully_scored:
        if not debug_mode:
            ok, why = validate_context(candidate)
            if not ok:
                if candidate.features.penalties.get("context_dependency", 0.0) >= MAX_CONTEXT_DEPENDENCY:
                    code = "context_dependency"
                    report["rejected_context"] += 1
                elif candidate.features.factors.get("standalone", 0.0) < MIN_STANDALONE:
                    code = "not_standalone"
                    report["rejected_other"] += 1
                else:
                    code = "low_confidence"
                    report["rejected_other"] += 1
                candidate.status = "rejected"
                candidate.rejection_code = code
                candidate.reason = (candidate.reason + f" · rejected: {why}").strip(" ·")
                continue
        valid.append(candidate)

    # 4. Remove same-moment overlaps. Debug mode disables only semantic
    #    similarity rejection; temporal overlap protection remains mandatory.
    phase("dedupe", f"removing overlaps and near-duplicates from {len(valid)} candidates")
    kept, duplicates = deduplicate(valid, semantic_threshold=2.0 if debug_mode else SEMANTIC_DUPLICATE_THRESHOLD)
    if duplicates and not debug_mode:
        kept, duplicates_again = deduplicate(kept)
        duplicates.extend(duplicates_again)
    report["duplicates"] = len(duplicates)
    report["overlap_rejections"] = sum(1 for candidate in duplicates if candidate.rejection_code == "overlap_duplicate")
    report["semantic_rejections"] = sum(1 for candidate in duplicates if candidate.rejection_code == "semantic_duplicate")

    # 5. Rank/select. Debug mode is effectively best mode at score 0 and does
    #    not cap the result count; only non-overlap is retained as a hard rule.
    phase("rank", f"ranking clips (mode: {'debug' if debug_mode else mode}, minimum score {effective_threshold:.0f})")
    selection_pool = kept if debug_mode or mode != "max" else kept + duplicates
    report["threshold_pass"] = sum(1 for candidate in successfully_scored if candidate.score >= effective_threshold)
    selected = select_by_mode(
        selection_pool,
        min_score=effective_threshold,
        mode="best" if debug_mode else mode,
        max_clips=0 if debug_mode else max_clips,
    )
    selected, overlap_dropped = enforce_non_overlap(selected)
    for candidate in overlap_dropped:
        candidate.rejection_code = "overlap_conflict"
        candidate.status = "rejected"
    report["dropped_overlap"] = len(overlap_dropped)
    report["overlap_rejections"] += len(overlap_dropped)

    selected_keys = {candidate.key() for candidate in selected}
    selection_keys = {candidate.key() for candidate in selection_pool}
    for candidate in selection_pool:
        if candidate.key() in selected_keys:
            candidate.status = "selected"
            candidate.rejection_code = ""
            continue
        if candidate.rejection_code == "overlap_conflict":
            continue
        if candidate.score < effective_threshold:
            candidate.status = "rejected"
            candidate.rejection_code = "below_threshold"
            candidate.reason = (
                candidate.reason + f" · rejected: score {candidate.score:.1f} is below threshold {effective_threshold:.1f}"
            ).strip(" ·")
        elif not debug_mode and max_clips > 0 and len(selected) >= max_clips:
            candidate.status = "rejected"
            candidate.rejection_code = "max_clips"
            candidate.reason = (candidate.reason + f" · not selected: limit of {max_clips} clips reached").strip(" ·")

    selected.sort(key=lambda candidate: -candidate.score)
    scored_scores = [candidate.score for candidate in successfully_scored]
    reason_counts: dict[str, int] = {}
    for candidate in candidate_list:
        if candidate.rejection_code:
            reason_counts[candidate.rejection_code] = reason_counts.get(candidate.rejection_code, 0) + 1

    report.update(
        {
            "usable": len(usable),
            "valid": len(valid),
            "kept": len(kept),
            "selected": len(selected),
            "final_accepted": len(selected),
            "best_score": round(max(scored_scores), 1) if scored_scores else 0.0,
            "highest_score": round(max(scored_scores), 1) if scored_scores else 0.0,
            "average_score": round(sum(scored_scores) / len(scored_scores), 1) if scored_scores else 0.0,
            "below_threshold": sum(1 for candidate in successfully_scored if candidate.score < effective_threshold),
            "mode": "debug" if debug_mode else mode,
            "rejection_reasons": reason_counts,
        }
    )
    log.info(
        "finalize: %d input, %d scored (%d errors), %d selected, %d context rejected, %d overlap rejected",
        report["input"], report["scored"], report["scoring_errors"], report["selected"],
        report["rejected_context"], report["overlap_rejections"],
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
