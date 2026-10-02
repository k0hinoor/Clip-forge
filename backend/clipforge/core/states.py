"""Job state machine (PRD §13) and pipeline stage catalogue (TRD §8).

The backend validates every transition; illegal transitions raise
``INVALID_STATE_TRANSITION``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from clipforge.core.errors import AppError, ErrorCode


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    VALIDATING = "VALIDATING"
    INGESTING = "INGESTING"
    EXTRACTING_AUDIO = "EXTRACTING_AUDIO"
    TRANSCRIBING = "TRANSCRIBING"
    SEGMENTING = "SEGMENTING"
    CANDIDATE_SCORING = "CANDIDATE_SCORING"
    BOUNDARY_OPTIMIZATION = "BOUNDARY_OPTIMIZATION"
    FRAME_ANALYSIS = "FRAME_ANALYSIS"
    CAPTION_GENERATION = "CAPTION_GENERATION"
    RENDERING = "RENDERING"
    FINALIZING = "FINALIZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


# Ordered processing states (excluding QUEUED and terminal states).
ACTIVE_STATES: tuple[JobStatus, ...] = (
    JobStatus.VALIDATING,
    JobStatus.INGESTING,
    JobStatus.EXTRACTING_AUDIO,
    JobStatus.TRANSCRIBING,
    JobStatus.SEGMENTING,
    JobStatus.CANDIDATE_SCORING,
    JobStatus.BOUNDARY_OPTIMIZATION,
    JobStatus.FRAME_ANALYSIS,
    JobStatus.CAPTION_GENERATION,
    JobStatus.RENDERING,
    JobStatus.FINALIZING,
)
TERMINAL_STATES = frozenset({JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.EXPIRED})
# States that count against concurrency limits / are "in flight".
IN_FLIGHT_STATES = frozenset({JobStatus.QUEUED, *ACTIVE_STATES})

_ORDER = {s: i for i, s in enumerate(ACTIVE_STATES)}


def is_active(status: str) -> bool:
    return JobStatus(status) in _ORDER


def can_transition(current: str, new: str) -> bool:
    cur, nxt = JobStatus(current), JobStatus(new)
    if cur == JobStatus.EXPIRED:
        return False
    if cur == JobStatus.QUEUED:
        return nxt in (JobStatus.VALIDATING, JobStatus.CANCELLED, JobStatus.FAILED,
                       JobStatus.EXPIRED) or nxt in _ORDER  # resume after recovery
    if cur in _ORDER:
        if nxt in _ORDER:
            return _ORDER[nxt] >= _ORDER[cur]  # forward (stages may be skipped)
        if nxt == JobStatus.COMPLETED:
            return cur == JobStatus.FINALIZING
        return nxt in (JobStatus.QUEUED, JobStatus.FAILED, JobStatus.CANCELLED)
    if cur == JobStatus.COMPLETED:
        return nxt == JobStatus.EXPIRED
    if cur in (JobStatus.FAILED, JobStatus.CANCELLED):
        return nxt in (JobStatus.QUEUED, JobStatus.EXPIRED)
    return False


def assert_transition(current: str, new: str) -> None:
    if current == new and JobStatus(current) in _ORDER:
        return
    if not can_transition(current, new):
        raise AppError(
            ErrorCode.INVALID_STATE_TRANSITION,
            f"Cannot move job from {current} to {new}.",
        )


@dataclass(frozen=True)
class StageInfo:
    name: str  # TRD pipeline stage
    status: JobStatus  # PRD job status displayed to the user
    weight: float  # relative share of overall progress


# TRD §8 pipeline. INGEST+MEDIA_PROBE are split into validate/ingest; QA runs
# per-clip after rendering but is also tracked as its own stage.
PIPELINE_STAGES: tuple[StageInfo, ...] = (
    StageInfo("validate", JobStatus.VALIDATING, 2),
    StageInfo("ingest", JobStatus.INGESTING, 2),
    StageInfo("audio_extract", JobStatus.EXTRACTING_AUDIO, 5),
    StageInfo("transcribe", JobStatus.TRANSCRIBING, 35),
    StageInfo("segment", JobStatus.SEGMENTING, 3),
    StageInfo("candidate_generation", JobStatus.CANDIDATE_SCORING, 2),
    StageInfo("candidate_scoring", JobStatus.CANDIDATE_SCORING, 6),
    StageInfo("boundary_optimization", JobStatus.BOUNDARY_OPTIMIZATION, 2),
    StageInfo("visual_analysis", JobStatus.FRAME_ANALYSIS, 8),
    StageInfo("caption_layout", JobStatus.CAPTION_GENERATION, 2),
    StageInfo("render", JobStatus.RENDERING, 28),
    StageInfo("qa", JobStatus.RENDERING, 3),
    StageInfo("finalize", JobStatus.FINALIZING, 2),
)
STAGE_BY_NAME = {s.name: s for s in PIPELINE_STAGES}
_TOTAL_WEIGHT = sum(s.weight for s in PIPELINE_STAGES)


def overall_progress(stage_name: str, stage_fraction: float) -> int:
    """Map (stage, fraction-of-stage) to 0..100 overall job progress."""
    done = 0.0
    for stage in PIPELINE_STAGES:
        if stage.name == stage_name:
            frac = min(max(stage_fraction, 0.0), 1.0)
            return min(99, int(round((done + stage.weight * frac) / _TOTAL_WEIGHT * 100)))
        done += stage.weight
    return 0


class RenderStatus(StrEnum):
    QUEUED = "QUEUED"
    RENDERING = "RENDERING"
    QA = "QA"
    READY = "READY"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class ClipStatus(StrEnum):
    PENDING = "PENDING"
    RENDERING = "RENDERING"
    READY = "READY"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"
    DELETED = "DELETED"


class AssetStatus(StrEnum):
    UPLOADING = "UPLOADING"
    READY = "READY"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    DELETED = "DELETED"
