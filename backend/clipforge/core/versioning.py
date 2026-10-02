"""Pipeline/model/config version recording (TRD §47)."""

from __future__ import annotations

PIPELINE_VERSION = "1.0.0"
MANIFEST_VERSION = 1
# Bump when a stage's artifact format changes so stale artifacts are not reused.
STAGE_VERSIONS: dict[str, str] = {
    "validate": "1",
    "ingest": "1",
    "audio_extract": "1",
    "transcribe": "1",
    "segment": "1",
    "candidate_generation": "1",
    "candidate_scoring": "1",
    "boundary_optimization": "1",
    "visual_analysis": "1",
    "caption_layout": "1",
    "render": "1",
    "qa": "1",
    "finalize": "1",
}
