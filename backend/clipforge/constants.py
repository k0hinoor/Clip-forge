"""Immutable analysis vocabulary: pipeline stages, categories, caption presets.

Keeping these in one module means the UI, the pipeline and the docs can never
drift apart, and it makes the progress bar a pure function of the stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# --------------------------------------------------------------------------- #
# Analysis stages (order = execution order, weight = share of the progress bar)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Stage:
    key: str
    label: str
    detail: str
    weight: float


STAGES: Final[tuple[Stage, ...]] = (
    Stage("metadata", "Video metadata", "Reading title, channel, duration and thumbnail", 0.02),
    Stage("download", "Downloading video", "Fetching the best available stream locally", 0.16),
    Stage("audio", "Extracting audio", "Normalising audio to 16 kHz mono for speech recognition", 0.05),
    Stage("language", "Detecting language", "Identifying the primary language and any code-switching", 0.03),
    Stage("transcribe", "Transcribing speech", "Running local speech-to-text with word timestamps", 0.30),
    Stage("diarize", "Detecting speakers", "Labelling who is speaking and where the speaker changes", 0.04),
    Stage("segment", "Semantic segmentation", "Splitting the transcript at topic and thought boundaries", 0.04),
    Stage("discover", "Discovering candidates", "Scanning every part of the transcript for valuable moments", 0.12),
    Stage("score", "Scoring candidates", "Rating hook, payoff, emotion, standalone quality and novelty", 0.08),
    Stage("dedupe", "Removing overlaps", "Dropping duplicate and overlapping candidates", 0.03),
    Stage("validate", "Validating context", "Rejecting moments that need the rest of the video to make sense", 0.03),
    Stage("boundaries", "Optimising boundaries", "Snapping in/out points to natural thought boundaries", 0.05),
    Stage("prepare", "Preparing clips", "Building captions, framing plans and silence trims", 0.05),
)

STAGE_INDEX: Final[dict[str, int]] = {stage.key: index for index, stage in enumerate(STAGES)}
STAGE_BY_KEY: Final[dict[str, Stage]] = {stage.key: stage for stage in STAGES}
TOTAL_WEIGHT: Final[float] = sum(stage.weight for stage in STAGES)


def progress_for(stage_key: str, fraction: float = 1.0) -> float:
    """Overall 0..1 progress for ``fraction`` of ``stage_key`` being complete."""
    fraction = max(0.0, min(1.0, fraction))
    index = STAGE_INDEX.get(stage_key)
    if index is None:
        return 0.0
    before = sum(stage.weight for stage in STAGES[:index])
    weight = STAGES[index].weight
    return min(1.0, (before + weight * fraction) / TOTAL_WEIGHT)


def stage_label(stage_key: str) -> str:
    stage = STAGE_BY_KEY.get(stage_key)
    return stage.label if stage else stage_key.replace("_", " ").title()


def stage_public_list() -> list[dict[str, object]]:
    return [
        {"key": stage.key, "label": stage.label, "detail": stage.detail, "weight": stage.weight}
        for stage in STAGES
    ]


# --------------------------------------------------------------------------- #
# Categories
# --------------------------------------------------------------------------- #

CATEGORIES: Final[dict[str, str]] = {
    "hook": "Hook",
    "story": "Personal Story",
    "emotional": "Emotional Moment",
    "controversial": "Controversial Take",
    "surprising": "Surprising Statement",
    "funny": "Funny Moment",
    "argument": "Argument / Debate",
    "advice": "Advice",
    "lesson": "Lesson",
    "opinion": "Strong Opinion",
    "information": "Useful Information",
    "experience": "Personal Experience",
    "inspirational": "Inspirational",
    "revelation": "Shocking Revelation",
    "curiosity": "Curiosity Gap",
    "quote": "Strong Quote",
    "punchline": "Punchline",
    "conclusion": "Unexpected Conclusion",
    "story_mini": "Mini Story",
    "qa": "Question & Answer",
    "other": "Other",
}

CATEGORY_KEYS: Final[tuple[str, ...]] = tuple(CATEGORIES.keys())

CATEGORY_HINTS: Final[dict[str, tuple[str, ...]]] = {
    "story": ("story", "personal story", "experience", "experienced"),
    "emotional": ("emotional", "emotional moment", "vulnerable"),
    "funny": ("funny", "humour", "humor", "comedy"),
    "advice": ("advice", "advice", "how to"),
    "lesson": ("lesson", "lessons", "learned"),
    "argument": ("argument", "debate"),
    "controversial": ("controversial", "controversial take", "hot take"),
    "opinion": ("opinion", "strong opinion"),
    "information": ("information", "informational", "useful"),
    "inspirational": ("inspirational", "motivational", "motivation"),
    "revelation": ("revelation", "shocking", "reveal"),
    "surprising": ("surprising", "surprise", "counterintuitive"),
    "quote": ("quote", "quotable"),
    "punchline": ("punchline", "joke"),
    "curiosity": ("curiosity", "curiosity gap"),
    "qa": ("qa", "question and answer", "q&a", "interview"),
    "experience": ("experience",),
    "story_mini": ("mini story", "story"),
    "conclusion": ("conclusion", "closing"),
    "hook": ("hook",),
}


def coerce_category(raw: str) -> str:
    """Map free-form LLM/heuristic category text onto our fixed vocabulary."""
    if not raw:
        return "other"
    value = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if value in CATEGORIES:
        return value
    for key, hints in CATEGORY_HINTS.items():
        if any(hint.replace(" ", "_") in value for hint in hints):
            return key
    return "other"


def category_label(key: str) -> str:
    return CATEGORIES.get(key, key.replace("_", " ").title())


# --------------------------------------------------------------------------- #
# Caption presets
# --------------------------------------------------------------------------- #

CAPTION_PRESETS: Final[dict[str, dict[str, object]]] = {
    "minimal": {
        "label": "Minimal",
        "description": "Clean single-line subtitles, no animation, no shouting.",
        "theme": {
            "font_size": 46, "uppercase": False, "animation": "none", "max_words_per_line": 6,
            "max_lines": 1, "outline_width": 3, "shadow": 1, "background_enabled": False,
            "primary_color": "#FFFFFF", "highlight_color": "#FFFFFF",
        },
    },
    "cinematic": {
        "label": "Cinematic",
        "description": "Elegant serif captions with a soft shadow, one line at a time.",
        "theme": {
            "font": "Georgia", "font_size": 52, "uppercase": False, "animation": "pop",
            "max_words_per_line": 5, "max_lines": 2, "outline_width": 2, "shadow": 3,
            "background_enabled": False, "primary_color": "#F6F3EA", "highlight_color": "#E7C878",
            "position": "lower-middle",
        },
    },
    "bold_creator": {
        "label": "Bold Creator",
        "description": "Huge uppercase word groups with a yellow highlight - the Shorts classic.",
        "theme": {
            "font": "Inter ExtraBold", "font_size": 68, "weight": 900, "uppercase": True,
            "animation": "word_by_word", "max_words_per_line": 3, "max_lines": 2,
            "outline_width": 6, "shadow": 2, "highlight_color": "#FFD400",
            "emphasis_scale": 118.0,
        },
    },
    "karaoke": {
        "label": "Karaoke",
        "description": "Words fill in as they are spoken, timed to the word timestamps.",
        "theme": {
            "font_size": 60, "uppercase": True, "animation": "karaoke",
            "max_words_per_line": 4, "max_lines": 2, "highlight_color": "#4CE0B3",
            "outline_width": 5,
        },
    },
    "highlight": {
        "label": "Highlight",
        "description": "Sentence at a time, with the currently spoken phrase emphasised.",
        "theme": {
            "font_size": 58, "uppercase": True, "animation": "word_by_word",
            "max_words_per_line": 4, "max_lines": 2, "highlight_color": "#FF5A5F",
            "outline_width": 5,
        },
    },
    "documentary": {
        "label": "Documentary",
        "description": "Lower-third subtitle look for interview footage.",
        "theme": {
            "font": "Georgia", "font_size": 44, "uppercase": False, "animation": "none",
            "max_words_per_line": 7, "max_lines": 2, "outline_width": 2, "shadow": 1,
            "position": "bottom", "margin_v": 140, "background_enabled": True,
            "background_opacity": 55, "highlight_color": "#FFFFFF",
        },
    },
}

# --------------------------------------------------------------------------- #
# Layouts
# --------------------------------------------------------------------------- #

LAYOUTS: Final[dict[str, str]] = {
    "split": "Podcast + gameplay split screen",
    "podcast": "Podcast fills the frame",
    "broll": "Podcast + B-roll split screen",
    "gameplay": "Gameplay background with the speaker on top",
    "cinematic": "Full-frame cinematic (original framing)",
    "blur": "Blurred background with the speaker centred",
}

SPLIT_RATIOS: Final[tuple[int, ...]] = (50, 60, 65, 70)

# --------------------------------------------------------------------------- #
# Speaker / silence thresholds
# --------------------------------------------------------------------------- #

DEFAULT_MIN_SCORE: Final[float] = 70.0
DEFAULT_TARGET_SECONDS: Final[float] = 60.0
DEFAULT_MIN_SECONDS: Final[float] = 35.0
DEFAULT_MAX_SECONDS: Final[float] = 75.0

__all__ = [
    "CAPTION_PRESETS",
    "CATEGORIES",
    "CATEGORY_HINTS",
    "CATEGORY_KEYS",
    "DEFAULT_MAX_SECONDS",
    "DEFAULT_MIN_SCORE",
    "DEFAULT_MIN_SECONDS",
    "DEFAULT_TARGET_SECONDS",
    "LAYOUTS",
    "SPLIT_RATIOS",
    "STAGES",
    "STAGE_BY_KEY",
    "STAGE_INDEX",
    "TOTAL_WEIGHT",
    "category_label",
    "coerce_category",
    "progress_for",
    "stage_label",
    "stage_public_list",
]
