"""Title / hook / description / keyword generation (PRD §11; TRD §15).

Uses the configured local LLM when available and falls back to deterministic
heuristics on any failure (graceful degradation, PRD principle 8). Output never
claims a clip is "viral".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from clipforge.core.errors import AppError
from clipforge.core.logging import get_logger
from clipforge.scoring.text import FILLERS, TfIdf, tokenize
from clipforge.worker.providers.llm import HeuristicProvider, LLMProvider

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You write metadata for short-form video clips cut from a longer video. "
    "You only see a transcript excerpt. Respond with a single JSON object with keys: "
    "title (max 70 chars, no clickbait, no emojis), hook (max 120 chars, a compelling first line), "
    "description (max 300 chars), keywords (3-6 lowercase strings), interest (integer 0-10: how "
    "self-contained and interesting the excerpt is), reasoning (max 200 chars). "
    "Never claim the clip is or will be viral."
)
_VIRAL_RE = re.compile(r"\b(go(es|ing)? viral|viral|guaranteed views?)\b", re.IGNORECASE)
MAX_EXCERPT_CHARS = 1800


@dataclass
class ClipMetadata:
    title: str
    hook: str
    description: str
    keywords: list[str] = field(default_factory=list)
    reasoning: str = ""
    interest: float | None = None  # 0..1 when an LLM rated it
    source: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _strip_fillers(text: str) -> str:
    words = text.split()
    while words and tokenize(words[0]) and tokenize(words[0])[0] in FILLERS | {"so", "and", "but", "okay", "ok"}:
        words.pop(0)
    out = " ".join(words)
    return out[:1].upper() + out[1:] if out else out


def _truncate(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0].rstrip(",;:-")
    return cut + "…"


def _sanitize(text: str, limit: int) -> str:
    text = _VIRAL_RE.sub("standout", str(text or ""))
    text = re.sub(r"[\x00-\x1f<>]", " ", text)
    return _truncate(text, limit)


def heuristic_metadata(excerpt: str, tfidf: TfIdf | None, reasons: list[str]) -> ClipMetadata:
    sents = _sentences(excerpt) or [excerpt]
    hook = _strip_fillers(sents[0]) if sents and sents[0] else "Watch this moment"
    question = next((s for s in sents[:3] if s.endswith("?")), None)
    title_src = _strip_fillers(question or sents[0] if sents else excerpt)
    title = _truncate(title_src.rstrip("."), 70) or "Highlight"
    keywords = tfidf.top_terms(excerpt, 5) if tfidf else []
    description = _truncate(" ".join(_strip_fillers(s) for s in sents[:2]), 300)
    reasoning = " · ".join(reasons) if reasons else "Selected by content signals"
    return ClipMetadata(title=_sanitize(title, 70), hook=_sanitize(hook, 120),
                        description=_sanitize(description, 300), keywords=keywords,
                        reasoning=_sanitize(reasoning, 200), source="heuristic")


class MetadataGenerator:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        self.failures = 0

    def generate(self, excerpt: str, *, duration: float, reasons: list[str], tfidf: TfIdf | None,
                 language: str | None) -> ClipMetadata:
        fallback = heuristic_metadata(excerpt, tfidf, reasons)
        if isinstance(self.provider, HeuristicProvider) or not excerpt.strip() or self.failures >= 3:
            return fallback
        prompt = (
            f"Clip duration: {duration:.0f} seconds. Language: {language or 'unknown'}.\n"
            f"Signals: {', '.join(reasons) or 'n/a'}.\n"
            f"Transcript excerpt:\n\"\"\"\n{excerpt[:MAX_EXCERPT_CHARS]}\n\"\"\""
        )
        try:
            data = self.provider.generate_json(SYSTEM_PROMPT, prompt)
        except AppError as err:
            self.failures += 1
            log.warning("LLM metadata failed; using heuristics", extra={"reason": (err.internal or "")[:200]})
            return fallback
        keywords = data.get("keywords") if isinstance(data.get("keywords"), list) else fallback.keywords
        keywords = [re.sub(r"[^\w\s-]", "", str(k)).strip().lower()[:30] for k in keywords][:6]
        try:
            interest = max(0.0, min(10.0, float(data.get("interest")))) / 10.0
        except (TypeError, ValueError):
            interest = None
        return ClipMetadata(
            title=_sanitize(data.get("title") or fallback.title, 70),
            hook=_sanitize(data.get("hook") or fallback.hook, 120),
            description=_sanitize(data.get("description") or fallback.description, 300),
            keywords=[k for k in keywords if k] or fallback.keywords,
            reasoning=_sanitize(data.get("reasoning") or fallback.reasoning, 200),
            interest=interest,
            source=f"{self.provider.name}:{self.provider.model}",
        )
