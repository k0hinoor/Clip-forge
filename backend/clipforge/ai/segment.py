"""Semantic segmentation.

Turns raw word-level ASR output into two layers the discovery engine can reason
about:

* **sentences** - complete thoughts with exact start/end times, speaker labels
  and confidence. Built from the real word timings (punctuation is not trusted
  blindly: an unpunctuated ASR run is split on pause structure as well).
* **topic blocks** - groups of consecutive sentences that belong together,
  found with a TextTiling-style lexical-cohesion walk (TF-IDF cosine between
  adjacent sentence windows, boundary where cohesion collapses), plus silence
  gaps, speaker changes and hard duration caps.

No language assumption is made anywhere: sentence splitting understands
Devanagari danda (।) as well as Latin punctuation, and mixed-script text is left
exactly as transcribed.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..logging_setup import get_logger
from .textutil import (
    classify_tokens,
    cosine,
    keywords,
    normalize_word,
    sentence_case,
    split_sentences,
    tfidf_matrix,
    words_only,
)
from .transcribe import Utterance, Word

log = get_logger("clipforge.ai")

MAX_SENTENCE_SECONDS = 22.0
MIN_SENTENCE_WORDS = 3
PAUSE_BOUNDARY_SECONDS = 0.9
MAX_BLOCK_SECONDS = 240.0
MIN_BLOCK_SECONDS = 12.0


@dataclass
class Sentence:
    index: int
    start: float
    end: float
    text: str
    speaker: str
    words: list[Word] = field(default_factory=list)
    confidence: float = 0.0
    features: dict[str, float] = field(default_factory=dict)
    block: int = -1

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def wpm(self) -> float:
        minutes = self.duration / 60.0
        return (self.word_count / minutes) if minutes > 0 else 0.0

    def to_dict(self, *, with_words: bool = False) -> dict[str, Any]:
        payload = {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "speaker": self.speaker,
            "confidence": round(self.confidence, 3),
            "word_count": self.word_count,
            "duration": round(self.duration, 2),
            "block": self.block,
            "features": self.features,
        }
        if with_words:
            payload["words"] = [word.to_dict() for word in self.words]
        return payload


@dataclass
class TopicBlock:
    index: int
    start: float
    end: float
    sentence_indexes: list[int] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    title: str = ""
    cohesion: float = 0.0
    average_energy: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 2),
            "sentences": len(self.sentence_indexes),
            "speakers": self.speakers,
            "keywords": self.keywords,
            "title": self.title,
            "cohesion": round(self.cohesion, 3),
            "energy": round(self.average_energy, 3),
        }


@dataclass
class SegmentationResult:
    sentences: list[Sentence]
    blocks: list[TopicBlock]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence_count": len(self.sentences),
            "block_count": len(self.blocks),
            "blocks": [block.to_dict() for block in self.blocks],
        }


# --------------------------------------------------------------------------- #
# Sentence construction
# --------------------------------------------------------------------------- #


def build_sentences(utterances: Sequence[Utterance]) -> list[Sentence]:
    sentences: list[Sentence] = []
    for utterance in utterances:
        sentences.extend(_sentences_from_utterance(utterance))
    sentences = _merge_fragments(sentences)
    for index, sentence in enumerate(sentences):
        sentence.index = index
        sentence.features = _sentence_features(sentence)
    return sentences


def _sentences_from_utterance(utterance: Utterance) -> list[Sentence]:
    words = list(utterance.words)
    text = (utterance.text or "").strip()

    if not words:
        if not text:
            return []
        return [
            Sentence(
                index=0,
                start=utterance.start,
                end=utterance.end,
                text=sentence_case(text),
                speaker=utterance.speaker,
                confidence=utterance.confidence,
            )
        ]

    parts = split_sentences(text) if text else []
    if len(parts) <= 1:
        return _split_long_run(words, text or " ".join(word.word for word in words), utterance)

    # Distribute the utterance's words across the punctuation-derived sentences.
    budgets = [len(words_only(part)) for part in parts]
    total_budget = sum(budgets) or len(words)
    scale = len(words) / total_budget
    out: list[Sentence] = []
    cursor = 0
    for part_index, part in enumerate(parts):
        take = max(1, int(round(budgets[part_index] * scale)))
        chunk = words[cursor: cursor + take]
        if not chunk and out:
            # Absorb orphan punctuation-only fragments into the previous sentence.
            out[-1].text = f"{out[-1].text} {part}".strip()
            continue
        cursor += take
        if not chunk:
            continue
        out.append(
            Sentence(
                index=0,
                start=chunk[0].start,
                end=chunk[-1].end,
                text=sentence_case(part),
                speaker=utterance.speaker,
                words=chunk,
                confidence=_mean_confidence(chunk),
            )
        )
    if cursor < len(words) and out:
        out[-1].words.extend(words[cursor:])
        out[-1].end = max(out[-1].end, words[-1].end)
        out[-1].text = f"{out[-1].text} {' '.join(word.word for word in words[cursor:])}".strip()
    return out


def _split_long_run(words: list[Word], text: str, utterance: Utterance) -> list[Sentence]:
    """No punctuation (or one run): split on long pauses and hard duration caps."""
    chunks: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        if current:
            gap = word.start - current[-1].end
            duration = word.end - current[0].start
            if gap >= PAUSE_BOUNDARY_SECONDS or duration >= MAX_SENTENCE_SECONDS:
                chunks.append(current)
                current = []
        current.append(word)
    if current:
        chunks.append(current)

    if len(chunks) == 1 and len(words) > 26:
        # No pauses either - fall back to even splits so blocks still have structure.
        target = max(8, len(words) // max(1, round(len(words) / 18)))
        chunks = [words[i: i + target] for i in range(0, len(words), target)]

    out: list[Sentence] = []
    for chunk in chunks:
        chunk_text = text if len(chunks) == 1 else " ".join(word.word for word in chunk)
        out.append(
            Sentence(
                index=0,
                start=chunk[0].start,
                end=chunk[-1].end,
                text=sentence_case(chunk_text.strip()),
                speaker=utterance.speaker,
                words=chunk,
                confidence=_mean_confidence(chunk),
            )
        )
    return out


def _merge_fragments(sentences: list[Sentence]) -> list[Sentence]:
    """Glue leftovers such as "Yeah." or dangling words onto a neighbour."""
    out: list[Sentence] = []
    for sentence in sentences:
        if not sentence.text.strip():
            continue
        too_short = sentence.word_count < MIN_SENTENCE_WORDS and not sentence.text.strip().endswith((".", "!", "?", "।"))
        if out and too_short and sentence.start - out[-1].end < 1.5:
            out[-1].text = f"{out[-1].text} {sentence.text}".strip()
            out[-1].words.extend(sentence.words)
            out[-1].end = max(out[-1].end, sentence.end)
            out[-1].confidence = _mean_confidence(out[-1].words)
            continue
        out.append(sentence)
    return out


def _mean_confidence(words: Sequence[Word]) -> float:
    values = [word.confidence for word in words if word.confidence]
    return float(sum(values) / len(values)) if values else 0.0


def _sentence_features(sentence: Sentence) -> dict[str, float]:
    lexical = classify_tokens(sentence.text)
    return {
        **lexical,
        "duration": round(sentence.duration, 3),
        "word_count": float(sentence.word_count),
        "wpm": round(sentence.wpm, 1),
        "ends_sentence": 1.0 if sentence.text.strip().endswith((".", "!", "?", "…", "।")) else 0.0,
        "question": 1.0 if sentence.text.strip().endswith("?") else lexical["question"],
    }


# --------------------------------------------------------------------------- #
# Topic blocks (TextTiling style)
# --------------------------------------------------------------------------- #


def build_blocks(sentences: Sequence[Sentence], *, window: int = 2) -> list[TopicBlock]:
    if not sentences:
        return []

    documents = [sentence.text for sentence in sentences]
    vectors, _ = tfidf_matrix(documents)

    cohesion: list[float] = []
    for index in range(len(sentences) - 1):
        left = _window_vector(vectors, index - window + 1, index + 1)
        right = _window_vector(vectors, index + 1, index + 1 + window)
        cohesion.append(cosine(left, right))

    similarity = np.array(cohesion) if cohesion else np.zeros(0)
    if similarity.size >= 4:
        mean = float(similarity.mean())
        std = float(similarity.std())
        threshold = max(0.02, mean - 0.55 * std)
    else:
        threshold = 0.08

    boundaries: list[int] = [0]
    for index in range(len(sentences) - 1):
        gap = sentences[index + 1].start - sentences[index].end
        speaker_change = sentences[index + 1].speaker != sentences[index].speaker
        block_span = sentences[index].end - sentences[boundaries[-1]].start
        dropped = similarity.size > index and similarity[index] < threshold
        long_pause_break = gap >= 1.6
        if dropped or long_pause_break or (speaker_change and gap > 0.7) or block_span >= MAX_BLOCK_SECONDS:
            if sentences[index].end - sentences[boundaries[-1]].start >= MIN_BLOCK_SECONDS:
                boundaries.append(index + 1)
    boundaries.append(len(sentences))

    blocks: list[TopicBlock] = []
    for index, (start_index, end_index) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        group = list(sentences[start_index:end_index])
        if not group:
            continue
        text = " ".join(sentence.text for sentence in group)
        block = TopicBlock(
            index=index,
            start=group[0].start,
            end=group[-1].end,
            sentence_indexes=[sentence.index for sentence in group],
            speakers=sorted({sentence.speaker for sentence in group}),
            keywords=keywords(text, limit=6),
            title=_block_title(group),
            cohesion=round(float(np.mean(similarity[start_index:end_index - 1])) if end_index - start_index > 1 and similarity.size else 0.0, 4),
            average_energy=round(float(np.mean([sentence.features.get("emotion", 0.0) for sentence in group])), 4),
        )
        blocks.append(block)
        for sentence in group:
            sentence.block = index

    log.info("segmentation: %d sentences -> %d topic blocks", len(sentences), len(blocks))
    return blocks


def _window_vector(vectors: list[dict[str, float]], start: int, end: int) -> dict[str, float]:
    combined: dict[str, float] = {}
    for vector in vectors[max(start, 0): max(end, 0)]:
        for term, value in vector.items():
            combined[term] = combined.get(term, 0.0) + value
    for term in combined:
        combined[term] /= max(end - start, 1)
    return combined


def _block_title(group: Sequence[Sentence]) -> str:
    """Short descriptive label from the block's own words (no LLM needed)."""
    first = group[0].text.strip()
    if len(first) <= 78:
        return first
    return first[:75].rsplit(" ", 1)[0] + "…"


# --------------------------------------------------------------------------- #
# Convenience analytics used by discovery + UI
# --------------------------------------------------------------------------- #


def sentence_map(sentences: Sequence[Sentence]) -> dict[str, Any]:
    """Summary statistics for the analysis screen."""
    if not sentences:
        return {"count": 0}
    durations = [sentence.duration for sentence in sentences]
    speakers = sorted({sentence.speaker for sentence in sentences})
    wpm_values = [sentence.wpm for sentence in sentences if sentence.wpm]
    return {
        "count": len(sentences),
        "duration_total": round(sum(durations), 2),
        "duration_mean": round(statistics.mean(durations), 2),
        "duration_median": round(statistics.median(durations), 2),
        "speakers": speakers,
        "mean_wpm": round(statistics.mean(wpm_values), 1) if wpm_values else 0.0,
    }


def words_in_range(sentences: Sequence[Sentence], start: float, end: float) -> list[Word]:
    out: list[Word] = []
    for sentence in sentences:
        if sentence.end < start - 0.05 or sentence.start > end + 0.05:
            continue
        for word in sentence.words:
            if word.end >= start - 0.05 and word.start <= end + 0.05:
                out.append(word)
    return out


def text_in_range(sentences: Sequence[Sentence], start: float, end: float) -> str:
    parts = [
        sentence.text.strip()
        for sentence in sentences
        if sentence.end > start and sentence.start < end
    ]
    return " ".join(part for part in parts if part).strip()


def sentence_index_at(sentences: Sequence[Sentence], timestamp: float) -> int:
    for index, sentence in enumerate(sentences):
        if sentence.start <= timestamp <= sentence.end:
            return index
    return -1


def normalize_for_dedupe(text: str, limit: int = 220) -> str:
    return " ".join(normalize_word(token) for token in words_only(text))[:limit]


__all__ = [
    "Sentence",
    "SegmentationResult",
    "TopicBlock",
    "build_blocks",
    "build_sentences",
    "normalize_for_dedupe",
    "sentence_index_at",
    "sentence_map",
    "text_in_range",
    "words_in_range",
]
