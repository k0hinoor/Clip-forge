"""Semantic segmentation (TRD §12).

1. Build *sentence units* from word timestamps (punctuation + pauses).
2. Group sentences into topics using lexical-cohesion dips (TextTiling-style
   over TF-IDF vectors, or optional embeddings), long pauses and speaker
   changes.

Embeddings are optional and never a mandatory cloud dependency.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from clipforge.scoring.text import TfIdf, cosine, ends_with_terminal

SENTENCE_PAUSE_SECONDS = 0.8
TOPIC_PAUSE_SECONDS = 2.5
MAX_SENTENCE_SECONDS = 15.0
HARD_MAX_SENTENCE_SECONDS = 25.0


@dataclass
class Sentence:
    idx: int
    start: float
    end: float
    text: str
    word_start: int  # index into the flat word list
    word_end: int  # inclusive
    pause_before: float
    pause_after: float
    speaker_id: str | None = None
    topic_id: int = 0
    terminal: bool = True
    avg_logprob: float | None = None
    no_speech_prob: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["segment_id"] = self.idx
        return d


@dataclass
class FlatWord:
    start: float
    end: float
    word: str
    segment_idx: int
    speaker_id: str | None = None
    probability: float | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def flatten_words(transcript: dict[str, Any]) -> list[FlatWord]:
    """Flatten transcript segments into a word list, synthesising word timings when absent."""
    words: list[FlatWord] = []
    for seg in transcript.get("segments", []):
        seg_words = seg.get("words") or []
        if not seg_words and seg.get("text", "").strip():
            # No word timestamps: spread words evenly across the segment.
            toks = seg["text"].split()
            span = max(seg["end"] - seg["start"], 0.01)
            step = span / max(len(toks), 1)
            seg_words = [{"start": seg["start"] + i * step, "end": seg["start"] + (i + 1) * step, "word": t}
                         for i, t in enumerate(toks)]
        for w in seg_words:
            text = str(w.get("word", "")).strip()
            if not text:
                continue
            words.append(FlatWord(
                start=float(w["start"]), end=float(max(w["end"], w["start"])), word=text,
                segment_idx=int(seg.get("id", 0)), speaker_id=seg.get("speaker_id"),
                probability=w.get("probability"), avg_logprob=seg.get("avg_logprob"),
                no_speech_prob=seg.get("no_speech_prob"),
            ))
    words.sort(key=lambda w: (w.start, w.end))
    return words


def build_sentences(words: Sequence[FlatWord]) -> list[Sentence]:
    sentences: list[Sentence] = []
    if not words:
        return sentences
    begin = 0
    for i, w in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt.start - w.end) if nxt else 999.0
        speaker_change = nxt is not None and nxt.speaker_id != w.speaker_id
        terminal = ends_with_terminal(w.word)
        span = w.end - words[begin].start
        too_long = (span >= MAX_SENTENCE_SECONDS and (gap > 0.25 or w.word.endswith(","))) or span >= HARD_MAX_SENTENCE_SECONDS
        if nxt is None or terminal or gap >= SENTENCE_PAUSE_SECONDS or speaker_change or too_long:
            chunk = words[begin:i + 1]
            prev_end = words[begin - 1].end if begin > 0 else 0.0
            logprobs = [c.avg_logprob for c in chunk if c.avg_logprob is not None]
            nsp = [c.no_speech_prob for c in chunk if c.no_speech_prob is not None]
            sentences.append(Sentence(
                idx=len(sentences), start=chunk[0].start, end=chunk[-1].end,
                text=" ".join(c.word for c in chunk).replace(" ,", ",").strip(),
                word_start=begin, word_end=i,
                pause_before=max(0.0, chunk[0].start - prev_end) if begin > 0 else chunk[0].start,
                pause_after=gap if nxt else 999.0,
                speaker_id=chunk[0].speaker_id, terminal=terminal,
                avg_logprob=sum(logprobs) / len(logprobs) if logprobs else None,
                no_speech_prob=sum(nsp) / len(nsp) if nsp else None,
            ))
            begin = i + 1
    return sentences


def assign_topics(
    sentences: list[Sentence],
    *,
    embed: Callable[[list[str]], list[dict[str, float] | list[float]]] | None = None,
    window: int = 3,
    depth_threshold: float = 0.12,
) -> list[Sentence]:
    """Assign ``topic_id`` to each sentence using cohesion dips + pauses + speakers."""
    n = len(sentences)
    if n == 0:
        return sentences
    texts = [s.text for s in sentences]
    if embed is not None:
        vectors = embed(texts)
        sim_fn = _dense_cosine if vectors and isinstance(vectors[0], list) else cosine
    else:
        tfidf = TfIdf(texts)
        vectors = [tfidf.vector(t) for t in texts]
        sim_fn = cosine

    def block(lo: int, hi: int):
        lo, hi = max(lo, 0), min(hi, n)
        if isinstance(vectors[0], dict):
            acc: dict[str, float] = {}
            for v in vectors[lo:hi]:
                for k, x in v.items():  # type: ignore[union-attr]
                    acc[k] = acc.get(k, 0.0) + x
            norm = sum(x * x for x in acc.values()) ** 0.5 or 1.0
            return {k: x / norm for k, x in acc.items()}
        dims = len(vectors[0])  # type: ignore[arg-type]
        acc_l = [sum(v[d] for v in vectors[lo:hi]) for d in range(dims)]  # type: ignore[index]
        norm = sum(x * x for x in acc_l) ** 0.5 or 1.0
        return [x / norm for x in acc_l]

    # Similarity across each gap between sentence i-1 and i.
    gaps = [1.0] + [sim_fn(block(i - window, i), block(i, i + window)) for i in range(1, n)]
    boundaries = set()
    for i in range(1, n):
        left = max(gaps[max(1, i - window):i + 1])
        right = max(gaps[i:min(n, i + window + 1)])
        depth = (left - gaps[i]) + (right - gaps[i])
        if depth >= depth_threshold and gaps[i] < 0.35:
            boundaries.add(i)
        if sentences[i].pause_before >= TOPIC_PAUSE_SECONDS:
            boundaries.add(i)
        if sentences[i].speaker_id != sentences[i - 1].speaker_id and sentences[i].pause_before >= 1.0:
            boundaries.add(i)
    topic = 0
    topic_start = sentences[0].start
    for i, s in enumerate(sentences):
        # Avoid micro-topics shorter than ~8 seconds.
        if i in boundaries and s.start - topic_start >= 8.0:
            topic += 1
            topic_start = s.start
        s.topic_id = topic
    return sentences


def _dense_cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))
