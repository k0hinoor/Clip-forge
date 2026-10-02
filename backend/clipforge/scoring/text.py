"""Lightweight, dependency-free text analysis used by segmentation and scoring.

Everything here is deterministic so scores are reproducible for a given
scoring version (TRD §14, §47).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

_WORD_RE = re.compile(r"[A-Za-z0-9']+")

STOPWORDS = frozenset(["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "just", "like", "really", "yeah", "okay", "ok", "um", "uh", "gonna", "wanna", "kind", "sort", "thing", "things", "get", "got", "go", "going", "know", "mean", "actually", "basically", "literally", "right", "well", "also", "one"])

FILLERS = frozenset({"um", "uh", "erm", "hmm", "like", "basically", "literally", "actually", "you know", "i mean"})
CONTINUATION_STARTERS = frozenset({"and", "but", "so", "or", "because", "which", "then", "also", "plus", "that"})

HOOK_WORDS = frozenset(["secret", "secrets", "mistake", "mistakes", "never", "always", "nobody", "everyone", "truth", "lie", "lies", "wrong", "worst", "best", "biggest", "crazy", "insane", "shocking", "surprising", "why", "how", "stop", "must", "need", "warning", "problem", "hack", "hacks", "trick", "tricks", "myth", "myths", "fail", "failed", "failure", "million", "billion", "money", "free", "fastest", "easiest", "simple", "simplest", "proven", "reason", "reasons", "rule", "rules", "important", "imagine", "listen", "look", "honestly", "seriously", "unbelievable", "incredible", "hidden", "real", "nobody's", "everybody"])

EMOTION_WORDS = frozenset(["love", "hate", "amazing", "awesome", "terrible", "horrible", "incredible", "insane", "crazy", "wow", "omg", "scared", "afraid", "fear", "angry", "mad", "furious", "happy", "sad", "cry", "cried", "crying", "laugh", "laughed", "laughing", "hilarious", "funny", "excited", "exciting", "shocked", "shocking", "surprised", "devastated", "painful", "beautiful", "brutal", "wild", "unbelievable", "obsessed", "hurt", "proud", "ashamed", "embarrassing", "nervous", "frustrated", "frustrating", "disgusting", "perfect", "worst", "best", "fantastic", "epic", "heartbreaking"])

INFO_MARKERS = (
    "how to", "the key", "step", "steps", "because", "the reason", "for example", "for instance", "means",
    "percent", "%", "research", "study", "data", "number one", "first", "second", "third", "tip", "tips",
    "lesson", "framework", "strategy", "here's", "here is", "the way", "the trick",
)
STORY_MARKERS = (
    "when i", "i was", "one day", "then", "suddenly", "turns out", "turned out", "years ago", "back then",
    "at first", "after that", "finally", "eventually", "so i", "and then", "that's when", "story",
)
CONCLUSION_MARKERS = (
    "that's why", "the lesson", "so the", "in the end", "bottom line", "that's how", "the point is",
    "which means", "so remember", "and that's", "the takeaway", "at the end of the day", "that's the",
)

_NUMBER_RE = re.compile(r"\b\d+([.,]\d+)?\b|\b(one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion)\b",
                        re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2 and not t.isdigit()]


_PUNCT_RE = re.compile(r"[^\w'%\s]")


def count_markers(text: str, markers: Iterable[str]) -> int:
    """Count whole-phrase occurrences of ``markers`` (punctuation-insensitive)."""
    low = " " + " ".join(_PUNCT_RE.sub(" ", (text or "").lower()).split()) + " "
    total = 0
    for m in markers:
        total += low.count(f" {m} ") if m[0].isalnum() else low.count(m)
    return total


def has_number(text: str) -> bool:
    return bool(_NUMBER_RE.search(text or ""))


def first_word(text: str) -> str:
    toks = tokenize(text)
    return toks[0] if toks else ""


def ends_with_terminal(text: str) -> bool:
    return bool(re.search(r"[.!?][\"')\]]*\s*$", (text or "").strip()))


class TfIdf:
    """Minimal TF-IDF over a fixed document collection (lexical embeddings)."""

    def __init__(self, documents: Sequence[str]) -> None:
        self.docs_tokens = [content_tokens(d) for d in documents]
        n = max(1, len(self.docs_tokens))
        df: Counter[str] = Counter()
        for toks in self.docs_tokens:
            df.update(set(toks))
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}

    def vector(self, text_or_tokens: str | Sequence[str]) -> dict[str, float]:
        toks = content_tokens(text_or_tokens) if isinstance(text_or_tokens, str) else list(text_or_tokens)
        tf = Counter(toks)
        vec = {t: c * self.idf.get(t, 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def top_terms(self, text: str, k: int = 5) -> list[str]:
        vec = self.vector(text)
        return [t for t, _ in sorted(vec.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())
