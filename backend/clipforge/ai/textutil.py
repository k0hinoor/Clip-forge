"""Lightweight text utilities used across the AI modules.

No heavy dependencies: everything is pure Python on top of the standard library
so analysis works even on a machine where NLP packages cannot be installed.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Tokenisation & sentences
# --------------------------------------------------------------------------- #

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|(?<=[।॥])\s+")
_WORD_RE = re.compile(r"[\w'’\-]+", re.UNICODE)

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
ARABIC_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f]")
CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
LATIN_RE = re.compile(r"[A-Za-z]")

FILLER_WORDS = {
    "um", "uh", "erm", "hmm", "mm", "ah", "eh", "like", "basically", "literally",
    "actually", "you know", "i mean", "sort of", "kind of", "right", "okay so",
    "matlab", "yaani", "wo", "arre", "haan", "toh",
}

ENGLISH_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "doing", "have",
    "has", "had", "having", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "of", "in", "on", "at", "to", "for", "with",
    "by", "from", "as", "about", "into", "over", "after", "so", "just", "not", "no", "yes", "what",
    "which", "who", "whom", "when", "where", "why", "how", "there", "here", "up", "down", "out",
    "will", "would", "can", "could", "should", "may", "might", "must", "get", "got", "go", "going",
    "know", "think", "really", "very", "much", "more", "most", "also", "because", "one", "two",
}

# Romanised Hindi/Hinglish markers seen in real Indian-English transcripts.
#
# Only *distinctive* tokens belong here: common English words that also happen to
# be valid Hindi romanisations ("the", "to", "ab", "log", "time", "problem",
# "life", "kam") are deliberately absent, otherwise every English transcript
# would be flagged as Hinglish.
HINGLISH_MARKERS = {
    "kya", "kyu", "kyun", "kaise", "kaisa", "kaisi", "hai", "hain", "nahi", "nahin", "haan",
    "mujhe", "mujhko", "mera", "meri", "mere", "tum", "tumhe", "tumhara", "aap", "aapko", "hum",
    "humko", "humara", "unka", "uska", "uski", "iske", "iski", "bhai", "yaar", "dost",
    "samajh", "samjha", "samajhna", "raha", "rahi", "rahe", "karna", "karta", "karti",
    "karte", "kiya", "kiye", "hoga", "hogi", "honge", "tha", "thi", "bhi", "toh",
    "phir", "fir", "abhi", "jab", "tab", "yaha", "yahan", "waha", "wahan", "bohot", "bahut",
    "zyada", "jyada", "sahi", "galat", "paisa", "paisaa", "kaam", "zindagi", "zindgi",
    "dil", "baat", "baatein", "cheez", "banda", "bande", "acha", "accha", "theek", "thik",
    "matlab", "yaani", "sachhi", "sach", "jhuth", "mazaa", "maza", "karo", "karo",
}

# Minimum number of marker hits before Hinglish is even considered, and the
# density that maps to a score of 1.0.
HINGLISH_MIN_HITS = 3
HINGLISH_DENSITY_SCALE = 2.4


def words_only(text: str) -> list[str]:
    return _WORD_RE.findall(text or "")


def normalize_word(word: str) -> str:
    """Lowercase, strip surrounding punctuation, keep internal apostrophes."""
    cleaned = word.lower().strip("\"'“”‘’()[]{}.,!?;:।॥…—–-")
    return cleaned


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").replace("\u200b", " ").strip()


def split_sentences(text: str) -> list[str]:
    """Split into sentences using punctuation for Latin and Devanagari scripts."""
    text = normalize_text(text)
    if not text:
        return []
    pieces = [piece.strip() for piece in _SENTENCE_END.split(text) if piece and piece.strip()]
    if len(pieces) <= 1 and len(text) > 220:
        # Long unpunctuated run (common in raw ASR): fall back to comma/clause splits.
        pieces = [piece.strip() for piece in re.split(r"(?<=[,;:])\s+", text) if piece.strip()]
    return pieces


def sentence_case(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    return text[0].upper() + text[1:]


def ellipsize(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return (cut or text[:limit]).rstrip(" ,;:") + "…"


# --------------------------------------------------------------------------- #
# Keywords / salience
# --------------------------------------------------------------------------- #

_EMOTION_WORDS = {
    "love", "hate", "cry", "cried", "crying", "afraid", "scared", "fear", "angry", "rage", "hurt",
    "pain", "beautiful", "amazing", "terrible", "awful", "incredible", "shocked", "shocking",
    "never", "always", "everyone", "nobody", "impossible", "regret", "proud", "shame", "hope",
    "hopeless", "break", "broke", "broken", "die", "died", "death", "life", "loved", "lonely",
    "happy", "sad", "crazy", "insane", "wow", "holy", "damn", "honestly", "truth", "failed",
    "failure", "success", "successful", "win", "won", "lost", "lose", "fight", "fought",
}

_INSIGHT_WORDS = {
    "because", "reason", "means", "actually", "realize", "realised", "learned", "learnt", "lesson",
    "mistake", "advice", "secret", "truth", "understand", "discovered", "figured", "principle",
    "rule", "framework", "step", "strategy", "method", "how", "why",
}

_QUESTION_WORDS = {
    "what", "why", "how", "when", "where", "who", "which", "does", "do", "did", "is", "are", "can",
    "could", "would", "should", "kya", "kyun", "kyu", "kaise", "kaun", "kab", "kahan",
}

_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b|\b(million|billion|thousand|crore|lakh|percent)\b", re.IGNORECASE)

_FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "our", "us", "mujhe", "mera", "meri", "main", "hum"}
_SECOND_PERSON = {"you", "your", "yours", "tum", "aap", "tumhara", "aapko"}


def classify_tokens(text: str) -> dict[str, float]:
    """Cheap lexical features used by the heuristic scorer and by LLM prompts."""
    tokens = [normalize_word(token) for token in words_only(text)]
    tokens = [token for token in tokens if token]
    total = max(len(tokens), 1)
    counts = Counter(tokens)

    emotion = sum(1 for token in tokens if token in _EMOTION_WORDS) / total
    insight = sum(1 for token in tokens if token in _INSIGHT_WORDS) / total
    question = sum(1 for token in tokens if token in _QUESTION_WORDS) / total
    first_person = sum(1 for token in tokens if token in _FIRST_PERSON) / total
    second_person = sum(1 for token in tokens if token in _SECOND_PERSON) / total
    numbers = len(_NUMBER_RE.findall(text)) / total
    filler = sum(1 for token in tokens if token in FILLER_WORDS) / total
    unique_ratio = len(counts) / total
    long_words = sum(1 for token in tokens if len(token) > 7) / total

    return {
        "emotion": round(min(emotion * 12, 1.0), 4),
        "insight": round(min(insight * 10, 1.0), 4),
        "question": round(min(question * 6, 1.0), 4),
        "first_person": round(min(first_person * 4, 1.0), 4),
        "second_person": round(min(second_person * 5, 1.0), 4),
        "numbers": round(min(numbers * 8, 1.0), 4),
        "filler_ratio": round(filler, 4),
        "unique_ratio": round(unique_ratio, 4),
        "long_word_ratio": round(long_words, 4),
        "token_count": float(len(tokens)),
    }


def keywords(text: str, limit: int = 8) -> list[str]:
    tokens = [normalize_word(token) for token in words_only(text)]
    counts = Counter(token for token in tokens if token and token not in ENGLISH_STOPWORDS and len(token) > 2)
    return [word for word, _ in counts.most_common(limit)]


def tfidf_matrix(documents: list[str]) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Return L2-normalised TF-IDF vectors plus the idf table."""
    tokenised = [
        [normalize_word(token) for token in words_only(doc)]
        for doc in documents
    ]
    tokenised = [[token for token in doc if token and token not in ENGLISH_STOPWORDS] for doc in tokenised]
    document_count = max(len(tokenised), 1)
    document_frequency: Counter[str] = Counter()
    for doc in tokenised:
        document_frequency.update(set(doc))
    idf = {term: math.log((1 + document_count) / (1 + freq)) + 1.0 for term, freq in document_frequency.items()}

    vectors: list[dict[str, float]] = []
    for doc in tokenised:
        counts = Counter(doc)
        total = max(sum(counts.values()), 1)
        vector = {
            term: (count / total) * idf.get(term, 1.0)
            for term, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        vectors.append({term: value / norm for term, value in vector.items()})
    return vectors, idf


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(term, 0.0) for term, value in a.items())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def script_counts(text: str) -> dict[str, int]:
    return {
        "devanagari": len(DEVANAGARI_RE.findall(text)),
        "latin": len(LATIN_RE.findall(text)),
        "arabic": len(ARABIC_RE.findall(text)),
        "cyrillic": len(CYRILLIC_RE.findall(text)),
        "cjk": len(CJK_RE.findall(text)),
    }


def hinglish_score(text: str) -> float:
    """How strongly a Latin-script transcript looks like romanised Hindi.

    Density based, with a floor on the absolute number of hits so a couple of
    stray words can never flip an English transcript into Hinglish mode.
    """
    tokens = [normalize_word(token) for token in words_only(text)]
    total = len(tokens)
    if total < 5:
        return 0.0
    hits = sum(1 for token in tokens if token in HINGLISH_MARKERS)
    if hits < HINGLISH_MIN_HITS:
        return 0.0
    density = hits / total
    return min(density * HINGLISH_DENSITY_SCALE, 1.0)


@dataclass
class TextStats:
    token_count: int
    word_count: int
    char_count: int
    sentence_count: int
    avg_word_length: float
    speech_rate_wpm: float


def text_stats(text: str, duration_seconds: float) -> TextStats:
    tokens = words_only(text)
    sentences = split_sentences(text)
    minutes = max(duration_seconds / 60.0, 1e-6)
    return TextStats(
        token_count=len(tokens),
        word_count=len(tokens),
        char_count=len(text),
        sentence_count=max(len(sentences), 1),
        avg_word_length=round(sum(len(t) for t in tokens) / max(len(tokens), 1), 2),
        speech_rate_wpm=round(len(tokens) / minutes, 1),
    )


__all__ = [
    "DEVANAGARI_RE",
    "ENGLISH_STOPWORDS",
    "FILLER_WORDS",
    "HINGLISH_MARKERS",
    "TextStats",
    "classify_tokens",
    "cosine",
    "ellipsize",
    "hinglish_score",
    "jaccard",
    "keywords",
    "normalize_text",
    "normalize_word",
    "script_counts",
    "sentence_case",
    "split_sentences",
    "text_stats",
    "tfidf_matrix",
    "words_only",
]
