"""Feature extraction for candidate moments.

Every number in here is computed from the transcript, the word timings and the
lexical content - there is no lookup table of "viral phrases" and no random
scoring. Each feature also produces human-readable *evidence* strings, which is
what the "WHY THIS CLIP?" panel is built from.

Feature vocabulary (product spec):

hook · emotion · curiosity · standalone · story · payoff · novelty ·
shareability · controversy · practical · humour · insight · unexpectedness ·
engagement

with penalties: context dependency, abrupt start/end, filler, ASR confidence,
repetition and weak delivery.

Performance: the whole transcript is pre-analysed once into per-sentence
statistics plus prefix vectors, so evaluating a candidate range is O(sentences
in range) with small constants - fast enough to score tens of thousands of
candidate ranges for a multi-hour video.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .segment import Sentence, TopicBlock
from .textutil import (
    ENGLISH_STOPWORDS,
    FILLER_WORDS,
    cosine,
    normalize_word,
    tfidf_matrix,
    words_only,
)

# --------------------------------------------------------------------------- #
# Marker vocabularies - cues, not clip lists
# --------------------------------------------------------------------------- #

_GROUPS: dict[str, tuple[str, ...]] = {
    "curiosity": (
        "nobody talks about", "nobody tells you", "no one talks about", "what happens", "the truth is",
        "the real reason", "here's why", "heres why", "turns out", "most people", "everyone thinks",
        "people don't realize", "people dont realise", "you won't believe", "you wont believe",
        "what if", "imagine if", "the secret", "nobody knows", "the problem is", "here's the thing",
        "the question is", "let me tell you", "the crazy part", "nobody asked",
    ),
    "contrast": (
        "but ", "however", "although", "actually", "turns out", "instead", "whereas", "yet ", "even though",
        "the thing is", "in reality", "surprisingly", "unexpectedly", "counterintuitive", "opposite",
        "except", "magar", "lekin", "phir bhi", "but the",
    ),
    "payoff": (
        "so that's why", "so thats why", "that's why", "thats why", "in the end", "eventually",
        "and that's when", "and thats when", "which means", "the lesson", "since then", "from then on",
        "long story short", "bottom line", "the whole point", "i realised", "i realized", "so i decided",
        "so finally", "the result", "that changed", "after that", "and that's how", "and thats how",
    ),
    "story": (
        "i remember", "when i was", "one day", "back then", "at the time", "suddenly", "so i ",
        "we were", "he said", "she said", "i started", "i walked", "i called", "i met", "i got",
        "there was this", "the first time", "my friend", "my mom", "my dad", "my wife", "my boss",
        "so basically", "after that day", "aur phir", "ek din",
    ),
    "advice": (
        "you should", "you need to", "you have to", "make sure", "the key is", "the point is",
        "here's how", "heres how", "one thing i learned", "the trick", "the mistake", "stop doing",
        "start doing", "focus on", "the best way", "my advice", "karna chahiye", "never do",
    ),
    "humour": (
        "[laughter]", "(laughter)", "haha", "hahaha", "lol", "lmao", "hilarious", "funny",
        "joke", "kidding", "crack up", "comedy", "seriously though", "punchline",
    ),
    "controversy": (
        "controversial", "unpopular opinion", "i disagree", "that's wrong", "thats wrong",
        "nobody wants to hear", "hate to say", "overrated", "underrated", "everyone is wrong",
        "the truth nobody", "hot take", "i don't agree", "i dont agree", "nonsense", "scam",
        "lie", "lying", "bullshit",
    ),
    "insight": (
        "the reason", "which means", "that means", "the principle", "the framework",
        "the difference between", "what i learned", "the takeaway", "in other words",
        "the mistake most", "the trick is", "rarely", "usually", "because of that",
    ),
    "intensity": (
        "never", "always", "everything", "nothing", "everyone", "nobody", "insane", "crazy",
        "huge", "massive", "incredible", "unbelievable", "terrible", "amazing", "absolutely",
        "completely", "totally", "desperately", "literally", "honestly",
    ),
    "absolutes": ("never", "always", "everyone", "nobody", "no one", "worst", "best ever", "every single"),
    "continuation": (
        "like i said", "as i mentioned", "as i said", "back to what", "so continuing",
        "coming back to", "as we discussed", "before we move", "jaisa maine bola", "pehle bola tha",
    ),
    "profanity": ("damn", "hell", "shit", "fuck", "fucking", "bitch", "crap", "bastard", "chutiya", "saala"),
    "emotion": (
        "love", "hate", "cry", "cried", "crying", "afraid", "scared", "angry", "hurt", "pain",
        "proud", "shame", "hope", "lonely", "regret", "grateful", "broke", "broken", "died",
        "death", "heart", "tears", "fight", "failed", "failure", "success", "successful",
    ),
    "life_theme": (
        "life", "money", "success", "failure", "fear", "purpose", "meaning", "relationship",
        "family", "career", "health", "discipline", "habit", "mindset", "confidence", "zindagi", "paisa",
    ),
}

QUOTABLE_ANCHORS = frozenset(
    {
        "never", "always", "everyone", "nobody", "truth", "people", "should", "reason", "problem",
        "life", "success", "failure", "money", "fear", "because", "never", "most",
    }
) | frozenset(_GROUPS["life_theme"])

NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b|\b(million|billion|thousand|crore|lakh|percent|dollars?)\b", re.IGNORECASE)
IMPERATIVE_RE = re.compile(r"\b(?:do|don't|dont|stop|start|try|listen|think|remember|imagine|focus|never|always|make)\b")

QUESTION_STARTS = frozenset({
    "what", "why", "how", "when", "where", "who", "which", "is", "are", "do", "does", "did", "can",
    "could", "would", "should", "have", "will", "kya", "kyun", "kaise", "kaun",
})
DEICTIC_STARTS = frozenset({
    "this", "that", "these", "those", "it", "they", "he", "she", "him", "her", "them", "there",
    "then", "so", "and", "but", "because", "also", "which", "who", "when", "while", "another",
    "iski", "uska", "uski", "iske", "wo", "yeh", "ye", "vo",
})
DIRECT_ADDRESS = frozenset({"you", "your", "you're", "youre", "tum", "aap", "tumhara", "aapko", "yaar", "guys", "bro"})
FIRST_PERSON = frozenset({"i", "me", "my", "mine", "we", "our", "us", "main", "mai", "mujhe", "mera", "meri", "hum", "humein"})
PAST_TENSE = frozenset({"was", "were", "had", "did", "went", "told", "said", "started", "thought", "felt", "came", "took", "gave", "knew", "tha", "thi", "gaya", "gayi", "hua", "hui"})
POSITIVE_WORDS = frozenset({"great", "amazing", "love", "best", "happy", "proud", "beautiful", "success", "won", "worked", "grateful", "excited", "incredible"})
NEGATIVE_WORDS = frozenset({"hate", "worst", "sad", "angry", "failed", "failure", "hurt", "pain", "lost", "scared", "afraid", "broke", "cry", "terrible"})

# Monotone calibration applied to the weighted score (keeps the ranking intact).
SCORE_CURVE = 0.80

WEIGHTS: dict[str, float] = {
    "hook": 0.20,
    "standalone": 0.20,
    "story": 0.10,
    "payoff": 0.10,
    "engagement": 0.15,
    "emotion": 0.10,
    "novelty": 0.10,
    "shareability": 0.05,
}

# Penalties are intentionally smaller than the positive weights: they should
# demote a clip, not make every clip fail. Calibrated so a genuinely strong
# moment lands in the 80s, a decent one in the 70s, and a weak one below 65.
PENALTY_WEIGHTS: dict[str, float] = {
    "context_dependency": 0.30,
    "abrupt_start": 0.14,
    "abrupt_end": 0.12,
    "filler": 0.10,
    "low_confidence": 0.18,
    "repetition": 0.14,
    "weak_delivery": 0.12,
}

HUMAN_LABELS: dict[str, str] = {
    "hook": "Strong opening hook",
    "emotion": "High emotional intensity",
    "curiosity": "Open curiosity gap",
    "standalone": "Complete and understandable on its own",
    "story": "Full story arc",
    "payoff": "Clear payoff",
    "novelty": "Fresh angle, not repeated elsewhere",
    "shareability": "Quotable and shareable",
    "controversy": "Opinionated - drives comments",
    "practical": "Actionable advice",
    "humour": "Funny moment",
    "insight": "Explains the underlying reason",
    "unexpectedness": "Unexpected turn",
    "engagement": "Strong delivery and pacing",
    "context_dependency": "Needs earlier context",
    "abrupt_start": "Starts mid-thought",
    "abrupt_end": "Ends mid-thought",
    "filler": "Rambling / filler-heavy",
    "low_confidence": "Unclear audio or wording",
    "repetition": "Repeats earlier material",
    "weak_delivery": "Low speaking energy",
}


# --------------------------------------------------------------------------- #
# Per-sentence pre-analysis
# --------------------------------------------------------------------------- #


@dataclass
class SentenceStats:
    index: int
    start: float
    end: float
    duration: float
    text: str
    lowered: str
    speaker: str
    tokens: list[str]
    token_set: set[str]
    token_count: int
    markers: dict[str, list[str]]
    numbers: int
    questions: int
    exclamations: int
    first_person: int
    second_person: int
    imperatives: int
    fillers: int
    emotion_pos: int
    emotion_neg: int
    ends_terminal: bool
    starts_deictic: bool
    starts_conjunction: bool
    quotable: bool
    past_tense: int
    confidence: float


def _analyse_sentence(sentence: Sentence, index: int) -> SentenceStats:
    text = sentence.text.strip()
    lowered = text.lower()
    tokens = [normalize_word(token) for token in words_only(text)]
    tokens = [token for token in tokens if token]
    token_set = set(tokens)

    markers = {name: [marker for marker in needles if marker in lowered] for name, needles in _GROUPS.items()}
    first_token = tokens[0] if tokens else ""

    return SentenceStats(
        index=index,
        start=sentence.start,
        end=sentence.end,
        duration=sentence.duration,
        text=text,
        lowered=lowered,
        speaker=sentence.speaker,
        tokens=tokens,
        token_set=token_set,
        token_count=len(tokens),
        markers=markers,
        numbers=len(NUMBER_RE.findall(text)),
        questions=text.count("?"),
        exclamations=text.count("!"),
        first_person=sum(1 for token in tokens if token in FIRST_PERSON),
        second_person=sum(1 for token in tokens if token in DIRECT_ADDRESS),
        imperatives=len(IMPERATIVE_RE.findall(lowered)),
        fillers=sum(1 for token in tokens if token in FILLER_WORDS),
        emotion_pos=len(token_set & POSITIVE_WORDS),
        emotion_neg=len(token_set & NEGATIVE_WORDS),
        ends_terminal=text.endswith((".", "!", "?", "…", "।")),
        starts_deictic=first_token in DEICTIC_STARTS,
        starts_conjunction=first_token in {"and", "but", "so", "because", "then", "also", "aur", "lekin", "phir"},
        quotable=(
            5 <= len(tokens) <= 22
            and not (token_set & FILLER_WORDS)
            and bool(token_set & QUOTABLE_ANCHORS)
        ),
        past_tense=sum(1 for token in tokens if token in PAST_TENSE),
        confidence=sentence.confidence or 0.0,
    )


# --------------------------------------------------------------------------- #
# Transcript index
# --------------------------------------------------------------------------- #


@dataclass
class TranscriptIndex:
    sentences: list[Sentence]
    blocks: list[TopicBlock]
    stats: list[SentenceStats]
    vectors: list[dict[str, float]]
    prefix_vectors: list[dict[str, float]]
    prefix_token_counts: list[int]
    duration: float
    word_count: int
    mean_word_confidence: float
    sentence_gaps: list[float]
    speech_rate_wpm: float

    # ------------------------------------------------------------- builders
    @classmethod
    def build(cls, sentences: Sequence[Sentence], blocks: Sequence[TopicBlock], duration: float) -> "TranscriptIndex":
        sentences = list(sentences)
        stats = [_analyse_sentence(sentence, index) for index, sentence in enumerate(sentences)]
        vectors, _ = tfidf_matrix([sentence.text for sentence in sentences])

        prefix_vectors: list[dict[str, float]] = []
        running: dict[str, float] = {}
        for vector in vectors:
            running = {**running}
            for term, value in vector.items():
                running[term] = running.get(term, 0.0) + value
            prefix_vectors.append(running)

        confidences = [word.confidence for sentence in sentences for word in sentence.words if word.confidence]
        gaps = [max(0.0, sentences[i + 1].start - sentences[i].end) for i in range(len(sentences) - 1)]
        word_count = sum(len(sentence.words) or len(words_only(sentence.text)) for sentence in sentences)
        # prefix_token_counts[i] = tokens in sentences[0..i-1] (cumulative, inclusive prefix).
        prefix_token_counts = [0]
        for sentence in sentences:
            prefix_token_counts.append(prefix_token_counts[-1] + len(words_only(sentence.text)))
        return cls(
            sentences=sentences,
            blocks=list(blocks),
            stats=stats,
            vectors=vectors,
            prefix_vectors=prefix_vectors,
            prefix_token_counts=prefix_token_counts,
            duration=duration,
            word_count=word_count,
            mean_word_confidence=float(np.mean(confidences)) if confidences else 0.7,
            sentence_gaps=gaps,
            speech_rate_wpm=round(word_count / max(duration / 60.0, 1e-6), 1),
        )

    # -------------------------------------------------------------- helpers
    def range_text(self, start_index: int, end_index: int) -> str:
        return " ".join(self.sentences[i].text for i in range(start_index, end_index) if 0 <= i < len(self.sentences))

    def range_duration(self, start_index: int, end_index: int) -> float:
        if end_index <= start_index:
            return 0.0
        return max(0.0, self.sentences[end_index - 1].end - self.sentences[start_index].start)

    def mean_vector_before(self, index: int, lookback_skip: int = 6) -> dict[str, float]:
        """Average TF-IDF vector of everything earlier than ``index`` (cheap)."""
        reference = index - lookback_skip
        if reference <= 0:
            return {}
        total = self.prefix_vectors[min(reference, len(self.prefix_vectors)) - 1]
        norm = float(reference)
        return {term: value / norm for term, value in total.items()}

    def range_vector(self, start_index: int, end_index: int) -> dict[str, float]:
        combined: dict[str, float] = {}
        count = 0
        for index in range(max(start_index, 0), min(end_index, len(self.vectors))):
            count += 1
            for term, value in self.vectors[index].items():
                combined[term] = combined.get(term, 0.0) + value
        if not count:
            return {}
        return {term: value / count for term, value in combined.items()}


# --------------------------------------------------------------------------- #
# Range aggregation
# --------------------------------------------------------------------------- #


@dataclass
class RangeAggregate:
    duration: float
    token_count: int
    words: int
    markers: dict[str, int]
    numbers: int
    questions: int
    exclamations: int
    first_person: int
    second_person: int
    imperatives: int
    fillers: int
    emotion_pos: int
    emotion_neg: int
    past_tense: int
    quotable: list[str]
    speakers: set[str]
    speaker_turns: int
    mean_confidence: float
    text: str


def aggregate(index: TranscriptIndex, start_index: int, end_index: int) -> RangeAggregate:
    markers: dict[str, int] = {name: 0 for name in _GROUPS}
    speakers: set[str] = set()
    quotable: list[str] = []
    token_count = words = 0
    numbers = questions = exclamations = 0
    first_person = second_person = imperatives = fillers = 0
    emotion_pos = emotion_neg = past_tense = 0
    confidences: list[float] = []
    turns = 0
    previous_speaker = ""

    for position in range(start_index, end_index):
        stat = index.stats[position]
        token_count += stat.token_count
        sentence = index.sentences[position]
        words += len(sentence.words) or len(words_only(sentence.text))
        numbers += stat.numbers
        questions += stat.questions
        exclamations += stat.exclamations
        first_person += stat.first_person
        second_person += stat.second_person
        imperatives += stat.imperatives
        fillers += stat.fillers
        emotion_pos += stat.emotion_pos
        emotion_neg += stat.emotion_neg
        past_tense += stat.past_tense
        for name, hits in stat.markers.items():
            markers[name] += len(hits)
        if stat.quotable and len(quotable) < 4:
            quotable.append(stat.text)
        speakers.add(stat.speaker)
        if previous_speaker and stat.speaker != previous_speaker:
            turns += 1
        previous_speaker = stat.speaker
        if stat.confidence:
            confidences.append(stat.confidence)

    return RangeAggregate(
        duration=index.range_duration(start_index, end_index),
        token_count=token_count,
        words=words,
        markers=markers,
        numbers=numbers,
        questions=questions,
        exclamations=exclamations,
        first_person=first_person,
        second_person=second_person,
        imperatives=imperatives,
        fillers=fillers,
        emotion_pos=emotion_pos,
        emotion_neg=emotion_neg,
        past_tense=past_tense,
        quotable=quotable,
        speakers=speakers,
        speaker_turns=turns,
        mean_confidence=float(np.mean(confidences)) if confidences else index.mean_word_confidence,
        text=index.range_text(start_index, end_index),
    )


# --------------------------------------------------------------------------- #
# Feature bundle
# --------------------------------------------------------------------------- #


@dataclass
class CandidateFeatures:
    factors: dict[str, float] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def weighted_score(self) -> float:
        """Weighted feature average minus penalties, then a monotone calibration.

        The positive weights follow the product spec (hook 20%, standalone 20%,
        story 10%, payoff 10%, engagement 15%, emotion 10%, novelty 10%,
        shareability 5%). ``SCORE_CURVE`` stretches the result so that the
        0-100 scale matches how the threshold is meant to read: ~70 is a
        genuinely usable clip, 80+ is a strong one, below 60 is filler.
        """
        base = sum(self.factors.get(name, 0.0) * weight for name, weight in WEIGHTS.items())
        penalty = sum(self.penalties.get(name, 0.0) * weight for name, weight in PENALTY_WEIGHTS.items())
        return max(0.0, min(1.0, base - penalty)) ** SCORE_CURVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "factors": {key: round(value, 4) for key, value in self.factors.items()},
            "penalties": {key: round(value, 4) for key, value in self.penalties.items()},
            "evidence": self.evidence,
            "warnings": self.warnings,
            "meta": self.meta,
            "weighted": round(self.weighted_score, 4),
        }


def compute_features(
    index: TranscriptIndex,
    start_index: int,
    end_index: int,
    *,
    target_seconds: float = 60.0,
    min_seconds: float = 35.0,
    max_seconds: float = 75.0,
) -> CandidateFeatures:
    """Score the candidate range ``[start_index, end_index)`` of sentences."""
    if end_index <= start_index or start_index < 0 or end_index > len(index.sentences):
        return CandidateFeatures(evidence=["Invalid range"])

    agg = aggregate(index, start_index, end_index)
    first = index.stats[start_index]
    last = index.stats[end_index - 1]
    total = max(agg.token_count, 1)
    duration = agg.duration or 1e-6
    evidence: list[str] = []
    warnings: list[str] = []

    # ---------------------------------------------------------------- hook
    hook_len = first.token_count
    hook = 0.30
    if first.questions or (first.tokens and first.tokens[0].rstrip(",;:") in QUESTION_STARTS):
        hook += 0.22
        evidence.append(f"Opens with a question: “{_trim(first.text)}”")
    if first.markers["curiosity"]:
        hook += min(0.20 + 0.04 * len(first.markers["curiosity"]), 0.30)
        evidence.append(f"Curiosity trigger in the first line: “{first.markers['curiosity'][0]}”")
    if first.markers["intensity"]:
        hook += 0.08
    if first.second_person:
        hook += 0.07
        evidence.append("Speaks directly to the viewer from the first line")
    if first.numbers:
        hook += 0.06
        evidence.append("Concrete number or statistic in the opening")
    if 4 <= hook_len <= 16:
        hook += 0.08
        evidence.append("Tight, quotable opening sentence")
    elif hook_len > 26:
        hook -= 0.12
        warnings.append("Opening sentence is long and slow to land")
    if first.markers["continuation"]:
        hook -= 0.15
    hook = _clamp(hook)

    # ----------------------------------------------------------- curiosity
    curiosity = _clamp(0.25 + 0.18 * agg.markers["curiosity"] + (0.15 if agg.questions else 0.0) + 0.06 * agg.markers["contrast"])

    # ------------------------------------------------------------- emotion
    emotion_terms = agg.markers["emotion"]
    emotion_swing = _emotion_swing(index, start_index, end_index)
    first_person_ratio = agg.first_person / total
    emotion = _clamp(
        0.18
        + 0.15 * emotion_terms
        + 0.04 * min(agg.exclamations, 4)
        + 0.04 * agg.markers["intensity"]
        + 0.05 * agg.markers["profanity"]
        + 0.55 * first_person_ratio
        + 0.25 * emotion_swing,
    )
    if emotion_terms:
        hits = [hit for hit in _GROUPS["emotion"] if hit in agg.text.lower()][:3]
        if hits:
            evidence.append("Emotional language: " + ", ".join(f"“{hit}”" for hit in hits))
    if emotion_swing > 0.45:
        evidence.append("Emotional intensity shifts through the moment")
    if first_person_ratio > 0.12:
        evidence.append("Personal, first-person account")

    # --------------------------------------------------------------- story
    narrative_ratio = agg.past_tense / total
    story = _clamp(0.16 + 0.055 * agg.markers["story"] + min(narrative_ratio * 3.2, 0.42) + (0.12 if first_person_ratio > 0.1 else 0.0))
    if agg.markers["story"]:
        hits = [hit for hit in _GROUPS["story"] if hit in agg.text.lower()][:2]
        if hits:
            evidence.append("Narrative structure: " + ", ".join(f"“{hit}”" for hit in hits))

    # -------------------------------------------------------------- payoff
    second_person_close = last.second_person > 0
    payoff = _clamp(
        0.20
        + 0.10 * agg.markers["payoff"]
        + (0.16 if 3 <= last.token_count <= 18 else 0.0)
        + (0.10 if last.ends_terminal else 0.0)
        + (0.08 if second_person_close else 0.0),
    )
    if agg.markers["payoff"]:
        hits = [hit for hit in _GROUPS["payoff"] if hit in agg.text.lower()][:2]
        if hits:
            evidence.append("Explicit resolution: " + ", ".join(f"“{hit}”" for hit in hits))
    if 3 <= last.token_count <= 18:
        evidence.append(f"Punchy closing line: “{_trim(last.text)}”")

    # ----------------------------------------------------------- practical
    practical = _clamp(0.12 + 0.11 * agg.markers["advice"] + 0.05 * min(agg.numbers, 4))
    if agg.markers["advice"]:
        hits = [hit for hit in _GROUPS["advice"] if hit in agg.text.lower()][:2]
        if hits:
            evidence.append("Actionable advice: " + ", ".join(f"“{hit}”" for hit in hits))

    # -------------------------------------------------------------- humour
    humour = _clamp(0.05 + 0.24 * agg.markers["humour"] + 0.05 * agg.exclamations)
    if agg.markers["humour"]:
        evidence.append("Humour cues detected (laughter / joke language)")

    # ---------------------------------------------------------- controversy
    controversy = _clamp(0.08 + 0.17 * agg.markers["controversy"] + 0.05 * agg.markers["absolutes"])
    if agg.markers["controversy"]:
        evidence.append("Opinionated, debate-triggering stance")
    elif agg.markers["absolutes"]:
        evidence.append("Absolute language (everyone / nobody / never) makes it quotable")

    # ------------------------------------------------------ unexpectedness
    unexpected = _clamp(0.13 * agg.markers["contrast"] + (0.18 if "turns out" in agg.text.lower() else 0.0))
    if agg.markers["contrast"] >= 2:
        evidence.append("Clear turning point mid-clip")

    # ------------------------------------------------------------- insight
    insight = _clamp(0.1 + 0.09 * agg.markers["insight"] + 0.08 * (1.0 if agg.markers["payoff"] else 0.0))
    if agg.markers["insight"]:
        hits = [hit for hit in _GROUPS["insight"] if hit in agg.text.lower()][:1]
        if hits:
            evidence.append(f"Explains the underlying reason: “{hits[0]}”")

    # ---------------------------------------------------------- engagement
    wpm = (agg.token_count / (duration / 60.0)) if duration else 0.0
    pace_score = _pace_score(wpm)
    direct_ratio = agg.second_person / total
    engagement = _clamp(
        0.20
        + 0.30 * pace_score
        + min(direct_ratio * 1.8, 0.25)
        + min(agg.questions * 0.05, 0.12)
        + min(agg.imperatives * 0.04, 0.14),
    )

    # -------------------------------------------------------------- novelty
    novelty = _novelty(index, start_index, end_index)

    # --------------------------------------------------------- shareability
    shareability = _clamp(0.15 + 0.13 * min(len(agg.quotable), 3) + min(direct_ratio * 1.6, 0.25) + (0.08 if agg.numbers else 0.0))
    if agg.quotable:
        evidence.append(f"Contains a quotable line: “{_trim(agg.quotable[0])}”")

    # ----------------------------------------------------------- standalone
    standalone, standalone_notes = _standalone(index, start_index, end_index, agg, first, last)
    warnings.extend(standalone_notes)

    # --------------------------------------------------- context dependency
    context_penalty, context_notes = _context_dependency(index, start_index, end_index, first)
    warnings.extend(context_notes)
    if context_penalty > 0.4:
        evidence.append("⚠ Needs a little earlier context")

    # ---------------------------------------------------------- abrupt edges
    abrupt_start = 0.0
    if first.starts_deictic:
        abrupt_start = 0.35
    if first.starts_conjunction:
        abrupt_start = max(abrupt_start, 0.4)
    if start_index > 0 and index.sentence_gaps[start_index - 1] < 0.12:
        abrupt_start = max(abrupt_start, 0.25)

    abrupt_end = 0.0
    if not last.ends_terminal:
        abrupt_end = 0.3
    if end_index < len(index.sentence_gaps) and index.sentence_gaps[end_index - 1] < 0.12:
        abrupt_end = max(abrupt_end, 0.2)

    # -------------------------------------------------------------- filler
    filler = _clamp((agg.fillers / total) * 4.5)
    if filler > 0.45:
        warnings.append("Plenty of filler words (um / like / basically)")

    # ---------------------------------------------------------- confidence
    low_confidence = _clamp(1.0 - agg.mean_confidence) if agg.mean_confidence else 0.0

    # ---------------------------------------------------------- repetition
    repetition = _repetition(index, start_index, end_index)

    # ------------------------------------------------------- weak delivery
    weak_delivery = _clamp((1.0 - pace_score) * 0.55 + (0.35 if duration < min_seconds * 0.8 else 0.0))

    factors = {
        "hook": hook, "emotion": emotion, "curiosity": curiosity, "standalone": standalone,
        "story": story, "payoff": payoff, "novelty": novelty, "shareability": shareability,
        "controversy": controversy, "practical": practical, "humour": humour, "insight": insight,
        "unexpectedness": unexpected, "engagement": engagement,
    }
    penalties = {
        "context_dependency": context_penalty, "abrupt_start": abrupt_start, "abrupt_end": abrupt_end,
        "filler": filler, "low_confidence": low_confidence, "repetition": repetition,
        "weak_delivery": weak_delivery,
    }
    meta = {
        "text": (agg.text or "")[:6000],
        "duration": round(duration, 2),
        "word_count": agg.token_count,
        "wpm": round(wpm, 1),
        "numbers": agg.numbers,
        "questions": agg.questions,
        "laughter": agg.markers["humour"],
        "mean_word_confidence": round(agg.mean_confidence, 3),
        "quotable_lines": [_trim(line) for line in agg.quotable[:3]],
        "speakers": sorted(agg.speakers),
        "speaker_turns": agg.speaker_turns,
        "pace_score": round(pace_score, 3),
        "sentences": end_index - start_index,
    }
    return CandidateFeatures(
        factors={key: round(value, 4) for key, value in factors.items()},
        penalties={key: round(value, 4) for key, value in penalties.items()},
        evidence=evidence,
        warnings=warnings,
        meta=meta,
    )


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def _pace_score(wpm: float) -> float:
    """Speech-rate sweet spot for short-form video is roughly 120-190 wpm."""
    if wpm <= 0:
        return 0.0
    if 120 <= wpm <= 190:
        return 1.0
    if wpm < 120:
        return _clamp(1.0 - (120 - wpm) / 110.0)
    return _clamp(1.0 - (wpm - 190) / 160.0)


def _emotion_swing(index: TranscriptIndex, start_index: int, end_index: int) -> float:
    scores = [
        index.stats[position].emotion_pos - index.stats[position].emotion_neg
        for position in range(start_index, end_index)
    ]
    if len(scores) < 2:
        return 0.0
    return _clamp((max(scores) - min(scores)) / 3.0)


def _novelty(index: TranscriptIndex, start_index: int, end_index: int) -> float:
    """1 - similarity with everything said before this point in the video."""
    vector = index.range_vector(start_index, end_index)
    if not vector or not index.prefix_vectors or start_index < 6:
        return 0.62 if start_index < 6 else 0.5
    reference = index.mean_vector_before(start_index, lookback_skip=6)
    if not reference:
        return 0.6
    return _clamp(1.0 - cosine(vector, reference) * 1.45)


def _repetition(index: TranscriptIndex, start_index: int, end_index: int) -> float:
    """How much this range re-treads a nearby earlier passage."""
    vector = index.range_vector(start_index, end_index)
    if not vector or start_index < 4:
        return 0.0
    length = max(1, end_index - start_index)
    best = 0.0
    for offset in (length, 2 * length, 3 * length):
        previous_start = start_index - offset
        if previous_start < 0:
            break
        best = max(best, cosine(vector, index.range_vector(previous_start, previous_start + length)))
    return _clamp((best - 0.52) / 0.48)


def _standalone(
    index: TranscriptIndex,
    start_index: int,
    end_index: int,
    agg: RangeAggregate,
    first: SentenceStats,
    last: SentenceStats,
) -> tuple[float, list[str]]:
    score = 0.65
    notes: list[str] = []

    if first.starts_deictic:
        score -= 0.12
        notes.append("Opens with a reference that was established earlier")
    if first.starts_conjunction:
        score -= 0.08
        notes.append("Starts with a conjunction, relying on the previous sentence")

    distinct_content = {token for token in first.token_set | last.token_set if token not in ENGLISH_STOPWORDS and len(token) > 3}
    if first.starts_deictic and len(distinct_content) < 5:
        score -= 0.08
        notes.append("People or things inside the clip are introduced only outside it")

    if last.ends_terminal:
        score += 0.18
    else:
        score -= 0.10
        notes.append("Ends mid-sentence in the source transcript")

    if agg.markers["continuation"]:
        score -= 0.22
        notes.append("Explicitly continues an earlier discussion")

    if agg.duration < 20:
        score -= 0.25
        notes.append("Too short to stand alone")
    elif agg.duration > 110:
        score -= 0.10
    return _clamp(score), notes


def _context_dependency(
    index: TranscriptIndex,
    start_index: int,
    end_index: int,
    first: SentenceStats,
) -> tuple[float, list[str]]:
    penalty = 0.08
    notes: list[str] = []

    if first.start > 600:
        penalty += 0.04
    if first.start > 2400:
        penalty += 0.05

    block_id = index.sentences[start_index].block
    if block_id >= 0 and block_id < len(index.blocks) and start_index > 0:
        previous = index.sentences[start_index - 1]
        if previous.block == block_id and first.start - previous.end < 0.6:
            penalty += 0.12
            notes.append("Starts in the middle of a continuous topic")

    if first.markers["continuation"]:
        penalty += 0.2
        notes.append("Continues a discussion from earlier in the episode")

    if start_index > 0 and index.sentence_gaps[start_index - 1] < 0.1:
        penalty += 0.1

    return _clamp(penalty), notes


def _trim(text: str, limit: int = 88) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# --------------------------------------------------------------------------- #
# Aggregation helpers for the UI
# --------------------------------------------------------------------------- #


def summarize_factors(features: Sequence[dict[str, float]]) -> dict[str, float]:
    if not features:
        return {}
    keys = {key for item in features for key in item}
    return {
        key: round(float(np.mean([item.get(key, 0.0) for item in features])), 3)
        for key in sorted(keys)
    }


def top_human_evidence(features: CandidateFeatures, limit: int = 8) -> list[str]:
    """Ordered, human-readable reasons - strongest positives first, then caveats."""
    entries: list[tuple[float, str]] = []
    for name, value in features.factors.items():
        if value >= 0.55 and name in HUMAN_LABELS:
            entries.append((value, HUMAN_LABELS[name]))
    for name, value in features.penalties.items():
        if value >= 0.35 and name in HUMAN_LABELS:
            entries.append((value * 0.55, HUMAN_LABELS[name] + " (penalty)"))
    entries.sort(key=lambda item: -item[0])
    ordered: list[str] = []
    for _, label in entries:
        if label not in ordered:
            ordered.append(label)
    for note in features.evidence:
        if note not in ordered:
            ordered.append(note)
    return ordered[:limit]


def evidence_only(features: CandidateFeatures, limit: int = 6) -> list[str]:
    """Only true transcript observations (no generic labels)."""
    return list(dict.fromkeys(features.evidence))[:limit]


__all__ = [
    "CandidateFeatures",
    "HUMAN_LABELS",
    "PENALTY_WEIGHTS",
    "RangeAggregate",
    "SentenceStats",
    "TranscriptIndex",
    "WEIGHTS",
    "aggregate",
    "compute_features",
    "evidence_only",
    "summarize_factors",
    "top_human_evidence",
]
