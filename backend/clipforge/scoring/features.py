"""Deterministic feature extraction for candidate scoring (TRD §14; PRD §6.3, §7).

Every feature is normalised to 0..1. Raw signals are returned alongside so
they can be persisted in ``candidate_features.raw`` for future model training.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from clipforge.scoring.candidates import CandidateWindow
from clipforge.scoring.segmentation import Sentence
from clipforge.scoring.text import (
    CONCLUSION_MARKERS,
    CONTINUATION_STARTERS,
    EMOTION_WORDS,
    FILLERS,
    HOOK_WORDS,
    INFO_MARKERS,
    STORY_MARKERS,
    TfIdf,
    content_tokens,
    count_markers,
    first_word,
    has_number,
    tokenize,
)

FEATURE_NAMES = (
    "hook", "completeness", "emotional_interest", "information_value", "narrative_completeness",
    "audio_quality", "visual_quality", "caption_suitability", "length_fit", "llm_interest",
)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def sat(x: float, full: float) -> float:
    """Saturating map: 0 → 0, ``full`` → ~0.86, ∞ → 1."""
    if full <= 0:
        return 0.0
    return 1.0 - math.exp(-2.0 * max(x, 0.0) / full)


@dataclass
class FeatureContext:
    sentences: Sequence[Sentence]
    tfidf: TfIdf | None
    audio_levels: Sequence[float] | None  # per-second RMS dBFS
    min_seconds: float
    target_min: float
    target_max: float
    max_seconds: float


def length_fit(duration: float, ctx: FeatureContext) -> float:
    if ctx.target_min <= duration <= ctx.target_max:
        return 1.0
    if duration < ctx.target_min:
        span = max(ctx.target_min - ctx.min_seconds, 1e-6)
        return clamp(1.0 - (ctx.target_min - duration) / span) * 0.8
    span = max(ctx.max_seconds - ctx.target_max, 1e-6)
    return clamp(1.0 - (duration - ctx.target_max) / span) * 0.8


def _audio_stats(levels: Sequence[float] | None, start: float, end: float) -> dict[str, float] | None:
    if not levels:
        return None
    lo, hi = int(max(0, math.floor(start))), int(min(len(levels), math.ceil(end)))
    window = [v for v in levels[lo:hi] if v is not None]
    if not window:
        return None
    mean = sum(window) / len(window)
    var = sum((v - mean) ** 2 for v in window) / len(window)
    silent = sum(1 for v in window if v < -45.0) / len(window)
    return {"mean_db": mean, "std_db": math.sqrt(var), "silence_ratio": silent}


def compute_features(cand: CandidateWindow, ctx: FeatureContext) -> tuple[dict[str, float], dict[str, Any]]:
    duration = cand.duration
    raw: dict[str, Any] = {"duration": round(duration, 3)}
    audio = _audio_stats(ctx.audio_levels, cand.start, cand.end)

    if cand.synthetic:
        f = {name: 0.3 for name in FEATURE_NAMES}
        f["length_fit"] = length_fit(duration, ctx)
        f["llm_interest"] = 0.5
        if audio:
            f["audio_quality"] = clamp((audio["mean_db"] + 45) / 30) * (1 - audio["silence_ratio"])
            f["emotional_interest"] = clamp(audio["std_db"] / 8.0)
            raw.update(audio)
        raw["synthetic"] = True
        return f, raw

    sents = list(ctx.sentences[cand.start_idx:cand.end_idx + 1])
    text = cand.text
    toks = tokenize(text)
    n_words = max(len(toks), 1)
    first = sents[0]
    last = sents[-1]

    # ------------------------------------------------------------- hook
    opener = first.text if first.duration >= 2.5 or len(sents) == 1 else f"{first.text} {sents[1].text}"
    opener_toks = tokenize(opener)
    question = "?" in first.text
    hook_hits = sum(1 for t in opener_toks if t in HOOK_WORDS)
    you = sum(1 for t in opener_toks if t in ("you", "your", "you're"))
    fw = first_word(first.text)
    continuation = fw in CONTINUATION_STARTERS
    filler_start = fw in FILLERS
    punchy = 4 <= len(tokenize(first.text)) <= 14
    hook = clamp(0.25 + 0.2 * question + 0.12 * min(hook_hits, 3) + 0.1 * has_number(opener)
                 + 0.1 * (you > 0) + 0.1 * punchy - 0.3 * continuation - 0.15 * filler_start)
    raw.update(question_open=question, hook_words=hook_hits, you_in_open=you, continuation_start=continuation,
               filler_start=filler_start)

    # ----------------------------------------------------- completeness
    start_clean = not continuation and not filler_start and (cand.start_idx == 0 or first.pause_before >= 0.3
                                                             or ctx.sentences[cand.start_idx - 1].terminal)
    end_clean = last.terminal
    single_topic = len(cand.topic_ids) <= 1
    completeness = clamp(0.3 * start_clean + 0.3 * end_clean + 0.15 * clamp(first.pause_before / 0.6)
                         + 0.15 * clamp(last.pause_after / 0.6) + 0.1 * single_topic)
    raw.update(start_clean=start_clean, end_clean=end_clean, topics=len(cand.topic_ids),
               pause_before=round(first.pause_before, 3), pause_after=round(min(last.pause_after, 99), 3))

    # ----------------------------------------------- emotional interest
    emo = sum(1 for t in toks if t in EMOTION_WORDS)
    exclaims = text.count("!")
    laughs = len(re.findall(r"\b(ha){2,}\b|\[laughter\]|\blol\b", text.lower()))
    energy = clamp(audio["std_db"] / 8.0) if audio else 0.4
    emotional = clamp(0.5 * sat(emo * 100 / n_words, 3) + 0.15 * sat(exclaims, 2) + 0.1 * sat(laughs, 1)
                      + 0.3 * energy)
    raw.update(emotion_words=emo, exclamations=exclaims, laughs=laughs)

    # ----------------------------------------------- information value
    content = content_tokens(text)
    density = len(content) / n_words
    unique_ratio = len(set(content)) / max(len(content), 1)
    info_hits = count_markers(text, INFO_MARKERS)
    salience = 0.0
    if ctx.tfidf is not None and content:
        # Mean IDF of content words: rarer vocabulary → more specific information.
        idfs = [ctx.tfidf.idf.get(t, 1.0) for t in content]
        max_idf = max(ctx.tfidf.idf.values(), default=1.0)
        salience = clamp((sum(idfs) / len(idfs)) / max(max_idf, 1e-6))
    information = clamp(0.3 * sat(density, 0.5) + 0.25 * sat(info_hits, 2) + 0.1 * has_number(text)
                        + 0.15 * unique_ratio + 0.2 * salience)
    raw.update(lexical_density=round(density, 3), info_markers=info_hits, unique_ratio=round(unique_ratio, 3),
               salience=round(salience, 3))

    # ------------------------------------------- narrative completeness
    story = count_markers(text, STORY_MARKERS)
    half = max(1, len(sents) // 2)
    qa = any("?" in s.text for s in sents[:half]) and any("?" not in s.text for s in sents[half:])
    tail = " ".join(s.text for s in sents[-2:])
    conclusion = count_markers(tail, CONCLUSION_MARKERS) > 0
    narrative = clamp(0.15 + 0.3 * sat(story, 2) + 0.3 * qa + 0.3 * conclusion)
    raw.update(story_markers=story, question_answer=qa, conclusion=conclusion)

    # --------------------------------------------------- audio quality
    lp = [s.avg_logprob for s in sents if s.avg_logprob is not None]
    nsp = [s.no_speech_prob for s in sents if s.no_speech_prob is not None]
    asr_conf = clamp((sum(lp) / len(lp) + 1.0) / 0.9) if lp else 0.7
    speechiness = 1.0 - (sum(nsp) / len(nsp) if nsp else 0.1)
    if audio:
        loud = clamp((audio["mean_db"] + 45) / 25)
        audio_q = clamp(0.35 * loud + 0.25 * (1 - audio["silence_ratio"]) + 0.25 * asr_conf + 0.15 * speechiness)
        raw.update(mean_db=round(audio["mean_db"], 2), std_db=round(audio["std_db"], 2),
                   silence_ratio=round(audio["silence_ratio"], 3))
    else:
        audio_q = clamp(0.6 * asr_conf + 0.4 * speechiness)
    raw.update(asr_confidence=round(asr_conf, 3))

    # ---------------------------------------------- caption suitability
    wps = len(toks) / max(duration, 1e-6)
    if 2.0 <= wps <= 4.0:
        pace = 1.0
    elif wps < 2.0:
        pace = clamp(wps / 2.0)
    else:
        pace = clamp(1.0 - (wps - 4.0) / 2.5)
    filler_ratio = sum(1 for t in toks if t in FILLERS) / n_words
    caption = clamp(pace - 2.0 * filler_ratio)
    raw.update(words_per_second=round(wps, 3), filler_ratio=round(filler_ratio, 3))

    features = {
        "hook": round(hook, 4),
        "completeness": round(completeness, 4),
        "emotional_interest": round(emotional, 4),
        "information_value": round(information, 4),
        "narrative_completeness": round(narrative, 4),
        "audio_quality": round(audio_q, 4),
        "visual_quality": 0.5,  # refined after visual analysis
        "caption_suitability": round(caption, 4),
        "length_fit": round(length_fit(duration, ctx), 4),
        "llm_interest": 0.5,  # neutral unless a local LLM rates the excerpt
    }
    return features, raw
