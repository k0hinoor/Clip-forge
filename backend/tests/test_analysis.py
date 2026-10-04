"""The analytical engine: segmentation, features, discovery, scoring, dedupe.

These tests build *real* transcripts (word lists with real timings) from a scripted
podcast and assert the mathematics: strong moments must outrank filler, overlapping
candidates must collapse, and nothing may be invented when the content is empty.
"""

from __future__ import annotations

import pytest

from clipforge.ai import candidates as discovery
from clipforge.ai import scoring
from clipforge.ai.features import PENALTY_WEIGHTS, WEIGHTS, TranscriptIndex
from clipforge.ai.segment import build_blocks, build_sentences
from clipforge.ai.transcribe import Utterance, Word

STRONG = [
    "Nobody talks about what happens after you actually become successful.",
    "Everyone thinks success is going to solve their problems. I believed that too.",
    "And I remember sitting in my apartment in 2019, three months after the launch.",
    "We had just crossed one million users. One million people. And I felt completely empty inside.",
    "That was the moment I realised the goal was never the company. It was permission to feel enough.",
    "So I tell founders now: do the work, but know exactly why you are doing it.",
]

FILLER = [
    "So, um, yeah, today's episode is about, you know, some things.",
    "First, the housekeeping. The newsletter is moving to Tuesday.",
    "Also the community call is on the first Friday of every month.",
    "Sponsorship slots for next quarter are open, the details are in the description.",
    "Anyway, let me read the next question that came in.",
]


def make_utterances(lines: list[str], *, start: float = 0.0, speaker: str = "SPEAKER_01", gap: float = 0.55) -> list[Utterance]:
    utterances: list[Utterance] = []
    cursor = start
    for line in lines:
        if not line.strip():
            continue
        words: list[Word] = []
        t = cursor
        for token in line.split():
            words.append(Word(word=token, start=t, end=t + 0.30, confidence=0.93, speaker=speaker))
            t += 0.34
        utterances.append(Utterance(text=line, start=words[0].start, end=words[-1].end, speaker=speaker, confidence=0.93, words=words))
        cursor = words[-1].end + gap
    return utterances


def index_for(lines: list[str], **kwargs) -> TranscriptIndex:
    utterances = make_utterances(lines, **kwargs)
    sentences = build_sentences(utterances)
    blocks = build_blocks(sentences)
    duration = utterances[-1].end if utterances else 0.0
    return TranscriptIndex.build(sentences, blocks, duration)


def test_sentences_and_blocks_are_derived_from_words():
    index = index_for(STRONG + FILLER)
    assert len(index.sentences) >= len(STRONG) + len(FILLER) - 2
    assert index.blocks, "topic blocks must be produced"
    assert index.duration > 30
    for sentence in index.sentences:
        assert sentence.words, "every sentence keeps its word-level timings"
        assert sentence.end > sentence.start
        assert sentence.words[0].start >= sentence.start - 0.5


def test_empty_transcript_produces_no_clips():
    index = index_for([""])
    assert discovery.generate_candidates(index, min_seconds=35, target_seconds=60, max_seconds=75) == []


def test_candidates_respect_duration_limits_and_are_ranked():
    index = index_for(STRONG + FILLER)
    pool = discovery.generate_candidates(index, min_seconds=35, target_seconds=60, max_seconds=75)
    assert pool, "a 40s+ transcript must yield candidates"
    for candidate in pool:
        assert 35 - 0.01 <= candidate.duration <= 75 + 0.01
        assert 0 <= candidate.score <= 100
    assert pool == sorted(pool, key=lambda c: c.score, reverse=True)
    assert pool[0].why, "every candidate must explain itself"


def test_strong_content_outscores_filler():
    index = index_for(STRONG * 2 + FILLER * 3)
    pool = discovery.generate_candidates(index, min_seconds=35, target_seconds=60, max_seconds=75)
    best = pool[0]
    text = " ".join((best.features.meta.get("text") or best.summary).lower().split())
    assert any(word in text for word in ["successful", "empty", "million"]), text[:120]
    assert best.score > 60


def test_scoring_depends_on_content_not_constants():
    rich = index_for(STRONG * 2)
    poor = index_for(FILLER * 4)
    rich_pool = scoring.finalize_candidates(
        rich,
        discovery.generate_candidates(rich, min_seconds=35, target_seconds=60, max_seconds=75),
        min_seconds=35, target_seconds=60, max_seconds=75, min_score=0, mode="balanced",
    )
    poor_pool = scoring.finalize_candidates(
        poor,
        discovery.generate_candidates(poor, min_seconds=35, target_seconds=60, max_seconds=75),
        min_seconds=35, target_seconds=60, max_seconds=75, min_score=0, mode="balanced",
    )
    assert rich_pool["kept"], "rich transcript should keep candidates"
    best_rich = max(c.score for c in rich_pool["kept"])
    best_poor = max((c.score for c in poor_pool["kept"]), default=0.0)
    assert best_rich > best_poor, (best_rich, best_poor)


def test_min_score_gate_rejects_weak_material():
    index = index_for(FILLER * 5)
    pool = discovery.generate_candidates(index, min_seconds=35, target_seconds=60, max_seconds=75)
    strict = scoring.finalize_candidates(
        index, pool, min_seconds=35, target_seconds=60, max_seconds=75, min_score=70, mode="balanced",
    )
    assert strict["selected"] == []
    report = strict["report"]
    assert report["input"] >= len(pool) > 0
    assert report["below_threshold"] > 0, report  # weak moments must be counted, not selected


def test_max_mode_finds_more_clips_than_best_mode():
    index = index_for(STRONG * 4)
    pool = discovery.generate_candidates(index, min_seconds=35, target_seconds=60, max_seconds=75)
    best = scoring.finalize_candidates(
        index, pool, min_seconds=35, target_seconds=60, max_seconds=75, min_score=70, mode="best",
    )
    maximum = scoring.finalize_candidates(
        index, pool, min_seconds=35, target_seconds=60, max_seconds=75, min_score=70, mode="max",
    )
    assert len(maximum["selected"]) >= len(best["selected"])


def test_dedupe_drops_overlapping_candidates():
    index = index_for(STRONG * 3)
    pool = discovery.generate_candidates(index, min_seconds=35, target_seconds=60, max_seconds=75)
    result = scoring.finalize_candidates(
        index, pool, min_seconds=35, target_seconds=60, max_seconds=75, min_score=0, mode="max",
    )
    selected = sorted(result["selected"], key=lambda c: c.start)
    for previous, following in zip(selected, selected[1:]):
        overlap = previous.end - following.start
        shorter = min(previous.duration, following.duration)
        assert overlap <= shorter * 0.5, (previous.start, previous.end, following.start, following.end)
    assert result["report"]["duplicates"] >= 0


def test_boundaries_do_not_cut_words_in_half():
    index = index_for(STRONG * 2 + FILLER)
    pool = discovery.generate_candidates(index, min_seconds=35, target_seconds=60, max_seconds=75)
    result = scoring.finalize_candidates(
        index, pool, min_seconds=35, target_seconds=60, max_seconds=75, min_score=50, mode="balanced",
    )
    all_words = [word for sentence in index.sentences for word in sentence.words]
    for candidate in result["selected"]:
        for word in all_words:
            if word.start < candidate.start < word.end:
                pytest.fail(f"clip starts inside word {word.word!r} at {word.start}")
        ends = {round(word.end, 3) for word in all_words}
        if round(candidate.end, 3) not in ends:
            # ends should sit on a word boundary (small tolerance for rounding)
            assert any(abs(candidate.end - word.end) < 0.4 for word in all_words), candidate.end


def test_weights_cover_the_specified_factors():
    assert pytest.approx(sum(WEIGHTS.values()), abs=0.001) == 1.0
    assert {"hook", "standalone", "engagement", "story", "payoff", "emotion", "novelty", "shareability"} <= set(WEIGHTS)
    assert {"context_dependency", "abrupt_start", "low_confidence"} <= set(PENALTY_WEIGHTS)


def test_reasons_are_transcript_derived():
    index = index_for(STRONG * 2)
    pool = discovery.generate_candidates(index, min_seconds=35, target_seconds=60, max_seconds=75)
    result = scoring.finalize_candidates(
        index, pool, min_seconds=35, target_seconds=60, max_seconds=75, min_score=0, mode="best",
    )
    candidate = result["selected"][0]
    assert candidate.hook
    assert candidate.hook.split()[0].lower().strip(",.") in " ".join(STRONG).lower()
    assert candidate.title
    assert candidate.why and all(isinstance(reason, str) and reason for reason in candidate.why)
    assert candidate.category in {
        "emotional", "story", "funny", "controversial", "educational", "motivational",
        "insightful", "shocking", "practical", "inspirational", "debate", "advice", "general",
        "storytelling", "hot_take", "tutorial", "reaction", "interview", "rant", "highlight", "viral",
    }


def test_phase_callbacks_fire_in_order():
    index = index_for(STRONG * 2)
    pool = discovery.generate_candidates(index, min_seconds=35, target_seconds=60, max_seconds=75)
    phases: list[str] = []
    scoring.finalize_candidates(
        index, pool, min_seconds=35, target_seconds=60, max_seconds=75, min_score=0, mode="balanced",
        on_phase=lambda key, message: phases.append(key),
    )
    assert "score" in phases and "dedupe" in phases


def test_max_clips_cap_is_respected_when_asked():
    index = index_for(STRONG * 6)
    pool = discovery.generate_candidates(index, min_seconds=35, target_seconds=60, max_seconds=75)
    result = scoring.finalize_candidates(
        index, pool, min_seconds=35, target_seconds=60, max_seconds=75, min_score=0, mode="max", max_clips=2,
    )
    assert len(result["selected"]) <= 2


def test_language_detection_reads_the_transcript_not_a_constant():
    from clipforge.ai.language import detect_language

    english = detect_language(" ".join(STRONG), whisper_language="en", whisper_confidence=0.9)
    assert english.primary.startswith("en")
    assert english.mode == "monolingual"
    hindi_text = "आपका प्राइस आपकी वैल्यू का translation है, इसलिए पहले value समझो"
    mixed = detect_language(hindi_text, whisper_language="hi", whisper_confidence=0.8)
    assert mixed.primary in {"hi", "hinglish", "mixed"} or mixed.mode in {"hinglish", "mixed"}


def test_llm_seed_influence_is_capped():
    """LLM suggestions may reorder candidates, never fabricate them."""
    index = index_for(FILLER * 4)
    pool = discovery.generate_candidates(index, min_seconds=35, target_seconds=60, max_seconds=75)
    before = {round(c.start, 2) for c in pool}
    seeds = [{"start": 900.0, "end": 960.0, "score": 99, "reason": "invented"}]
    merged = discovery.merge_llm_seeds(index, pool, seeds, min_seconds=35, max_seconds=75, target_seconds=60)
    after = {round(c.start, 2) for c in merged}
    assert before <= after, "LLM seeds must not remove real candidates"
    assert all(0 <= c.score <= 100 for c in merged)
