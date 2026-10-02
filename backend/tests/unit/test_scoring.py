from __future__ import annotations

import pytest

from clipforge.core.config import Settings
from clipforge.core.presets import scoring_config
from clipforge.scoring.boundaries import optimize_boundary
from clipforge.scoring.candidates import generate_candidates, time_based_candidates
from clipforge.scoring.engine import ScoredCandidate, ScoringEngine, select_top
from clipforge.scoring.features import FeatureContext, compute_features
from clipforge.scoring.segmentation import assign_topics, build_sentences, flatten_words
from clipforge.scoring.text import TfIdf

SCRIPT = [
    "Why do most people fail at learning something new?",
    "The secret is not talent, it is consistency.",
    "I tried this for thirty days and the results shocked me.",
    "First, you need a tiny habit you can do every single day.",
    "Second, track it, because what gets measured gets managed.",
    "Honestly, the biggest mistake is quitting after one bad week.",
    "Let me tell you a story about my first marathon.",
    "I was terrified, my legs hurt, and I almost gave up.",
    "But then something changed at mile twenty.",
    "So here is the lesson: progress beats perfection every time.",
]


def transcript(repeat: int = 3, wps: float = 2.6) -> dict:
    segs, t = [], 0.5
    for i, sentence in enumerate(SCRIPT * repeat):
        words = []
        for tok in sentence.split():
            d = 1 / wps
            words.append({"start": round(t, 3), "end": round(t + d * 0.9, 3), "word": tok, "probability": 0.95})
            t += d
        segs.append({"id": i, "start": words[0]["start"], "end": words[-1]["end"], "text": sentence, "words": words})
        t += 0.7
    return {"segments": segs, "language": "en"}


@pytest.fixture
def sentences():
    words = flatten_words(transcript())
    return words, assign_topics(build_sentences(words))


def test_sentences_follow_punctuation(sentences):
    _, sents = sentences
    assert len(sents) == len(SCRIPT) * 3
    assert sents[0].text.startswith("Why do most people")
    assert all(s.end > s.start for s in sents)


def test_flatten_synthesises_word_timings():
    words = flatten_words({"segments": [{"id": 0, "start": 0.0, "end": 2.0, "text": "hello there world"}]})
    assert [w.word for w in words] == ["hello", "there", "world"]
    assert words[0].start == 0.0 and words[-1].end <= 2.0


def test_candidates_respect_duration_limits(sentences):
    _, sents = sentences
    cands = generate_candidates(sents, min_seconds=10, target_min=20, target_max=60, max_seconds=90)
    assert cands
    assert all(10 <= c.duration <= 90 for c in cands)
    # candidates start and end on sentence boundaries
    starts = {s.start for s in sents}
    ends = {s.end for s in sents}
    assert all(c.start in starts and c.end in ends for c in cands)


def test_time_based_fallback():
    cands = time_based_candidates(120, length=30)
    assert cands and all(c.synthetic for c in cands)
    assert all(0 <= c.start < c.end <= 120 for c in cands)


def test_scoring_is_deterministic_and_explained(sentences):
    _, sents = sentences
    engine = ScoringEngine(scoring_config(Settings()))
    cands = generate_candidates(sents, min_seconds=10, target_min=20, target_max=60, max_seconds=90)
    ctx = FeatureContext(sentences=sents, tfidf=TfIdf([s.text for s in sents]), audio_levels=[],
                         min_seconds=10, target_min=20, target_max=60, max_seconds=90)
    scored = [engine.apply(ScoredCandidate(c, *compute_features(c, ctx))) for c in cands]
    again = [engine.apply(ScoredCandidate(c, *compute_features(c, ctx))) for c in cands]
    assert [s.score for s in scored] == [s.score for s in again]
    for s in scored:
        assert 0.0 <= s.score <= 100.0  # scores are reported on a 0-100 scale
        assert all(0.0 <= v <= 1.0 for v in s.features.values())
    best = max(scored, key=lambda s: s.score)
    assert best.reasons


def test_weights_sum_and_hook_feature():
    engine = ScoringEngine(scoring_config(Settings()))
    weighted = {k: v for k, v in engine.weights.items() if v > 0}
    assert abs(sum(weighted.values()) - 1.0) < 1e-6
    base = {k: 0.5 for k in weighted}
    assert engine.score({**base, "hook": 1.0}) > engine.score(base)


def test_select_top_avoids_overlap(sentences):
    _, sents = sentences
    cands = generate_candidates(sents, min_seconds=10, target_min=20, target_max=60, max_seconds=90)
    scored = [ScoredCandidate(c, {}, {}, score=1.0 - i * 0.001) for i, c in enumerate(cands)]
    top = select_top(scored, 5, 0.25)
    for i, a in enumerate(top):
        for b in top[i + 1:]:
            inter = max(0.0, min(a.window.end, b.window.end) - max(a.window.start, b.window.start))
            union = max(a.window.end, b.window.end) - min(a.window.start, b.window.start)
            assert inter / union <= 0.25 + 1e-9


def test_boundary_optimisation_stays_in_range(sentences):
    words, sents = sentences
    cand = generate_candidates(sents, min_seconds=10, target_min=20, target_max=60, max_seconds=90)[3]
    b = optimize_boundary(cand, sents, words, source_duration=sents[-1].end + 1, min_seconds=10, max_seconds=90)
    assert 0 <= b.start < b.end
    assert 10 <= b.end - b.start <= 90
    # does not cut a word in half
    for w in words:
        assert not (w.start < b.start < w.end - 0.01)
