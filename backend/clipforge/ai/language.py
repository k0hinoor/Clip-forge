"""Language identification for transcription and captions.

Whisper already returns a language guess with a probability. That is not enough
for real Indian-podcast content, where speakers slip between Hindi and English
mid-sentence. This module combines:

* Whisper's language probability distribution,
* Unicode script analysis of the actual transcript,
* a romanised-Hindi ("Hinglish") lexicon score,
* a function-word fingerprint of the transcript,

into a :class:`LanguageProfile` that tells the caption engine which language the
words are in - so nothing gets translated or normalised behind the user's back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .textutil import (
    ARABIC_RE,
    CJK_RE,
    CYRILLIC_RE,
    DEVANAGARI_RE,
    LATIN_RE,
    hinglish_score,
    normalize_word,
    script_counts,
    words_only,
)

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English", "hi": "Hindi", "ur": "Urdu", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi",
    "ne": "Nepali", "si": "Sinhala", "ar": "Arabic", "fa": "Persian", "tr": "Turkish",
    "es": "Spanish", "pt": "Portuguese", "fr": "French", "de": "German", "it": "Italian",
    "nl": "Dutch", "pl": "Polish", "ru": "Russian", "uk": "Ukrainian", "id": "Indonesian",
    "ms": "Malay", "vi": "Vietnamese", "th": "Thai", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "sv": "Swedish", "ro": "Romanian", "el": "Greek", "he": "Hebrew",
    "cs": "Czech", "da": "Danish", "fi": "Finnish", "no": "Norwegian", "hu": "Hungarian",
}

# Distinctive function words → language. Deliberately small: this only ever acts
# as a tie-breaker next to Whisper's own probabilities.
FUNCTION_WORDS: dict[str, set[str]] = {
    "en": {"the", "and", "you", "that", "with", "have", "this", "what", "your", "about", "there", "would"},
    "es": {"que", "los", "para", "con", "una", "por", "pero", "muy", "porque", "también"},
    "pt": {"que", "não", "uma", "para", "com", "isso", "você", "muito", "está", "então"},
    "fr": {"que", "les", "des", "une", "pour", "dans", "avec", "c'est", "pas", "vous"},
    "de": {"und", "der", "die", "das", "nicht", "ist", "mit", "auch", "aber", "wir"},
    "it": {"che", "non", "una", "però", "sono", "questo", "molto", "anche", "perché"},
    "nl": {"het", "een", "niet", "maar", "ook", "wij", "dit", "zijn", "voor"},
    "id": {"yang", "dan", "tidak", "itu", "untuk", "dengan", "saya", "bisa", "kamu"},
    "tr": {"bir", "ama", "çok", "için", "değil", "ben", "sen", "var", "yok"},
    "ru": {"что", "это", "как", "для", "они", "если", "было", "очень", "только"},
    "ar": {"هذا", "التي", "على", "هو", "لا", "من", "في", "أن", "كان"},
    "vi": {"không", "được", "người", "nhưng", "cũng", "này", "với", "của"},
}

# Latin-script Hindi written in Devanagari by Whisper is common; these markers
# decide whether "hi" output should be treated as Hinglish for captions.
HINGLISH_MODE_THRESHOLD = 0.12


@dataclass
class LanguageProfile:
    primary: str = "en"
    primary_name: str = "English"
    secondary: str = ""
    secondary_name: str = ""
    mode: str = "monolingual"  # monolingual | mixed | hinglish
    confidence: float = 0.0
    whistle_language: str = ""
    probabilities: dict[str, float] = field(default_factory=dict)
    script: dict[str, int] = field(default_factory=dict)
    hinglish_score: float = 0.0
    secondary_share: float = 0.0
    notes: list[str] = field(default_factory=list)

    def caption_language(self) -> str:
        """Language tag the caption engine should assume (never auto-translates)."""
        return self.primary or "en"

    def describe(self) -> str:
        if self.mode == "hinglish":
            return f"{self.primary_name} + English (Hinglish)"
        if self.mode == "mixed" and self.secondary_name:
            return f"{self.primary_name} (mixed with {self.secondary_name})"
        return self.primary_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary,
            "primary_name": self.primary_name,
            "secondary": self.secondary,
            "secondary_name": self.secondary_name,
            "mode": self.mode,
            "confidence": round(self.confidence, 3),
            "whisper_language": self.whistle_language,
            "hinglish_score": round(self.hinglish_score, 3),
            "secondary_share": round(self.secondary_share, 3),
            "script": self.script,
            "probabilities": {k: round(v, 4) for k, v in list(self.probabilities.items())[:6]},
            "description": self.describe(),
            "notes": self.notes,
        }


def language_name(code: str) -> str:
    if not code:
        return "Unknown"
    base = code.split("-")[0].lower()
    return LANGUAGE_NAMES.get(base, code.upper())


SCRIPT_LANGUAGES = (
    ("devanagari", "hi"),
    ("arabic", "ar"),
    ("cyrillic", "ru"),
    ("cjk", "zh"),
)


def _language_for_script(counts: dict[str, int], letters: int) -> str:
    """The language implied by a clearly dominant script (>= 50% of the letters)."""
    if letters <= 1:
        return ""
    for script, language in SCRIPT_LANGUAGES:
        if counts.get(script, 0) / letters >= 0.5:
            return language
    return ""


def detect_language(
    text: str,
    *,
    whisper_language: str = "",
    whisper_confidence: float = 0.0,
    probabilities: dict[str, float] | None = None,
    sample_limit: int = 40000,
) -> LanguageProfile:
    """Combine Whisper's guess with script and lexical evidence."""
    sample = (text or "")[:sample_limit]
    counts = script_counts(sample)
    tokens = [normalize_word(token) for token in words_only(sample)]
    tokens = [token for token in tokens if token]

    profile = LanguageProfile(
        probabilities=dict(probabilities or {}),
        whistle_language=whisper_language or "",
        script=counts,
        confidence=whisper_confidence,
    )

    devanagari = counts["devanagari"]
    latin = counts["latin"]
    letters = max(devanagari + latin + counts["arabic"] + counts["cyrillic"] + counts["cjk"], 1)
    script_language = _language_for_script(counts, letters)

    fingerprint = _fingerprint_language(tokens)
    primary = (whisper_language or "").split("-")[0].lower()
    if not primary:
        # Script evidence outranks the Latin-word fingerprint: a Devanagari
        # transcript has no Latin tokens at all, and defaulting it to English
        # would caption Hindi audio as English.
        primary = script_language or (fingerprint[0] if fingerprint else "en")
    if primary:
        # Whisper's guess is the starting point; the checks below can still
        # override it (romanised Hindi reported as English, mixed scripts, ...).
        profile.primary = primary

    # Whisper sometimes reports a confident English guess for romanised Hindi.
    hinglish = hinglish_score(sample)
    profile.hinglish_score = round(hinglish, 3)

    if script_language and primary in {"", "en"} and primary != script_language and not whisper_language:
        profile.primary = script_language
        profile.notes.append(f"Script evidence points at {language_name(script_language)}.")

    if primary == "hi" and latin > devanagari:
        # Hindi reported but the transcript is mostly romanised.
        if hinglish >= HINGLISH_MODE_THRESHOLD:
            profile.mode = "hinglish"
            profile.secondary = "en"
            profile.notes.append("Romanised Hindi detected - captions keep the original Hinglish wording.")
        else:
            profile.notes.append("Hindi audio written in Latin script.")
    elif primary in {"en", ""} and hinglish >= HINGLISH_MODE_THRESHOLD and devanagari < 0.2 * letters:
        profile.primary = "hi"
        profile.mode = "hinglish"
        profile.secondary = "en"
        profile.notes.append("Hinglish detected: Hindi speech with English words, written in Latin script.")

    if devanagari > 0.15 * letters and latin > 0.15 * letters:
        profile.mode = "hinglish" if profile.mode == "hinglish" else "mixed"
        if profile.primary in {"", "en"} and devanagari > latin:
            profile.primary = "hi"
        if not profile.secondary:
            profile.secondary = "en" if profile.primary != "en" else "hi"
        profile.secondary_share = round(min(latin, devanagari) / letters, 3)
        profile.notes.append("Mixed-script transcript: the caption engine keeps both languages as spoken.")

    if fingerprint and not whisper_language and not script_language:
        profile.primary = fingerprint[0]
        profile.confidence = fingerprint[1]

    if not profile.primary:
        profile.primary = "en"
    if profile.primary == "hi" and profile.mode == "monolingual" and latin and devanagari:
        profile.secondary_share = round(min(latin, devanagari) / letters, 3)

    profile.primary_name = language_name(profile.primary)
    profile.secondary_name = language_name(profile.secondary) if profile.secondary else ""
    if not profile.confidence:
        profile.confidence = float((probabilities or {}).get(profile.primary, 0.0)) or 0.6
    return profile


def _fingerprint_language(tokens: Iterable[str]) -> tuple[str, float] | None:
    tokens = list(tokens)
    if len(tokens) < 25:
        return None
    best: tuple[str, float] | None = None
    for code, words in FUNCTION_WORDS.items():
        hits = sum(1 for token in tokens if token in words)
        score = hits / len(tokens)
        if best is None or score > best[1]:
            best = (code, score)
    if best is None or best[1] < 0.02:
        return None
    return best


def dominant_script(text: str) -> str:
    counts = script_counts(text)
    if not any(counts.values()):
        return "unknown"
    return max(counts.items(), key=lambda item: item[1])[0]


def script_language_hint(text: str) -> str:
    """Guess a language from dominant script alone (used for caption fonts)."""
    script = dominant_script(text)
    return {
        "devanagari": "hi",
        "arabic": "ar",
        "cyrillic": "ru",
        "cjk": "zh",
        "latin": "en",
    }.get(script, "en")


def contains_script(text: str, language: str) -> bool:
    if language == "hi":
        return bool(DEVANAGARI_RE.search(text))
    if language == "ar":
        return bool(ARABIC_RE.search(text))
    if language == "ru":
        return bool(CYRILLIC_RE.search(text))
    if language in {"zh", "ja", "ko"}:
        return bool(CJK_RE.search(text))
    return bool(LATIN_RE.search(text))


def needs_multilingual_font(profile: LanguageProfile) -> bool:
    """Devanagari/Arabic/CJK need a font with wider glyph coverage than Latin-only fonts."""
    return profile.primary in {"hi", "ar", "ur", "fa", "ru", "zh", "ja", "ko", "th", "he"} or profile.mode in {"hinglish", "mixed"}


__all__ = [
    "FUNCTION_WORDS",
    "LANGUAGE_NAMES",
    "LanguageProfile",
    "contains_script",
    "detect_language",
    "dominant_script",
    "language_name",
    "needs_multilingual_font",
    "script_language_hint",
]
