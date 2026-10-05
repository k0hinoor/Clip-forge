"""Caption engine: word-level subtitles burned in with ffmpeg (libass).

The engine does four things properly instead of dumping raw ASR text on screen:

1. **Line breaking** - lines are assembled from real word timings and broken at
   semantic phrase boundaries (punctuation, pause length, conjunctions, clause
   length), never mid-phrase just to hit a word count.
2. **Animation** - each preset renders through its own animation model
   (word-by-word highlight, karaoke fill, pop, slide-up, or none) using libass
   override tags, still synchronised to the original word timestamps.
3. **Emphasis** - words that carry the meaning (numbers, intensity, the LLM's
   emphasis list, zoom points) get a larger, colour-highlighted treatment.
4. **Multilingual rendering** - Hindi/Devanagari, Arabic, Cyrillic and CJK text
   select a font with real coverage, and nothing is transliterated or translated.

Time remapping from silence removal is applied *before* this module runs, so the
timings here are already in output (rendered) time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..constants import CAPTION_PRESETS
from ..logging_setup import get_logger
from ..config import CaptionTheme

log = get_logger("clipforge.render")

MAX_LINE_SECONDS = 2.6
PAUSE_BREAK_SECONDS = 0.32
TERMINAL_PUNCTUATION = (".", "!", "?", "…", "।", "॥")

# ASS override sequence for a hard line break inside one subtitle event.
NEWLINE_TAG = "\\N"

# Optional words that should never end a line on their own.
TRAILING_WORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with", "is",
    "are", "was", "were", "my", "your", "his", "her", "their", "our", "this", "that", "i",
    "you", "he", "she", "it", "we", "they", "as", "if", "so", "ka", "ke", "ki", "mein", "me",
}
CONJUNCTIONS = {"and", "but", "so", "because", "then", "while", "when", "if", "though", "although", "aur", "lekin", "toh", "kyunki"}

EMPHASIS_WORDS = {
    "never", "always", "everyone", "nobody", "everything", "nothing", "impossible", "insane",
    "crazy", "huge", "massive", "incredible", "unbelievable", "terrible", "amazing", "truth",
    "secret", "mistake", "failure", "success", "money", "life", "death", "love", "hate", "fear",
    "biggest", "worst", "best", "first", "last", "only", "free", "real", "actually", "literally",
}

NUMBER_RE = re.compile(r"[\d%$₹€£]")

FONT_STACKS: dict[str, list[str]] = {
    "devanagari": ["Noto Sans Devanagari", "Nirmala UI", "Mangal", "Kohinoor Devanagari", "DejaVu Sans"],
    "arabic": ["Noto Naskh Arabic", "Segoe UI", "Geeza Pro", "DejaVu Sans"],
    "cyrillic": ["Noto Sans", "Segoe UI", "DejaVu Sans"],
    "cjk": ["Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "DejaVu Sans"],
    "latin": ["Inter", "Arial Black", "Segoe UI", "DejaVu Sans"],
}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class CaptionWord:
    text: str
    start: float
    end: float
    emphasis: bool = False
    highlight: bool = False
    speaker: str = ""

    @property
    def duration(self) -> float:
        return max(0.04, self.end - self.start)


@dataclass
class CaptionLine:
    words: list[CaptionWord]
    start: float
    end: float
    index: int = 0
    text_override: str = ""

    @property
    def text(self) -> str:
        if self.text_override:
            return self.text_override
        return " ".join(word.text for word in self.words).strip()

    @property
    def has_word_timings(self) -> bool:
        return bool(self.words)

    @property
    def duration(self) -> float:
        return max(0.05, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "has_word_timings": self.has_word_timings,
            "words": [
                {
                    "text": word.text,
                    "start": round(word.start, 3),
                    "end": round(word.end, 3),
                    "emphasis": word.emphasis,
                    "speaker": word.speaker,
                }
                for word in self.words
            ],
        }


@dataclass
class CaptionPlan:
    lines: list[CaptionLine]
    theme: CaptionTheme
    language: str = "en"
    font: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return sum(len(line.words) or len(line.text.split()) for line in self.lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "font": self.font,
            "preset": self.theme.preset,
            "animation": self.theme.animation,
            "word_count": self.word_count,
            "line_count": len(self.lines),
            "timing_granularity": "word" if self.lines and all(line.has_word_timings for line in self.lines) else "cue",
            "lines": [line.to_dict() for line in self.lines],
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
# Building lines
# --------------------------------------------------------------------------- #


def build_lines(
    words: Sequence[Any],
    theme: CaptionTheme,
    *,
    emphasis_words: Sequence[str] = (),
    zoom_times: Sequence[float] = (),
) -> list[CaptionLine]:
    """Group word-level timings into caption lines at semantic boundaries."""
    prepared: list[CaptionWord] = []
    emphasis_pool = {word.lower() for word in theme.emphasis_words} | {word.lower() for word in emphasis_words}

    for word in words:
        text = (getattr(word, "text", None) or getattr(word, "word", "") or "").strip()
        if not text:
            continue
        start = float(getattr(word, "start", 0.0))
        end = float(getattr(word, "end", start + 0.15))
        prepared.append(
            CaptionWord(
                text=text,
                start=start,
                end=end,
                emphasis=False,
                speaker=str(getattr(word, "speaker", "") or ""),
            )
        )

    if not prepared:
        return []

    max_words = max(1, int(theme.max_words_per_line))
    lines: list[CaptionLine] = []
    current: list[CaptionWord] = []

    def flush() -> None:
        nonlocal current
        if current:
            lines.append(CaptionLine(words=current, start=current[0].start, end=current[-1].end, index=len(lines)))
            current = []

    for index, word in enumerate(prepared):
        if current:
            gap = word.start - current[-1].end
            projected = current[-1].end - current[0].start + (word.end - word.start)
            previous_text = current[-1].text
            break_reason = (
                gap >= PAUSE_BREAK_SECONDS
                or len(current) >= max_words
                or projected > MAX_LINE_SECONDS
                or previous_text.endswith(TERMINAL_PUNCTUATION)
            )
            # Do not end a line on a word that needs its neighbour.
            if break_reason and previous_text.strip(".,!?;:").lower() in TRAILING_WORDS and len(current) > 1:
                current.pop()
                flush()
                current.append(prepared[index - 1])
            elif break_reason:
                flush()
            # Prefer starting a new line at a conjunction clause boundary.
            elif len(current) >= max(2, max_words - 1) and word.text.strip(".,!?;:").lower() in CONJUNCTIONS:
                flush()
        current.append(word)
    flush()

    # Merge orphan lines (a single short word) into neighbours.
    merged: list[CaptionLine] = []
    for line in lines:
        if (
            merged
            and len(line.words) <= 1
            and len(merged[-1].words) < max_words + 1
            and line.start - merged[-1].end < 0.35
        ):
            merged[-1].words.extend(line.words)
            merged[-1].end = line.end
            continue
        merged.append(line)

    for index, line in enumerate(merged):
        line.index = index

    _apply_emphasis(merged, emphasis_pool, zoom_times)
    return merged


def build_cue_lines(cues: Sequence[Any]) -> list[CaptionLine]:
    """Build static subtitle events from real cue timings; do not infer word times."""
    lines: list[CaptionLine] = []
    for cue in cues:
        if isinstance(cue, dict):
            text = str(cue.get("text") or "").strip()
            start = float(cue.get("start", 0.0))
            end = float(cue.get("end", 0.0))
        else:
            text = str(getattr(cue, "text", "") or "").strip()
            start = float(getattr(cue, "start", 0.0))
            end = float(getattr(cue, "end", 0.0))
        if not text or start < 0 or end <= start:
            continue
        lines.append(CaptionLine(words=[], start=start, end=end, index=len(lines), text_override=text))
    return lines


def _apply_emphasis(lines: Sequence[CaptionLine], emphasis_pool: set[str], zoom_times: Sequence[float]) -> None:
    for line in lines:
        content_words = [word for word in line.words if word.text.strip(".,!?;:")]
        longest = max(content_words, key=lambda word: len(word.text), default=None)
        for word in line.words:
            token = word.text.strip(".,!?;:\"'“”").lower()
            word.emphasis = bool(
                token in emphasis_pool
                or token in EMPHASIS_WORDS
                or NUMBER_RE.search(word.text)
                or (word.text.isupper() and len(word.text) > 2)
                or (longest is not None and word is longest and len(word.text) > 6)
                or any(abs(word.start - moment) < 0.35 for moment in zoom_times)
            )
            # A line where everything is emphasised emphasises nothing.
        if sum(1 for word in line.words if word.emphasis) > max(1, len(line.words) - 1):
            for word in line.words:
                word.emphasis = word.text.strip(".,!?;:") in emphasis_pool
        if not any(word.emphasis for word in line.words) and content_words:
            content_words[0].emphasis = True


def describe_emphasis(line: CaptionLine) -> str:
    return " ".join(f"[{word.text}]" if word.emphasis else word.text for word in line.words)


# --------------------------------------------------------------------------- #
# Colours & fonts
# --------------------------------------------------------------------------- #


def hex_to_ass_color(value: str, *, alpha: int = 0) -> str:
    """``#RRGGBB`` → ASS ``&HAABBGGRR``."""
    text = (value or "#FFFFFF").lstrip("#")
    if len(text) == 3:
        text = "".join(char * 2 for char in text)
    if len(text) != 6:
        text = "FFFFFF"
    try:
        red, green, blue = (int(text[index: index + 2], 16) for index in (0, 2, 4))
    except ValueError:
        red = green = blue = 255
    alpha = max(0, min(255, int(alpha)))
    return f"&H{alpha:02X}{blue:02X}{green:02X}{red:02X}"


def resolve_font(theme: CaptionTheme, language: str = "en") -> str:
    """Pick a font that can actually render the transcript's script."""
    base = language.split("-")[0].lower()
    needs_devanagari = base in {"hi", "mr", "ne", "sa"} or "hinglish" in language.lower()
    stack: list[str] = []
    if needs_devanagari:
        stack.extend(FONT_STACKS["devanagari"])
    elif base in {"ar", "ur", "fa"}:
        stack.extend(FONT_STACKS["arabic"])
    elif base in {"ru", "uk", "bg", "sr"}:
        stack.extend(FONT_STACKS["cyrillic"])
    elif base in {"zh", "ja", "ko"}:
        stack.extend(FONT_STACKS["cjk"])
    else:
        stack.extend(FONT_STACKS["latin"])

    # Search for a usable installed font (fc-match on POSIX, registry-free name check).
    installed = _installed_fonts()
    for candidate in [theme.resolved_font(), *stack, *theme.fallback_fonts]:
        if not candidate:
            continue
        if not installed or any(candidate.lower() in name.lower() for name in installed):
            return candidate
    return stack[0]


_FONT_CACHE: list[str] = []


def _installed_fonts() -> list[str]:
    if _FONT_CACHE:
        return _FONT_CACHE
    try:
        import subprocess

        completed = subprocess.run(  # noqa: S603 - fixed argv
            ["fc-list", "--format", "%{family}\n"],
            capture_output=True,
            text=True,
            timeout=6,
            shell=False,
        )
        if completed.returncode == 0:
            families: set[str] = set()
            for line in completed.stdout.splitlines():
                for family in line.split(","):
                    family = family.strip()
                    if family:
                        families.add(family)
            _FONT_CACHE.extend(sorted(families))
    except Exception:  # noqa: BLE001 - fc-list is optional
        pass
    return _FONT_CACHE


# --------------------------------------------------------------------------- #
# ASS generation
# --------------------------------------------------------------------------- #


def _escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()
    )


def _timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        centis = 99
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def ass_header(theme: CaptionTheme, width: int, height: int, font: str) -> str:
    primary = hex_to_ass_color(theme.primary_color)
    secondary = hex_to_ass_color(theme.highlight_color)
    outline = hex_to_ass_color(theme.outline_color)
    background = hex_to_ass_color(theme.background_color, alpha=int((1 - theme.background_opacity / 100.0) * 255))
    border_style = 3 if theme.background_enabled else 1
    alignment = {"top": 8, "middle": 5, "lower-middle": 2, "bottom": 2}.get(theme.position, 2)
    margin_v = theme.margin_v
    if theme.position == "lower-middle":
        margin_v = max(theme.margin_v, int(height * 0.30))
    elif theme.position == "bottom":
        margin_v = max(theme.margin_v, int(height * 0.10))
    margin_h = max(int(width * theme.safe_margin_pct / 100.0), 40)
    bold = -1 if theme.weight >= 700 else 0
    # The theme's font size is authored for a 1080px-wide frame; every other
    # output size (previews, 1:1, 16:9) scales with it so the look is identical
    # at any resolution.
    font_size = max(12, int(round(theme.font_size * width / 1080.0)))
    outline_width = max(1, int(round(theme.outline_width * width / 1080.0)))
    shadow = max(0, int(round(theme.shadow * width / 1080.0)))

    return "\n".join(
        [
            "[Script Info]",
            "; Generated by CLIPFORGE AI - subtitle timing follows the selected transcript source.",
            "ScriptType: v4.00+",
            "WrapStyle: 2",
            "ScaledBorderAndShadow: yes",
            "YCbCr Matrix: TV.709",
            f"PlayResX: {width}",
            f"PlayResY: {height}",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
            "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: CF,{font},{font_size},{primary},{secondary},{outline},{background},"
            f"{bold},0,0,0,100,100,0,0,{border_style},{outline_width},{shadow},"
            f"{alignment},{margin_h},{margin_h},{margin_v},1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, Effect, Text",
        ]
    )


def build_ass(
    plan: CaptionPlan,
    *,
    width: int,
    height: int,
    font: str | None = None,
    offset: float = 0.0,
) -> str:
    """Render the caption plan to an ASS subtitle document."""
    theme = plan.theme
    chosen_font = font or plan.font or resolve_font(theme, plan.language)
    lines = plan.lines
    events: list[str] = []

    if not lines:
        return ass_header(theme, width, height, chosen_font) + "\n"

    max_lines = max(1, int(theme.max_lines))
    blocks = _blocks(lines, max_lines)
    animation = theme.animation

    for block in blocks:
        block_start = block[0].start + offset
        block_end = block[-1].end + offset
        if any(not line.has_word_timings for line in block):
            # Cue-level sources keep their original full text and event timing;
            # karaoke/word animation is disabled rather than fabricating times.
            events.append(_plain_event(block, theme, block_start, block_end))
            continue
        if animation == "karaoke":
            events.append(_karaoke_event(block, theme, block_start, block_end))
            continue
        if animation == "none":
            events.append(_plain_event(block, theme, block_start, block_end))
            continue

        # Word-by-word / pop / slide-up / highlight: one event per word with the
        # line visible and the current word highlighted.
        words = [word for line in block for word in line.words]
        for index, word in enumerate(words):
            start = word.start + offset
            end = (words[index + 1].start + offset) if index + 1 < len(words) else block_end
            if end <= start:
                end = start + 0.12
            text = _compose_words(block, theme, active=word, animation=animation)
            events.append(
                f"Dialogue: 0,{_timestamp(start)},{_timestamp(end)},CF,,0,0,0,,{text}"
            )

    return ass_header(theme, width, height, chosen_font) + "\n" + "\n".join(events) + "\n"


def _blocks(lines: Sequence[CaptionLine], max_lines: int) -> list[list[CaptionLine]]:
    if max_lines <= 1:
        return [[line] for line in lines]
    blocks: list[list[CaptionLine]] = []
    index = 0
    while index < len(lines):
        window = list(lines[index: index + max_lines])
        total = window[-1].end - window[0].start
        if len(window) > 1 and total > MAX_LINE_SECONDS * 1.8:
            window = window[:1]
        blocks.append(window)
        index += len(window)
    return blocks


def _wrap(line: CaptionLine, theme: CaptionTheme) -> str:
    text = _escape(line.text)
    return text.upper() if theme.uppercase else text


def _plain_event(block: Sequence[CaptionLine], theme: CaptionTheme, start: float, end: float) -> str:
    text = "\\N".join(_wrap(line, theme) for line in block)
    intro = ""
    if theme.animation == "pop":
        intro = f"{{\\fad(80,80)\\fscx88\\fscy88\\t(0,140,\\fscx100\\fscy100)}}"
    elif theme.animation == "slide_up":
        intro = "{\\fad(70,70)\\move(0,-1,0,0,0,140)\\pos(0,0)}"
    return f"Dialogue: 0,{_timestamp(start)},{_timestamp(end)},CF,,0,0,0,,{intro}{text}"


def _compose_words(
    block: Sequence[CaptionLine],
    theme: CaptionTheme,
    *,
    active: CaptionWord,
    animation: str,
) -> str:
    """Assemble the block text with the active word highlighted."""
    primary = hex_to_ass_color(theme.primary_color)
    highlight = hex_to_ass_color(theme.highlight_color)
    secondary = hex_to_ass_color(theme.secondary_color)
    scale = int(theme.emphasis_scale)
    rendered_lines: list[str] = []

    for line in block:
        pieces: list[str] = []
        for word in line.words:
            text = _escape(word.text)
            if theme.uppercase:
                text = text.upper()
            if word is active:
                pop = f"\\fscx{scale if word.emphasis else 106}\\fscy{scale if word.emphasis else 106}"
                pieces.append(f"{{\\c{highlight}{pop}}}{text}{{\\c{primary}\\fscx100\\fscy100}}")
            elif word.emphasis:
                pieces.append(f"{{\\c{highlight}}}{text}{{\\c{primary}}}")
            else:
                pieces.append(f"{{\\c{primary}}}{text}")
        # The spaces join the words; without them the line renders as one blob.
        rendered_lines.append(" ".join(pieces))
    joined = "\\N".join(rendered_lines)

    intro = ""
    if animation == "pop":
        intro = "{\\fad(60,60)}"
    elif animation == "slide_up":
        intro = "{\\fad(60,60)}"
    return f"{intro}{joined}"


def _karaoke_event(block: Sequence[CaptionLine], theme: CaptionTheme, start: float, end: float) -> str:
    """Karaoke fill: words brighten as they are spoken (uses the real timings)."""
    del end
    rendered: list[str] = []
    for line in block:
        pieces: list[str] = []
        for word in line.words:
            duration_cs = max(4, int(round(word.duration * 100)))
            text = _escape(word.text)
            if theme.uppercase:
                text = text.upper()
            tag = "\\kf" if word.emphasis else "\\k"
            pieces.append(f"{{{tag}{duration_cs}}}{text}")
        rendered.append(" ".join(pieces))
    body = NEWLINE_TAG.join(rendered)
    return f"Dialogue: 0,{_timestamp(start)},{_timestamp(block[-1].end)},CF,,0,0,0,,{body}"


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #


def write_ass(plan: CaptionPlan, destination: Path, *, width: int, height: int, offset: float = 0.0) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        build_ass(plan, width=width, height=height, offset=offset),
        encoding="utf-8",
    )
    log.debug("wrote %d caption lines to %s", len(plan.lines), destination.name)
    return destination


def build_srt(plan: CaptionPlan, *, offset: float = 0.0) -> str:
    """Plain SRT export (for platforms that take their own subtitle upload)."""

    def stamp(seconds: float) -> str:
        seconds = max(0.0, seconds)
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        if millis >= 1000:
            millis = 999
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    rows: list[str] = []
    for index, line in enumerate(plan.lines, start=1):
        rows.append(str(index))
        rows.append(f"{stamp(line.start + offset)} --> {stamp(line.end + offset)}")
        rows.append(line.text)
        rows.append("")
    return "\n".join(rows)


def write_srt(plan: CaptionPlan, destination: Path, *, offset: float = 0.0) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_srt(plan, offset=offset), encoding="utf-8-sig")
    return destination


def plan_captions(
    words: Sequence[Any],
    *,
    theme: CaptionTheme,
    language: str = "en",
    emphasis_words: Sequence[str] = (),
    zoom_times: Sequence[float] = (),
    cues: Sequence[Any] | None = None,
) -> CaptionPlan:
    lines = build_cue_lines(cues) if cues is not None else build_lines(words, theme, emphasis_words=emphasis_words, zoom_times=zoom_times)
    font = resolve_font(theme, language)
    notes: list[str] = []
    if not lines:
        notes.append("No words to caption in this range.")
    if language.lower().startswith("hi") or "hinglish" in language.lower():
        notes.append("Devanagari-capable font selected so Hindi/Hinglish text renders correctly.")
    return CaptionPlan(lines=lines, theme=theme, language=language, font=font, notes=notes)


def preset_theme(preset: str, base: CaptionTheme | None = None) -> CaptionTheme:
    """Merge a named preset into a caption theme (used by the editor UI)."""
    theme = (base or CaptionTheme()).model_copy(deep=True)
    definition = CAPTION_PRESETS.get(preset)
    if not definition:
        return theme
    for key, value in dict(definition["theme"]).items():
        if hasattr(theme, key):
            setattr(theme, key, value)
    theme.preset = preset
    return theme


def ass_escape_filter_path(path: Path) -> str:
    """Escape a path for use inside an ffmpeg filter argument."""
    text = str(path).replace("\\", "/")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\\'")
    text = text.replace("[", "\\[").replace("]", "\\]")
    text = text.replace(",", "\\,")
    return text


__all__ = [
    "CaptionLine",
    "CaptionPlan",
    "CaptionWord",
    "ass_escape_filter_path",
    "ass_header",
    "build_ass",
    "build_cue_lines",
    "build_lines",
    "build_srt",
    "hex_to_ass_color",
    "plan_captions",
    "preset_theme",
    "resolve_font",
    "write_ass",
    "write_srt",
]
