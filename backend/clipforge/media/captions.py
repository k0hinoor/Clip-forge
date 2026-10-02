"""Caption engine (TRD §19; PRD §9).

Input: transcript words, clip start/end, style preset.
Output: a caption timeline (JSON-serialisable) and an ASS subtitle document.
Rendering is deterministic from the saved configuration.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from clipforge.scoring.segmentation import FlatWord
from clipforge.scoring.text import EMOTION_WORDS, HOOK_WORDS

MAX_CAPTION_SECONDS = 3.0
MIN_CAPTION_SECONDS = 0.4
GAP_BREAK_SECONDS = 0.5
_NUM_RE = re.compile(r"^\$?\d[\d,.%]*$")
_STRIP_RE = re.compile(r"^[^\w$]+|[^\w%]+$")


def _clean(word: str) -> str:
    return _STRIP_RE.sub("", word).lower()


def is_emphasis_word(word: str, extra_terms: set[str] | None = None) -> bool:
    w = _clean(word)
    if not w:
        return False
    return bool(_NUM_RE.match(w)) or w in HOOK_WORDS or w in EMOTION_WORDS or (extra_terms is not None and w in extra_terms)


def wrap_lines(words: Sequence[str], max_chars: int, max_lines: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for w in words:
        candidate = f"{current} {w}".strip()
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)
    if len(lines) > max_lines:  # balance into max_lines by merging the tail
        head, tail = lines[: max_lines - 1], lines[max_lines - 1:]
        lines = [*head, " ".join(tail)]
    return lines


def build_caption_timeline(
    words: Sequence[FlatWord],
    clip_start: float,
    clip_end: float,
    preset: dict[str, Any],
    *,
    emphasis_terms: Iterable[str] = (),
) -> list[dict[str, Any]]:
    max_chars = int(preset.get("max_chars_per_line", 24))
    max_lines = int(preset.get("max_lines", 2))
    capacity = max_chars * max_lines
    use_emphasis = bool(preset.get("emphasis", False))
    extra = {t.lower() for t in emphasis_terms}
    duration = clip_end - clip_start

    clip_words = [w for w in words if w.end > clip_start + 0.05 and w.start < clip_end - 0.05]
    groups: list[list[FlatWord]] = []
    current: list[FlatWord] = []
    for w in clip_words:
        if current:
            text_len = len(" ".join(x.word for x in current)) + 1 + len(w.word)
            gap = w.start - current[-1].end
            sentence_end = bool(re.search(r"[.!?]$", current[-1].word)) and len(current) >= 2
            too_long = (w.end - current[0].start) > MAX_CAPTION_SECONDS
            if text_len > capacity or gap > GAP_BREAK_SECONDS or sentence_end or too_long:
                groups.append(current)
                current = []
        current.append(w)
    if current:
        groups.append(current)

    timeline: list[dict[str, Any]] = []
    for gi, g in enumerate(groups):
        start = max(0.0, g[0].start - clip_start)
        end = min(duration, g[-1].end - clip_start)
        next_start = (groups[gi + 1][0].start - clip_start) if gi + 1 < len(groups) else duration
        end = min(max(end, start + MIN_CAPTION_SECONDS), next_start, duration)
        if end - start < 0.05:
            continue
        tokens = [w.word for w in g]
        emphasis: list[str] = []
        if use_emphasis:
            emphasis = [_clean(t) for t in tokens if is_emphasis_word(t, extra)][:2]
        timeline.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "text": " ".join(tokens),
            "lines": wrap_lines(tokens, max_chars, max_lines),
            "emphasis": emphasis,
            "speaker_id": g[0].speaker_id,
        })
    return timeline


# ----------------------------------------------------------------------- ASS
def _ass_color(hex_rgb: str, alpha: int = 0) -> str:
    h = hex_rgb.lstrip("#")
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", h):
        raise ValueError(f"Invalid colour {hex_rgb!r}")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def _ass_time(t: float) -> str:
    cs = int(round(max(t, 0.0) * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape(text: str) -> str:
    # Neutralise ASS override syntax coming from transcript text.
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


# Average advance width (in em) of DejaVu Sans glyphs: (uppercase, bold) -> em.
CHAR_WIDTH_EM = {(True, True): 0.74, (True, False): 0.68, (False, True): 0.62, (False, False): 0.56}


def render_ass(timeline: Sequence[dict[str, Any]], preset: dict[str, Any], width: int, height: int,
               *, speaker_colors: dict[str, str] | None = None) -> str:
    font = re.sub(r"[^A-Za-z0-9 _-]", "", str(preset.get("font", "DejaVu Sans"))) or "DejaVu Sans"
    size = max(8, round(float(preset.get("size", 0.05)) * height))
    outline = round(float(preset.get("outline", 0.004)) * height, 1)
    shadow = round(float(preset.get("shadow", 0.0)) * height, 1)
    bold = -1 if preset.get("bold") else 0
    primary = _ass_color(preset.get("primary_color", "#FFFFFF"))
    outline_c = _ass_color(preset.get("outline_color", "#000000"))
    back = _ass_color(preset.get("box_color", preset.get("outline_color", "#000000")), alpha=0x60)
    border_style = 1
    if preset.get("box"):
        border_style = 3
        outline_c = _ass_color(preset.get("box_color", "#000000"), alpha=0x30)
        outline = round(0.012 * height, 1)
    margin_v = round(float(preset.get("position", 0.25)) * height)
    margin_lr = round(0.06 * width)
    # Size is a fraction of height, but the longest allowed line must also fit
    # horizontally (narrow 9:16 frames): shrink using an average glyph width.
    uppercase_flag = bool(preset.get("uppercase", False))
    glyph_em = CHAR_WIDTH_EM[(uppercase_flag, bool(preset.get("bold")))]
    max_chars = int(preset.get("max_chars_per_line", 24))
    fit = (width - 2 * margin_lr - 2 * outline) / (max_chars * glyph_em)
    size = max(8, min(size, int(fit)))
    emph_color = _ass_color(preset.get("emphasis_color", preset.get("primary_color", "#FFFFFF")))
    uppercase = bool(preset.get("uppercase", False))

    header = (
        "[Script Info]\n"
        "; Generated by ClipForge AI\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {int(width)}\n"
        f"PlayResY: {int(height)}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},{size},{primary},{primary},{outline_c},{back},{bold},0,0,0,100,100,0,0,"
        f"{border_style},{outline},{shadow},2,{margin_lr},{margin_lr},{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = []
    for cap in timeline:
        emph = set(cap.get("emphasis") or [])
        line_color = None
        if speaker_colors and cap.get("speaker_id") in speaker_colors:
            line_color = _ass_color(speaker_colors[cap["speaker_id"]])
        rendered_lines = []
        for line in cap.get("lines") or [cap["text"]]:
            parts = []
            for tok in line.split(" "):
                shown = _escape(tok.upper() if uppercase else tok)
                if emph and _clean(tok) in emph:
                    parts.append(f"{{\\c{emph_color}&}}{shown}{{\\r}}")
                else:
                    parts.append(shown)
            text = " ".join(parts)
            if line_color:
                text = f"{{\\c{line_color}&}}{text}"
            rendered_lines.append(text)
        events.append(f"Dialogue: 0,{_ass_time(cap['start'])},{_ass_time(cap['end'])},Default,,0,0,0,,"
                      + "\\N".join(rendered_lines))
    return header + "\n".join(events) + ("\n" if events else "")
