"""Edit timeline: silence removal and time remapping.

Short-form video lives and dies on pacing, but cutting audio also moves every
other time-based element: captions, zoom keyframes, gameplay loops. So the trim
step produces an explicit :class:`Timeline` (source interval → output interval)
and everything downstream maps through it. Nothing is estimated twice, and the
speech itself is never altered - only silence between sentences is compressed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..config import AppSettings, get_settings
from ..logging_setup import get_logger

log = get_logger("clipforge.render")

MIN_KEEP_SECONDS = 0.12
MAX_SEGMENTS = 90


@dataclass
class Segment:
    src_start: float
    src_end: float
    out_start: float
    out_end: float

    @property
    def src_duration(self) -> float:
        return max(0.0, self.src_end - self.src_start)

    @property
    def out_duration(self) -> float:
        return max(0.0, self.out_end - self.out_start)

    def to_dict(self) -> dict[str, float]:
        return {
            "src_start": round(self.src_start, 3),
            "src_end": round(self.src_end, 3),
            "out_start": round(self.out_start, 3),
            "out_end": round(self.out_end, 3),
        }


@dataclass
class Timeline:
    segments: list[Segment] = field(default_factory=list)
    speed: float = 1.0
    removed: list[tuple[float, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source_start: float = 0.0
    source_end: float = 0.0
    source_duration: float = 0.0

    @property
    def output_duration(self) -> float:
        if not self.segments:
            return self.source_duration / max(self.speed, 0.01)
        return self.segments[-1].out_end

    @property
    def removed_seconds(self) -> float:
        return sum(end - start for start, end in self.removed)

    @property
    def is_trimmed(self) -> bool:
        return len(self.segments) > 1

    def source_to_output(self, timestamp: float) -> float | None:
        """Map a source timestamp into the rendered clip; ``None`` if it was cut."""
        for segment in self.segments:
            if segment.src_start - 0.001 <= timestamp <= segment.src_end + 0.001:
                offset = min(max(timestamp - segment.src_start, 0.0), segment.src_duration)
                return segment.out_start + offset / max(self.speed, 0.01)
        return None

    def output_to_source(self, timestamp: float) -> float | None:
        for segment in self.segments:
            if segment.out_start <= timestamp <= segment.out_end + 0.001:
                offset = (timestamp - segment.out_start) * max(self.speed, 0.01)
                return segment.src_start + offset
        return None

    def map_words(self, words: Sequence[Any]) -> list[Any]:
        """Remap word timings, dropping words that fell inside removed silence."""
        mapped: list[Any] = []
        for word in words:
            start = self.source_to_output(word.start)
            end = self.source_to_output(word.end)
            if start is None and end is None:
                continue
            if start is None:
                start = end
            if end is None or end < start:
                end = start + max(0.08, (word.end - word.start) / max(self.speed, 0.01))
            word.start = float(start)
            word.end = float(end)
            mapped.append(word)
        return _repair_sequence(mapped)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": [segment.to_dict() for segment in self.segments],
            "speed": round(self.speed, 4),
            "output_duration": round(self.output_duration, 3),
            "removed_seconds": round(self.removed_seconds, 3),
            "trimmed": self.is_trimmed,
            "notes": self.notes,
        }


def _repair_sequence(words: list[Any]) -> list[Any]:
    for index in range(len(words) - 1):
        if words[index].end > words[index + 1].start:
            words[index].end = max(words[index].start + 0.05, words[index + 1].start)
    return words


# --------------------------------------------------------------------------- #
# Building the timeline
# --------------------------------------------------------------------------- #


def build_timeline(
    start: float,
    end: float,
    silences: Sequence[tuple[float, float]],
    *,
    settings: AppSettings | None = None,
    min_output_seconds: float = 0.0,
    max_output_seconds: float = 0.0,
) -> Timeline:
    """Create the keep/remove plan for one clip."""
    settings = settings or get_settings()
    timeline = Timeline(source_start=start, source_end=end, source_duration=max(0.0, end - start))

    if not settings.remove_silence or not silences:
        timeline.segments = [Segment(start, end, 0.0, end - start)]
        if not settings.remove_silence:
            timeline.notes.append("Silence removal is off - the clip keeps its natural pacing.")
        return timeline

    keeps = _keep_intervals(start, end, silences, padding=settings.keep_natural_pauses, max_cut=settings.silence_max_cut)
    if len(keeps) > MAX_SEGMENTS:
        keeps = _coarsen(keeps, MAX_SEGMENTS)
        timeline.notes.append("Long clip: silence removal was coarsened to keep the render fast.")

    if not keeps:
        timeline.segments = [Segment(start, end, 0.0, end - start)]
        timeline.notes.append("Silence detection returned nothing usable - keeping the original pacing.")
        return timeline

    total_keep = sum(stop - begin for begin, stop in keeps)
    if total_keep < 3.0:
        timeline.segments = [Segment(start, end, 0.0, end - start)]
        timeline.notes.append("Almost the whole clip is silence - nothing was cut.")
        return timeline

    speed = 1.0
    if max_output_seconds and total_keep > max_output_seconds * 1.02:
        speed = min(total_keep / max_output_seconds, 1.25)
        timeline.notes.append(f"Speed x{speed:.2f} applied to fit the maximum duration.")

    out_cursor = 0.0
    for begin, stop in keeps:
        duration = (stop - begin) / speed
        timeline.segments.append(Segment(begin, stop, out_cursor, out_cursor + duration))
        out_cursor += duration

    removed = _removed_from_keeps(start, end, keeps)
    timeline.removed = removed
    timeline.speed = speed

    trimmed = sum(stop - begin for begin, stop in removed)
    if trimmed > 0.3:
        timeline.notes.append(
            f"Removed {trimmed:.1f}s of silence in {len(removed)} cuts (threshold {settings.silence_min_duration:.2f}s)."
        )

    # Never let silence removal destroy the pacing floor.
    if min_output_seconds and timeline.output_duration < min_output_seconds * 0.85 and speed == 1.0:
        timeline.segments = [Segment(start, end, 0.0, end - start)]
        timeline.removed = []
        timeline.notes = ["Silence removal would have made the clip too short - keeping the original pacing."]
    return timeline


def _keep_intervals(
    start: float,
    end: float,
    silences: Sequence[tuple[float, float]],
    *,
    padding: float,
    max_cut: float,
) -> list[tuple[float, float]]:
    """Complement of the silence intervals, padded so speech never clips."""
    clip_start, clip_end = start, end
    cuts: list[tuple[float, float]] = []
    for silence_start, silence_end in sorted(silences):
        cut_start = min(max(silence_start, clip_start), clip_end)
        cut_end = min(max(silence_end, clip_start), clip_end)
        if cut_end - cut_start <= 0:
            continue
        if max_cut and cut_end - cut_start > max_cut:
            # Very long gap (e.g. a music break): keep a natural breath instead of a hard jump.
            keep_middle = (cut_start + cut_end) / 2
            cuts.append((keep_middle - max_cut / 2, keep_middle + max_cut / 2))
        else:
            cuts.append((cut_start, cut_end))

    keeps: list[tuple[float, float]] = []
    cursor = clip_start
    for cut_start, cut_end in cuts:
        if cut_start > cursor:
            keeps.append((cursor, cut_start))
        cursor = max(cursor, cut_end)
    if cursor < clip_end:
        keeps.append((cursor, clip_end))

    padded: list[tuple[float, float]] = []
    for begin, stop in keeps:
        if stop - begin < MIN_KEEP_SECONDS:
            continue
        padded.append((max(clip_start, begin - padding), min(clip_end, stop + padding)))

    merged: list[tuple[float, float]] = []
    for begin, stop in padded:
        if merged and begin <= merged[-1][1] + 0.02:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((begin, stop))
    return merged


def _removed_from_keeps(start: float, end: float, keeps: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    removed: list[tuple[float, float]] = []
    cursor = start
    for begin, stop in keeps:
        if begin - cursor > 0.05:
            removed.append((cursor, begin))
        cursor = stop
    if end - cursor > 0.05:
        removed.append((cursor, end))
    return removed


def _coarsen(keeps: list[tuple[float, float]], limit: int) -> list[tuple[float, float]]:
    """Merge the shortest gaps so the filter graph stays reasonable."""
    while len(keeps) > limit:
        gaps = [keeps[index + 1][0] - keeps[index][1] for index in range(len(keeps) - 1)]
        smallest = int(min(range(len(gaps)), key=lambda index: gaps[index]))
        keeps[smallest] = (keeps[smallest][0], keeps[smallest + 1][1])
        del keeps[smallest + 1]
    return keeps


def plan_from_silences(
    start: float,
    end: float,
    *,
    silence_db: float = -32.0,
    min_silence: float = 0.35,
    settings: AppSettings | None = None,
    min_output_seconds: float = 0.0,
    max_output_seconds: float = 0.0,
    detect=None,
    source_path: str | None = None,
) -> Timeline:
    """Detect silence in ``[start, end]`` and build the timeline for one clip."""
    settings = settings or get_settings()
    if not settings.remove_silence or not source_path:
        return build_timeline(start, end, [], settings=settings)

    detector = detect
    if detector is None:
        from .ffmpeg import detect_silence

        detector = detect_silence
    try:
        silences = detector(
            source_path,
            noise_db=silence_db,
            min_duration=min_silence,
            start=start,
            end=end,
        )
    except Exception as exc:  # noqa: BLE001 - silence detection is best-effort
        log.warning("silence detection failed for %.1f-%.1f: %s", start, end, exc)
        silences = []

    return build_timeline(
        start,
        end,
        silences,
        settings=settings,
        min_output_seconds=min_output_seconds,
        max_output_seconds=max_output_seconds,
    )


def merge_nearby_words(words: Sequence[Any], *, gap: float = 0.35) -> list[list[Any]]:
    """Group words into caption lines at natural pauses (used by the caption engine)."""
    lines: list[list[Any]] = []
    current: list[Any] = []
    for word in words:
        if current and word.start - current[-1].end > gap:
            lines.append(current)
            current = []
        current.append(word)
    if current:
        lines.append(current)
    return lines


__all__ = [
    "MAX_SEGMENTS",
    "Segment",
    "Timeline",
    "build_timeline",
    "merge_nearby_words",
    "plan_from_silences",
]
