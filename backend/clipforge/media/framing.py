"""Smart reframing: horizontal footage → vertical short, without decapitating anyone.

The framer converts visual analysis (:mod:`clipforge.ai.vision`) into a *crop
plan*: a small number of horizontal centre keyframes plus a smooth ffmpeg
expression that interpolates between them. Faces win over motion, motion wins
over a plain centre crop, and when two people are too far apart to fit in a 9:16
window the framer recommends the blurred-background layout instead of pretending
a crop would work.

Everything degrades gracefully: with no OpenCV and no frames, the plan is a
static centre crop - the same thing every other clipper does - and the notes say
so explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..logging_setup import get_logger
from ..ai.vision import FrameAnalysis, face_center_x, face_span, multi_speaker_layout

log = get_logger("clipforge.render")

MIN_CROP_CONFIDENCE = 0.25


@dataclass
class CropKeyframe:
    time: float      # seconds, relative to the clip
    center_x: float  # 0..1 in source coordinates

    def to_dict(self) -> dict[str, float]:
        return {"time": round(self.time, 3), "x": round(self.center_x, 4)}


@dataclass
class CropPlan:
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    crop_width: int
    crop_height: int
    mode: str = "static"          # static | tracked | fit_blur
    keyframes: list[CropKeyframe] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    face_ratio: float = 0.0
    shot_type: str = "unknown"

    @property
    def is_tracked(self) -> bool:
        return self.mode == "tracked" and len(self.keyframes) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "source": [self.source_width, self.source_height],
            "target": [self.target_width, self.target_height],
            "crop": [self.crop_width, self.crop_height],
            "keyframes": [keyframe.to_dict() for keyframe in self.keyframes[:40]],
            "notes": self.notes,
            "face_ratio": round(self.face_ratio, 3),
            "shot_type": self.shot_type,
        }

    # ------------------------------------------------------------------ ffmpeg
    def crop_x_expression(self) -> str:
        """Piecewise-linear ``x`` expression for ffmpeg's crop filter (in source px)."""
        max_x = max(0, self.source_width - self.crop_width)
        if max_x == 0:
            return "0"
        if not self.keyframes:
            return str(max_x // 2)

        points = [
            (keyframe.time, float(np.clip(keyframe.center_x * self.source_width - self.crop_width / 2.0, 0, max_x)))
            for keyframe in self.keyframes
        ]
        if len(points) == 1:
            return f"{points[0][1]:.2f}"

        expression = f"{points[-1][1]:.2f}"
        for index in range(len(points) - 1, 0, -1):
            t0, x0 = points[index - 1]
            t1, x1 = points[index]
            span = max(t1 - t0, 1e-3)
            # Hold the first value before the first keyframe and interpolate after.
            expression = f"if(lt(t,{t1:.3f}),{x0:.2f}+({x1 - x0:.2f})*(t-{t0:.3f})/{span:.3f},{expression})"
        return f"if(lt(t,{points[0][0]:.3f}),{points[0][1]:.2f},{expression})"

    def crop_y_expression(self) -> str:
        max_y = max(0, self.source_height - self.crop_height)
        if max_y == 0:
            return "0"
        # Bias slightly above centre: faces usually sit in the upper third.
        return str(int(max_y * 0.42))


# --------------------------------------------------------------------------- #
# Plan builders
# --------------------------------------------------------------------------- #


def plan_crop(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    analyses: Sequence[FrameAnalysis],
    *,
    duration: float,
    smart: bool = True,
    tracking: bool = True,
) -> CropPlan:
    """Decide how to cut the source frame for the target aspect ratio."""
    target_ratio = target_width / max(target_height, 1)
    source_ratio = source_width / max(source_height, 1)

    if source_ratio <= target_ratio + 0.02:
        # Source is already as tall (or taller) than the target: no horizontal crop.
        plan = CropPlan(
            source_width=source_width,
            source_height=source_height,
            target_width=target_width,
            target_height=target_height,
            crop_width=source_width,
            crop_height=source_height,
            mode="static",
            notes=["Source is already vertical or square - the frame is scaled, not cropped."],
        )
        return plan

    crop_height = source_height
    crop_width = int(round(crop_height * target_ratio))
    crop_width = min(crop_width, source_width, target_width * 4)

    plan = CropPlan(
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
        crop_width=crop_width,
        crop_height=crop_height,
    )

    if not analyses:
        plan.mode = "static"
        plan.notes.append("No frame analysis available - using a centre crop.")
        return plan

    usable = [analysis for analysis in analyses if analysis.width and not analysis.error]
    if not usable:
        plan.mode = "static"
        plan.notes.append("Frame analysis produced no usable frames - using a centre crop.")
        return plan

    plan.face_ratio = sum(1 for analysis in usable if analysis.faces) / len(usable)
    plan.shot_type = multi_speaker_layout(usable[len(usable) // 2])

    # Two speakers too far apart for a 9:16 crop → suggest the blur layout.
    spans = [face_span(analysis) for analysis in usable]
    spans = [span for span in spans if span is not None]
    if spans:
        widest = max(span[1] - span[0] for span in spans)
        if widest * source_width > crop_width * 1.06:
            plan.mode = "fit_blur"
            plan.notes.append(
                "Two speakers are further apart than a 9:16 crop can hold - "
                "the blurred-background layout keeps both of them visible."
            )
            return plan

    if not smart:
        plan.mode = "static"
        plan.notes.append("Smart reframing is disabled in settings - using a centre crop.")
        return plan

    centers = [_center_for(analysis, tracking) for analysis in usable]
    times = [analysis.time for analysis in usable]
    centers = _smooth(centers, window=3)

    # Static when the subject barely moves; tracked when it does.
    spread = float(np.max(centers) - np.min(centers)) if centers else 0.0
    if spread < 0.045 or len(centers) < 3:
        plan.mode = "static"
        plan.keyframes = [CropKeyframe(time=0.0, center_x=float(np.median(centers)) if centers else 0.5)]
        if plan.face_ratio > 0.3:
            plan.notes.append(f"Steady framing on the speaker ({int(plan.face_ratio * 100)}% of sampled frames contain faces).")
        return plan

    # Keyframes at most one per 1.5s: smooth movement instead of jitter.
    step = max(1, int(round(len(centers) / max(duration / 1.5, 1))))
    keyframes = [
        CropKeyframe(time=times[index] - times[0], center_x=float(np.clip(centers[index], 0.0, 1.0)))
        for index in range(0, len(centers), step)
    ]
    if keyframes and keyframes[-1].time < duration:
        keyframes.append(
            CropKeyframe(time=duration, center_x=float(np.clip(centers[-1], 0.0, 1.0)))
        )
    plan.mode = "tracked" if tracking else "static"
    plan.keyframes = keyframes if tracking else [CropKeyframe(time=0.0, center_x=0.5)]
    plan.notes.append(
        f"Speaker tracking enabled: {len(plan.keyframes)} smooth framing keyframes across {duration:.0f}s."
        if tracking
        else "Speaker tracking is off in settings - static framing."
    )
    return plan


def _center_for(analysis: FrameAnalysis, tracking: bool) -> float:
    if not tracking:
        return 0.5
    center = face_center_x(analysis)
    if center is not None:
        # Bias toward keeping the body in frame, not just the face.
        return float(np.clip(center, 0.15, 0.85))
    return float(np.clip(analysis.subject_x, 0.2, 0.8))


def _smooth(values: Sequence[float], *, window: int = 3) -> list[float]:
    """Median filter + exponential smoothing: removes detector flicker."""
    if not values:
        return []
    array = np.asarray(values, dtype=np.float64)
    if array.size >= window:
        padded = np.pad(array, (window // 2, window // 2), mode="edge")
        filtered = np.array([np.median(padded[index: index + window]) for index in range(array.size)])
    else:
        filtered = array

    smoothed: list[float] = []
    previous = float(filtered[0])
    alpha = 0.45
    for value in filtered:
        previous = alpha * float(value) + (1 - alpha) * previous
        smoothed.append(previous)
    return smoothed


def crop_plan_from_dict(payload: dict[str, Any] | None) -> CropPlan | None:
    """Rebuild a stored crop plan (used when re-rendering an existing clip)."""
    if not payload:
        return None
    try:
        plan = CropPlan(
            source_width=int(payload.get("source", [0, 0])[0]),
            source_height=int(payload.get("source", [0, 0])[1]),
            target_width=int(payload.get("target", [0, 0])[0]),
            target_height=int(payload.get("target", [0, 0])[1]),
            crop_width=int(payload.get("crop", [0, 0])[0]),
            crop_height=int(payload.get("crop", [0, 0])[1]),
            mode=str(payload.get("mode", "static")),
            face_ratio=float(payload.get("face_ratio", 0.0)),
            shot_type=str(payload.get("shot_type", "unknown")),
            notes=list(payload.get("notes") or []),
        )
    except (TypeError, ValueError, IndexError):
        return None
    plan.keyframes = [
        CropKeyframe(time=float(item.get("time", 0.0)), center_x=float(item.get("x", 0.5)))
        for item in payload.get("keyframes") or []
    ]
    if plan.crop_width <= 0 or plan.crop_height <= 0:
        return None
    return plan


def fit_crop_to_panel(plan: CropPlan | None, panel_width: int, panel_height: int) -> CropPlan | None:
    """Return ``plan`` with its crop window re-cut for a ``panel_width x panel_height`` panel.

    Crop plans are stored per clip and may have been computed for another
    panel: the full 9:16 frame when the clip is now a split screen, or a ratio
    the user has since changed. The subject keyframes stay valid (they are
    relative positions), so only the window size is recomputed: full height
    when the source is wider than the panel, full width when it is taller.
    """
    if plan is None or plan.source_width <= 0 or plan.source_height <= 0 or panel_width <= 0 or panel_height <= 0:
        return plan
    target_ratio = panel_width / panel_height
    if plan.source_width / plan.source_height > target_ratio:
        crop_height = plan.source_height
        crop_width = min(plan.source_width, int(round(crop_height * target_ratio)))
    else:
        crop_width = plan.source_width
        crop_height = min(plan.source_height, int(round(crop_width / target_ratio)))
    crop_width = max(2, crop_width - crop_width % 2)
    crop_height = max(2, crop_height - crop_height % 2)
    if (crop_width, crop_height, panel_width, panel_height) == (plan.crop_width, plan.crop_height, plan.target_width, plan.target_height):
        return plan
    return CropPlan(
        source_width=plan.source_width,
        source_height=plan.source_height,
        target_width=panel_width,
        target_height=panel_height,
        crop_width=crop_width,
        crop_height=crop_height,
        mode=plan.mode,
        keyframes=list(plan.keyframes),
        notes=list(plan.notes),
        face_ratio=plan.face_ratio,
        shot_type=plan.shot_type,
    )


def plan_for_layout(
    layout: str,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    analyses: Sequence[FrameAnalysis],
    *,
    duration: float,
    smart: bool = True,
    tracking: bool = True,
) -> CropPlan:
    """Layout-aware wrapper: some layouts do not need a crop at all."""
    if layout in {"blur", "cinematic"}:
        plan = plan_crop(
            source_width, source_height, target_width, target_height, analyses,
            duration=duration, smart=smart, tracking=tracking,
        )
        if layout == "blur":
            plan.mode = "fit_blur"
            plan.notes = ["Blurred background layout: the full source frame stays visible."]
        return plan

    plan = plan_crop(
        source_width, source_height, target_width, target_height, analyses,
        duration=duration, smart=smart, tracking=tracking,
    )
    if plan.mode == "fit_blur" and layout in {"split", "podcast"}:
        # Split layouts pan-and-scan the podcast panel using the subject position.
        plan.mode = "tracked" if plan.keyframes else "static"
        if not plan.keyframes:
            plan.keyframes = [CropKeyframe(time=0.0, center_x=0.5)]
        plan.notes.append("Wide two-shot: each speaker is followed with a gentle pan inside the podcast panel.")
    return plan


__all__ = ["CropKeyframe", "CropPlan", "crop_plan_from_dict", "fit_crop_to_panel", "plan_crop", "plan_for_layout"]
