"""Optional visual analysis.

CLIPFORGE is fully functional without a vision model: if OpenCV is not installed
the reframing engine falls back to a centre crop with motion awareness. When
OpenCV *is* available, this module measures real per-frame signals that the
framer uses:

* face boxes (Haar cascade that ships with OpenCV) → active-speaker framing,
* frame brightness and contrast → exposure-aware overlays,
* Laplacian sharpness → "is this a still frame or movement?",
* gradient-energy centre of mass → subject position when no face is found,
* scene-change score → camera-cut detection for smooth vs hard reframes.

A user-supplied vision model can be wired in later through ``vision_model`` in
settings; the interface here (``analyse_frames``) is intentionally narrow so
swapping the implementation does not touch the rest of the pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..config import AppSettings, get_settings
from ..logging_setup import get_logger
from ..system import ai_stack

log = get_logger("clipforge.ai")


@dataclass
class FrameAnalysis:
    time: float
    width: int
    height: int
    faces: list[tuple[int, int, int, int]] = field(default_factory=list)  # x, y, w, h
    brightness: float = 0.5
    sharpness: float = 0.0
    subject_x: float = 0.5   # normalised horizontal centre of visual interest
    subject_y: float = 0.45
    scene_change: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": round(self.time, 3),
            "faces": [list(face) for face in self.faces],
            "brightness": round(self.brightness, 3),
            "sharpness": round(self.sharpness, 1),
            "subject_x": round(self.subject_x, 4),
            "subject_y": round(self.subject_y, 4),
            "scene_change": round(self.scene_change, 3),
        }


def vision_available() -> bool:
    return bool(ai_stack().get("opencv"))


def analyse_frames(
    frames: Sequence[tuple[float, Path]],
    *,
    previous: FrameAnalysis | None = None,
) -> list[FrameAnalysis]:
    """Analyse grabbed frames. Returns an empty list when OpenCV is unavailable."""
    if not vision_available():
        return []
    try:
        import cv2  # type: ignore
    except ImportError:  # pragma: no cover - guarded above
        return []

    cascade = _face_cascade(cv2)
    results: list[FrameAnalysis] = []
    last_gray = None
    last_analysis = previous

    for timestamp, path in frames:
        try:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        except Exception as exc:  # noqa: BLE001 - unreadable frame is skippable
            results.append(FrameAnalysis(time=timestamp, width=0, height=0, error=str(exc)))
            continue
        if image is None:
            continue

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (0, 0), fx=0.35, fy=0.35) if width > 480 else gray

        faces: list[tuple[int, int, int, int]] = []
        if cascade is not None:
            detected = cascade.detectMultiScale(small, scaleFactor=1.12, minNeighbors=5, minSize=(24, 24))
            scale = width / small.shape[1] if small.shape[1] else 1.0
            for (x, y, w, h) in detected:
                faces.append((int(x * scale), int(y * scale), int(w * scale), int(h * scale)))
            faces.sort(key=lambda box: -box[2] * box[3])

        brightness = float(np.mean(gray) / 255.0)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        subject_x, subject_y = _subject_position(cv2, gray, width, height)
        scene_change = _scene_change(cv2, gray, last_gray)
        last_gray = gray

        results.append(
            FrameAnalysis(
                time=timestamp,
                width=width,
                height=height,
                faces=faces[:4],
                brightness=brightness,
                sharpness=sharpness,
                subject_x=subject_x,
                subject_y=subject_y,
                scene_change=scene_change,
            )
        )
        last_analysis = results[-1]

    if not vision_available():
        return []
    return results


def _face_cascade(cv2: Any) -> Any:
    try:
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if path.exists():
            cascade = cv2.CascadeClassifier(str(path))
            if not cascade.empty():
                return cascade
    except Exception:  # noqa: BLE001
        pass
    return None


def _subject_position(cv2: Any, gray: np.ndarray, width: int, height: int) -> tuple[float, float]:
    """Gradient-energy centre of mass, centre-biased (a decent proxy when no face is found)."""
    if gray.size == 0:
        return 0.5, 0.45
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = np.abs(gradient_x) + np.abs(gradient_y)
    energy = cv2.GaussianBlur(energy, (0, 0), sigmaX=max(width / 60.0, 1.0))

    ys = np.linspace(0.0, 1.0, energy.shape[0])[:, None]
    xs = np.linspace(0.0, 1.0, energy.shape[1])[None, :]
    # Centre bias keeps a talking-head framing stable rather than chasing edges.
    bias = np.exp(-((xs - 0.5) ** 2) / 0.08 - ((ys - 0.42) ** 2) / 0.14)
    weighted = energy * bias
    total = float(weighted.sum())
    if total <= 1e-6:
        return 0.5, 0.45
    subject_x = float((weighted * xs).sum() / total)
    subject_y = float((weighted * ys).sum() / total)
    return subject_x, subject_y


def _scene_change(cv2: Any, gray: np.ndarray, previous: np.ndarray | None) -> float:
    if previous is None or previous.shape != gray.shape:
        return 0.0
    diff = cv2.absdiff(gray, previous)
    return float(np.mean(diff) / 255.0)


# --------------------------------------------------------------------------- #
# Interpretations used by the framer
# --------------------------------------------------------------------------- #


def face_center_x(analysis: FrameAnalysis) -> float | None:
    """Normalised x centre of the primary face (largest box), if any."""
    if not analysis.faces or not analysis.width:
        return None
    x, _, w, _ = analysis.faces[0]
    return (x + w / 2.0) / analysis.width


def face_span(analysis: FrameAnalysis) -> tuple[float, float] | None:
    """Horizontal extent covered by all detected faces (normalised)."""
    if not analysis.faces or not analysis.width:
        return None
    left = min(box[0] for box in analysis.faces) / analysis.width
    right = max(box[0] + box[2] for box in analysis.faces) / analysis.width
    return left, right


def multi_speaker_layout(analysis: FrameAnalysis) -> str:
    """Classify the shot so the framer can choose a strategy."""
    if not analysis.faces:
        return "no_faces"
    if len(analysis.faces) >= 2:
        span = face_span(analysis)
        if span and (span[1] - span[0]) > 0.55:
            return "wide_two_shot"
        return "two_shot"
    span = face_span(analysis)
    if span and (span[1] - span[0]) < 0.22:
        return "close_up"
    return "single"


def summarise(analyses: Sequence[FrameAnalysis]) -> dict[str, Any]:
    if not analyses:
        return {"frames": 0, "vision": False}
    with_faces = [analysis for analysis in analyses if analysis.faces]
    brightness = [analysis.brightness for analysis in analyses]
    scenes = [analysis.scene_change for analysis in analyses]
    return {
        "frames": len(analyses),
        "vision": True,
        "frames_with_faces": len(with_faces),
        "face_ratio": round(len(with_faces) / len(analyses), 3),
        "mean_brightness": round(float(np.mean(brightness)), 3) if brightness else 0.5,
        "mean_scene_change": round(float(np.mean(scenes)), 3) if scenes else 0.0,
        "cuts": sum(1 for value in scenes if value > 0.25),
        "shot_type": multi_speaker_layout(analyses[len(analyses) // 2]) if analyses else "unknown",
    }


def sample_times(start: float, end: float, *, interval: float = 2.0, max_samples: int = 60) -> list[float]:
    duration = max(0.0, end - start)
    if duration <= 0:
        return []
    count = max(1, min(int(duration / max(interval, 0.25)), max_samples))
    step = duration / count
    return [start + step * index + step / 2 for index in range(count)]


__all__ = [
    "FrameAnalysis",
    "analyse_frames",
    "face_center_x",
    "face_span",
    "multi_speaker_layout",
    "sample_times",
    "summarise",
    "vision_available",
]
