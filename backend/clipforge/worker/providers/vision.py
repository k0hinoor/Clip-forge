"""Visual analysis providers (TRD §17).

Frames are *sampled* (default 2 fps at 320 px wide) by piping raw grayscale
frames out of FFmpeg — never every frame at full resolution.

* ``OpenCVVisionProvider`` — face detection (Haar cascade), scene changes, motion.
* ``FFmpegVisionProvider`` — no OpenCV: scene changes + motion only.
* ``NullVisionProvider`` — disabled.

Output per frame: ``{t, face_boxes: [[x, y, w, h] normalised], scene_id, motion_score}``.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Protocol

from clipforge.core.config import Settings
from clipforge.core.errors import AppError, ErrorCode
from clipforge.core.logging import get_logger
from clipforge.media.ffmpeg import resolve_binary
from clipforge.security.files import assert_safe_media_path

log = get_logger(__name__)
SCENE_THRESHOLD = 0.35


class VisionProvider(Protocol):
    name: str

    def analyze(self, video: Path, *, start: float, end: float, src_w: int, src_h: int,
                cancel_check: Callable[[], bool] | None = None) -> list[dict[str, Any]]: ...


def iter_gray_frames(ffmpeg_path: str, video: Path, *, start: float, end: float, fps: float, width: int,
                     height: int, timeout: float = 600,
                     cancel_check: Callable[[], bool] | None = None) -> Iterator[tuple[float, bytes]]:
    binary = resolve_binary(ffmpeg_path)
    src = assert_safe_media_path(video)
    cmd = [binary, "-hide_banner", "-nostdin", "-loglevel", "error", "-ss", f"{max(start, 0):.3f}",
           "-t", f"{max(end - start, 0.1):.3f}", "-i", str(src),
           "-vf", f"fps={fps:.3f},scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    frame_size = width * height
    started = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                            shell=False)
    try:
        idx = 0
        assert proc.stdout is not None
        while True:
            if cancel_check and cancel_check():
                raise AppError(ErrorCode.JOB_CANCELLED)
            if time.monotonic() - started > timeout:
                raise AppError(ErrorCode.UNKNOWN_ERROR, internal="frame extraction timeout")
            buf = proc.stdout.read(frame_size)
            if len(buf) < frame_size:
                break
            yield idx / fps, buf
            idx += 1
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)


def _dims(src_w: int, src_h: int, width: int) -> tuple[int, int]:
    width = min(width, src_w) // 2 * 2
    height = max(2, round(width * src_h / max(src_w, 1) / 2) * 2)
    return width, height


class NullVisionProvider:
    name = "none"

    def analyze(self, video: Path, *, start: float, end: float, src_w: int, src_h: int,
                cancel_check=None) -> list[dict[str, Any]]:
        return []


class FFmpegVisionProvider:
    name = "ffmpeg"

    def __init__(self, ffmpeg_path: str, fps: float) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.fps = fps

    def analyze(self, video: Path, *, start: float, end: float, src_w: int, src_h: int,
                cancel_check=None) -> list[dict[str, Any]]:
        w, h = _dims(src_w, src_h, 96)
        frames: list[dict[str, Any]] = []
        prev: bytes | None = None
        scene = 0
        step = 4  # subsample pixels for speed in pure Python
        for t, buf in iter_gray_frames(self.ffmpeg_path, video, start=start, end=end, fps=self.fps,
                                       width=w, height=h, cancel_check=cancel_check):
            motion = 0.0
            if prev is not None:
                diffs = [abs(buf[i] - prev[i]) for i in range(0, len(buf), step)]
                motion = sum(diffs) / (len(diffs) * 255.0)
                if motion > SCENE_THRESHOLD:
                    scene += 1
            frames.append({"t": round(t, 3), "face_boxes": [], "scene_id": scene, "motion_score": round(motion, 4)})
            prev = buf
        return frames


class OpenCVVisionProvider:
    name = "opencv"

    def __init__(self, ffmpeg_path: str, fps: float, width: int) -> None:
        import cv2  # noqa: F401  (ImportError → caller falls back)
        import numpy  # noqa: F401

        self.ffmpeg_path = ffmpeg_path
        self.fps = fps
        self.width = width
        self._cascade = None

    def _detector(self):
        import cv2

        if self._cascade is None:
            path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if not path.exists():
                raise AppError(ErrorCode.MODEL_UNAVAILABLE, internal="Haar cascade missing (use opencv<5)")
            self._cascade = cv2.CascadeClassifier(str(path))
        return self._cascade

    def analyze(self, video: Path, *, start: float, end: float, src_w: int, src_h: int,
                cancel_check=None) -> list[dict[str, Any]]:
        import cv2
        import numpy as np

        w, h = _dims(src_w, src_h, self.width)
        detector = self._detector()
        frames: list[dict[str, Any]] = []
        prev_hist = None
        prev_small = None
        scene = 0
        min_face = max(12, int(min(w, h) * 0.08))
        for t, buf in iter_gray_frames(self.ffmpeg_path, video, start=start, end=end, fps=self.fps,
                                       width=w, height=h, cancel_check=cancel_check):
            img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w)
            eq = cv2.equalizeHist(img)
            faces = detector.detectMultiScale(eq, scaleFactor=1.1, minNeighbors=5, minSize=(min_face, min_face))
            boxes = [[round(x / w, 4), round(y / h, 4), round(fw / w, 4), round(fh / h, 4)]
                     for (x, y, fw, fh) in (faces if len(faces) else [])]
            hist = cv2.calcHist([img], [0], None, [32], [0, 256])
            cv2.normalize(hist, hist)
            small = cv2.resize(img, (64, max(2, int(64 * h / w))))
            motion = 0.0
            if prev_hist is not None:
                dist = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                motion = float(np.mean(cv2.absdiff(small, prev_small))) / 255.0
                if dist > SCENE_THRESHOLD:
                    scene += 1
            frames.append({"t": round(t, 3), "face_boxes": boxes, "scene_id": scene,
                           "motion_score": round(motion, 4)})
            prev_hist, prev_small = hist, small
        return frames


def build_vision_provider(settings: Settings) -> VisionProvider:
    choice = settings.VISION_PROVIDER
    if choice == "none":
        return NullVisionProvider()
    if choice in ("auto", "opencv"):
        try:
            return OpenCVVisionProvider(settings.FFMPEG_PATH, settings.VISION_SAMPLE_FPS, settings.VISION_FRAME_WIDTH)
        except ImportError:
            if choice == "opencv":
                log.warning("OpenCV not installed; falling back to FFmpeg-only visual analysis")
    return FFmpegVisionProvider(settings.FFMPEG_PATH, settings.VISION_SAMPLE_FPS)


def visual_quality(frames: list[dict[str, Any]], framing_mode: str) -> float:
    """Visual signal for scoring: face presence, scene stability, moderate motion."""
    if not frames:
        return 0.5
    face_ratio = sum(1 for f in frames if f.get("face_boxes")) / len(frames)
    scenes = len({f.get("scene_id") for f in frames})
    cuts_per_10s = (scenes - 1) / max(frames[-1]["t"] - frames[0]["t"], 1.0) * 10
    stability = max(0.0, 1.0 - cuts_per_10s / 4)
    motion = sum(f.get("motion_score", 0.0) for f in frames) / len(frames)
    motion_fit = 1.0 - min(abs(motion - 0.05) / 0.2, 1.0)
    score = 0.45 * face_ratio + 0.35 * stability + 0.2 * motion_fit
    if framing_mode == "fit":
        score = max(score, 0.45)
    return round(max(0.0, min(1.0, score)), 4)
