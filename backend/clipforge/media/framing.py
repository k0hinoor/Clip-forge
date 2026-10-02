"""Smart reframing (TRD §18; PRD §10).

1. determine source dimensions
2. determine target crop
3. detect faces/subjects (from sampled-frame visual analysis)
4. generate crop trajectory
5. smooth trajectory
6. render via FFmpeg (piecewise-linear crop expressions)

The resulting plan is stored on the clip so renders are reproducible.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

FRAMING_VERSION = "1"
MAX_KEYFRAMES = 30


def parse_aspect(aspect: str) -> tuple[int, int]:
    a, b = aspect.split(":")
    return int(a), int(b)


def _even(x: float) -> int:
    return max(2, int(x) // 2 * 2)


def cover_crop(src_w: int, src_h: int, aspect: str) -> tuple[int, int]:
    """Largest crop of the target aspect ratio that fits inside the source."""
    aw, ah = parse_aspect(aspect)
    target = aw / ah
    if src_w / src_h > target:
        return _even(src_h * target), _even(src_h)
    return _even(src_w), _even(src_w / target)


def _primary_face(frame: dict[str, Any]) -> list[float] | None:
    faces = frame.get("face_boxes") or []
    if not faces:
        return None
    return max(faces, key=lambda b: b[2] * b[3])  # normalised [x, y, w, h]


def smooth_positions(samples: Sequence[tuple[float, float]], *, crop_size: float, max_speed: float,
                     dead_zone: float, alpha: float = 0.35) -> list[tuple[float, float]]:
    """Dead-zone + exponential smoothing + speed clamp for a 1-D crop offset."""
    out: list[tuple[float, float]] = []
    pos: float | None = None
    last_t = 0.0
    for t, target in samples:
        if pos is None:
            pos = target
        else:
            if abs(target - pos) > dead_zone * crop_size:
                desired = pos + alpha * (target - pos)
                limit = max_speed * crop_size * max(t - last_t, 1e-3)
                pos = pos + max(-limit, min(limit, desired - pos))
        out.append((t, pos))
        last_t = t
    return out


def _reduce(points: list[tuple[float, float, float]], tolerance: float) -> list[tuple[float, float, float]]:
    if len(points) <= 2:
        return points
    kept = [points[0]]
    for p in points[1:-1]:
        if abs(p[1] - kept[-1][1]) > tolerance or abs(p[2] - kept[-1][2]) > tolerance:
            kept.append(p)
    kept.append(points[-1])
    if len(kept) > MAX_KEYFRAMES:
        step = (len(kept) - 1) / (MAX_KEYFRAMES - 1)
        kept = [kept[round(i * step)] for i in range(MAX_KEYFRAMES)]
    return kept


def plan_framing(
    *,
    src_w: int,
    src_h: int,
    out_w: int,
    out_h: int,
    aspect: str,
    mode: str,
    frames: Sequence[dict[str, Any]] | None,
    duration: float,
) -> dict[str, Any]:
    crop_w, crop_h = cover_crop(src_w, src_h, aspect)
    needs_crop = crop_w < src_w - 2 or crop_h < src_h - 2
    frames = list(frames or [])
    with_face = [f for f in frames if f.get("face_boxes")]
    face_ratio = len(with_face) / len(frames) if frames else 0.0
    resolved = mode
    if mode == "auto":
        if not needs_crop or not frames:
            resolved = "center"
        elif face_ratio >= 0.3:
            resolved = "face"
        elif face_ratio > 0:
            resolved = "center"
        else:
            # Screen recordings / gameplay: avoid destructive cropping.
            resolved = "fit"
    if mode == "face" and not with_face:
        resolved = "center"

    plan: dict[str, Any] = {
        "version": FRAMING_VERSION, "mode": resolved, "requested_mode": mode, "aspect_ratio": aspect,
        "src_w": src_w, "src_h": src_h, "crop_w": crop_w, "crop_h": crop_h, "out_w": out_w, "out_h": out_h,
        "face_ratio": round(face_ratio, 3), "keyframes": [],
    }
    cx, cy = (src_w - crop_w) / 2, (src_h - crop_h) / 2
    if resolved == "fit":
        return plan
    if resolved == "center" or not needs_crop:
        plan["keyframes"] = [{"t": 0.0, "x": round(cx), "y": round(cy)}]
        return plan

    # Face tracking: target offsets per sampled frame; hold last known when no face.
    samples_x: list[tuple[float, float]] = []
    samples_y: list[tuple[float, float]] = []
    last_x, last_y = cx, cy
    for f in sorted(frames, key=lambda fr: fr["t"]):
        face = _primary_face(f)
        if face is not None:
            fx = (face[0] + face[2] / 2) * src_w
            fy = (face[1] + face[3] * 0.4) * src_h  # keep eyes in upper part (headroom)
            last_x = min(max(fx - crop_w / 2, 0), src_w - crop_w)
            last_y = min(max(fy - crop_h * 0.4, 0), src_h - crop_h)
        samples_x.append((float(f["t"]), last_x))
        samples_y.append((float(f["t"]), last_y))
    sx = smooth_positions(samples_x, crop_size=crop_w, max_speed=0.6, dead_zone=0.08)
    sy = smooth_positions(samples_y, crop_size=crop_h, max_speed=0.6, dead_zone=0.08)
    points = [(t, x, y) for (t, x), (_, y) in zip(sx, sy, strict=True)]
    if points and points[0][0] > 0:
        points.insert(0, (0.0, points[0][1], points[0][2]))
    if points and points[-1][0] < duration:
        points.append((round(duration, 3), points[-1][1], points[-1][2]))
    points = _reduce(points, tolerance=max(2.0, 0.01 * crop_w))
    plan["keyframes"] = [{"t": round(t, 3),
                          "x": int(min(max(round(x), 0), src_w - crop_w)),
                          "y": int(min(max(round(y), 0), src_h - crop_h))} for t, x, y in points]
    return plan


def piecewise_expr(keyframes: Sequence[dict[str, Any]], axis: str) -> str:
    """Build a numeric-only FFmpeg expression interpolating ``axis`` over ``t``."""
    pts = [(float(k["t"]), float(k[axis])) for k in keyframes]
    if not pts:
        return "0"
    for t, v in pts:
        if not (0 <= t < 1e6 and -1 < v < 1e5):
            raise ValueError("Invalid keyframe values")
    expr = f"{pts[-1][1]:.1f}"
    for (t0, v0), (t1, v1) in reversed(list(zip(pts[:-1], pts[1:], strict=True))):
        if t1 <= t0:
            continue
        seg = f"{v0:.1f}+({v1 - v0:.1f})*(t-{t0:.3f})/{t1 - t0:.3f}"
        expr = f"if(lt(t\\,{t1:.3f})\\,{seg}\\,{expr})"
    if len(pts) == 1:
        return f"{pts[0][1]:.1f}"
    return expr
