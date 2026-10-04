"""Rendering: clip row → finished MP4 (plus preview, thumbnail and subtitles).

Rendering rebuilds the edit plan from the database (pace, framing, captions,
layout, assets), hands it to the ffmpeg composer and writes the result back onto
the clip row. Re-rendering after an edit therefore uses exactly the same code
path as the first render.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from ..config import AppSettings, get_settings
from ..db import Clip, Project, session_scope, utcnow
from ..errors import ClipForgeError, ErrorCode
from ..logging_setup import get_logger
from ..media.compose import RenderSpec, ZoomPoint, estimate_render_seconds, graph_summary, render as compose_render
from ..media.ffmpeg import make_thumbnail, probe_media
from ..media.framing import crop_plan_from_dict
from ..system import ensure_disk_space, ffmpeg_info
from .context import ProgressReporter, ProjectPaths, find_media
from .edit import layout_payload, rebuild_captions, rebuild_layout, resolve_asset

log = get_logger("clipforge.render")


def project_settings_for(project: Project, overrides: dict[str, Any] | None = None) -> AppSettings:
    snapshot = project.to_dict()
    from .analyze import project_settings_from_snapshot

    settings = project_settings_from_snapshot(snapshot)
    if overrides:
        merged = {**settings.model_dump(), **{key: value for key, value in overrides.items() if key in settings.model_dump()}}
        merged.update({key: value for key, value in overrides.items() if key == "caption"})
        settings = AppSettings.model_validate(merged)
    return settings


def load_clip_context(clip_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise ClipForgeError(code=ErrorCode.NOT_FOUND, message="Clip not found.", status_code=404)
        project = session.get(Project, clip.project_id)
        if project is None:
            raise ClipForgeError(code=ErrorCode.NOT_FOUND, message="Project for this clip no longer exists.", status_code=404)
        return clip.to_dict(include_words=False), project.to_dict()


def render_clip(
    clip_id: str,
    *,
    report: ProgressReporter | None = None,
    quality: str = "final",
    overrides: dict[str, Any] | None = None,
    export: bool = False,
    cancel_key: str = "",
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Render one clip (``quality='preview'`` produces a fast low-res version)."""
    from .context import NullReporter

    report = report or NullReporter()
    started = time.time()

    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise ClipForgeError(code=ErrorCode.NOT_FOUND, message="Clip not found.", status_code=404)
        project = session.get(Project, clip.project_id)
        if project is None:
            raise ClipForgeError(code=ErrorCode.NOT_FOUND, message="Project not found.", status_code=404)
        settings = project_settings_for(project, overrides)
        project_snapshot = project.to_dict()
        clip_status = clip.status
        paths = ProjectPaths.for_project(project).ensure()

    if clip_status == "rendering":
        raise ClipForgeError(
            code=ErrorCode.CONFLICT,
            message="This clip is already rendering.",
            hint="Wait for the current render to finish or cancel it from the queue.",
            status_code=409,
        )

    source_path = find_media(paths.source)
    if source_path is None:
        raise ClipForgeError(
            code=ErrorCode.NOT_FOUND,
            message="The source video for this project is missing.",
            hint="Re-run the analysis, or check that the data directory was not moved.",
            status_code=404,
        )

    if quality == "final":
        ensure_disk_space(1.5)

    _update_clip(
        clip_id,
        status="rendering" if quality == "final" else clip_status,
        progress=0.0,
        stage="preparing",
        error_code="",
        error_message="",
    )
    report.stage("prepare", "rebuilding the edit plan", fraction=0.05)

    try:
        spec = _build_spec_from_db(
            clip_id=clip_id,
            source_path=source_path,
            paths=paths,
            settings=settings,
            quality=quality,
            project_snapshot=project_snapshot,
        )
    except ClipForgeError as exc:
        _fail_clip(clip_id, exc, quality)
        raise

    plan_lines = graph_summary(spec)
    for line in plan_lines:
        report.log(line)
    report.stage("render", "encoding", fraction=0.1)

    def on_progress(fraction: float, message: str) -> None:
        if should_cancel and should_cancel():
            from ..media.runner import kill_process

            kill_process(cancel_key or clip_id)
        report.sub(max(0.0, min(1.0, fraction)), message)
        _update_clip(clip_id, progress=round(fraction, 3), stage=message[:40])

    try:
        stats = compose_render(spec, progress=on_progress)
    except ClipForgeError as exc:
        _fail_clip(clip_id, exc, quality)
        if exc.code == ErrorCode.RENDER_CANCELLED:
            raise ClipForgeError(code=ErrorCode.CANCELLED, message="Render cancelled.", status_code=409) from exc
        raise

    # --------------------------------------------------------------- finishing
    report.stage("thumbnail", "generating a thumbnail", fraction=0.9)
    thumbnail = None
    try:
        thumb_dir = paths.renders / "thumbnails"
        thumbnail = make_thumbnail(spec.output_path, thumb_dir / f"{clip_id}.jpg", time=min(1.5, stats["duration"] / 3), width=720)
    except Exception as exc:  # noqa: BLE001 - thumbnail is cosmetic
        log.debug("thumbnail failed: %s", exc)

    exported: Path | None = None
    if export and quality == "final":
        try:
            exported = _export_copy(spec.output_path, settings, clip_id)
            report.log(f"exported to {exported}")
        except OSError as exc:
            log.warning("export copy failed: %s", exc)
            report.log("Could not copy the clip to the export folder (see logs/app.log).")

    fields: dict[str, Any] = {
        "progress": 1.0,
        "stage": "rendered",
        "render_seconds": round(time.time() - started, 2),
        "file_size": stats["size_bytes"],
        "width": stats["width"],
        "height": stats["height"],
        "fps": int(round(stats["fps"] or spec.fps)),
        "subtitle_path": str(spec.subtitle_path) if spec.subtitle_path else "",
        "error_code": "",
        "error_message": "",
    }
    if quality == "final":
        fields.update({"status": "rendered", "file_path": str(spec.output_path), "rendered_at": utcnow()})
    else:
        fields.update({"preview_path": str(spec.output_path)})
    if thumbnail is not None:
        fields["thumb_path"] = str(thumbnail)
    _update_clip(clip_id, **fields)

    log.info(
        "rendered clip %s (%.1fs source -> %.1fs output in %.1fs, %.1f MB)",
        clip_id,
        spec.timeline.source_duration,
        stats["duration"],
        stats["encode_seconds"],
        stats["size_bytes"] / 1e6,
    )
    return {
        "clip_id": clip_id,
        "quality": quality,
        "output": str(spec.output_path),
        "exported_to": str(exported) if exported else "",
        "duration": stats["duration"],
        "width": stats["width"],
        "height": stats["height"],
        "fps": stats["fps"],
        "size_bytes": stats["size_bytes"],
        "encode_seconds": stats["encode_seconds"],
        "render_seconds": round(time.time() - started, 2),
        "speed_ratio": stats["speed_ratio"],
        "plan": plan_lines,
        "notes": stats.get("notes", []),
    }


def _build_spec_from_db(
    *,
    clip_id: str,
    source_path: Path,
    paths: ProjectPaths,
    settings: AppSettings,
    quality: str,
    project_snapshot: dict[str, Any],
) -> RenderSpec:
    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise ClipForgeError(code=ErrorCode.NOT_FOUND, message="Clip not found.", status_code=404)
        plan = rebuild_layout(clip, settings)
        layout_settings = layout_payload(plan)
        timeline = _load_timeline(clip, settings)
        zoom_points = [
            ZoomPoint(time=float(point.get("time", 0)), strength=float(point.get("strength", 1.0)), reason=str(point.get("reason", "")))
            for point in plan.get("zoom_points") or []
        ]
        captions = rebuild_captions(clip, settings, plan.get("language", settings.language_hint or "en"), [p.time for p in zoom_points])
        crop_plan = crop_plan_from_dict(plan.get("crop"))
        layout = str(layout_settings.get("layout") or settings.layout)
        split_ratio = int(layout_settings.get("split_ratio") or settings.split_ratio)
        gameplay = resolve_asset(layout_settings.get("gameplay"))
        broll = resolve_asset(layout_settings.get("broll"))
        music = resolve_asset(layout_settings.get("music"))
        title = clip.title
        index = clip.index

    source_fps = float(getattr(probe_media(source_path), "fps", 0.0) or 0.0)

    width, height = settings.aspect_dims()
    fps = settings.output_fps
    if quality == "preview":
        width, height = _preview_dims(width, height)
        fps = min(fps, 30)

    if quality == "preview":
        folder = paths.renders / "previews"
        output = folder / f"{clip_id}_preview.mp4"
    else:
        output = paths.render_path(index, title or f"clip_{index:02d}")
    subtitle_path = paths.clip_dir(index) / "captions.ass"

    return RenderSpec(
        source_path=source_path,
        output_path=output,
        timeline=timeline,
        layout=layout,
        split_ratio=split_ratio,
        width=width,
        height=height,
        fps=fps,
        crop_plan=crop_plan,
        captions=captions,
        gameplay_path=gameplay,
        broll_path=broll,
        music_path=music,
        zoom_points=zoom_points,
        source_fps=source_fps,
        language=plan.get("language", "en"),
        title=title,
        subtitle_path=subtitle_path,
        settings=settings,
        quality=quality,
        cancel_key=clip_id,
        notes=list(layout_settings.get("notes") or []),
    )


def _preview_dims(width: int, height: int) -> tuple[int, int]:
    scale = 480.0 / max(width, height)
    return max(2, int(round(width * scale / 2) * 2)), max(2, int(round(height * scale / 2) * 2))


def _load_timeline(clip: Clip, settings: AppSettings):
    from .edit import load_timeline

    return load_timeline(clip, settings)


def _export_copy(source: Path, settings: AppSettings, clip_id: str) -> Path:
    export_dir = settings.resolved_export_dir()
    export_dir.mkdir(parents=True, exist_ok=True)
    target = export_dir / source.name
    if target.exists() and target.resolve() != source.resolve():
        target = export_dir / f"{source.stem}_{clip_id[-4:]}{source.suffix}"
    if target.resolve() != source.resolve():
        shutil.copy2(source, target)
    return target


def _update_clip(clip_id: str, **fields: Any) -> None:
    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            return
        for key, value in fields.items():
            setattr(clip, key, value)


def _fail_clip(clip_id: str, error: ClipForgeError, quality: str) -> None:
    fields: dict[str, Any] = {
        "status": "failed" if quality == "final" else "pending",
        "stage": "failed",
        "error_code": error.code,
        "error_message": error.message,
    }
    _update_clip(clip_id, **fields)


def render_queue_estimate(clip_ids: list[str], settings: AppSettings | None = None) -> dict[str, Any]:
    """Rough ETA for a batch render (used by the queue UI)."""
    settings = settings or get_settings()
    total_seconds = 0.0
    with session_scope() as session:
        for clip_id in clip_ids:
            clip = session.get(Clip, clip_id)
            if clip is None:
                continue
            duration = clip.duration or settings.target_clip_seconds
            total_seconds += estimate_render_seconds(
                duration,
                settings.output_width,
                settings.output_height,
                hw_accel=settings.hw_accel != "none" and ffmpeg_info().available,
            )
    return {
        "clips": len(clip_ids),
        "estimated_seconds": round(total_seconds, 1),
        "encoder": "hardware" if settings.hw_accel != "none" else "cpu",
    }


__all__ = [
    "load_clip_context",
    "project_settings_for",
    "render_clip",
    "render_queue_estimate",
]
