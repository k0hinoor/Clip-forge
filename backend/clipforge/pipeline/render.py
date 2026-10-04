"""Rendering: clip row → finished MP4 (plus preview, thumbnail and subtitles).

Rendering rebuilds the edit plan from the database (pace, framing, captions,
layout, assets), hands it to the ffmpeg composer and writes the result back onto
the clip row. Re-rendering after an edit therefore uses exactly the same code
path as the first render.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from ..config import AppSettings, Env, describe_validation_error, get_settings, merge_settings_patch
from ..db import Clip, Project, session_scope, utcnow
from ..errors import ClipForgeError, ErrorCode
from ..logging_setup import get_logger
from ..media.compose import RenderSpec, ZoomPoint, estimate_render_seconds, graph_summary, render as compose_render
from ..media.ffmpeg import make_thumbnail, probe_media
from ..media.framing import crop_plan_from_dict
from ..system import ensure_disk_space, ffmpeg_info, open_in_file_manager
from .context import ProgressReporter, ProjectPaths, find_media
from .edit import clip_settings, layout_payload, rebuild_captions, rebuild_layout, resolve_asset

log = get_logger("clipforge.render")


def project_settings_for(project: Project, overrides: dict[str, Any] | None = None) -> AppSettings:
    """Project settings, plus one-off render ``overrides`` (unknown keys ignored)."""
    from .analyze import project_settings_from_snapshot

    settings = project_settings_from_snapshot({"settings": project.settings})
    if overrides:
        usable = {key: value for key, value in overrides.items() if key in AppSettings.model_fields}
        try:
            settings = AppSettings.model_validate(merge_settings_patch(settings.model_dump(), usable))
        except ValidationError as exc:
            raise ClipForgeError(
                code=ErrorCode.INVALID_INPUT,
                message=f"Invalid render override - {describe_validation_error(exc)}",
                status_code=422,
            ) from exc
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
        previous_file = clip.file_path or ""
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

    if quality == "final":
        _update_clip(clip_id, status="rendering", progress=0.0, stage="preparing", error_code="", error_message="")
    else:
        # A preview reports through its job; the clip keeps its own state.
        _update_clip(clip_id, error_code="", error_message="")
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

    if cancel_key:
        spec.cancel_key = cancel_key

    def on_progress(fraction: float, message: str) -> None:
        if should_cancel and should_cancel():
            from ..media.runner import kill_process

            # The encoder is registered under the spec's key; killing anything
            # else would leave it running.
            kill_process(spec.cancel_key)
        report.sub(max(0.0, min(1.0, fraction)), message)
        if quality == "final":
            _update_clip(clip_id, progress=round(fraction, 3), stage=message[:40])

    # Encode into a sibling ".part" file and swap it in only on success, so a
    # failed or cancelled re-render never destroys the clip's previous file.
    final_output = spec.output_path
    spec.output_path = final_output.with_name(f"{final_output.stem}.part{final_output.suffix}")
    try:
        stats = compose_render(spec, progress=on_progress)
        os.replace(spec.output_path, final_output)
    except ClipForgeError as exc:
        spec.output_path.unlink(missing_ok=True)
        if exc.code in {ErrorCode.RENDER_CANCELLED, ErrorCode.CANCELLED}:
            # The job layer restores the clip (previous file or pending).
            raise ClipForgeError(code=ErrorCode.CANCELLED, message="Render cancelled.", status_code=409) from exc
        _fail_clip(clip_id, exc, quality)
        raise
    except OSError as exc:
        spec.output_path.unlink(missing_ok=True)
        error = ClipForgeError(
            code=ErrorCode.RENDER_FAILED,
            message="The rendered clip could not be saved.",
            hint="Check that the data folder is writable and has free space.",
            detail=str(exc),
            status_code=500,
        )
        _fail_clip(clip_id, error, quality)
        raise error from exc
    spec.output_path = final_output
    stats["output"] = str(final_output)

    # --------------------------------------------------------------- finishing
    thumbnail = None
    # A low-res preview only provides the thumbnail while there is no final render.
    if quality == "final" or not previous_file:
        report.stage("thumbnail", "generating a thumbnail", fraction=0.9)
        try:
            thumb_dir = paths.renders / "thumbnails"
            thumbnail = make_thumbnail(spec.output_path, thumb_dir / f"{clip_id}.jpg", time=min(1.5, stats["duration"] / 3), width=720)
        except Exception as exc:  # noqa: BLE001 - thumbnail is cosmetic
            log.debug("thumbnail failed: %s", exc)

    if quality == "final":
        _remove_replaced_render(previous_file, spec.output_path, paths)

    exported: Path | None = None
    if export and quality == "final":
        try:
            exported = _export_copy(spec.output_path, settings, clip_id)
            report.log(f"exported to {exported}")
        except OSError as exc:
            log.warning("export copy failed: %s", exc)
            report.log("Could not copy the clip to the export folder (see logs/app.log).")
        if exported is not None and settings.auto_open_folder and Env.local_paths_allowed():
            try:
                open_in_file_manager(exported.parent)
            except ClipForgeError as exc:
                report.log(f"{exc.message} {exc.hint}".strip())

    fields: dict[str, Any] = {"error_code": "", "error_message": ""}
    if quality == "final":
        # Size, dimensions and timings describe the deliverable - a preview
        # (480p) must never overwrite them.
        fields.update(
            {
                "status": "rendered",
                "stage": "rendered",
                "progress": 1.0,
                "render_seconds": round(time.time() - started, 2),
                "file_size": stats["size_bytes"],
                "width": stats["width"],
                "height": stats["height"],
                "file_path": str(spec.output_path),
                "rendered_at": utcnow(),
                "fps": int(round(stats["fps"] or spec.fps)),
                "subtitle_path": str(spec.subtitle_path) if spec.subtitle_path else "",
            }
        )
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
        # Per-clip edits (caption style, captions on/off, aspect ratio, ...) live
        # in the stored plan and must reach the encoder.
        settings = clip_settings(settings, layout_settings)
        timeline = _load_timeline(clip, settings)
        # Punch-ins stay in the plan; "Automatic punch-ins" decides whether they render.
        zoom_points = [
            ZoomPoint(time=float(point.get("time", 0)), strength=float(point.get("strength", 1.0)), reason=str(point.get("reason", "")))
            for point in (plan.get("zoom_points") or layout_settings.get("zoom_points") or [])
        ] if settings.auto_zoom else []
        captions = rebuild_captions(clip, settings, plan.get("language", settings.language_hint or "en"), [p.time for p in zoom_points])
        crop_plan = crop_plan_from_dict(plan.get("crop"))
        layout = str(layout_settings.get("layout") or settings.layout)
        split_ratio = int(layout_settings.get("split_ratio") or settings.split_ratio)
        notes = list(layout_settings.get("notes") or [])
        gameplay = resolve_asset(layout_settings.get("gameplay"))
        if layout in {"split", "gameplay"} and not settings.gameplay_enabled:
            layout, gameplay = "podcast", None
            notes.append("Gameplay is switched off for this clip, so the speaker fills the frame.")
        broll = resolve_asset(layout_settings.get("broll"))
        music = resolve_asset(layout_settings.get("music"))
        title = clip.title
        index = clip.index
        previous_file = clip.file_path or ""

    source_fps = float(getattr(probe_media(source_path), "fps", 0.0) or 0.0)

    width, height = settings.aspect_dims()
    fps = settings.output_fps if settings.allow_60fps else min(settings.output_fps, 30)
    if quality == "preview":
        width, height = _preview_dims(width, height)
        fps = min(fps, 30)

    if quality == "preview":
        folder = paths.renders / "previews"
        output = folder / f"{clip_id}_preview.mp4"
    else:
        output = paths.render_path(
            index,
            title or f"clip_{index:02d}",
            project_title=str(project_snapshot.get("title") or ""),
            reuse=previous_file,
        )
    subtitle_path = paths.clip_dir(index) / ("preview.ass" if quality == "preview" else "captions.ass")

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
        notes=notes,
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
    """Record a render failure. A failed *preview* never changes the clip's status."""
    if quality == "final":
        _update_clip(clip_id, status="failed", stage="failed", error_code=error.code, error_message=error.message)
    else:
        _update_clip(clip_id, error_code=error.code, error_message=f"Preview failed: {error.message}")


def _remove_replaced_render(previous: str, current: Path, paths: ProjectPaths) -> None:
    """Delete the clip's old render once a new one (e.g. after a rename) replaced it."""
    if not previous:
        return
    old = Path(previous)
    try:
        if old.resolve() == current.resolve() or not old.is_file():
            return
        old.resolve().relative_to(paths.renders.resolve())  # only ever inside renders/
    except (OSError, ValueError):
        return
    try:
        old.unlink()
        log.info("removed replaced render %s", old.name)
    except OSError as exc:
        log.warning("could not remove the replaced render %s: %s", old, exc)


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
