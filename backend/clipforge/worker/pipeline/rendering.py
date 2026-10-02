"""Rendering (TRD §18-§22): framing plan + captions + FFmpeg render + QA + thumbnail.

``ClipRenderer`` is used both by the pipeline RENDER stage and by user-triggered
re-renders (PRD Flow B), which reuse the persisted transcript and analysis
instead of re-running the whole pipeline (TRD §49).
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from clipforge.core.config import Settings
from clipforge.core.errors import AppError, ErrorCode
from clipforge.core.logging import get_logger
from clipforge.core.presets import caption_presets
from clipforge.core.states import ClipStatus, RenderStatus
from clipforge.core.timeutil import hours_from_now, utcnow
from clipforge.db.models import Clip, Render, RenderProfile
from clipforge.db.session import session_scope
from clipforge.media.captions import build_caption_timeline, render_ass
from clipforge.media.ffmpeg import run_ffmpeg
from clipforge.media.framing import plan_framing
from clipforge.media.render import build_render_args, build_thumbnail_args
from clipforge.scoring.segmentation import FlatWord
from clipforge.security.files import clip_filename
from clipforge.services.clips import profile_for_aspect
from clipforge.storage import Keys, StorageProvider
from clipforge.worker.pipeline.base import PipelineContext, Stage, sha256_file
from clipforge.worker.pipeline.qa import validate_render_output
from clipforge.worker.providers.vision import VisionProvider

log = get_logger(__name__)


def profile_dict(p: RenderProfile) -> dict[str, Any]:
    return {"name": p.name, "version": p.version, "aspect_ratio": p.aspect_ratio, "width": p.width,
            "height": p.height, "fps": p.fps, "video_codec": p.video_codec, "audio_codec": p.audio_codec,
            "crf": p.crf, "preset": p.preset, "audio_bitrate_kbps": p.audio_bitrate_kbps}


@dataclass
class RenderInputs:
    source_path: Path
    media: dict[str, Any]  # probe dict (display_width/height, has audio, ...)
    words: Sequence[FlatWord]
    work_dir: Path


class ClipRenderer:
    def __init__(self, settings: Settings, storage: StorageProvider, session_factory: sessionmaker[Session],
                 vision: VisionProvider, cancel_check: Callable[[], bool] | None = None) -> None:
        self.settings = settings
        self.storage = storage
        self.session_factory = session_factory
        self.vision = vision
        self.cancel_check = cancel_check

    # ------------------------------------------------------------- planning
    def ensure_visual(self, clip: Clip, inputs: RenderInputs) -> list[dict[str, Any]]:
        if clip.visual and clip.visual.get("range") == [clip.start, clip.end]:
            return clip.visual.get("frames", [])
        frames = self.vision.analyze(inputs.source_path, start=clip.start, end=clip.end,
                                     src_w=inputs.media["display_width"], src_h=inputs.media["display_height"],
                                     cancel_check=self.cancel_check)
        clip.visual = {"provider": self.vision.name, "range": [clip.start, clip.end], "frames": frames}
        return frames

    def ensure_framing(self, clip: Clip, inputs: RenderInputs, profile: dict[str, Any]) -> dict[str, Any]:
        crop = clip.crop or {}
        if (crop.get("aspect_ratio") == clip.aspect_ratio and crop.get("requested_mode") == clip.framing_mode
                and crop.get("out_w") == profile["width"] and crop.get("range") == [clip.start, clip.end]):
            return crop
        frames = self.ensure_visual(clip, inputs)
        plan = plan_framing(src_w=inputs.media["display_width"], src_h=inputs.media["display_height"],
                            out_w=profile["width"], out_h=profile["height"], aspect=clip.aspect_ratio,
                            mode=clip.framing_mode, frames=frames, duration=clip.end - clip.start)
        plan["range"] = [clip.start, clip.end]
        clip.crop = plan
        return plan

    def ensure_captions(self, clip: Clip, words: Sequence[FlatWord]) -> list[dict[str, Any]]:
        cached = clip.captions or {}
        if cached.get("preset") == clip.caption_preset and cached.get("range") == [clip.start, clip.end]:
            return cached.get("items", [])
        preset = caption_presets(self.settings)[clip.caption_preset]
        items = build_caption_timeline(words, clip.start, clip.end, preset, emphasis_terms=clip.keywords or [])
        clip.captions = {"preset": clip.caption_preset, "range": [clip.start, clip.end], "items": items}
        return items

    # ------------------------------------------------------------- render
    def render(self, clip_id: str, render_id: str, inputs: RenderInputs,
               on_progress: Callable[[float], None] | None = None) -> dict[str, Any]:
        """Render one clip end-to-end. Returns a summary dict. Raises on final failure."""
        with session_scope(self.session_factory) as s:
            clip = s.get(Clip, clip_id)
            render = s.get(Render, render_id)
            if clip is None or render is None:
                raise AppError(ErrorCode.RENDER_FAILED, internal="clip/render vanished")
            profile_row = (s.get(RenderProfile, render.render_profile_id) if render.render_profile_id else None) \
                or profile_for_aspect(s, clip.aspect_ratio)
            if profile_row.aspect_ratio != clip.aspect_ratio:
                profile_row = profile_for_aspect(s, clip.aspect_ratio)
            profile = profile_dict(profile_row)
            render.render_profile_id = profile_row.id
            render.render_profile_name = profile_row.name
            render.render_profile_version = profile_row.version
            render.status = RenderStatus.RENDERING.value
            render.started_at = utcnow()
            clip.status = ClipStatus.RENDERING.value
            plan = self.ensure_framing(clip, inputs, profile)
            captions = self.ensure_captions(clip, inputs.words) if clip.captions_enabled else []
            snapshot = {
                "clip_id": clip.id, "job_id": clip.job_id, "rank": clip.rank, "slug": clip.slug,
                "start": clip.start, "end": clip.end, "aspect_ratio": clip.aspect_ratio,
                "caption_preset": clip.caption_preset, "captions_enabled": clip.captions_enabled,
                "framing": plan, "profile": profile, "previous_render_id": clip.current_render_id,
            }
            render.params = {k: v for k, v in snapshot.items() if k not in ("previous_render_id",)}

        out_dir = inputs.work_dir / "renders"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{render_id}.mp4"
        subs_path: Path | None = None
        if snapshot["captions_enabled"] and captions:
            preset = caption_presets(self.settings)[snapshot["caption_preset"]]
            subs_path = out_dir / f"{render_id}.ass"
            subs_path.write_text(render_ass(captions, preset, profile["width"], profile["height"]), "utf-8")

        duration = round(snapshot["end"] - snapshot["start"], 3)
        fonts_dir = Path(self.settings.CAPTION_FONTS_DIR) if self.settings.CAPTION_FONTS_DIR else None
        args = build_render_args(source=inputs.source_path, output=out_path, start=snapshot["start"],
                                 duration=duration, plan=plan, profile=profile,
                                 has_audio=bool(inputs.media.get("has_audio", True)),
                                 subtitles=subs_path, fonts_dir=fonts_dir)
        qa_report: dict[str, Any] | None = None
        last_error: AppError | None = None
        attempts = 0
        for attempt in (1, 2):  # PRD §14: failed QA automatically triggers one retry
            attempts = attempt
            try:
                run_ffmpeg(args, ffmpeg_path=self.settings.FFMPEG_PATH, timeout=self.settings.FFMPEG_TIMEOUT_SECONDS,
                           cancel_check=self.cancel_check, on_progress=on_progress, duration=duration,
                           error_code=ErrorCode.RENDER_FAILED)
                qa_report = validate_render_output(
                    out_path, settings=self.settings, expected_width=profile["width"],
                    expected_height=profile["height"], expected_duration=duration, subtitles=subs_path,
                    captions_expected=bool(snapshot["captions_enabled"] and captions))
                break
            except AppError as err:
                if err.code == ErrorCode.JOB_CANCELLED:
                    raise
                last_error = err
                log.warning("render attempt failed", extra={"clip_id": clip_id, "attempt": attempt,
                                                            "error_code": err.code.value})
                out_path.unlink(missing_ok=True)
        if qa_report is None:
            self._mark_failed(clip_id, render_id, last_error, attempts)
            raise last_error or AppError(ErrorCode.RENDER_FAILED)

        thumb_path = out_dir / f"{render_id}.jpg"
        try:
            run_ffmpeg(build_thumbnail_args(out_path, thumb_path, at=min(duration / 2, max(duration - 0.5, 0))),
                       ffmpeg_path=self.settings.FFMPEG_PATH, timeout=120)
        except AppError:
            thumb_path = None  # thumbnails are best-effort

        filename = clip_filename(snapshot["job_id"], snapshot["rank"], snapshot["slug"])
        out_key = Keys.output(snapshot["job_id"], f"{render_id}/{filename}")
        checksum = sha256_file(out_path)
        stored = self.storage.put_file(out_key, out_path)
        thumb_key = None
        if thumb_path and thumb_path.exists():
            thumb_key = Keys.thumbnail(snapshot["job_id"], clip_id, render_id)
            self.storage.put_file(thumb_key, thumb_path)
        subs_key = None
        if subs_path is not None:
            subs_key = Keys.subtitles(snapshot["job_id"], clip_id, render_id)
            self.storage.put_file(subs_key, subs_path)

        with session_scope(self.session_factory) as s:
            clip = s.get(Clip, clip_id)
            render = s.get(Render, render_id)
            previous_id = clip.current_render_id if clip else None
            render.status = RenderStatus.READY.value
            render.attempt = attempts
            render.storage_key = out_key
            render.thumbnail_key = thumb_key
            render.subtitles_key = subs_key
            render.filename = filename
            render.size_bytes = stored.size
            render.checksum_sha256 = checksum
            render.duration_seconds = qa_report["checks"]["duration"]
            render.width, render.height = qa_report["checks"]["dimensions"]
            render.qa = qa_report
            render.completed_at = utcnow()
            render.expires_at = hours_from_now(self.settings.OUTPUT_RETENTION_HOURS)
            render.error_code = render.error_message = None
            if clip is not None and clip.deleted_at is None:
                clip.current_render_id = render.id
                clip.status = ClipStatus.READY.value
                clip.expires_at = render.expires_at
            if previous_id and previous_id != render_id:
                self._delete_render_files(s, previous_id)
        # Local copies are no longer needed once stored.
        if self.storage.local_path(out_key) != out_path:
            out_path.unlink(missing_ok=True)
        return {"clip_id": clip_id, "render_id": render_id, "attempts": attempts, "size_bytes": stored.size,
                "duration": qa_report["checks"]["duration"]}

    def _mark_failed(self, clip_id: str, render_id: str, err: AppError | None, attempts: int) -> None:
        with session_scope(self.session_factory) as s:
            clip = s.get(Clip, clip_id)
            render = s.get(Render, render_id)
            if render is not None:
                render.status = RenderStatus.FAILED.value
                render.attempt = attempts
                render.error_code = (err.code.value if err else ErrorCode.RENDER_FAILED.value)
                render.error_message = err.user_message if err else None
                render.completed_at = utcnow()
            if clip is not None:
                # Keep serving the previous render if there is one.
                prev = s.get(Render, clip.current_render_id) if clip.current_render_id else None
                clip.status = ClipStatus.READY.value if prev and prev.status == RenderStatus.READY.value \
                    else ClipStatus.FAILED.value

    def _delete_render_files(self, s: Session, render_id: str) -> None:
        old = s.get(Render, render_id)
        if old is None:
            return
        for key in (old.storage_key, old.thumbnail_key, old.subtitles_key):
            if key:
                self.storage.delete(key)
        old.status = RenderStatus.EXPIRED.value


class RenderStage(Stage):
    name = "render"

    def execute(self, ctx: PipelineContext) -> dict[str, Any]:
        renderer = ClipRenderer(ctx.settings, ctx.storage, ctx.session_factory, ctx.models.vision,
                                cancel_check=ctx.is_cancelled)
        inputs = RenderInputs(source_path=ctx.source_path, media=ctx.media, words=ctx.words, work_dir=ctx.work_dir)
        with session_scope(ctx.session_factory) as s:
            clips = s.scalars(select(Clip).where(Clip.job_id == ctx.job_id, Clip.deleted_at.is_(None))
                              .order_by(Clip.rank)).all()
            todo: list[tuple[str, str]] = []
            for clip in clips:
                current = s.get(Render, clip.current_render_id) if clip.current_render_id else None
                if current and current.status == RenderStatus.READY.value and current.storage_key \
                        and ctx.storage.exists(current.storage_key):
                    continue  # idempotent resume: reuse valid rendered clips
                profile = profile_for_aspect(s, clip.aspect_ratio)
                render = Render(clip_id=clip.id, job_id=clip.job_id, user_id=clip.user_id,
                                render_profile_id=profile.id, render_profile_name=profile.name,
                                render_profile_version=profile.version, status=RenderStatus.QUEUED.value,
                                source="pipeline", params={})
                s.add(render)
                s.flush()
                todo.append((clip.id, render.id))
            total = len(clips)
            done_already = total - len(todo)
        results, failures = [], []
        for i, (clip_id, render_id) in enumerate(todo):
            ctx.check_cancel()
            base = (done_already + i) / max(total, 1)

            def prog(frac: float, base: float = base) -> None:
                ctx.report(base + frac / max(total, 1))

            try:
                results.append(renderer.render(clip_id, render_id, inputs, on_progress=prog))
            except AppError as err:
                if err.code in (ErrorCode.JOB_CANCELLED, ErrorCode.JOB_TIMEOUT, ErrorCode.STORAGE_FULL):
                    raise
                failures.append({"clip_id": clip_id, "error_code": err.code.value})
            ctx.report((done_already + i + 1) / max(total, 1))
        if total and len(failures) == total:
            raise AppError(ErrorCode.RENDER_FAILED, internal=f"all {total} clips failed to render")
        return {"rendered": results, "failed": failures, "reused": done_already}

    def persist(self, ctx: PipelineContext, output: dict[str, Any]) -> list[str]:
        ctx.manifest.data["rendered_clips"] = [r["render_id"] for r in output["rendered"]]
        p = ctx.save_json("render.json", output)
        ctx.manifest.add_artifact("render_summary", p, stage=self.name)
        return ["render_summary"]

    def cleanup(self, ctx: PipelineContext) -> None:
        tmp = ctx.work_dir / "renders"
        if tmp.exists() and not any(tmp.glob("*.mp4")):
            shutil.rmtree(tmp, ignore_errors=True)
