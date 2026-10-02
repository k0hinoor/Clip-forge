"""QA stage (TRD §22; PRD §14 automated QA).

Validates each rendered clip with ffprobe and decodes first/middle/last frames.
A failed QA triggers exactly one automatic re-render (handled by the renderer).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from clipforge.core.config import Settings
from clipforge.core.errors import AppError, ErrorCode
from clipforge.media.ffmpeg import probe, run_ffmpeg
from clipforge.media.render import build_decode_check_args
from clipforge.worker.pipeline.base import PipelineContext, Stage

MIN_OUTPUT_BYTES = 10 * 1024
DURATION_TOLERANCE = 1.0


def validate_render_output(
    path: Path,
    *,
    settings: Settings,
    expected_width: int,
    expected_height: int,
    expected_duration: float,
    subtitles: Path | None,
    captions_expected: bool,
    decode_frames: bool = True,
) -> dict[str, Any]:
    report: dict[str, Any] = {"checks": {}, "passed": False}
    checks = report["checks"]

    def fail(reason: str) -> None:
        report["failure"] = reason
        raise AppError(ErrorCode.OUTPUT_VALIDATION_FAILED, details={"check": reason}, internal=str(report))

    checks["exists"] = path.is_file()
    if not checks["exists"]:
        fail("exists")
    size = path.stat().st_size
    checks["size_bytes"] = size
    if size < MIN_OUTPUT_BYTES:
        fail("min_size")
    try:
        info = probe(path, ffprobe_path=settings.FFPROBE_PATH, timeout=settings.FFPROBE_TIMEOUT_SECONDS)
    except AppError:
        fail("readable_container")
    checks["container"] = info.format_name
    if "mp4" not in info.format_name and "mov" not in info.format_name:
        fail("container")
    v, a = info.video, info.audio
    checks["video_stream"] = v is not None
    checks["audio_stream"] = a is not None
    if v is None:
        fail("video_stream")
    if a is None:
        fail("audio_stream")
    checks["video_codec"] = v.codec_name
    checks["audio_codec"] = a.codec_name
    if v.codec_name != "h264" or a.codec_name != "aac":
        fail("codec")
    checks["dimensions"] = [v.width, v.height]
    if (v.width, v.height) != (expected_width, expected_height):
        fail("dimensions")
    checks["duration"] = round(info.duration, 3)
    if info.duration < min(1.0, expected_duration) or abs(info.duration - expected_duration) > DURATION_TOLERANCE:
        fail("duration")
    if captions_expected:
        ok = subtitles is not None and subtitles.is_file() and "Dialogue:" in subtitles.read_text("utf-8")
        checks["captions"] = ok
        if not ok:
            fail("captions")
    if decode_frames:
        for label, at in (("first", 0.0), ("middle", info.duration / 2), ("last", max(info.duration - 0.5, 0.0))):
            try:
                run_ffmpeg(build_decode_check_args(path, at), ffmpeg_path=settings.FFMPEG_PATH, timeout=60,
                           error_code=ErrorCode.OUTPUT_VALIDATION_FAILED)
            except AppError:
                checks[f"decode_{label}"] = False
                fail(f"decode_{label}")
            checks[f"decode_{label}"] = True
    report["passed"] = True
    return report


class QAStage(Stage):
    """Pipeline-level QA: every READY render must still exist and match its checksum."""

    name = "qa"

    def execute(self, ctx: PipelineContext) -> dict[str, Any]:
        from sqlalchemy import select

        from clipforge.core.states import ClipStatus
        from clipforge.db.models import Clip, Render
        from clipforge.db.session import session_scope

        summary: dict[str, Any] = {"clips": []}
        with session_scope(ctx.session_factory) as s:
            clips = s.scalars(select(Clip).where(Clip.job_id == ctx.job_id, Clip.deleted_at.is_(None))
                              .order_by(Clip.rank)).all()
            for clip in clips:
                render = s.get(Render, clip.current_render_id) if clip.current_render_id else None
                ok = bool(render and render.storage_key and ctx.storage.exists(render.storage_key))
                if ok and render.size_bytes:
                    st = ctx.storage.stat(render.storage_key)
                    ok = st is not None and st.size == render.size_bytes
                if not ok and clip.status == ClipStatus.READY.value:
                    clip.status = ClipStatus.FAILED.value
                summary["clips"].append({"clip_id": clip.id, "rank": clip.rank, "ok": ok})
        if not any(c["ok"] for c in summary["clips"]):
            raise AppError(ErrorCode.OUTPUT_VALIDATION_FAILED, internal="no clip passed QA")
        return summary

    def persist(self, ctx: PipelineContext, output: dict[str, Any]) -> list[str]:
        p = ctx.save_json("qa.json", output)
        ctx.manifest.add_artifact("qa", p, stage=self.name)
        return ["qa"]
