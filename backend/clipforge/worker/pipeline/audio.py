"""AUDIO_EXTRACT stage (TRD §10): 16 kHz mono PCM WAV + per-second loudness envelope."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from clipforge.core.errors import AppError, ErrorCode
from clipforge.media.ffmpeg import run_ffmpeg
from clipforge.media.render import build_audio_extract_args, build_audio_levels_args
from clipforge.worker.pipeline.base import PipelineContext, Stage

_LEVEL_RE = re.compile(r"lavfi\.astats\.Overall\.RMS_level=(-?inf|-?[\d.]+)")


def parse_levels(path: Path) -> list[float]:
    levels: list[float] = []
    if not path.exists():
        return levels
    for line in path.read_text("utf-8", "replace").splitlines():
        m = _LEVEL_RE.search(line)
        if m:
            raw = m.group(1)
            value = -90.0 if "inf" in raw else float(raw)
            levels.append(round(max(value, -90.0), 2) if math.isfinite(value) else -90.0)
    return levels


class AudioExtractStage(Stage):
    name = "audio_extract"

    def execute(self, ctx: PipelineContext) -> dict[str, Any]:
        if not ctx.media.get("has_audio"):
            return {"has_audio": False, "levels": []}
        wav = ctx.path("audio.wav")
        duration = float(ctx.media.get("duration") or 0)
        run_ffmpeg(build_audio_extract_args(ctx.source_path, wav), ffmpeg_path=ctx.settings.FFMPEG_PATH,
                   timeout=ctx.settings.FFMPEG_TIMEOUT_SECONDS, cancel_check=ctx.is_cancelled,
                   on_progress=lambda f: ctx.report(f * 0.8), duration=duration,
                   error_code=ErrorCode.MEDIA_CORRUPTED)
        levels_txt = ctx.path("audio_levels.txt")
        try:
            run_ffmpeg(build_audio_levels_args(wav, levels_txt), ffmpeg_path=ctx.settings.FFMPEG_PATH,
                       timeout=ctx.settings.FFMPEG_TIMEOUT_SECONDS, cancel_check=ctx.is_cancelled)
            levels = parse_levels(levels_txt)
        except AppError as err:
            if err.code == ErrorCode.JOB_CANCELLED:
                raise
            levels = []  # loudness is an optional signal
        finally:
            levels_txt.unlink(missing_ok=True)
        return {"has_audio": True, "levels": levels}

    def persist(self, ctx: PipelineContext, output: dict[str, Any]) -> list[str]:
        ctx.audio_levels = output["levels"]
        p = ctx.save_json("audio_levels.json", {"has_audio": output["has_audio"], "levels": output["levels"]})
        ctx.manifest.add_artifact("audio_levels", p, stage=self.name)
        names = ["audio_levels"]
        if output["has_audio"]:
            ctx.manifest.add_artifact("audio", ctx.work_dir / "audio.wav", stage=self.name)
            names.append("audio")
        return names

    def validate_output(self, ctx: PipelineContext, output: dict[str, Any]) -> None:
        if output["has_audio"]:
            wav = ctx.work_dir / "audio.wav"
            if not wav.is_file() or wav.stat().st_size < 1000:
                raise AppError(ErrorCode.MEDIA_CORRUPTED, internal="audio extraction produced no data")

    def load(self, ctx: PipelineContext) -> None:
        ctx.audio_levels = ctx.load_json("audio_levels.json").get("levels", [])
