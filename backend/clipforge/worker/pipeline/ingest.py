"""VALIDATE (media probe) and INGEST stages (TRD §9)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from clipforge.core.errors import AppError, ErrorCode
from clipforge.db.models import Asset
from clipforge.db.session import session_scope
from clipforge.media.ffmpeg import probe
from clipforge.media.validation import validate_media_info
from clipforge.security.files import validate_signature
from clipforge.worker.pipeline.base import PipelineContext, Stage, sha256_file


def resolve_source(ctx: PipelineContext) -> Path:
    """Local path of the immutable source. Downloads from object storage if needed."""
    with session_scope(ctx.session_factory) as s:
        asset = s.get(Asset, ctx.asset_id)
        if asset is None or asset.deleted_at is not None:
            raise AppError(ErrorCode.SOURCE_EXPIRED)
        key, ext = asset.storage_key, asset.extension
    local = ctx.storage.local_path(key)
    if local is not None:
        if not local.is_file():
            raise AppError(ErrorCode.SOURCE_EXPIRED)
        return local
    cached = ctx.work_dir / f"source{ext}"
    if not cached.is_file():
        if not ctx.storage.exists(key):
            raise AppError(ErrorCode.SOURCE_EXPIRED)
        ctx.storage.get_to_path(key, cached)
    return cached


class ValidateStage(Stage):
    name = "validate"

    def execute(self, ctx: PipelineContext) -> dict[str, Any]:
        src = resolve_source(ctx)
        with session_scope(ctx.session_factory) as s:
            ext = s.get(Asset, ctx.asset_id).extension
        with open(src, "rb") as fh:
            validate_signature(fh.read(64), ext)
        info = probe(src, ffprobe_path=ctx.settings.FFPROBE_PATH, timeout=ctx.settings.FFPROBE_TIMEOUT_SECONDS)
        validate_media_info(info, ctx.settings)
        ctx.source_path = src
        data = info.to_dict()
        data["has_audio"] = info.audio is not None
        data["has_video"] = info.video is not None
        return data

    def persist(self, ctx: PipelineContext, output: dict[str, Any]) -> list[str]:
        ctx.media = output
        p = ctx.save_json("probe.json", output)
        ctx.manifest.add_artifact("probe", p, stage=self.name)
        return ["probe"]

    def validate_output(self, ctx: PipelineContext, output: dict[str, Any]) -> None:
        if not output.get("display_width") or not output.get("duration"):
            raise AppError(ErrorCode.MEDIA_CORRUPTED)

    def load(self, ctx: PipelineContext) -> None:
        ctx.media = ctx.load_json("probe.json")
        ctx.source_path = resolve_source(ctx)


class IngestStage(Stage):
    """Verify source integrity and record immutable media metadata + source artifact."""

    name = "ingest"

    def execute(self, ctx: PipelineContext) -> dict[str, Any]:
        src = ctx.source_path or resolve_source(ctx)
        checksum = sha256_file(src)
        with session_scope(ctx.session_factory) as s:
            asset = s.get(Asset, ctx.asset_id)
            if asset.checksum_sha256 and asset.checksum_sha256 != checksum:
                raise AppError(ErrorCode.MEDIA_CORRUPTED, internal="source checksum mismatch")
            if not asset.media_metadata:  # metadata is immutable once stored
                asset.media_metadata = ctx.media
                asset.duration_seconds = ctx.media.get("duration")
        return {"checksum": checksum, "path": src}

    def persist(self, ctx: PipelineContext, output: dict[str, Any]) -> list[str]:
        ctx.manifest.add_artifact("source", output["path"], stage=self.name, checksum=output["checksum"])
        return ["source"]

    def load(self, ctx: PipelineContext) -> None:
        if ctx.source_path is None:
            ctx.source_path = resolve_source(ctx)
