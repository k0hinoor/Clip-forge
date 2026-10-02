"""Upload orchestration (PRD §6.1, TRD §6, §9, §42).

Supports:
* single-request streaming uploads (raw body or multipart), never loading the
  whole file into RAM;
* resumable chunked uploads (init → append chunks at offsets → complete);
* checksum + per-user duplicate detection;
* extension, magic-byte, and ffprobe validation.
"""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from clipforge.core.config import Settings
from clipforge.core.errors import AppError, ErrorCode
from clipforge.core.logging import get_logger
from clipforge.core.states import AssetStatus, JobStatus
from clipforge.core.timeutil import utcnow
from clipforge.db.models import Asset, Job, User
from clipforge.media.ffmpeg import probe
from clipforge.media.validation import validate_media_info
from clipforge.security.files import (
    mime_for_extension,
    sanitize_display_filename,
    validate_extension,
    validate_signature,
)
from clipforge.services.analytics import track
from clipforge.services.auth import ensure_default_project
from clipforge.storage import Keys, StorageProvider
from clipforge.storage.base import file_sha256

log = get_logger(__name__)


class UploadService:
    def __init__(self, session: Session, settings: Settings, storage: StorageProvider) -> None:
        self.session = session
        self.settings = settings
        self.storage = storage

    # ------------------------------------------------------------ creation
    def create_asset(self, user: User, filename: str | None, expected_size: int | None = None,
                     project_id: str | None = None) -> Asset:
        display = sanitize_display_filename(filename)
        ext = validate_extension(display, self.settings.ALLOWED_EXTENSIONS)
        if expected_size is not None:
            if expected_size <= 0:
                raise AppError(ErrorCode.VALIDATION_ERROR, "File size must be positive.")
            if expected_size > self.settings.MAX_UPLOAD_BYTES:
                raise AppError(ErrorCode.MEDIA_TOO_LARGE)
        if project_id is None:
            project_id = ensure_default_project(self.session, user).id
        asset = Asset(user_id=user.id, project_id=project_id, original_filename=display, extension=ext,
                      storage_key="pending", expected_size_bytes=expected_size, status=AssetStatus.UPLOADING.value,
                      mime_type=mime_for_extension(ext), bytes_received=0,
                      upload_expires_at=utcnow() + timedelta(hours=self.settings.ABANDONED_UPLOAD_HOURS))
        self.session.add(asset)
        self.session.flush()
        asset.storage_key = Keys.upload(user.id, asset.id, ext)
        return asset

    # ------------------------------------------------------------ streaming
    async def receive(self, asset: Asset, chunks: AsyncIterator[bytes], *, offset: int = 0,
                      chunked: bool = False) -> tuple[int, str | None]:
        """Append bytes at ``offset``. Returns (bytes_written, sha256-or-None)."""
        if asset.status != AssetStatus.UPLOADING.value:
            raise AppError(ErrorCode.CONFLICT, "This upload is already complete.")
        if offset != (asset.bytes_received or 0):
            raise AppError(ErrorCode.CONFLICT, "Chunk offset does not match the bytes received so far.",
                           details={"expected_offset": asset.bytes_received or 0})
        hasher = None if chunked else hashlib.sha256()
        limit = self.settings.MAX_UPLOAD_BYTES
        if asset.expected_size_bytes:
            limit = min(limit, asset.expected_size_bytes)
        try:
            written = await self.storage.put_stream(asset.storage_key, chunks, max_bytes=limit,
                                                    append=chunked, hasher=hasher, offset=offset)
        except AppError as err:
            if err.code == ErrorCode.MEDIA_TOO_LARGE:
                self.reject(asset, err)
            raise
        asset.bytes_received = offset + written
        asset.upload_expires_at = utcnow() + timedelta(hours=self.settings.ABANDONED_UPLOAD_HOURS)
        return written, (hasher.hexdigest() if hasher else None)

    # ------------------------------------------------------------- finalise
    def finalize(self, asset: Asset, checksum: str | None = None) -> tuple[Asset, bool]:
        """Validate the stored file. Returns (asset, is_duplicate).

        On duplicate the *existing* ready asset is returned and the new bytes
        are discarded (PRD §6.1 duplicate detection).
        """
        if asset.status != AssetStatus.UPLOADING.value:
            return asset, False
        if asset.expected_size_bytes and asset.bytes_received != asset.expected_size_bytes:
            raise AppError(ErrorCode.CONFLICT, "Upload is incomplete.",
                           details={"bytes_received": asset.bytes_received,
                                    "expected_size_bytes": asset.expected_size_bytes})
        if not asset.bytes_received:
            self.reject(asset, AppError(ErrorCode.MEDIA_CORRUPTED, "The uploaded file is empty."))
            raise AppError(ErrorCode.MEDIA_CORRUPTED, "The uploaded file is empty.")
        try:
            with self._local_copy(asset) as path:
                with open(path, "rb") as fh:
                    validate_signature(fh.read(64), asset.extension)
                checksum = checksum or file_sha256(path)
                existing = self.session.scalar(select(Asset).where(
                    Asset.user_id == asset.user_id, Asset.checksum_sha256 == checksum,
                    Asset.status == AssetStatus.READY.value, Asset.id != asset.id, Asset.deleted_at.is_(None)))
                if existing is not None and self.storage.exists(existing.storage_key):
                    self.storage.delete(asset.storage_key)
                    asset.status = AssetStatus.DELETED.value
                    asset.deleted_at = utcnow()
                    asset.checksum_sha256 = checksum
                    return existing, True
                info = probe(path, ffprobe_path=self.settings.FFPROBE_PATH,
                             timeout=self.settings.FFPROBE_TIMEOUT_SECONDS)
                validate_media_info(info, self.settings)
        except AppError as err:
            self.reject(asset, err)
            raise
        width, height = info.display_size
        asset.checksum_sha256 = checksum
        asset.size_bytes = asset.bytes_received
        asset.container = info.format_name
        asset.duration_seconds = round(info.duration, 3)
        asset.width, asset.height = width, height
        asset.fps = info.video.fps if info.video else None
        asset.video_codec = info.video.codec_name if info.video else None
        asset.audio_codec = info.audio.codec_name if info.audio else None
        asset.has_video = info.video is not None
        asset.has_audio = info.audio is not None
        asset.media_metadata = info.to_dict()
        asset.status = AssetStatus.READY.value
        asset.upload_expires_at = None
        # Unprocessed uploads also expire (privacy-first default).
        asset.expires_at = utcnow() + timedelta(hours=self.settings.FAILED_SOURCE_RETENTION_HOURS)
        track(self.session, "upload_completed", user_id=asset.user_id,
              properties={"duration": asset.duration_seconds, "size_bytes": asset.size_bytes})
        return asset, False

    def reject(self, asset: Asset, err: AppError) -> None:
        try:
            self.storage.delete(asset.storage_key)
        except Exception:  # pragma: no cover - best effort
            log.warning("failed to delete rejected upload", exc_info=True)
        asset.status = AssetStatus.REJECTED.value
        asset.error_code = err.code.value
        track(self.session, "upload_rejected", user_id=asset.user_id, properties={"error_code": err.code.value})

    def delete(self, asset: Asset) -> None:
        active = self.session.scalar(select(Job.id).where(
            Job.source_asset_id == asset.id,
            Job.status.notin_([JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value,
                               JobStatus.EXPIRED.value]),
            Job.deleted_at.is_(None)).limit(1))
        if active:
            raise AppError(ErrorCode.CONFLICT, "Cancel the running job before deleting its source video.")
        self.storage.delete(asset.storage_key)
        asset.status = AssetStatus.DELETED.value
        asset.deleted_at = utcnow()

    # -------------------------------------------------------------- helpers
    class _LocalCopy:
        def __init__(self, svc: UploadService, asset: Asset) -> None:
            self.svc, self.asset, self.tmp = svc, asset, None

        def __enter__(self) -> Path:
            local = self.svc.storage.local_path(self.asset.storage_key)
            if local is not None:
                return local
            self.tmp = tempfile.TemporaryDirectory()
            return self.svc.storage.get_to_path(self.asset.storage_key,
                                                Path(self.tmp.name) / f"src{self.asset.extension}")

        def __exit__(self, *exc) -> None:
            if self.tmp is not None:
                self.tmp.cleanup()

    def _local_copy(self, asset: Asset) -> UploadService._LocalCopy:
        return UploadService._LocalCopy(self, asset)
