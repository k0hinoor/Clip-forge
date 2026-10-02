"""Storage providers and storage key conventions."""

from __future__ import annotations

from clipforge.core.config import Settings
from clipforge.storage.base import StorageProvider, StoredObject, file_sha256
from clipforge.storage.local import LocalFilesystemStorage

_instances: dict[tuple, StorageProvider] = {}


def get_storage(settings: Settings) -> StorageProvider:
    cache_key = (settings.STORAGE_BACKEND, str(settings.STORAGE_ROOT), settings.S3_BUCKET)
    if cache_key not in _instances:
        if settings.STORAGE_BACKEND == "s3":
            from clipforge.storage.s3 import S3CompatibleStorage

            _instances[cache_key] = S3CompatibleStorage(
                settings.S3_BUCKET,
                endpoint_url=settings.S3_ENDPOINT_URL,
                region=settings.S3_REGION,
                access_key_id=settings.S3_ACCESS_KEY_ID,
                secret_access_key=settings.S3_SECRET_ACCESS_KEY,
            )
        else:
            _instances[cache_key] = LocalFilesystemStorage(settings.STORAGE_ROOT)
    return _instances[cache_key]


class Keys:
    """UUID-based storage key layout (TRD §6)."""

    @staticmethod
    def upload(user_id: str, asset_id: str, ext: str) -> str:
        return f"uploads/{user_id}/{asset_id}/source{ext}"

    @staticmethod
    def work_prefix(job_id: str) -> str:
        return f"work/{job_id}"

    @staticmethod
    def output(job_id: str, filename: str) -> str:
        return f"outputs/{job_id}/{filename}"

    @staticmethod
    def output_prefix(job_id: str) -> str:
        return f"outputs/{job_id}"

    @staticmethod
    def thumbnail(job_id: str, clip_id: str, render_id: str) -> str:
        return f"thumbnails/{job_id}/{clip_id}_{render_id}.jpg"

    @staticmethod
    def thumbnail_prefix(job_id: str) -> str:
        return f"thumbnails/{job_id}"

    @staticmethod
    def transcript(job_id: str) -> str:
        return f"transcripts/{job_id}/transcript.json"

    @staticmethod
    def transcript_prefix(job_id: str) -> str:
        return f"transcripts/{job_id}"

    @staticmethod
    def subtitles(job_id: str, clip_id: str, render_id: str) -> str:
        return f"outputs/{job_id}/subtitles/{clip_id}_{render_id}.ass"


__all__ = ["Keys", "LocalFilesystemStorage", "StorageProvider", "StoredObject", "file_sha256", "get_storage"]
