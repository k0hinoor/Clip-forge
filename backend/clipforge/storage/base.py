"""Storage abstraction (TRD §6).

``StorageProvider`` hides whether bytes live on the local filesystem or in an
S3-compatible bucket. Keys are internal, UUID-based and validated; they are
never derived from user input and never returned to clients.
"""

from __future__ import annotations

import abc
import hashlib
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO


@dataclass(frozen=True)
class StoredObject:
    key: str
    size: int
    modified_at: datetime


class StorageProvider(abc.ABC):
    name: str = "abstract"
    supports_local_paths: bool = False

    @abc.abstractmethod
    def put_file(self, key: str, local_path: Path) -> StoredObject: ...

    @abc.abstractmethod
    def put_bytes(self, key: str, data: bytes) -> StoredObject: ...

    @abc.abstractmethod
    async def put_stream(
        self, key: str, chunks: AsyncIterator[bytes], *, max_bytes: int, append: bool = False,
        hasher: hashlib._Hash | None = None, offset: int = 0,
    ) -> int:
        """Stream bytes into ``key`` enforcing ``max_bytes``. Returns bytes written."""

    @abc.abstractmethod
    def get_to_path(self, key: str, local_path: Path) -> Path: ...

    @abc.abstractmethod
    def open(self, key: str) -> BinaryIO: ...

    @abc.abstractmethod
    def iter_bytes(self, key: str, start: int = 0, end: int | None = None,
                   chunk_size: int = 1024 * 1024) -> Iterator[bytes]: ...

    @abc.abstractmethod
    def delete(self, key: str) -> bool: ...

    @abc.abstractmethod
    def delete_prefix(self, prefix: str) -> int: ...

    @abc.abstractmethod
    def exists(self, key: str) -> bool: ...

    @abc.abstractmethod
    def stat(self, key: str) -> StoredObject | None: ...

    def signed_url(self, key: str, expires_seconds: int, filename: str | None = None) -> str | None:
        """Pre-signed URL when the backend supports it; ``None`` otherwise.

        Local mode uses access-controlled API endpoints with HMAC-signed query
        parameters instead (see ``api/routes/clips.py``).
        """
        return None

    def local_path(self, key: str) -> Path | None:
        """Direct filesystem path (local backend only)."""
        return None

    def usage_bytes(self, prefix: str = "") -> int:
        return 0

    def checksum(self, key: str) -> str:
        h = hashlib.sha256()
        for chunk in self.iter_bytes(key):
            h.update(chunk)
        return h.hexdigest()


def file_sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def storage_full_guard(exc: OSError) -> None:
    """Translate disk-full OS errors into the stable STORAGE_FULL error."""
    import errno

    from clipforge.core.errors import AppError, ErrorCode

    if exc.errno in (errno.ENOSPC, errno.EDQUOT):
        raise AppError(ErrorCode.STORAGE_FULL, internal=str(exc)) from exc
