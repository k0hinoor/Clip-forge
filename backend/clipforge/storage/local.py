"""Local filesystem storage rooted at ``STORAGE_ROOT`` (outside any web root)."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from clipforge.core.errors import AppError, ErrorCode
from clipforge.security.files import safe_join
from clipforge.storage.base import StorageProvider, StoredObject, storage_full_guard


class LocalFilesystemStorage(StorageProvider):
    name = "local"
    supports_local_paths = True

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return safe_join(self.root, key)

    def _stat(self, key: str, path: Path) -> StoredObject:
        st = path.stat()
        return StoredObject(key=key, size=st.st_size, modified_at=datetime.fromtimestamp(st.st_mtime, UTC).replace(tzinfo=None))

    def put_file(self, key: str, local_path: Path) -> StoredObject:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            if Path(local_path).resolve() != dest:
                tmp = dest.with_name(dest.name + ".part")
                shutil.copyfile(local_path, tmp)
                os.replace(tmp, dest)
        except OSError as exc:
            storage_full_guard(exc)
            raise
        return self._stat(key, dest)

    def put_bytes(self, key: str, data: bytes) -> StoredObject:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, dest)
        except OSError as exc:
            storage_full_guard(exc)
            raise
        return self._stat(key, dest)

    async def put_stream(
        self, key: str, chunks: AsyncIterator[bytes], *, max_bytes: int, append: bool = False,
        hasher: hashlib._Hash | None = None, offset: int = 0,
    ) -> int:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        mode = "r+b" if append and dest.exists() else "wb"
        try:
            with open(dest, mode) as fh:
                if append:
                    fh.seek(offset)
                    fh.truncate()
                async for chunk in chunks:
                    if not chunk:
                        continue
                    written += len(chunk)
                    if offset + written > max_bytes:
                        raise AppError(ErrorCode.MEDIA_TOO_LARGE)
                    fh.write(chunk)
                    if hasher is not None:
                        hasher.update(chunk)
        except OSError as exc:
            storage_full_guard(exc)
            raise
        return written

    def get_to_path(self, key: str, local_path: Path) -> Path:
        src = self._path(key)
        if not src.exists():
            raise FileNotFoundError(key)
        if Path(local_path).resolve() != src:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, local_path)
        return Path(local_path)

    def open(self, key: str) -> BinaryIO:
        return open(self._path(key), "rb")

    def iter_bytes(self, key: str, start: int = 0, end: int | None = None,
                   chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        with open(self._path(key), "rb") as fh:
            fh.seek(start)
            remaining = None if end is None else end - start + 1
            while remaining is None or remaining > 0:
                size = chunk_size if remaining is None else min(chunk_size, remaining)
                data = fh.read(size)
                if not data:
                    break
                if remaining is not None:
                    remaining -= len(data)
                yield data

    def delete(self, key: str) -> bool:
        path = self._path(key)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    def delete_prefix(self, prefix: str) -> int:
        path = self._path(prefix.rstrip("/"))
        if not path.exists():
            return 0
        if path.is_file():
            path.unlink()
            return 1
        count = sum(1 for p in path.rglob("*") if p.is_file())
        shutil.rmtree(path, ignore_errors=True)
        return count

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def stat(self, key: str) -> StoredObject | None:
        path = self._path(key)
        return self._stat(key, path) if path.is_file() else None

    def local_path(self, key: str) -> Path | None:
        return self._path(key)

    def usage_bytes(self, prefix: str = "") -> int:
        base = self._path(prefix.rstrip("/")) if prefix else self.root
        if not base.exists():
            return 0
        if base.is_file():
            return base.stat().st_size
        total = 0
        for p in base.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue
        return total
