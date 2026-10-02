"""S3-compatible object storage (AWS S3, MinIO, R2, ...). Requires ``boto3``.

Install with ``pip install clipforge[s3]``. Used in cloud mode; downloads use
pre-signed URLs (TRD §41).
"""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, BinaryIO

from clipforge.core.errors import AppError, ErrorCode
from clipforge.security.files import validate_storage_key
from clipforge.storage.base import StorageProvider, StoredObject


class S3CompatibleStorage(StorageProvider):
    name = "s3"

    def __init__(self, bucket: str, *, endpoint_url: str = "", region: str = "us-east-1",
                 access_key_id: str = "", secret_access_key: str = "") -> None:
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("S3 storage requires boto3: pip install 'clipforge[s3]'") from exc
        if not bucket:
            raise RuntimeError("S3_BUCKET must be configured for S3 storage")
        self.bucket = bucket
        self.client: Any = boto3.client(
            "s3",
            endpoint_url=endpoint_url or None,
            region_name=region,
            aws_access_key_id=access_key_id or None,
            aws_secret_access_key=secret_access_key or None,
        )

    def put_file(self, key: str, local_path: Path) -> StoredObject:
        validate_storage_key(key)
        self.client.upload_file(str(local_path), self.bucket, key)
        return self.stat(key)  # type: ignore[return-value]

    def put_bytes(self, key: str, data: bytes) -> StoredObject:
        validate_storage_key(key)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)
        return self.stat(key)  # type: ignore[return-value]

    async def put_stream(
        self, key: str, chunks: AsyncIterator[bytes], *, max_bytes: int, append: bool = False,
        hasher: hashlib._Hash | None = None, offset: int = 0,
    ) -> int:
        # Spool to a local temp file (bounded by max_bytes), then upload.
        validate_storage_key(key)
        written = 0
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            if append and offset and self.exists(key):
                self.client.download_fileobj(self.bucket, key, tmp)
                tmp.truncate(offset)
                tmp.seek(offset)
            async for chunk in chunks:
                written += len(chunk)
                if offset + written > max_bytes:
                    raise AppError(ErrorCode.MEDIA_TOO_LARGE)
                tmp.write(chunk)
                if hasher is not None:
                    hasher.update(chunk)
            tmp.flush()
            self.client.upload_file(tmp.name, self.bucket, key)
        return written

    def get_to_path(self, key: str, local_path: Path) -> Path:
        validate_storage_key(key)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(local_path))
        return Path(local_path)

    def open(self, key: str) -> BinaryIO:
        validate_storage_key(key)
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"]

    def iter_bytes(self, key: str, start: int = 0, end: int | None = None,
                   chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        validate_storage_key(key)
        rng = f"bytes={start}-{'' if end is None else end}"
        body = self.client.get_object(Bucket=self.bucket, Key=key, Range=rng)["Body"]
        yield from body.iter_chunks(chunk_size)

    def delete(self, key: str) -> bool:
        validate_storage_key(key)
        self.client.delete_object(Bucket=self.bucket, Key=key)
        return True

    def delete_prefix(self, prefix: str) -> int:
        validate_storage_key(prefix.rstrip("/"))
        count = 0
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix.rstrip("/") + "/"):
            objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objs:
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": objs})
                count += len(objs)
        return count

    def exists(self, key: str) -> bool:
        return self.stat(key) is not None

    def stat(self, key: str) -> StoredObject | None:
        validate_storage_key(key)
        try:
            head = self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception:
            return None
        return StoredObject(key=key, size=int(head["ContentLength"]),
                            modified_at=head["LastModified"].replace(tzinfo=None))

    def signed_url(self, key: str, expires_seconds: int, filename: str | None = None) -> str | None:
        validate_storage_key(key)
        params: dict[str, Any] = {"Bucket": self.bucket, "Key": key}
        if filename:
            params["ResponseContentDisposition"] = f'attachment; filename="{filename}"'
        return self.client.generate_presigned_url("get_object", Params=params, ExpiresIn=expires_seconds)
