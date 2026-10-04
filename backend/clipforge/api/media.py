"""Media streaming helpers.

Serves local files to the browser with HTTP range support so ``<video>`` can
seek, an aggressive cache header for renders, and a path guard that refuses to
serve anything outside the data directory.
"""

from __future__ import annotations

import mimetypes
import os
import re
from pathlib import Path
from typing import Iterator

from fastapi import Request, Response
from fastapi.responses import FileResponse, StreamingResponse

from ..config import Env
from ..errors import ClipForgeError, ErrorCode
from ..logging_setup import get_logger

log = get_logger(__name__)

CHUNK_SIZE = 1024 * 512
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def ensure_within(path: Path, roots: tuple[Path, ...] | None = None) -> Path:
    """Resolve ``path`` and refuse anything outside the allowed roots."""
    resolved = Path(path).expanduser().resolve()
    allowed = roots or (Env.DATA_DIR.resolve(), Env.LOG_DIR.resolve())
    for root in allowed:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise ClipForgeError(
        code=ErrorCode.INVALID_INPUT,
        message="That file is not inside the CLIPFORGE data directory.",
        hint="Only project files, renders and assets can be served.",
        status_code=403,
    )


def require_file(path: str | Path | None) -> Path:
    if not path:
        raise ClipForgeError(code=ErrorCode.NOT_FOUND, message="No file is linked to this item.", status_code=404)
    resolved = ensure_within(Path(path))
    if not resolved.exists() or not resolved.is_file():
        raise ClipForgeError(
            code=ErrorCode.NOT_FOUND,
            message="That file is no longer on disk.",
            hint="It may have been deleted or the project may need re-rendering.",
            status_code=404,
        )
    return resolved


def guess_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".mp4":
        return "video/mp4"
    if suffix in {".webm", ".mkv"}:
        return "video/webm" if suffix == ".webm" else "video/x-matroska"
    if suffix in {".mp3", ".m4a", ".aac", ".wav", ".ogg", ".opus"}:
        return mimetypes.guess_type(str(path))[0] or "audio/mpeg"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix in {".ass", ".srt"}:
        return "text/plain; charset=utf-8"
    if suffix == ".json":
        return "application/json"
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def stream_file(path: Path, request: Request, *, download_name: str | None = None, cache_seconds: int = 0) -> Response:
    """Range-aware file response (needed for video scrubbing)."""
    size = path.stat().st_size
    media_type = guess_media_type(path)
    range_header = request.headers.get("range") or request.headers.get("Range")
    headers: dict[str, str] = {"Accept-Ranges": "bytes"}
    if cache_seconds:
        headers["Cache-Control"] = f"private, max-age={cache_seconds}"
    if download_name:
        headers["Content-Disposition"] = f'attachment; filename="{download_name}"'

    if range_header:
        match = RANGE_RE.match(range_header.strip())
        if match:
            start_text, end_text = match.groups()
            start = int(start_text) if start_text else 0
            end = int(end_text) if end_text else size - 1
            start = max(0, min(start, max(size - 1, 0)))
            end = max(start, min(end, size - 1))
            length = end - start + 1
            headers.update(
                {
                    "Content-Range": f"bytes {start}-{end}/{size}",
                    "Content-Length": str(length),
                }
            )
            return StreamingResponse(
                _iter_range(path, start, length),
                status_code=206,
                media_type=media_type,
                headers=headers,
            )

    headers["Content-Length"] = str(size)
    return FileResponse(path, media_type=media_type, headers=headers)


def _iter_range(path: Path, start: int, length: int, chunk: int = CHUNK_SIZE) -> Iterator[bytes]:
    remaining = length
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining > 0:
            data = handle.read(min(chunk, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


__all__ = ["ensure_within", "guess_media_type", "human_size", "require_file", "stream_file"]
