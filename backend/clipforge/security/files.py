"""File/path safety helpers (TRD §6, §9, §40).

User-supplied filenames are *never* used as filesystem paths; they are only
kept (sanitised) for display. Storage keys are generated from UUIDs.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path, PurePosixPath

from clipforge.core.errors import AppError, ErrorCode

_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-./]*$")
_SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9_\-./]+$")

# Container signatures (magic numbers) for allowed formats.
_MIME_BY_EXT = {
    ".mp4": "video/mp4",
    ".m4v": "video/x-m4v",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
}
_ISO_BMFF_EXTS = {".mp4", ".m4v", ".mov"}
_EBML_EXTS = {".mkv", ".webm"}
_QT_ATOMS = {b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot"}


def sanitize_display_filename(name: str | None, max_len: int = 200) -> str:
    """Return a safe display-only filename (no path components or control chars)."""
    name = (name or "upload").replace("\\", "/").split("/")[-1]
    name = unicodedata.normalize("NFKC", name)
    name = "".join(ch for ch in name if ch.isprintable() and ch not in '<>:"|?*\x00')
    name = name.strip().strip(".") or "upload"
    if len(name) > max_len:
        stem, dot, ext = name.rpartition(".")
        name = (stem[: max_len - len(ext) - 1] + dot + ext) if dot else name[:max_len]
    return name


def extension_of(name: str) -> str:
    return PurePosixPath(name.lower()).suffix


def validate_extension(name: str, allowed: list[str]) -> str:
    ext = extension_of(name)
    if ext not in allowed:
        raise AppError(ErrorCode.UNSUPPORTED_FORMAT, details={"extension": ext or None})
    return ext


def mime_for_extension(ext: str) -> str:
    return _MIME_BY_EXT.get(ext, "application/octet-stream")


def detect_container(header: bytes) -> str | None:
    """Detect the container family from the first bytes of a file."""
    if len(header) >= 4 and header[:4] == b"\x1a\x45\xdf\xa3":
        return "ebml"  # Matroska / WebM
    if len(header) >= 8 and header[4:8] in _QT_ATOMS:
        return "isobmff"  # MP4 / MOV / M4V
    return None


def validate_signature(header: bytes, ext: str) -> str:
    """Ensure the file's magic bytes match its extension family."""
    family = detect_container(header)
    if family is None:
        raise AppError(ErrorCode.UNSUPPORTED_FORMAT, "The file is not a recognised video container.")
    if (family == "ebml" and ext not in _EBML_EXTS) or (family == "isobmff" and ext not in _ISO_BMFF_EXTS):
        raise AppError(ErrorCode.UNSUPPORTED_FORMAT, "The file content does not match its extension.")
    return family


def validate_storage_key(key: str) -> str:
    """Storage keys are internal, relative, and restricted to a safe charset."""
    if not key or len(key) > 512 or not _KEY_RE.match(key):
        raise ValueError(f"Invalid storage key: {key!r}")
    parts = PurePosixPath(key).parts
    if any(p in ("..", ".", "") for p in parts) or key.startswith("/"):
        raise ValueError(f"Invalid storage key: {key!r}")
    return key


def safe_join(root: Path, key: str) -> Path:
    """Resolve ``key`` under ``root`` and guarantee it cannot escape it."""
    validate_storage_key(key)
    root = root.resolve()
    path = (root / key).resolve()
    if path != root and root not in path.parents:
        raise ValueError("Path escapes storage root")
    return path


def assert_safe_media_path(path: Path) -> Path:
    """Paths handed to FFmpeg must be absolute and contain only safe characters.

    This keeps filter-graph strings (e.g. ``subtitles=``) free of injection.
    """
    p = Path(path)
    s = str(p)
    if not p.is_absolute() or not _SAFE_PATH_RE.match(s) or ".." in p.parts:
        raise AppError(ErrorCode.UNKNOWN_ERROR, internal=f"unsafe media path: {s!r}")
    return p


def slugify(text: str, max_len: int = 40) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    text = text[:max_len].rstrip("-")
    return text or "clip"


def clip_filename(job_id: str, rank: int, slug: str) -> str:
    """Deterministic output filename (PRD §11): clip_<job_id>_<rank>_<short_slug>.mp4"""
    return f"clip_{job_id}_{int(rank):02d}_{slugify(slug, 40)}.mp4"
