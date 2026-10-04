"""Typed, user-facing errors.

Everything that can go wrong during analysis or rendering is mapped onto a
:class:`ClipForgeError` so the API can always return a stable ``code``, a human
readable ``message`` and an actionable ``hint`` instead of leaking a Python
traceback into the UI. Raw tracebacks still go to ``logs/*.log``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


class ErrorCode:
    """Stable machine-readable error identifiers (kept flat on purpose)."""

    INVALID_URL = "invalid_url"
    VIDEO_UNAVAILABLE = "video_unavailable"
    VIDEO_PRIVATE = "video_private"
    VIDEO_AGE_RESTRICTED = "video_age_restricted"
    VIDEO_GEO_BLOCKED = "video_geo_blocked"
    LIVE_STREAM = "live_stream"
    VIDEO_TOO_LONG = "video_too_long"
    VIDEO_TOO_SHORT = "video_too_short"
    DOWNLOAD_FAILED = "download_failed"
    NO_AUDIO = "no_audio"
    UNSUPPORTED_FORMAT = "unsupported_format"
    FFMPEG_NOT_FOUND = "ffmpeg_not_found"
    FFMPEG_FAILED = "ffmpeg_failed"
    FFMPEG_NO_LIBASS = "ffmpeg_no_libass"
    TRANSCRIPTION_FAILED = "transcription_failed"
    TRANSCRIPTION_UNAVAILABLE = "transcription_unavailable"
    WHISPER_FAILED = "whisper_failed"
    NO_SPEECH = "no_speech"
    OLLAMA_UNAVAILABLE = "ollama_unavailable"
    OLLAMA_ERROR = "ollama_error"
    ANALYSIS_FAILED = "analysis_failed"
    NO_CANDIDATES = "no_candidates"
    RENDER_FAILED = "render_failed"
    RENDER_CANCELLED = "render_cancelled"
    CANCELLED = "cancelled"
    OUT_OF_MEMORY = "out_of_memory"
    LOW_DISK = "low_disk"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    INVALID_INPUT = "invalid_input"
    INTERNAL = "internal_error"
    UPLOAD_TOO_LARGE = "upload_too_large"
    ASSET_FAILED = "asset_failed"


@dataclass
class ClipForgeError(Exception):
    """Base error with a friendly surface for the frontend."""

    code: str = ErrorCode.INTERNAL
    message: str = "Something went wrong."
    hint: str = ""
    detail: str = ""
    status_code: int = 400
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self, *, include_detail: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": {
                "code": self.code,
                "message": self.message,
                "hint": self.hint,
            }
        }
        if self.context:
            payload["error"]["context"] = self.context
        if include_detail and self.detail:
            payload["error"]["detail"] = self.detail[:4000]
        return payload


# --------------------------------------------------------------------------- #
# Convenience constructors
# --------------------------------------------------------------------------- #

def not_found(what: str, ident: str = "") -> ClipForgeError:
    return ClipForgeError(
        code=ErrorCode.NOT_FOUND,
        message=f"{what} not found" + (f": {ident}" if ident else "."),
        status_code=404,
    )


def invalid_input(message: str, hint: str = "") -> ClipForgeError:
    return ClipForgeError(code=ErrorCode.INVALID_INPUT, message=message, hint=hint, status_code=422)


def conflict(message: str, hint: str = "") -> ClipForgeError:
    return ClipForgeError(code=ErrorCode.CONFLICT, message=message, hint=hint, status_code=409)


# --------------------------------------------------------------------------- #
# Third-party error translation
# --------------------------------------------------------------------------- #

_YTDLP_SIGNATURES: Iterable[tuple[tuple[str, ...], str, str, str]] = (
    (
        ("private video", "this video is private"),
        ErrorCode.VIDEO_PRIVATE,
        "That video is private, so it cannot be downloaded.",
        "Use a public video, or point CLIPFORGE at a cookies file in Settings to access it.",
    ),
    (
        ("age-restricted", "age restricted", "sign in to confirm your age"),
        ErrorCode.VIDEO_AGE_RESTRICTED,
        "That video is age restricted.",
        "Export YouTube cookies to a file and set the cookies path in Settings -> Download.",
    ),
    (
        ("video unavailable", "removed by the uploader", "no longer available"),
        ErrorCode.VIDEO_UNAVAILABLE,
        "That video is unavailable.",
        "Double-check the URL; the video may have been deleted or made private.",
    ),
    (
        ("not available in your country", "geo restricted", "blocked in your country"),
        ErrorCode.VIDEO_GEO_BLOCKED,
        "That video is not available from your region.",
        "Try a different source or use a proxy for the download step.",
    ),
    (
        ("is live", "live event will begin", "premieres in"),
        ErrorCode.LIVE_STREAM,
        "That URL is a live stream or an upcoming premiere.",
        "Wait until the stream ends and a VOD is available.",
    ),
    (
        ("unsupported url", "is not a valid url"),
        ErrorCode.INVALID_URL,
        "That does not look like a supported video URL.",
        "Paste a YouTube watch/share URL, e.g. https://www.youtube.com/watch?v=...",
    ),
    (
        ("sign in to confirm you", "confirm you're not a bot", "http error 429", "too many requests"),
        ErrorCode.DOWNLOAD_FAILED,
        "YouTube rate-limited this download or asked for sign-in.",
        "Retry in a few minutes, or supply a cookies file in Settings -> Download.",
    ),
    (
        ("requested format is not available", "no video formats found"),
        ErrorCode.UNSUPPORTED_FORMAT,
        "No downloadable video stream was found for that URL.",
        "Try again after updating yt-dlp (pip install -U yt-dlp).",
    ),
)


def from_download_error(exc: BaseException | str) -> ClipForgeError:
    """Translate a yt-dlp failure into a friendly error."""
    text = str(exc)
    low = text.lower()
    for needles, code, message, hint in _YTDLP_SIGNATURES:
        if any(n in low for n in needles):
            return ClipForgeError(code=code, message=message, hint=hint, detail=text, status_code=422)
    return ClipForgeError(
        code=ErrorCode.DOWNLOAD_FAILED,
        message="The video could not be downloaded.",
        hint="Check your connection, update yt-dlp, then retry the analysis.",
        detail=text,
        status_code=502,
    )


_FFMPEG_SIGNATURES: Iterable[tuple[tuple[str, ...], str, str, str]] = (
    (
        ("no space left on device",),
        ErrorCode.LOW_DISK,
        "The disk ran out of space while processing.",
        "Free up space or point the data directory at a larger drive in Settings -> Storage.",
    ),
    (
        ("cannot allocate memory", "out of memory", "std::bad_alloc", "cuda out of memory"),
        ErrorCode.OUT_OF_MEMORY,
        "Ran out of memory while processing.",
        "Close other apps, lower the Whisper model size, or disable GPU acceleration.",
    ),
    (
        ("invalid data found when processing input", "moov atom not found", "could not find codec parameters"),
        ErrorCode.UNSUPPORTED_FORMAT,
        "That media file could not be decoded.",
        "The download may be incomplete or corrupt. Delete the project and analyse again.",
    ),
    (
        ("no such filter", "filter not found"),
        ErrorCode.FFMPEG_NO_LIBASS,
        "Your ffmpeg build is missing a filter CLIPFORGE needs.",
        "Install a full ffmpeg build (libass/drawtext enabled) or enable the bundled binary in Settings -> Video.",
    ),
)


def from_ffmpeg_error(stderr_tail: str, code: str = ErrorCode.FFMPEG_FAILED) -> ClipForgeError:
    low = stderr_tail.lower()
    for needles, mapped_code, message, hint in _FFMPEG_SIGNATURES:
        if any(n in low for n in needles):
            return ClipForgeError(code=mapped_code, message=message, hint=hint, detail=stderr_tail, status_code=500)
    return ClipForgeError(
        code=code,
        message="The video engine (ffmpeg) failed while processing this file.",
        hint="Open Settings -> Diagnostics for the full ffmpeg output, then retry.",
        detail=stderr_tail,
        status_code=500,
    )


__all__ = [
    "ClipForgeError",
    "ErrorCode",
    "conflict",
    "from_download_error",
    "from_ffmpeg_error",
    "invalid_input",
    "not_found",
]
