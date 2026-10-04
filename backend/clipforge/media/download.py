"""Source acquisition: YouTube (yt-dlp) and local uploads.

Downloads never scrape copyrighted gameplay or music; they only fetch the video
the user asked for. Everything is written inside the project directory so the
rest of the app can assume paths are local and containment-checked.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from ..config import AppSettings, get_settings
from ..errors import ClipForgeError, ErrorCode, from_download_error, invalid_input
from ..logging_setup import get_logger

log = get_logger("clipforge.ai")

YOUTUBE_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
    "youtu.be", "www.youtu.be", "youtube-nocookie.com", "www.youtube-nocookie.com",
}

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,20}$")


@dataclass
class SourceMetadata:
    url: str
    video_id: str
    title: str
    channel: str
    duration: float
    thumbnail_url: str
    description: str
    upload_date: str = ""
    view_count: int = 0
    is_live: bool = False
    has_audio: bool = True
    webpage_url: str = ""
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "video_id": self.video_id,
            "title": self.title,
            "channel": self.channel,
            "duration": self.duration,
            "thumbnail_url": self.thumbnail_url,
            "description": (self.description or "")[:4000],
            "upload_date": self.upload_date,
            "view_count": self.view_count,
            "is_live": self.is_live,
            "webpage_url": self.webpage_url or self.url,
        }


# --------------------------------------------------------------------------- #
# URL validation
# --------------------------------------------------------------------------- #


def parse_youtube_url(url: str) -> str:
    """Validate a YouTube URL and return the canonical 11-character video id."""
    raw = (url or "").strip()
    if not raw:
        raise invalid_input("Paste a YouTube URL first.", "Example: https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    if "://" not in raw:
        raw = "https://" + raw
    parsed = urlparse(raw)
    host = (parsed.netloc or "").lower().split(":")[0]
    if host not in YOUTUBE_HOSTS:
        raise ClipForgeError(
            code=ErrorCode.INVALID_URL,
            message=f"{host or 'That URL'} is not a supported source.",
            hint="CLIPFORGE accepts YouTube watch, share and Shorts URLs. For anything else, upload the file instead.",
            status_code=422,
        )

    video_id = ""
    if host.endswith("youtu.be"):
        video_id = parsed.path.lstrip("/").split("/")[0]
    elif parsed.path.startswith("/watch"):
        video_id = (parse_qs(parsed.query).get("v") or [""])[0]
    elif parsed.path.startswith(("/shorts/", "/embed/", "/live/", "/v/")):
        parts = [part for part in parsed.path.split("/") if part]
        video_id = parts[1] if len(parts) > 1 else ""

    video_id = (video_id or "").split("&")[0].strip()
    if not VIDEO_ID_RE.match(video_id):
        raise ClipForgeError(
            code=ErrorCode.INVALID_URL,
            message="That YouTube URL does not contain a valid video id.",
            hint="Copy the link straight from the YouTube share button, e.g. https://youtu.be/VIDEOID",
            status_code=422,
        )
    return video_id


def canonical_watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


# --------------------------------------------------------------------------- #
# yt-dlp options
# --------------------------------------------------------------------------- #


def _base_options(settings: AppSettings) -> dict[str, Any]:
    options: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "retries": 5,
        "fragment_retries": 5,
        "socket_timeout": 30,
        "consoletitle": False,
        "nocheckcertificate": False,
        "ignoreerrors": False,
        "age_limit": 99,
        "extractor_args": {"youtube": {"player_client": ["default", "web_safari"]}},
    }
    if settings.cookies_path and Path(settings.cookies_path).exists():
        options["cookiefile"] = settings.cookies_path
    if settings.proxy:
        options["proxy"] = settings.proxy
    return options


def format_selector(settings: AppSettings) -> str:
    height = max(360, min(int(settings.max_download_height or 1080), 2160))
    if settings.prefer_mp4:
        return (
            f"bv*[height<={height}][ext=mp4][vcodec^=avc1]+ba[ext=m4a]/"
            f"bv*[height<={height}][ext=mp4]+ba[ext=m4a]/"
            f"b[height<={height}][ext=mp4]/bv*[height<={height}]+ba/b[height<={height}]/bv*+ba/b"
        )
    return f"bv*[height<={height}]+ba/b[height<={height}]/bv*+ba/b"


def fetch_metadata(url: str, *, progress: Callable[[str], None] | None = None) -> SourceMetadata:
    """Read metadata without downloading (fast: used to show the video card)."""
    yt_dlp = _import_ytdlp()
    settings = get_settings()
    video_id = parse_youtube_url(url)
    options = _base_options(settings) | {"skip_download": True}
    if progress:
        progress("Reading video metadata")

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(canonical_watch_url(video_id), download=False)
    except Exception as exc:  # noqa: BLE001 - translated below
        raise from_download_error(exc) from exc

    if not info:
        raise ClipForgeError(
            code=ErrorCode.VIDEO_UNAVAILABLE,
            message="YouTube returned no information for that video.",
            status_code=404,
        )

    if info.get("is_live"):
        raise ClipForgeError(
            code=ErrorCode.LIVE_STREAM,
            message="That video is still live.",
            hint="Wait for the stream to finish, then analyse it again.",
            status_code=422,
        )

    duration = float(info.get("duration") or 0)
    if duration and duration > settings.max_source_hours * 3600:
        raise ClipForgeError(
            code=ErrorCode.VIDEO_TOO_LONG,
            message=f"That video is {duration / 3600:.1f} hours long; the limit is {settings.max_source_hours:g} hours.",
            hint="Raise the limit in Settings -> Video, or pick a shorter video.",
            status_code=422,
        )

    thumbnail = info.get("thumbnail") or ""
    if not thumbnail:
        thumbs = info.get("thumbnails") or []
        if thumbs:
            thumbnail = thumbs[-1].get("url", "")

    return SourceMetadata(
        url=canonical_watch_url(video_id),
        video_id=video_id,
        title=(info.get("title") or "Untitled video").strip(),
        channel=(info.get("uploader") or info.get("channel") or "").strip(),
        duration=duration,
        thumbnail_url=thumbnail,
        description=info.get("description") or "",
        upload_date=str(info.get("upload_date") or ""),
        view_count=int(info.get("view_count") or 0),
        is_live=bool(info.get("is_live")),
        webpage_url=info.get("webpage_url") or canonical_watch_url(video_id),
        raw={
            "id": info.get("id"),
            "extractor": info.get("extractor"),
            "subtitles": sorted((info.get("subtitles") or {}).keys()),
            "automatic_captions": sorted((info.get("automatic_captions") or {}).keys()),
            "chapters": info.get("chapters") or [],
            "categories": info.get("categories") or [],
            "tags": (info.get("tags") or [])[:20],
        },
    )


def _import_ytdlp():
    try:
        import yt_dlp  # type: ignore

        return yt_dlp
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise ClipForgeError(
            code=ErrorCode.INTERNAL,
            message="yt-dlp is not installed, so YouTube downloads are unavailable.",
            hint="Run: pip install yt-dlp",
            status_code=500,
        ) from exc


class DownloadCancelled(Exception):
    """Raised from inside a yt-dlp progress hook to abort a download."""


def download_video(
    url: str,
    destination: Path,
    *,
    progress: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Download the best stream for ``url`` into ``destination``.

    Returns the media path plus the last progress snapshot (speed/eta).
    """
    yt_dlp = _import_ytdlp()
    settings = get_settings()
    destination.mkdir(parents=True, exist_ok=True)
    video_id = parse_youtube_url(url)

    state: dict[str, Any] = {"filename": "", "speed": 0.0, "eta": 0.0, "downloaded": 0, "total": 0}

    def hook(payload: dict[str, Any]) -> None:
        if should_cancel and should_cancel():
            raise DownloadCancelled()
        status = payload.get("status")
        if status == "downloading":
            total = payload.get("total_bytes") or payload.get("total_bytes_estimate") or 0
            downloaded = payload.get("downloaded_bytes") or 0
            state.update(
                {
                    "filename": payload.get("filename", ""),
                    "speed": payload.get("speed") or 0.0,
                    "eta": payload.get("eta") or 0.0,
                    "downloaded": downloaded,
                    "total": total,
                }
            )
            if progress:
                fraction = (downloaded / total) if total else 0.0
                speed_mb = (payload.get("speed") or 0) / 1e6
                progress(fraction, f"{downloaded / 1e6:.1f} MB · {speed_mb:.1f} MB/s" if total else "downloading")
        elif status == "finished":
            if progress:
                progress(1.0, "merging streams")

    options = _base_options(settings) | {
        "format": format_selector(settings),
        "outtmpl": {"default": str(destination / f"{video_id}.%(ext)s")},
        "merge_output_format": "mp4",
        "progress_hooks": [hook],
        "concurrent_fragment_downloads": max(1, settings.download_concurrency),
        "postprocessors": [{"key": "FFmpegMetadata", "add_metadata": False}] if _ffmpeg_available() else [],
        "overwrites": False,
        "keepvideo": False,
    }

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(canonical_watch_url(video_id), download=True)
    except DownloadCancelled as exc:
        raise ClipForgeError(
            code=ErrorCode.CANCELLED,
            message="Download cancelled.",
            status_code=409,
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise from_download_error(exc) from exc

    media_path = _resolve_downloaded_path(destination, video_id, info)
    if media_path is None:
        raise ClipForgeError(
            code=ErrorCode.DOWNLOAD_FAILED,
            message="The download finished but no media file was produced.",
            hint="Check free disk space, then retry. Details are in logs/render.log.",
            status_code=500,
        )
    log.info("downloaded %s (%.1f MB)", media_path.name, media_path.stat().st_size / 1e6)
    return media_path, state


def _ffmpeg_available() -> bool:
    try:
        from ..system import ffmpeg_info

        return ffmpeg_info().available
    except Exception:
        return False


def _resolve_downloaded_path(destination: Path, video_id: str, info: dict[str, Any] | None) -> Path | None:
    candidates: list[Path] = []
    for key in ("requested_downloads", "entries"):
        for entry in (info or {}).get(key) or []:
            for field in ("filepath", "_filename", "filename"):
                value = entry.get(field) if isinstance(entry, dict) else None
                if value:
                    candidates.append(Path(value))
    if info is not None:
        for field in ("filepath", "_filename"):
            value = info.get(field)
            if value:
                candidates.append(Path(value))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate

    media_exts = {".mp4", ".mkv", ".webm", ".m4a", ".mov", ".mp3", ".opus", ".aac", ".flv", ".ts"}
    matches = [
        path for path in destination.glob(f"{video_id}*")
        if path.suffix.lower() in media_exts and path.exists() and path.stat().st_size > 1024
    ]
    if not matches:
        matches = [
            path for path in destination.iterdir()
            if path.is_file() and path.suffix.lower() in media_exts and path.stat().st_size > 1024
        ]
    if not matches:
        return None
    # Prefer a merged mp4/mkv over leftover audio-only fragments.
    matches.sort(key=lambda p: (p.suffix.lower() not in {".mp4", ".mkv", ".webm", ".mov"}, -p.stat().st_size))
    return matches[0]


# --------------------------------------------------------------------------- #
# Captions from YouTube (fallback when no local Whisper is installed)
# --------------------------------------------------------------------------- #

_SUBTITLE_LANGS = ("en", "en-US", "en-GB", "hi", "hi-IN", "en-orig", "a.en")


def download_captions(url: str, destination: Path, *, languages: tuple[str, ...] = _SUBTITLE_LANGS) -> Path | None:
    """Download YouTube captions (manual first, automatic as fallback).

    Used only when local speech-to-text is unavailable, and clearly labelled in
    the UI as an external-caption fallback.
    """
    yt_dlp = _import_ytdlp()
    settings = get_settings()
    destination.mkdir(parents=True, exist_ok=True)
    video_id = parse_youtube_url(url)
    options = _base_options(settings) | {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": list(languages),
        "subtitlesformat": "json3",
        "outtmpl": {"default": str(destination / f"{video_id}.%(ext)s")},
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([canonical_watch_url(video_id)])
    except Exception as exc:  # noqa: BLE001
        log.warning("caption download failed: %s", exc)
        return None

    candidates = sorted(destination.glob(f"{video_id}*.json3"), key=lambda p: (len(p.name), p.name))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 64:
            return candidate
    return None


def parse_json3_captions(path: Path) -> list[dict[str, Any]]:
    """Convert a YouTube ``json3`` caption file into word records."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    words: list[dict[str, Any]] = []
    for event in payload.get("events") or []:
        base_ms = event.get("tStartMs") or 0
        duration = event.get("dDurationMs") or 0
        segments = event.get("segs") or []
        if not segments:
            continue
        plain = "".join(seg.get("utf8", "") for seg in segments).strip()
        if not plain:
            continue
        confidences = [seg.get("acAsrConf") for seg in segments if isinstance(seg.get("acAsrConf"), (int, float))]
        confidence = (sum(confidences) / len(confidences)) if confidences else 0.7

        # Auto captions are word-level; manual captions carry whole lines.
        if len(segments) > 1:
            for seg in segments:
                text = (seg.get("utf8") or "").strip()
                if not text:
                    continue
                start = (base_ms + (seg.get("tOffsetMs") or 0)) / 1000
                words.append(
                    {
                        "word": text,
                        "start": round(start, 3),
                        "end": round(start + 0.4, 3),
                        "confidence": round(float(confidence), 3),
                    }
                )
        else:
            start = base_ms / 1000
            end = (base_ms + duration) / 1000 if duration else start + 1.5
            tokens = plain.split()
            span = max(end - start, 0.2) / max(len(tokens), 1)
            for index, token in enumerate(tokens):
                words.append(
                    {
                        "word": token,
                        "start": round(start + index * span, 3),
                        "end": round(start + (index + 1) * span, 3),
                        "confidence": round(float(confidence), 3),
                    }
                )
    # Fix overlapping timings produced by coarse caption tracks.
    for index in range(len(words) - 1):
        if words[index]["end"] > words[index + 1]["start"]:
            words[index]["end"] = max(words[index]["start"] + 0.05, words[index + 1]["start"])
    return words


# --------------------------------------------------------------------------- #
# Transcript files (SRT / VTT / json3) provided by the user
# --------------------------------------------------------------------------- #

TRANSCRIPT_EXTENSIONS = (".srt", ".vtt", ".json3", ".txt")

_SRT_TIME_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
)
_TAG_RE = re.compile(r"<[^>]+>")


def _clock_to_seconds(value: str) -> float:
    value = value.strip().replace(",", ".")
    parts = value.split(":")
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return 0.0
    while len(numbers) < 3:
        numbers.insert(0, 0.0)
    hours, minutes, seconds = numbers[-3], numbers[-2], numbers[-1]
    return hours * 3600 + minutes * 60 + seconds


def _words_from_cue(text: str, start: float, end: float, confidence: float = 0.6) -> list[dict[str, Any]]:
    """Split one subtitle cue into word records with proportional timings.

    Subtitle cues only carry line-level timings, so per-word times are estimated
    by character length inside the cue. That estimate is clearly reported back to
    the UI (``word timings estimated from subtitle cues``) and is good enough for
    caption display, while real ASR remains the accurate path.
    """
    tokens = [token for token in _TAG_RE.sub(" ", text).split() if token]
    if not tokens:
        return []
    total_chars = sum(len(token) for token in tokens) or 1
    span = max(end - start, 0.08)
    words: list[dict[str, Any]] = []
    cursor = start
    for token in tokens:
        share = span * (len(token) / total_chars)
        words.append(
            {
                "word": token,
                "start": round(cursor, 3),
                "end": round(min(cursor + share, end), 3),
                "confidence": round(confidence, 3),
            }
        )
        cursor += share
    return words


def parse_subtitle_captions(path: str | Path) -> list[dict[str, Any]]:
    """Parse an SRT or WebVTT file into word records (line timings → words)."""
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    words: list[dict[str, Any]] = []
    cues = _SRT_TIME_RE.split(raw)
    # ``re.split`` with groups yields [preamble, start, end, body, start, end, body, ...]
    for index in range(1, len(cues) - 2, 3):
        start = _clock_to_seconds(cues[index])
        end = _clock_to_seconds(cues[index + 1])
        body = cues[index + 2].strip().splitlines()
        text = " ".join(line for line in body if line.strip() and not line.strip().isdigit())
        if not text:
            continue
        if end <= start:
            end = start + max(0.6, 0.05 * len(text.split()))
        words.extend(_words_from_cue(text, start, end))
    return _repair_caption_timings(words)


def parse_transcript_file(path: str | Path) -> list[dict[str, Any]]:
    """Parse any supported transcript file (``.json3``, ``.srt``, ``.vtt``, ``.txt``)."""
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix == ".json3":
        return parse_json3_captions(target)
    if suffix in {".srt", ".vtt"}:
        return parse_subtitle_captions(target)
    if suffix == ".txt":
        return _parse_plain_transcript(target)
    return []


def _parse_plain_transcript(path: Path) -> list[dict[str, Any]]:
    """Parse ``[hh:mm:ss] text`` or ``mm:ss text`` style plain transcripts."""
    words: list[dict[str, Any]] = []
    pattern = re.compile(r"^\s*\[?(?P<stamp>\d{1,2}:\d{2}(?::\d{2})?)\]?\s*[-–:]?\s*(?P<text>.+)$")
    for line in (path.read_text(encoding="utf-8", errors="replace").splitlines() if path.exists() else []):
        match = pattern.match(line)
        if not match:
            continue
        start = _clock_to_seconds(match.group("stamp"))
        words.extend(_words_from_cue(match.group("text"), start, start + 3.0))
    return _repair_caption_timings(words)


def _repair_caption_timings(words: list[dict[str, Any]], *, min_gap: float = 0.02) -> list[dict[str, Any]]:
    """Make a caption-derived word list strictly increasing (no overlaps)."""
    for index in range(len(words) - 1):
        current, following = words[index], words[index + 1]
        limit = max(following["start"] - min_gap, current["start"] + 0.05)
        if current["end"] > following["start"]:
            current["end"] = round(limit, 3)
        if current["end"] <= current["start"]:
            current["end"] = round(current["start"] + 0.05, 3)
    return words


def transcript_files(folder: Path) -> list[Path]:
    """Transcript/caption files a user dropped into a project folder (newest first)."""
    found: list[Path] = []
    if not folder.exists():
        return found
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in TRANSCRIPT_EXTENSIONS and path.stat().st_size > 32:
            found.append(path)
    priority = {".json3": 0, ".srt": 1, ".vtt": 2, ".txt": 3}
    return sorted(found, key=lambda item: (priority.get(item.suffix.lower(), 9), -item.stat().st_mtime))


# --------------------------------------------------------------------------- #
# Uploads
# --------------------------------------------------------------------------- #

ALLOWED_UPLOAD_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".mp3", ".m4a", ".wav"}


def safe_upload_path(destination_dir: Path, filename: str) -> Path:
    """Resolve an uploaded filename inside ``destination_dir`` (no traversal)."""
    cleaned = Path(filename or "upload.mp4").name
    suffix = Path(cleaned).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise ClipForgeError(
            code=ErrorCode.UNSUPPORTED_FORMAT,
            message=f"{suffix or 'That file type'} cannot be analysed.",
            hint="Supported: " + ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS)),
            status_code=422,
        )
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(cleaned).stem)[:80] or "upload"
    target = (destination_dir / f"{stem}{suffix}").resolve()
    destination_root = destination_dir.resolve()
    if destination_root not in target.parents and target != destination_root:
        raise ClipForgeError(
            code=ErrorCode.INVALID_INPUT,
            message="That upload path is not allowed.",
            status_code=403,
        )
    return target


def import_local_file(source: str | Path, destination: Path) -> Path:
    """Copy a user-picked local file into the project directory."""
    src = Path(source).expanduser()
    if not src.exists() or not src.is_file():
        raise ClipForgeError(
            code=ErrorCode.NOT_FOUND,
            message=f"File not found: {src.name}",
            status_code=404,
        )
    target = safe_upload_path(destination, src.name)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)
    return target


__all__ = [
    "ALLOWED_UPLOAD_EXTENSIONS",
    "TRANSCRIPT_EXTENSIONS",
    "SourceMetadata",
    "canonical_watch_url",
    "download_captions",
    "download_video",
    "fetch_metadata",
    "format_selector",
    "import_local_file",
    "parse_json3_captions",
    "parse_subtitle_captions",
    "parse_transcript_file",
    "parse_youtube_url",
    "safe_upload_path",
    "transcript_files",
]
