"""Source acquisition: any video link (yt-dlp) and local uploads.

Three kinds of link are accepted, and they all end up as a local file:

* **YouTube** - watch, share, Shorts, embed and live URLs;
* **direct media** - a plain ``.mp4`` / ``.mov`` / ``.mkv`` / ``.webm`` / HLS link;
* **any other site** - everything yt-dlp knows how to extract (Vimeo, X/Twitter,
  TikTok, Instagram, Dailymotion, Twitch VODs, …).

Downloads never scrape copyrighted gameplay or music; they only fetch the video
the user asked for. Everything is written inside the project directory so the
rest of the app can assume paths are local and containment-checked.
"""

from __future__ import annotations

import hashlib
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

#: Extensions that mean "this URL *is* the media file".
DIRECT_MEDIA_SUFFIXES = {
    ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".flv", ".ts", ".m2ts",
    ".mpg", ".mpeg", ".3gp", ".ogv", ".m3u8", ".mpd", ".mp3", ".m4a", ".wav", ".aac",
}

#: ``Content-Type`` prefixes that mark a URL as a direct media file.
DIRECT_MEDIA_TYPES = ("video/", "audio/", "application/x-mpegurl", "application/vnd.apple.mpegurl", "application/dash+xml")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


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


@dataclass(frozen=True)
class SourceRef:
    """A validated video link, whatever site (or CDN) it points at."""

    kind: str          # "youtube" | "direct" | "web"
    url: str           # absolute URL handed to yt-dlp
    source_id: str     # stable identifier (YouTube id, or a hash of the URL)
    filename: str      # safe filename stem used for the downloaded file
    label: str         # short human description for logs and errors
    suffix: str = ""   # media extension, when the link exposes one

    @property
    def is_youtube(self) -> bool:
        return self.kind == "youtube"

    @property
    def is_direct(self) -> bool:
        return self.kind == "direct"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "url": self.url,
            "source_id": self.source_id,
            "filename": self.filename,
            "label": self.label,
            "suffix": self.suffix,
        }


def _stem_from_url(parsed: Any, fallback: str) -> str:
    """A safe, readable filename stem derived from a URL path or query."""
    path_name = Path(parsed.path or "").name
    candidate = path_name if Path(path_name).suffix.lower() in DIRECT_MEDIA_SUFFIXES else ""
    if not candidate:
        # Delivery URLs often hide the real name in a query parameter
        # (?file=episode.mp4); that beats an endpoint name like "get".
        for key in ("file", "filename", "name", "title", "src", "url"):
            value = (parse_qs(parsed.query or "").get(key) or [""])[0]
            if value:
                candidate = Path(value).name
                break
    candidate = candidate or path_name
    stem = Path(candidate).stem if candidate else fallback
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")[:80]
    return stem or fallback


def classify_url(url: str, *, probe: bool = False) -> SourceRef:
    """Turn any pasted link into a :class:`SourceRef` (or raise a friendly error).

    Nothing here needs the network unless ``probe`` is set: a link is treated as
    a direct media file when its path ends in a media extension, and otherwise it
    is handed to yt-dlp, which knows how to extract from thousands of sites.
    """
    raw = (url or "").strip().strip("<>")
    if not raw:
        raise invalid_input(
            "Paste a video link first.",
            "Example: https://www.youtube.com/watch?v=… or https://example.com/talk.mp4",
        )
    if "://" not in raw:
        raw = "https://" + raw

    parsed = urlparse(raw)
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        raise ClipForgeError(
            code=ErrorCode.INVALID_URL,
            message=f"'{scheme or raw}' links cannot be downloaded.",
            hint="Paste an http(s) link to a video, or upload the file instead.",
            status_code=422,
        )

    host = (parsed.netloc or "").lower().split(":")[0]
    if not host or "." not in host:
        raise ClipForgeError(
            code=ErrorCode.INVALID_URL,
            message="That link does not point at a website.",
            hint="Example: https://www.youtube.com/watch?v=… or https://example.com/talk.mp4",
            status_code=422,
        )

    if host in YOUTUBE_HOSTS:
        video_id = youtube_video_id(raw, parsed=parsed)
        return SourceRef(kind="youtube", url=canonical_watch_url(video_id), source_id=video_id,
                         filename=video_id, label="YouTube")

    suffix = Path(parsed.path or "").suffix.lower()
    if suffix not in DIRECT_MEDIA_SUFFIXES:
        for value in parse_qs(parsed.query or "").values():
            candidate = Path((value or [""])[0]).suffix.lower()
            if candidate in DIRECT_MEDIA_SUFFIXES:
                suffix = candidate
                break
    if probe and not suffix:
        suffix = _probe_media_suffix(raw)

    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    stem = _stem_from_url(parsed, digest)
    if suffix:
        return SourceRef(kind="direct", url=raw, source_id=digest, filename=stem, label=f"direct file · {host}", suffix=suffix)
    return SourceRef(kind="web", url=raw, source_id=digest, filename=stem, label=host)


def _probe_media_suffix(url: str) -> str:
    """Best-effort ``Content-Type`` check for links with no extension."""
    try:
        import httpx

        with httpx.Client(follow_redirects=True, timeout=12.0, headers={"User-Agent": USER_AGENT}) as client:
            response = client.head(url)
            if response.status_code >= 400:
                response = client.get(url, headers={"Range": "bytes=0-0"})
            content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
    except Exception as exc:  # noqa: BLE001 - a failed probe only changes the label
        log.debug("HEAD probe failed for %s: %s", url, exc)
        return ""
    if not content_type.startswith(DIRECT_MEDIA_TYPES):
        return ""
    guesses = {
        "video/mp4": ".mp4", "video/webm": ".webm", "video/quicktime": ".mov",
        "video/x-matroska": ".mkv", "audio/mpeg": ".mp3", "audio/mp4": ".m4a",
        "audio/wav": ".wav", "audio/x-wav": ".wav",
    }
    return guesses.get(content_type, ".mp4")


def youtube_video_id(url: str, *, parsed: Any | None = None) -> str:
    """Return the canonical 11-character id of a YouTube URL."""
    raw = (url or "").strip()
    if "://" not in raw:
        raw = "https://" + raw
    parsed = parsed or urlparse(raw)
    host = (parsed.netloc or "").lower().split(":")[0]

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


def parse_youtube_url(url: str) -> str:
    """Backwards-compatible helper: YouTube link in, video id out."""
    return youtube_video_id(url)


def canonical_watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


# --------------------------------------------------------------------------- #
# yt-dlp options
# --------------------------------------------------------------------------- #


def _base_options(settings: AppSettings, source: SourceRef | None = None) -> dict[str, Any]:
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
        "http_headers": {"User-Agent": USER_AGENT},
        "extractor_args": {"youtube": {"player_client": ["default", "web_safari"]}},
    }
    if source is not None and source.is_direct:
        # A plain .mp4/.m3u8 link must not be scraped as a web page: force the
        # generic extractor so yt-dlp streams the file instead of guessing.
        options["force_generic_extractor"] = True
        options["extractor_args"] = {}
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
    source = classify_url(url)
    if progress:
        progress("Reading video metadata")

    if source.is_direct:
        # A direct file has no page to scrape: the real metadata (duration,
        # resolution) comes from ffprobe right after the download.
        info: dict[str, Any] = {}
        try:
            info = _head_info(source.url)
        except Exception as exc:  # noqa: BLE001 - metadata is a nicety
            log.debug("HEAD failed for %s: %s", source.url, exc)
        return SourceMetadata(
            url=source.url,
            video_id=source.source_id,
            title=info.get("title") or source.filename,
            channel=urlparse(source.url).netloc,
            duration=float(info.get("duration") or 0),
            thumbnail_url="",
            description="",
            webpage_url=source.url,
            raw={"id": source.source_id, "extractor": "direct", "size_bytes": info.get("size_bytes", 0)},
        )

    options = _base_options(settings, source) | {"skip_download": True}
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(source.url, download=False) or {}
    except Exception as exc:  # noqa: BLE001 - translated below
        raise from_download_error(exc) from exc

    if not info:
        raise ClipForgeError(
            code=ErrorCode.VIDEO_UNAVAILABLE,
            message=f"{source.label} returned no information for that video.",
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
    _check_duration(duration, settings)

    thumbnail = info.get("thumbnail") or ""
    if not thumbnail:
        thumbs = info.get("thumbnails") or []
        if thumbs:
            thumbnail = thumbs[-1].get("url", "")

    return SourceMetadata(
        url=source.url,
        video_id=info.get("id") or source.source_id,
        title=(info.get("title") or source.filename or "Untitled video").strip(),
        channel=(info.get("uploader") or info.get("channel") or urlparse(source.url).netloc or "").strip(),
        duration=duration,
        thumbnail_url=thumbnail,
        description=info.get("description") or "",
        upload_date=str(info.get("upload_date") or ""),
        view_count=int(info.get("view_count") or 0),
        is_live=bool(info.get("is_live")),
        webpage_url=info.get("webpage_url") or source.url,
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


def _check_duration(duration: float, settings: AppSettings) -> None:
    if duration and duration > settings.max_source_hours * 3600:
        raise ClipForgeError(
            code=ErrorCode.VIDEO_TOO_LONG,
            message=f"That video is {duration / 3600:.1f} hours long; the limit is {settings.max_source_hours:g} hours.",
            hint="Raise the limit in Settings -> Video, or pick a shorter video.",
            status_code=422,
        )


def _head_info(url: str) -> dict[str, Any]:
    """Cheap ``HEAD`` read: filename, size and (rarely) duration for direct links."""
    import httpx

    with httpx.Client(follow_redirects=True, timeout=15.0, headers={"User-Agent": USER_AGENT}) as client:
        response = client.head(url)
        if response.status_code >= 400:
            # Some CDNs and dev servers refuse HEAD; a ranged GET says the same thing.
            response = client.get(url, headers={"Range": "bytes=0-0"})
    headers = response.headers
    disposition = headers.get("content-disposition") or ""
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', disposition, re.IGNORECASE)
    name = Path(match.group(1)).stem if match else ""
    return {
        "title": re.sub(r"[_+]+", " ", name).strip(),
        "size_bytes": int(headers.get("content-length") or 0),
        "duration": float(headers.get("x-content-duration") or 0),
    }



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
    source = classify_url(url)
    stem = source.filename or source.source_id

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

    options = _base_options(settings, source) | {
        "format": format_selector(settings),
        "outtmpl": {"default": str(destination / f"{stem}.%(ext)s")},
        "merge_output_format": "mp4",
        "progress_hooks": [hook],
        "concurrent_fragment_downloads": max(1, settings.download_concurrency),
        "postprocessors": [{"key": "FFmpegMetadata", "add_metadata": False}] if _ffmpeg_available() else [],
        "overwrites": False,
        "keepvideo": False,
    }

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(source.url, download=True)
    except DownloadCancelled as exc:
        raise ClipForgeError(
            code=ErrorCode.CANCELLED,
            message="Download cancelled.",
            status_code=409,
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise from_download_error(exc) from exc

    media_path = _resolve_downloaded_path(destination, stem, info)
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


def _resolve_downloaded_path(destination: Path, stem: str, info: dict[str, Any] | None) -> Path | None:
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

    media_exts = {".mp4", ".mkv", ".webm", ".m4a", ".mov", ".mp3", ".opus", ".aac", ".flv", ".ts", ".mpg", ".mpeg", ".ogv", ".3gp", ".m4v", ".avi"}
    # Glob patterns treat [] as a character class, so an arbitrary filename stem
    # is matched literally instead.
    matches = [
        path for path in destination.iterdir()
        if path.is_file() and path.suffix.lower() in media_exts and path.stat().st_size > 1024
        and (path.stem == stem or path.stem.startswith(f"{stem}."))
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
    """Download the site's captions (manual first, automatic as fallback).

    Works for any yt-dlp source that publishes a subtitle track - YouTube, Vimeo
    and friends. A plain media file has no track, so it returns ``None`` and the
    pipeline falls back to local speech recognition.
    """
    source = classify_url(url)
    if source.is_direct:
        return None
    yt_dlp = _import_ytdlp()
    settings = get_settings()
    destination.mkdir(parents=True, exist_ok=True)
    stem = source.filename or source.source_id
    options = _base_options(settings, source) | {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": list(languages),
        "subtitlesformat": "json3/vtt/srt/best",
        "outtmpl": {"default": str(destination / f"{stem}.%(ext)s")},
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([source.url])
    except Exception as exc:  # noqa: BLE001
        log.warning("caption download failed: %s", exc)
        return None

    for suffix in ("json3", "vtt", "srt"):
        candidates = sorted(destination.glob(f"{stem}*.{suffix}"), key=lambda p: (len(p.name), p.name))
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
    "DIRECT_MEDIA_SUFFIXES",
    "TRANSCRIPT_EXTENSIONS",
    "SourceMetadata",
    "SourceRef",
    "canonical_watch_url",
    "classify_url",
    "download_captions",
    "download_video",
    "fetch_metadata",
    "format_selector",
    "import_local_file",
    "parse_json3_captions",
    "parse_subtitle_captions",
    "parse_transcript_file",
    "parse_youtube_url",
    "youtube_video_id",
    "safe_upload_path",
    "transcript_files",
]
