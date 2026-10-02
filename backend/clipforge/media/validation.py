"""Media validation rules shared by the upload API and the worker's VALIDATE stage."""

from __future__ import annotations

from clipforge.core.config import Settings
from clipforge.core.errors import AppError, ErrorCode
from clipforge.media.ffmpeg import MediaInfo

ALLOWED_FORMAT_TOKENS = ("mov", "mp4", "m4a", "matroska", "webm")
ALLOWED_VIDEO_CODECS = {
    "h264", "hevc", "h265", "vp8", "vp9", "av1", "mpeg4", "mpeg2video", "prores", "mjpeg", "dnxhd",
    "theora", "vc1", "wmv3",
}
ALLOWED_AUDIO_CODECS = {
    "aac", "mp3", "opus", "vorbis", "flac", "ac3", "eac3", "pcm_s16le", "pcm_s24le", "pcm_f32le", "alac",
    "mp2", "pcm_s16be",
}


def validate_media_info(info: MediaInfo, settings: Settings) -> None:
    """Raise a stable AppError if the probed media must be rejected (TRD §9)."""
    if not any(tok in info.format_name for tok in ALLOWED_FORMAT_TOKENS):
        raise AppError(ErrorCode.UNSUPPORTED_FORMAT, details={"container": info.format_name})
    video = info.video
    if video is None:
        raise AppError(ErrorCode.UNSUPPORTED_FORMAT, "The file has no video stream.")
    if (video.codec_name or "").lower() not in ALLOWED_VIDEO_CODECS:
        raise AppError(ErrorCode.UNSUPPORTED_FORMAT, "The video codec is not supported.",
                       details={"video_codec": video.codec_name})
    audio = info.audio
    if audio is not None and (audio.codec_name or "").lower() not in ALLOWED_AUDIO_CODECS:
        raise AppError(ErrorCode.UNSUPPORTED_FORMAT, "The audio codec is not supported.",
                       details={"audio_codec": audio.codec_name})
    width, height = info.display_size
    if width <= 0 or height <= 0:
        raise AppError(ErrorCode.MEDIA_CORRUPTED, "The video dimensions could not be read.")
    if width > settings.MAX_VIDEO_WIDTH or height > settings.MAX_VIDEO_HEIGHT:
        raise AppError(ErrorCode.UNSUPPORTED_FORMAT,
                       f"Resolution {width}x{height} exceeds the maximum of "
                       f"{settings.MAX_VIDEO_WIDTH}x{settings.MAX_VIDEO_HEIGHT}.")
    if info.duration <= 0:
        raise AppError(ErrorCode.MEDIA_CORRUPTED, "The video duration could not be read.")
    if info.duration > settings.MAX_VIDEO_SECONDS:
        raise AppError(ErrorCode.MEDIA_TOO_LONG,
                       f"Videos can be at most {settings.MAX_VIDEO_SECONDS // 60} minutes long.")
    if info.duration < settings.MIN_VIDEO_SECONDS:
        raise AppError(ErrorCode.MEDIA_TOO_SHORT)
