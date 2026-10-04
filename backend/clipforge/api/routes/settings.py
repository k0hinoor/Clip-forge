"""Settings and template routes.

``GET /api/settings`` returns the current values *plus* a machine-readable schema
(types, ranges, options, section, label) so the Settings page can be rendered
from the backend definition and never drifts from what the app actually uses.
"""

from __future__ import annotations

import json
from typing import Any, get_args

from fastapi import APIRouter, Body, Query
from sqlalchemy import select

from ...config import SECTION_FIELDS, SERVER_PATH_FIELDS, AppSettings, Env, get_settings, settings_store
from ...constants import CAPTION_PRESETS
from ...db import Template, session_scope
from ...errors import ClipForgeError, ErrorCode, not_found
from ...logging_setup import get_logger
from ...services import events
from ...system import invalidate_cache
from ..schemas import TemplateCreate

router = APIRouter(tags=["settings"])
log = get_logger(__name__)

FIELD_LABELS: dict[str, str] = {
    "concurrency": "Worker threads",
    "max_concurrent_renders": "Concurrent renders",
    "auto_start_worker": "Start the worker automatically",
    "llm_enabled": "Use the local LLM (Ollama)",
    "ollama_base_url": "Ollama URL",
    "ollama_model": "Ollama model",
    "ollama_timeout_seconds": "Model timeout (seconds)",
    "ollama_temperature": "Model temperature",
    "ollama_num_ctx": "Context window (tokens)",
    "llm_max_workers": "Parallel model requests",
    "llm_chunk_chars": "Transcript chunk size",
    "llm_max_chunks": "Maximum chunks analysed",
    "vision_enabled": "Vision analysis",
    "vision_model": "Vision model",
    "whisper_model": "Whisper model",
    "whisper_device": "Device",
    "whisper_compute_type": "Compute type",
    "whisper_beam_size": "Beam size",
    "whisper_batch_size": "Batch size",
    "whisper_vad": "Voice activity detection",
    "whisper_vad_min_silence_ms": "VAD min silence (ms)",
    "whisper_condition_on_previous_text": "Condition on previous text",
    "whisper_initial_prompt": "Initial prompt",
    "whisper_cache_dir": "Model cache directory",
    "word_alignment": "Word alignment",
    "diarization": "Speaker detection",
    "max_speakers": "Maximum speakers",
    "min_word_confidence": "Minimum word confidence",
    "language_hint": "Force language (blank = auto)",
    "keep_source_audio": "Keep extracted audio",
    "ffmpeg_path": "ffmpeg path",
    "ffprobe_path": "ffprobe path",
    "hw_accel": "Hardware acceleration",
    "output_fps": "Output frame rate",
    "allow_60fps": "Allow 60 fps",
    "render_preset": "x264 preset",
    "crf": "Quality (CRF, lower is better)",
    "video_bitrate_kbps": "Video bitrate (0 = CRF)",
    "audio_bitrate_kbps": "Audio bitrate",
    "cookies_path": "YouTube cookies file",
    "proxy": "Download proxy",
    "max_download_height": "Maximum download height",
    "prefer_mp4": "Prefer MP4 streams",
    "max_source_hours": "Maximum source length (hours)",
    "download_concurrency": "Parallel download fragments",
    "clip_mode": "Clip selection mode",
    "target_clip_seconds": "Target clip length (s)",
    "min_clip_seconds": "Minimum clip length (s)",
    "max_clip_seconds": "Maximum clip length (s)",
    "min_score": "Minimum quality score",
    "max_clips": "Maximum clips (0 = unlimited)",
    "aspect_ratio": "Aspect ratio",
    "smart_reframe": "Smart reframing",
    "auto_zoom": "Automatic punch-ins",
    "speaker_tracking": "Speaker tracking",
    "remove_silence": "Silence removal",
    "silence_threshold_db": "Silence threshold (dB)",
    "silence_min_duration": "Minimum pause to remove (s)",
    "silence_max_cut": "Maximum cut per pause (s)",
    "keep_natural_pauses": "Keep natural pauses (s)",
    "captions_enabled": "Burn captions into the video",
    "translate_captions": "Translate captions",
    "translation_language": "Translation target",
    "caption": "Caption style",
    "gameplay_enabled": "Use gameplay footage",
    "gameplay_mode": "Gameplay selection",
    "gameplay_category": "Gameplay category",
    "layout": "Default layout",
    "split_ratio": "Split ratio (podcast %)",
    "gameplay_volume": "Gameplay volume",
    "gameplay_random": "Randomise gameplay",
    "gameplay_pace": "Gameplay pacing",
    "broll_enabled": "Use B-roll",
    "broll_category": "B-roll category",
    "normalize_loudness": "Loudness normalisation",
    "target_lufs": "Target loudness (LUFS)",
    "true_peak_db": "True peak (dB)",
    "noise_reduction": "Noise reduction",
    "voice_boost": "Voice enhancement",
    "voice_gain_db": "Voice gain (dB)",
    "music_enabled": "Background music",
    "music_mood": "Music mood",
    "music_volume": "Music volume",
    "ducking": "Duck music under speech",
    "ducking_db": "Ducking amount (dB)",
    "export_dir": "Export folder",
    "export_filename_template": "Filename template",
    "auto_open_folder": "Open folder after export",
    "keep_source_video": "Keep the source video",
    "cache_transcripts": "Cache transcripts",
    "cache_downloads": "Cache downloads",
    "cleanup_days": "Clean up after (days)",
    "max_cache_gb": "Cache budget (GB)",
    "uploads_enabled": "Allow file uploads",
    "max_upload_gb": "Maximum upload (GB)",
}


FIELD_HELP: dict[str, str] = {
    "concurrency": "Analyses run one at a time; extra threads let renders run alongside an analysis.",
    "max_concurrent_renders": "How many clips may encode at the same time.",
    "auto_start_worker": "Process the queue as soon as the app starts.",
    "llm_enabled": "Ask a local Ollama model for its own read of the transcript. Analysis works without it.",
    "ollama_base_url": "Address of the Ollama server, e.g. http://127.0.0.1:11434.",
    "ollama_model": "Any chat model you have pulled in Ollama (ollama pull qwen3:4b).",
    "whisper_model": "tiny/base are fastest; small is the default; medium and large-v3 are most accurate but need more memory.",
    "whisper_device": "auto picks CUDA when an NVIDIA GPU is available.",
    "whisper_compute_type": "auto = int8 on CPU, float16 on GPU.",
    "whisper_beam_size": "Higher is slightly more accurate and slower (1-5).",
    "whisper_initial_prompt": "Names or jargon that appear in the video help recognition.",
    "word_alignment": "WhisperX alignment gives tighter word timings when it is installed.",
    "diarization": "energy = built-in speaker detection; off treats everything as one speaker.",
    "language_hint": "ISO code such as en, hi or es. Leave blank to detect automatically.",
    "keep_source_audio": "Keep the extracted 16 kHz WAV after analysis (it is re-extracted when needed).",
    "hw_accel": "GPU encoders are much faster; auto falls back to x264 when none is available.",
    "output_fps": "Frame rate of the rendered clips.",
    "allow_60fps": "When off, 50/60 fps settings render at 30 fps.",
    "render_preset": "Slower presets give smaller files at the same quality.",
    "crf": "18-23 is visually lossless to high quality; higher numbers mean smaller files.",
    "video_bitrate_kbps": "0 = constant quality (CRF). Set a bitrate only if a platform requires one.",
    "cookies_path": "Netscape cookies.txt for videos that need a login (age-restricted or members-only).",
    "proxy": "Used only for downloads, e.g. socks5://127.0.0.1:1080.",
    "max_source_hours": "Longer videos are rejected before downloading.",
    "clip_mode": "best = only the strongest moments, balanced = a sensible set, max = everything above the score.",
    "min_score": "Moments scoring below this (0-100) do not become clips.",
    "max_clips": "0 = no limit.",
    "aspect_ratio": "9:16 for Shorts/Reels/TikTok, 1:1 for feeds, 16:9 for YouTube.",
    "output_width": "Must match the aspect ratio; otherwise the ratio's default size is used.",
    "output_height": "Must match the aspect ratio; otherwise the ratio's default size is used.",
    "smart_reframe": "Follow faces and motion when cropping to vertical (uses OpenCV when installed).",
    "auto_zoom": "Subtle punch-ins on emphatic moments.",
    "speaker_tracking": "Pan between speakers instead of a fixed crop.",
    "remove_silence": "Cut long pauses to tighten the pacing.",
    "silence_threshold_db": "Audio quieter than this counts as silence.",
    "silence_min_duration": "Only pauses longer than this are shortened.",
    "captions_enabled": "Burn word-by-word captions into every clip.",
    "translate_captions": "Have Whisper translate the speech into English captions. Uploaded transcript files are used as-is.",
    "gameplay_enabled": "Fill the lower part of split-screen clips with gameplay footage from your library.",
    "layout": "split = speaker over gameplay, podcast = speaker only, blur = full frame over a blurred copy.",
    "split_ratio": "Share of the frame used by the speaker in split layouts.",
    "gameplay_volume": "0 mutes the gameplay audio.",
    "normalize_loudness": "Match the loudness platforms expect.",
    "target_lufs": "-14 LUFS suits most short-form platforms.",
    "voice_boost": "Gentle EQ and compression for clearer speech.",
    "music_enabled": "Add background music from your library (Assets -> Music).",
    "music_volume": "Music level before ducking (0-1).",
    "ducking": "Lower the music while someone is talking.",
    "ducking_db": "How far the music dips under speech (-6 subtle, -12 typical, -20 strong).",
    "export_dir": "Copies of exported clips go here. Blank = the data folder's exports directory.",
    "export_filename_template": "Placeholders: {project_slug}, {index} (e.g. {index:02d}) and {title_slug}.",
    "auto_open_folder": "Open the export folder after a clip is exported (desktop app only).",
    "keep_source_video": "When off, Clean up also deletes the source video of projects whose clips are all rendered.",
    "cache_transcripts": "Re-analysing the same video with the same speech settings skips speech recognition.",
    "cache_downloads": "Keep downloaded videos so another project from the same link starts instantly.",
    "cleanup_days": "Clean up removes cached downloads and unfinished projects older than this.",
    "max_cache_gb": "Clean up trims the download cache to this size, oldest first.",
    "uploads_enabled": "Allow uploading video files from the browser.",
    "max_upload_gb": "Largest accepted upload.",
}

# Fields that exist for compatibility but have no effect in this version, plus
# purely informational ones; the UI does not offer them.
HIDDEN_FIELDS: frozenset[str] = frozenset({"telemetry", "translation_language", "vision_enabled", "vision_model", "min_word_confidence"})


def _field_schema() -> dict[str, Any]:
    schema: dict[str, Any] = {}
    settings = get_settings()
    values = settings.model_dump()
    local_paths = Env.local_paths_allowed()
    for name, field in AppSettings.model_fields.items():
        if name in HIDDEN_FIELDS:
            continue
        annotation = field.annotation
        origin = getattr(annotation, "__origin__", None)
        options: list[Any] = []
        kind = "string"
        if origin is not None:
            options = list(get_args(annotation))
            kind = "enum"
        elif annotation is bool:
            kind = "boolean"
        elif annotation is int:
            kind = "integer"
        elif annotation is float:
            kind = "number"
        elif annotation is str:
            kind = "string"
        elif hasattr(annotation, "model_fields"):
            kind = "object"
            options = [
                {"name": sub, "type": (sub_field.annotation is bool and "boolean") or (sub_field.annotation is int and "integer") or "string", "default": sub_field.default}
                for sub, sub_field in annotation.model_fields.items()
            ]

        constraints: dict[str, Any] = {}
        for meta in field.metadata:
            for attribute in ("ge", "le", "gt", "lt"):
                value = getattr(meta, attribute, None)
                if value is not None:
                    constraints[attribute] = value

        default = field.get_default(call_default_factory=True)
        schema[name] = {
            "name": name,
            "label": FIELD_LABELS.get(name, name.replace("_", " ").capitalize()),
            "type": kind,
            "options": options,
            "default": default.model_dump() if hasattr(default, "model_dump") else default,
            "value": values.get(name),
            "section": next((section for section, fields in SECTION_FIELDS.items() if name in fields), "advanced"),
            "help": FIELD_HELP.get(name, (field.description or "").strip()),
            "readonly": name in SERVER_PATH_FIELDS and not local_paths,
            **constraints,
        }
    return schema


def _guard_server_paths(patch: dict[str, Any]) -> None:
    """Server paths may only change when local-path features are allowed."""
    if Env.local_paths_allowed():
        return
    current = get_settings().model_dump()
    blocked = sorted(key for key in SERVER_PATH_FIELDS if key in patch and patch[key] != current.get(key))
    if blocked:
        raise ClipForgeError(
            code=ErrorCode.INVALID_INPUT,
            message=f"{', '.join(blocked)} cannot be changed on this server.",
            hint="These settings point at files on the server. Set CLIPFORGE_ALLOW_LOCAL_PATHS=true to allow it.",
            status_code=403,
        )


@router.get("/settings")
def get_settings_route():
    settings = get_settings()
    return {
        "settings": settings.to_public_dict(),
        "sections": settings_store().sections(),
        "schema": _field_schema(),
        "paths": {
            "data_dir": str(Env.DATA_DIR),
            "exports": str(settings.resolved_export_dir()),
            "logs": str(Env.LOG_DIR),
        },
        "caption_presets": {key: {"label": value["label"], "description": value["description"], "theme": value["theme"]} for key, value in CAPTION_PRESETS.items()},
    }


@router.put("/settings")
def update_settings(payload: dict[str, Any] = Body(...)):  # noqa: B008
    """Partial update: send only the fields you want to change."""
    if not isinstance(payload, dict):
        raise ClipForgeError(code=ErrorCode.INVALID_INPUT, message="Settings payload must be an object.", status_code=422)
    patch = {key: value for key, value in payload.items() if value is not None}
    _guard_server_paths(patch)
    updated = settings_store().update(patch)  # raises a friendly 422 for invalid values
    invalidate_cache()
    events.publish("settings.updated", {"settings": updated.to_public_dict()})
    return {"settings": updated.to_public_dict(), "updated": sorted(patch.keys())}


@router.post("/settings/reset")
def reset_settings():
    store = settings_store()
    if Env.local_paths_allowed():
        updated = store.reset()
    else:  # keep the server's path configuration; reset everything else
        current = store.current.model_dump()
        updated = store.update({**AppSettings().model_dump(), **{key: current[key] for key in SERVER_PATH_FIELDS}})
    invalidate_cache()
    events.publish("settings.updated", {"settings": updated.to_public_dict(), "reset": True})
    return {"settings": updated.to_public_dict()}


@router.get("/settings/schema")
def settings_schema():
    return {"schema": _field_schema(), "sections": list(SECTION_FIELDS.keys()) + ["advanced"]}


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #


@router.get("/templates")
def list_templates():
    from ...db import BUILTIN_TEMPLATES

    with session_scope() as session:
        rows = session.execute(select(Template).order_by(Template.created_at)).scalars().all()
        payload = [row.to_dict() for row in rows]
    return {"templates": payload, "count": len(payload), "builtin": [item["id"] for item in BUILTIN_TEMPLATES]}


@router.post("/templates")
def create_template(payload: TemplateCreate):
    with session_scope() as session:
        template = Template(name=payload.name[:120], description=payload.description[:400], config_json=json.dumps(payload.config))
        session.add(template)
        session.flush()
        result = template.to_dict()
    events.publish("template.created", {"template": result})
    return result


@router.put("/templates/{template_id}")
def update_template(template_id: str, payload: TemplateCreate):
    with session_scope() as session:
        template = session.get(Template, template_id)
        if template is None:
            raise not_found("Template", template_id)
        template.name = payload.name[:120]
        template.description = payload.description[:400]
        template.config_json = json.dumps(payload.config)
        session.flush()
        return template.to_dict()


@router.delete("/templates/{template_id}")
def delete_template(template_id: str):
    with session_scope() as session:
        template = session.get(Template, template_id)
        if template is None:
            raise not_found("Template", template_id)
        if template.builtin:
            raise ClipForgeError(
                code=ErrorCode.CONFLICT,
                message="Built-in templates cannot be deleted.",
                hint="Duplicate it first, then delete the copy.",
                status_code=409,
            )
        session.delete(template)
    return {"deleted": template_id}


@router.post("/templates/{template_id}/apply")
def apply_template(template_id: str, project_id: str = Query("")):
    """Apply a template's configuration to the global settings (and a project)."""
    with session_scope() as session:
        template = session.get(Template, template_id)
        if template is None:
            raise not_found("Template", template_id)
        config = json.loads(template.config_json or "{}")

    patch = {key: value for key, value in config.items() if key in AppSettings.model_fields}
    if not Env.local_paths_allowed():
        patch = {key: value for key, value in patch.items() if key not in SERVER_PATH_FIELDS}
    settings = settings_store().update(patch)
    if project_id:
        from ...services.projects import merge_project_options

        merge_project_options(project_id, config)
    invalidate_cache()
    events.publish("settings.updated", {"settings": settings.to_public_dict(), "template": template_id})
    return {"applied": template_id, "config": config, "settings": settings.to_public_dict()}


__all__ = ["router"]
