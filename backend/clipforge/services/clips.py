"""Clip APIs: listing, manual correction and re-rendering (PRD §12 Flow B)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from clipforge.core.config import Settings
from clipforge.core.errors import AppError, ErrorCode, not_found
from clipforge.core.presets import ASPECT_RATIOS, FRAMING_MODES, caption_presets
from clipforge.core.states import ClipStatus, JobStatus, RenderStatus
from clipforge.core.timeutil import utcnow
from clipforge.db.models import Clip, Job, Render, RenderProfile, User
from clipforge.queue.base import QueueProvider
from clipforge.services.analytics import track
from clipforge.services.events import record_job_event
from clipforge.storage import StorageProvider

RENDER_AFFECTING = {"start", "end", "caption_preset", "aspect_ratio", "framing_mode", "captions_enabled"}
METADATA_FIELDS = {"title", "hook", "description"}


def profile_for_aspect(session: Session, aspect_ratio: str) -> RenderProfile:
    profile = session.scalar(select(RenderProfile).where(
        RenderProfile.aspect_ratio == aspect_ratio, RenderProfile.is_active.is_(True)
    ).order_by(RenderProfile.is_default.desc(), RenderProfile.name))
    if profile is None:
        raise AppError(ErrorCode.VALIDATION_ERROR, f"No active render profile for aspect ratio {aspect_ratio}.")
    return profile


class ClipService:
    def __init__(self, session: Session, settings: Settings, storage: StorageProvider,
                 queue: QueueProvider | None = None) -> None:
        self.session = session
        self.settings = settings
        self.storage = storage
        self.queue = queue

    def get(self, user: User, clip_id: str) -> Clip:
        clip = self.session.scalar(select(Clip).where(Clip.id == clip_id, Clip.user_id == user.id,
                                                      Clip.deleted_at.is_(None)))
        if clip is None:
            raise not_found("Clip")
        return clip

    def list_for_job(self, user: User, job: Job) -> list[Clip]:
        return list(self.session.scalars(select(Clip).where(
            Clip.job_id == job.id, Clip.user_id == user.id, Clip.deleted_at.is_(None)).order_by(Clip.rank)))

    def current_render(self, clip: Clip) -> Render | None:
        if not clip.current_render_id:
            return None
        return self.session.get(Render, clip.current_render_id)

    def pending_render(self, clip: Clip) -> Render | None:
        return self.session.scalar(select(Render).where(
            Render.clip_id == clip.id,
            Render.status.in_([RenderStatus.QUEUED.value, RenderStatus.RENDERING.value, RenderStatus.QA.value]),
        ).order_by(Render.created_at.desc()))

    # -------------------------------------------------------------- update
    def update(self, user: User, clip: Clip, patch: dict[str, Any]) -> tuple[Clip, bool]:
        """Apply a manual correction. Returns (clip, render_required)."""
        patch = {k: v for k, v in patch.items() if v is not None}
        job = self.session.get(Job, clip.job_id)
        duration = float(job.source_seconds or 0) if job else 0.0
        start = float(patch.get("start", clip.start))
        end = float(patch.get("end", clip.end))
        if "start" in patch or "end" in patch:
            if start < 0 or end <= start or (duration and end > duration + 0.05):
                raise AppError(ErrorCode.VALIDATION_ERROR, "Clip start/end are outside the source video.")
            length = end - start
            if not self.settings.MIN_CLIP_SECONDS <= length <= self.settings.MAX_CLIP_SECONDS:
                raise AppError(ErrorCode.VALIDATION_ERROR,
                               f"Clip length must be between {self.settings.MIN_CLIP_SECONDS:g}s and "
                               f"{self.settings.MAX_CLIP_SECONDS:g}s.")
        if "aspect_ratio" in patch and patch["aspect_ratio"] not in ASPECT_RATIOS:
            raise AppError(ErrorCode.VALIDATION_ERROR, f"Aspect ratio must be one of {', '.join(ASPECT_RATIOS)}.")
        if "framing_mode" in patch and patch["framing_mode"] not in FRAMING_MODES:
            raise AppError(ErrorCode.VALIDATION_ERROR, f"Framing mode must be one of {', '.join(FRAMING_MODES)}.")
        if "caption_preset" in patch and patch["caption_preset"] not in caption_presets(self.settings):
            raise AppError(ErrorCode.VALIDATION_ERROR, "Unknown caption preset.")
        changed: set[str] = set()
        for field in RENDER_AFFECTING | METADATA_FIELDS:
            if field not in patch:
                continue
            value = patch[field]
            if field in ("start", "end"):
                value = round(float(value), 3)
            if field in METADATA_FIELDS:
                value = str(value).strip()[: {"title": 200, "hook": 300, "description": 2000}[field]]
            if getattr(clip, field) != value:
                setattr(clip, field, value)
                changed.add(field)
        if {"start", "end"} & changed:
            clip.visual = None  # range changed → re-analyse frames for framing
            clip.captions = None
        if {"aspect_ratio", "framing_mode"} & changed:
            clip.crop = None
        if changed:
            track(self.session, "clip_edited", user_id=user.id, job_id=clip.job_id, clip_id=clip.id,
                  properties={"fields": sorted(changed)})
        return clip, bool(changed & RENDER_AFFECTING)

    # -------------------------------------------------------------- render
    def request_render(self, user: User, clip: Clip) -> Render:
        job = self.session.get(Job, clip.job_id)
        if job is None or job.status != JobStatus.COMPLETED.value:
            raise AppError(ErrorCode.INVALID_STATE_TRANSITION, "Clips can be re-rendered after the job completes.")
        asset = job.source_asset
        if asset is None or asset.deleted_at is not None or not self.storage.exists(asset.storage_key):
            raise AppError(ErrorCode.SOURCE_EXPIRED)
        pending = self.pending_render(clip)
        if pending is not None:
            if pending.status == RenderStatus.QUEUED.value:
                return pending  # idempotent: the queued render will pick up latest clip settings
            raise AppError(ErrorCode.CONFLICT, "This clip is already rendering.")
        profile = profile_for_aspect(self.session, clip.aspect_ratio)
        render = Render(clip_id=clip.id, job_id=clip.job_id, user_id=user.id, render_profile_id=profile.id,
                        render_profile_name=profile.name, render_profile_version=profile.version,
                        status=RenderStatus.QUEUED.value, source="user", attempt=0, params={})
        self.session.add(render)
        self.session.flush()
        clip.status = ClipStatus.RENDERING.value
        record_job_event(self.session, job, "render.queued", stage="render",
                         message=f"Re-render queued for clip #{clip.rank}",
                         metadata={"clip_id": clip.id, "render_id": render.id})
        track(self.session, "clip_rerendered", user_id=user.id, job_id=job.id, clip_id=clip.id,
              properties={"aspect_ratio": clip.aspect_ratio, "caption_preset": clip.caption_preset})
        if self.queue is not None:
            self.queue.enqueue_render(render.id)
        return render

    # -------------------------------------------------------------- delete
    def delete(self, user: User, clip: Clip) -> None:
        renders = self.session.scalars(select(Render).where(Render.clip_id == clip.id)).all()
        for r in renders:
            for key in (r.storage_key, r.thumbnail_key, r.subtitles_key):
                if key:
                    self.storage.delete(key)
            if r.status == RenderStatus.QUEUED.value:
                r.status = RenderStatus.CANCELLED.value
        clip.status = ClipStatus.DELETED.value
        clip.deleted_at = utcnow()
        track(self.session, "clip_deleted", user_id=user.id, job_id=clip.job_id, clip_id=clip.id)
