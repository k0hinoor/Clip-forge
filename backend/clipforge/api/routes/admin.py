"""Admin routes (TRD §28, PRD §16): job/user oversight and system status."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from clipforge.api.deps import admin_user, client_ip, get_db, get_queue_dep, get_settings_dep, get_storage_dep
from clipforge.api.schemas import AdminUserPatch
from clipforge.api.serializers import event_out, job_out, user_out
from clipforge.core.config import Settings
from clipforge.core.errors import AppError, ErrorCode, not_found
from clipforge.core.presets import plans_config
from clipforge.db.models import Job, JobEvent, User
from clipforge.queue.base import QueueProvider
from clipforge.services import system
from clipforge.services.audit import audit
from clipforge.services.auth import AuthService
from clipforge.services.cleanup import CleanupService
from clipforge.services.jobs import JobService
from clipforge.services.quotas import UsageService
from clipforge.storage import StorageProvider

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/jobs")
def admin_jobs(status: str | None = Query(default=None, max_length=30), user_id: str | None = None,
               limit: int = Query(default=50, ge=1, le=200), offset: int = Query(default=0, ge=0),
               _: User = Depends(admin_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    q = select(Job)
    if status:
        q = q.where(Job.status == status.upper())
    if user_id:
        q = q.where(Job.user_id == user_id)
    total = int(db.scalar(select(func.count()).select_from(q.subquery())) or 0)
    rows = db.scalars(q.order_by(Job.created_at.desc()).limit(limit).offset(offset)).all()
    return {"items": [{**job_out(j), "user_id": j.user_id, "lock_owner": j.lock_owner,
                       "deleted": j.deleted_at is not None} for j in rows], "total": total}


@router.get("/jobs/{job_id}")
def admin_job(job_id: str, _: User = Depends(admin_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    job = db.get(Job, job_id)
    if job is None:
        raise not_found("Job")
    events = db.scalars(select(JobEvent).where(JobEvent.job_id == job_id).order_by(JobEvent.id)).all()
    return {**job_out(job, db), "user_id": job.user_id, "lock_owner": job.lock_owner, "manifest": job.manifest,
            "events": [event_out(e) for e in events]}


@router.post("/jobs/{job_id}/retry")
def admin_retry(job_id: str, request: Request, admin: User = Depends(admin_user), db: Session = Depends(get_db),
                settings: Settings = Depends(get_settings_dep), queue: QueueProvider = Depends(get_queue_dep),
                storage: StorageProvider = Depends(get_storage_dep)) -> dict[str, Any]:
    job = db.get(Job, job_id)
    if job is None:
        raise not_found("Job")
    svc = JobService(db, settings, queue, storage)
    owner = db.get(User, job.user_id)
    job = svc.retry(owner, job)
    audit(db, "admin.job_retried", actor_user_id=admin.id, target_type="job", target_id=job.id,
          ip_address=client_ip(request))
    db.commit()
    svc.notify_queue(job)
    return job_out(job, db)


@router.post("/jobs/{job_id}/cancel")
def admin_cancel(job_id: str, request: Request, admin: User = Depends(admin_user), db: Session = Depends(get_db),
                 settings: Settings = Depends(get_settings_dep), queue: QueueProvider = Depends(get_queue_dep),
                 storage: StorageProvider = Depends(get_storage_dep)) -> dict[str, Any]:
    job = db.get(Job, job_id)
    if job is None:
        raise not_found("Job")
    job = JobService(db, settings, queue, storage).cancel(db.get(User, job.user_id), job)
    audit(db, "admin.job_cancelled", actor_user_id=admin.id, target_type="job", target_id=job.id,
          ip_address=client_ip(request))
    db.commit()
    return job_out(job, db)


@router.get("/users")
def admin_users(q: str | None = Query(default=None, max_length=100), limit: int = Query(default=50, ge=1, le=200),
                offset: int = Query(default=0, ge=0), _: User = Depends(admin_user),
                db: Session = Depends(get_db)) -> dict[str, Any]:
    query = select(User).where(User.deleted_at.is_(None))
    if q:
        query = query.where(User.email.contains(q.lower()))
    total = int(db.scalar(select(func.count()).select_from(query.subquery())) or 0)
    rows = db.scalars(query.order_by(User.created_at.desc()).limit(limit).offset(offset)).all()
    usage = UsageService(db)
    return {"items": [{**user_out(u), "is_active": u.is_active, "last_login_at": u.last_login_at and
                       u.last_login_at.isoformat() + "Z", "usage": usage.summary(u.id)} for u in rows],
            "total": total}


@router.patch("/users/{user_id}")
def admin_patch_user(user_id: str, body: AdminUserPatch, request: Request, admin: User = Depends(admin_user),
                     db: Session = Depends(get_db), settings: Settings = Depends(get_settings_dep)) -> dict[str, Any]:
    target = db.get(User, user_id)
    if target is None or target.deleted_at is not None:
        raise not_found("User")
    patch = body.model_dump(exclude_unset=True)
    if "plan" in patch and patch["plan"] not in plans_config(settings):
        raise AppError(ErrorCode.VALIDATION_ERROR, "Unknown plan.")
    if target.id == admin.id and (patch.get("role") == "user" or patch.get("is_active") is False):
        raise AppError(ErrorCode.CONFLICT, "Admins cannot demote or deactivate themselves.")
    for k, v in patch.items():
        setattr(target, k, v)
    if patch.get("is_active") is False:
        AuthService(db, settings).revoke_all(target.id)
    audit(db, "admin.user_updated", actor_user_id=admin.id, target_type="user", target_id=target.id,
          ip_address=client_ip(request), metadata={k: v for k, v in patch.items() if k != "quota_override"})
    db.commit()
    return user_out(target)


@router.get("/system")
def admin_system(_: User = Depends(admin_user), db: Session = Depends(get_db),
                 settings: Settings = Depends(get_settings_dep)) -> dict[str, Any]:
    metrics = system.collect_metrics(db, settings)
    return {"metrics": metrics, "disk": system.disk_status(settings), "memory": system.memory_status(),
            "workers": system.workers_status(db, settings), "alerts": system.alerts(db, settings, metrics),
            "stage_durations_seconds": system.stage_durations(db), "cleanup": system.cleanup_status(db),
            "config": {"deployment_mode": settings.DEPLOYMENT_MODE, "storage_backend": settings.STORAGE_BACKEND,
                       "queue_backend": "redis+database" if settings.REDIS_URL else "database", "whisper_model": settings.WHISPER_MODEL,
                       "llm_provider": settings.LLM_PROVIDER, "vision_provider": settings.VISION_PROVIDER,
                       "worker_concurrency": settings.WORKER_CONCURRENCY}}


@router.post("/cleanup")
def admin_cleanup(request: Request, admin: User = Depends(admin_user), db: Session = Depends(get_db),
                  settings: Settings = Depends(get_settings_dep),
                  storage: StorageProvider = Depends(get_storage_dep)) -> dict[str, Any]:
    result = CleanupService(settings, request.app.state.session_factory, storage).run_once()
    audit(db, "admin.cleanup_run", actor_user_id=admin.id, ip_address=client_ip(request))
    db.commit()
    return result if isinstance(result, dict) else {"result": result}
