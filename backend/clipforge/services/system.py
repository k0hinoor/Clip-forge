"""System status, metrics and alerts for admin/health endpoints (TRD §38, §39; PRD §18)."""

from __future__ import annotations

import shutil
from datetime import timedelta
from typing import Any

import psutil
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from clipforge.core.config import Settings
from clipforge.core.states import ACTIVE_STATES, JobStatus, RenderStatus
from clipforge.core.timeutil import day_start, iso, utcnow
from clipforge.db.models import Job, JobEvent, Render, SystemSetting, UsageCounter, User, WorkerHeartbeat


def disk_status(settings: Settings) -> dict[str, Any]:
    settings.STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(settings.STORAGE_ROOT)
    return {
        "total_bytes": usage.total,
        "free_bytes": usage.free,
        "used_percent": round(usage.used / usage.total * 100, 1) if usage.total else 0.0,
    }


def memory_status() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {"total_bytes": vm.total, "available_bytes": vm.available, "used_percent": vm.percent}


def workers_status(session: Session, settings: Settings) -> list[dict[str, Any]]:
    cutoff = utcnow() - timedelta(seconds=settings.WORKER_OFFLINE_AFTER_SECONDS)
    rows = session.scalars(select(WorkerHeartbeat).order_by(WorkerHeartbeat.last_seen_at.desc())).all()
    return [{
        "worker_id": w.worker_id,
        "hostname": w.hostname,
        "status": w.status if w.last_seen_at >= cutoff else "offline",
        "online": w.last_seen_at >= cutoff and w.status != "stopped",
        "last_seen_at": iso(w.last_seen_at),
        "started_at": iso(w.started_at),
        "active_jobs": w.active_jobs,
        "capabilities": w.capabilities,
    } for w in rows]


def any_worker_online(session: Session, settings: Settings) -> bool:
    return any(w["online"] for w in workers_status(session, settings))


def job_counts(session: Session) -> dict[str, int]:
    rows = session.execute(select(Job.status, func.count()).where(Job.deleted_at.is_(None)).group_by(Job.status))
    counts = {s.value: 0 for s in JobStatus}
    counts.update({status: int(n) for status, n in rows})
    return counts


def stage_durations(session: Session, since_hours: int = 24 * 7) -> dict[str, float]:
    since = utcnow() - timedelta(hours=since_hours)
    rows = session.execute(
        select(JobEvent.stage, func.avg(JobEvent.duration_ms))
        .where(JobEvent.event_type == "stage.completed", JobEvent.timestamp >= since)
        .group_by(JobEvent.stage)
    )
    return {stage: round(float(avg or 0), 1) for stage, avg in rows if stage}


def collect_metrics(session: Session, settings: Settings) -> dict[str, Any]:
    counts = job_counts(session)
    active = sum(counts[s.value] for s in ACTIVE_STATES)
    today = day_start()
    minutes_today = session.scalar(select(func.coalesce(func.sum(UsageCounter.source_seconds), 0.0)).where(
        UsageCounter.period == "day", UsageCounter.period_start == today)) or 0.0
    processing_seconds = session.scalar(select(func.coalesce(func.sum(JobEvent.duration_ms), 0)).where(
        JobEvent.event_type == "stage.completed")) or 0
    model_errors = session.scalar(select(func.count()).select_from(JobEvent).where(
        JobEvent.error_code.in_(["MODEL_UNAVAILABLE", "TRANSCRIPTION_FAILED", "INSUFFICIENT_MEMORY"]),
        JobEvent.timestamp >= utcnow() - timedelta(hours=24))) or 0
    failed_renders = session.scalar(select(func.count()).select_from(Render).where(
        Render.status == RenderStatus.FAILED.value, Render.created_at >= utcnow() - timedelta(hours=24))) or 0
    started = session.scalar(select(func.count()).select_from(JobEvent).where(
        JobEvent.event_type == "stage.started", JobEvent.stage == "validate")) or 0
    return {
        "jobs_by_status": counts,
        "jobs_started": int(started),
        "jobs_completed": counts[JobStatus.COMPLETED.value],
        "jobs_failed": counts[JobStatus.FAILED.value],
        "jobs_active": active,
        "queue_length": counts[JobStatus.QUEUED.value],
        "processing_seconds_total": round(float(processing_seconds) / 1000, 1),
        "average_stage_duration_ms": stage_durations(session),
        "source_minutes_today": round(float(minutes_today) / 60, 2),
        "model_errors_24h": int(model_errors),
        "failed_renders_24h": int(failed_renders),
        "users_total": int(session.scalar(select(func.count()).select_from(User).where(User.deleted_at.is_(None))) or 0),
    }


def alerts(session: Session, settings: Settings, metrics: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    metrics = metrics or collect_metrics(session, settings)
    out: list[dict[str, Any]] = []
    disk = disk_status(settings)
    if disk["used_percent"] >= settings.ALERT_DISK_PERCENT:
        out.append({"type": "disk_high", "severity": "critical", "message": f"Disk usage at {disk['used_percent']}%"})
    if not any_worker_online(session, settings):
        out.append({"type": "worker_offline", "severity": "critical", "message": "No processing worker is online"})
    if metrics["model_errors_24h"] >= 3:
        out.append({"type": "model_failures", "severity": "warning",
                    "message": f"{metrics['model_errors_24h']} model failures in the last 24h"})
    if metrics["queue_length"] >= settings.ALERT_QUEUE_LENGTH:
        out.append({"type": "queue_large", "severity": "warning", "message": f"{metrics['queue_length']} jobs queued"})
    if metrics["failed_renders_24h"] >= settings.ALERT_FAILED_RENDERS:
        out.append({"type": "render_failures", "severity": "warning",
                    "message": f"{metrics['failed_renders_24h']} failed renders in the last 24h"})
    long_running = session.scalar(select(func.count()).select_from(Job).where(
        Job.status.in_([s.value for s in ACTIVE_STATES]),
        Job.started_at < utcnow() - timedelta(seconds=settings.MAX_JOB_SECONDS))) or 0
    if long_running:
        out.append({"type": "abnormal_processing_time", "severity": "warning",
                    "message": f"{long_running} job(s) exceeded the maximum processing time"})
    return out


def cleanup_status(session: Session) -> dict[str, Any] | None:
    row = session.get(SystemSetting, "cleanup.last_run")
    return row.value if row else None


def prometheus_text(metrics: dict[str, Any], disk: dict[str, Any]) -> str:
    lines = [
        "# HELP clipforge_jobs Jobs by status",
        "# TYPE clipforge_jobs gauge",
    ]
    for status, n in metrics["jobs_by_status"].items():
        lines.append(f'clipforge_jobs{{status="{status}"}} {n}')
    scalar = {
        "clipforge_queue_length": metrics["queue_length"],
        "clipforge_jobs_started_total": metrics["jobs_started"],
        "clipforge_processing_seconds_total": metrics["processing_seconds_total"],
        "clipforge_source_minutes_today": metrics["source_minutes_today"],
        "clipforge_model_errors_24h": metrics["model_errors_24h"],
        "clipforge_failed_renders_24h": metrics["failed_renders_24h"],
        "clipforge_disk_free_bytes": disk["free_bytes"],
        "clipforge_disk_used_percent": disk["used_percent"],
    }
    for name, value in scalar.items():
        lines += [f"# TYPE {name} gauge", f"{name} {value}"]
    lines.append("# TYPE clipforge_stage_duration_ms gauge")
    for stage, ms in metrics["average_stage_duration_ms"].items():
        lines.append(f'clipforge_stage_duration_ms{{stage="{stage}"}} {ms}')
    return "\n".join(lines) + "\n"
