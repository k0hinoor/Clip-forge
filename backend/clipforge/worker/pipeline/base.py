"""Pipeline contract (TRD §8), artifact manifest (TRD §46) and stage context.

Every stage implements ``validate → execute → persist → validate_output →
cleanup``. Completed stages are recorded in the manifest with their version
and output artifacts (path, checksum, size, created_at, stage, version), so a
retried/recovered job resumes from the last valid stage (TRD §7, §45).
"""

from __future__ import annotations

import abc
import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from clipforge.core.config import Settings
from clipforge.core.errors import AppError, ErrorCode
from clipforge.core.timeutil import iso, utcnow
from clipforge.core.versioning import MANIFEST_VERSION, PIPELINE_VERSION, STAGE_VERSIONS
from clipforge.storage import StorageProvider
from clipforge.worker.models.manager import ModelManager
from clipforge.worker.resources.guard import ResourceGuard

CHECKSUM_MAX_BYTES = 256 * 1024**2  # verify checksums fully for artifacts up to this size


class LeaseLost(Exception):
    """Another worker owns the job now — stop without touching job state."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class Manifest:
    """Per-job artifact manifest persisted to ``work/<job_id>/manifest.json`` and the DB."""

    def __init__(self, job_id: str, storage_root: Path, data: dict[str, Any] | None = None) -> None:
        self.job_id = job_id
        self.storage_root = storage_root
        self.path = storage_root / "work" / job_id / "manifest.json"
        self.data: dict[str, Any] = data or {
            "job_id": job_id, "version": MANIFEST_VERSION, "pipeline_version": PIPELINE_VERSION,
            "artifacts": {}, "stages": {}, "rendered_clips": [],
        }

    @classmethod
    def load(cls, job_id: str, storage_root: Path, db_copy: dict[str, Any] | None) -> Manifest:
        path = storage_root / "work" / job_id / "manifest.json"
        if path.exists():
            try:
                return cls(job_id, storage_root, json.loads(path.read_text("utf-8")))
            except (json.JSONDecodeError, OSError):
                pass
        if db_copy and db_copy.get("artifacts") is not None:
            return cls(job_id, storage_root, dict(db_copy))
        return cls(job_id, storage_root)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True), "utf-8")
        os.replace(tmp, self.path)

    def rel(self, path: Path) -> str:
        return str(Path(path).resolve().relative_to(self.storage_root.resolve()))

    def abs(self, rel_path: str) -> Path:
        return self.storage_root / rel_path

    def add_artifact(self, name: str, path: Path, *, stage: str, checksum: str | None = None) -> dict[str, Any]:
        size = path.stat().st_size
        if checksum is None and size <= CHECKSUM_MAX_BYTES:
            checksum = sha256_file(path)
        entry = {"path": self.rel(path), "checksum": checksum, "size": size, "created_at": iso(utcnow()),
                 "stage": stage, "version": STAGE_VERSIONS.get(stage, "1")}
        self.data["artifacts"][name] = entry
        return entry

    def artifact_path(self, name: str) -> Path | None:
        entry = self.data["artifacts"].get(name)
        return self.abs(entry["path"]) if entry else None

    def artifact_valid(self, name: str, *, verify_checksum: bool = True) -> bool:
        entry = self.data["artifacts"].get(name)
        if not entry:
            return False
        path = self.abs(entry["path"])
        try:
            if not path.is_file() or path.stat().st_size != entry["size"]:
                return False
        except OSError:
            return False
        if verify_checksum and entry.get("checksum") and entry["size"] <= CHECKSUM_MAX_BYTES:
            return sha256_file(path) == entry["checksum"]
        return True

    def mark_stage(self, stage: str, outputs: list[str], duration_ms: int) -> None:
        self.data["stages"][stage] = {"version": STAGE_VERSIONS.get(stage, "1"), "outputs": outputs,
                                      "completed_at": iso(utcnow()), "duration_ms": duration_ms}

    def stage_done(self, stage: str) -> bool:
        info = self.data["stages"].get(stage)
        if not info or info.get("version") != STAGE_VERSIONS.get(stage, "1"):
            return False
        return all(self.artifact_valid(a) for a in info.get("outputs", []))

    def invalidate_from(self, stage_names: list[str]) -> None:
        for s in stage_names:
            self.data["stages"].pop(s, None)


@dataclass
class PipelineContext:
    job_id: str
    user_id: str
    asset_id: str
    job_settings: dict[str, Any]
    settings: Settings
    storage: StorageProvider
    session_factory: sessionmaker[Session]
    models: ModelManager
    guard: ResourceGuard
    work_dir: Path
    manifest: Manifest
    worker_id: str
    started_monotonic: float = field(default_factory=time.monotonic)
    cancel_flag: Callable[[], bool] | None = None
    progress_cb: Callable[[str, float, str | None], None] | None = None
    lease_lost: threading.Event = field(default_factory=threading.Event)
    # Hydrated state shared between stages
    source_path: Path | None = None
    media: dict[str, Any] = field(default_factory=dict)
    transcript: dict[str, Any] | None = None
    words: list[Any] = field(default_factory=list)
    sentences: list[Any] = field(default_factory=list)
    audio_levels: list[float] = field(default_factory=list)
    candidates: list[Any] = field(default_factory=list)
    selected: list[dict[str, Any]] = field(default_factory=list)
    versions: dict[str, Any] = field(default_factory=dict)
    current_stage: str = ""
    _last_cancel_check: float = 0.0
    _cancelled: bool = False

    def path(self, name: str) -> Path:
        p = self.work_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save_json(self, name: str, data: Any) -> Path:
        p = self.path(name)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":"), default=str), "utf-8")
        os.replace(tmp, p)
        return p

    def load_json(self, name: str) -> Any:
        return json.loads((self.work_dir / name).read_text("utf-8"))

    def is_cancelled(self) -> bool:
        """Cheap, throttled cancellation check used inside long operations."""
        if self.lease_lost.is_set():
            raise LeaseLost(self.job_id)
        if time.monotonic() - self.started_monotonic > self.settings.MAX_JOB_SECONDS:
            raise AppError(ErrorCode.JOB_TIMEOUT, retryable=False)
        now = time.monotonic()
        if self.cancel_flag and now - self._last_cancel_check >= 1.0:
            self._last_cancel_check = now
            self._cancelled = bool(self.cancel_flag())
        return self._cancelled

    def check_cancel(self) -> None:
        if self.is_cancelled():
            raise AppError(ErrorCode.JOB_CANCELLED)

    def report(self, fraction: float, message: str | None = None) -> None:
        if self.progress_cb:
            self.progress_cb(self.current_stage, fraction, message)


class Stage(abc.ABC):
    """Conceptual stage interface from TRD §8."""

    name: str = "stage"

    def validate(self, ctx: PipelineContext) -> None:  # noqa: B027 - optional hook
        """Check preconditions; raise AppError if the input is invalid."""

    @abc.abstractmethod
    def execute(self, ctx: PipelineContext) -> Any:
        """Do the work. Must be idempotent."""

    def persist(self, ctx: PipelineContext, output: Any) -> list[str]:
        """Persist output artifacts / DB rows. Returns manifest artifact names."""
        return []

    def validate_output(self, ctx: PipelineContext, output: Any) -> None:  # noqa: B027
        """Raise if the produced output is not valid."""

    def cleanup(self, ctx: PipelineContext) -> None:  # noqa: B027
        """Remove temporary files created by this stage."""

    def load(self, ctx: PipelineContext) -> None:  # noqa: B027
        """Hydrate ``ctx`` from persisted artifacts when the stage is skipped on resume."""
