"""Resource management (TRD §32, §57).

Before claiming a job the worker checks disk, RAM, temporary disk usage and
its own capacity. If resources are insufficient the job simply stays QUEUED.
"""

from __future__ import annotations

import shutil
import threading
import time
from dataclasses import dataclass

import psutil

from clipforge.core.config import Settings
from clipforge.services.cleanup import directory_size


@dataclass(frozen=True)
class ResourceCheck:
    ok: bool
    reason: str | None = None


class ResourceGuard:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.transcription_slots = threading.BoundedSemaphore(max(1, settings.MAX_CONCURRENT_TRANSCRIPTIONS))
        self._active = 0
        self._lock = threading.Lock()
        self._usage = 0
        self._usage_checked_at = -1e9

    def can_start(self) -> ResourceCheck:
        with self._lock:
            if self._active >= max(1, self.settings.WORKER_CONCURRENCY):
                return ResourceCheck(False, "worker_at_capacity")
        free = shutil.disk_usage(self.settings.STORAGE_ROOT).free
        if free < self.settings.MIN_FREE_DISK_BYTES:
            return ResourceCheck(False, "disk_low")
        if psutil.virtual_memory().available < self.settings.MIN_FREE_RAM_BYTES:
            return ResourceCheck(False, "memory_low")
        if self._work_usage() > self.settings.MAX_WORK_DISK_BYTES:
            return ResourceCheck(False, "work_disk_quota")
        return ResourceCheck(True)

    def _work_usage(self) -> int:
        now = time.monotonic()
        if now - self._usage_checked_at > 30:  # rglob is not free; cache for 30 s
            self._usage = directory_size(self.settings.STORAGE_ROOT / "work")
            self._usage_checked_at = now
        return self._usage

    def acquire(self) -> None:
        with self._lock:
            self._active += 1

    def release(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)

    @property
    def active(self) -> int:
        with self._lock:
            return self._active
