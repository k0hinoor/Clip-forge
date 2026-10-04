"""Structured logging.

Four rotating files under ``logs/``:

* ``app.log``     - API + general activity
* ``worker.log``  - pipeline stages, progress, timings
* ``render.log``  - every ffmpeg/yt-dlp command and its output tail
* ``ai.log``      - transcription and LLM interaction

Every record lands in exactly one file, chosen by its logger name
(``clipforge.worker*``, ``clipforge.render*``, ``clipforge.ai*``; everything
else goes to ``app.log``). Console output stays compact and colourised; files get
JSON lines so the Diagnostics page can parse them and show recent errors without
grepping.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from typing import Any

from .config import Env, ensure_dirs

LOG_FILES = ("app", "worker", "render", "ai")
_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for attr in ("project_id", "job_id", "clip_id", "stage", "command_hash"):
            value = getattr(record, attr, None)
            if value:
                payload[attr] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[38;5;245m",
        "INFO": "\033[38;5;39m",
        "WARNING": "\033[38;5;214m",
        "ERROR": "\033[38;5;203m",
        "CRITICAL": "\033[48;5;203m\033[38;5;231m",
    }
    RESET = "\033[0m"

    def __init__(self, colour: bool = True) -> None:
        super().__init__()
        self.colour = colour

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        name = record.name.replace("clipforge.", "")
        prefix = ""
        for attr, label in (("project_id", "proj"), ("job_id", "job"), ("stage", "stage")):
            value = getattr(record, attr, None)
            if value:
                prefix += f" [{label}={value}]"
        message = f"{ts} {record.levelname:<7} {name}{prefix} · {record.getMessage()}"
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)
        if self.colour and sys.stderr.isatty():
            colour = self.COLORS.get(record.levelname, "")
            return f"{colour}{message}{self.RESET}"
        return message


# Logger subtrees that own a dedicated file; everything else goes to app.log.
_SUBSYSTEM_LOGGERS = {
    "worker": "clipforge.worker",
    "render": "clipforge.render",
    "ai": "clipforge.ai",
}


def log_file_for(logger_name: str) -> str:
    """Which of :data:`LOG_FILES` a logger writes to."""
    for file_name, prefix in _SUBSYSTEM_LOGGERS.items():
        if logger_name == prefix or logger_name.startswith(prefix + "."):
            return file_name
    return "app"


class LogFileFilter(logging.Filter):
    """Routes each record to exactly one log file (no duplicated lines)."""

    def __init__(self, file_name: str) -> None:
        super().__init__()
        self.file_name = file_name

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        return log_file_for(record.name) == self.file_name


class ContextAdapter(logging.LoggerAdapter):
    """Logger that stamps job/project/clip ids onto every record."""

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = kwargs.setdefault("extra", {})
        for key, value in (self.extra or {}).items():
            extra.setdefault(key, value)
        return msg, kwargs


def configure_logging(*, level: str | None = None, force: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    ensure_dirs()
    Env.LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("clipforge")
    root.setLevel(getattr(logging, (level or Env.LOG_LEVEL), logging.INFO))
    root.handlers.clear()
    root.propagate = False

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(ConsoleFormatter(colour=True))
    console.setLevel(logging.INFO)
    root.addHandler(console)

    for name in LOG_FILES:
        handler = logging.handlers.RotatingFileHandler(
            Env.LOG_DIR / f"{name}.log",
            maxBytes=8 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(JsonFormatter())
        handler.setLevel(logging.DEBUG)
        handler.addFilter(LogFileFilter(name))
        root.addHandler(handler)

    # Quiet down noisy third party libraries on the console; keep the errors.
    for noisy in ("httpx", "httpcore", "urllib3", "watchfiles", "multipart", "asyncio", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str, **context: Any) -> logging.LoggerAdapter:
    """Return a contextualised logger. ``get_logger(__name__, job_id=...)``.

    Names outside the ``clipforge`` tree are moved into it so they share the
    configured handlers; :func:`log_file_for` decides which file they land in.
    """
    configure_logging()
    if name != "clipforge" and not name.startswith("clipforge."):
        name = f"clipforge.{name}"
    return ContextAdapter(logging.getLogger(name), context)


def tail_log(name: str, lines: int = 200, *, level: str | None = None) -> list[str]:
    """Return the last ``lines`` of a log file (used by Diagnostics)."""
    path = Env.LOG_DIR / f"{name}.log"
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            content = handle.readlines()
    except OSError:
        return []
    if level:
        wanted = level.upper()
        content = [line for line in content if f'"level": "{wanted}"' in line or f" {wanted} " in line]
    return [line.rstrip("\n") for line in content[-lines:]]


def recent_errors(limit: int = 40) -> list[dict[str, Any]]:
    """Structured recent errors across all logs, newest first.

    Identical records (same time, logger and message) are reported once, so logs
    written by older versions - which copied every line into all four files - do
    not show each error four times.
    """
    found: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for name in LOG_FILES:
        for line in tail_log(name, lines=2000):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("level") not in {"ERROR", "CRITICAL"}:
                continue
            key = (str(record.get("ts", "")), str(record.get("logger", "")), str(record.get("msg", "")))
            if key in seen:
                continue
            seen.add(key)
            record["file"] = f"{name}.log"
            found.append(record)
    found.sort(key=lambda item: str(item.get("ts", "")), reverse=True)
    return found[:limit]


def log_environment() -> dict[str, Any]:
    info = {
        "data_dir": str(Env.DATA_DIR),
        "log_dir": str(Env.LOG_DIR),
        "python": sys.version.split()[0],
        "pid": os.getpid(),
    }
    get_logger(__name__).info("clipforge environment: %s", json.dumps(info))
    return info


def log_file_paths() -> dict[str, str]:
    return {name: str((Path(Env.LOG_DIR) / f"{name}.log")) for name in LOG_FILES}
