"""Structured JSON logging (TRD §38).

Context such as ``job_id``, ``stage``, ``request_id`` and ``user_id`` is carried in
contextvars so every log line emitted while handling a job/request includes it.
Never log secrets, media content or raw transcripts.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from collections.abc import Iterator
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

_context: ContextVar[dict[str, Any] | None] = ContextVar("clipforge_log_context", default=None)
_service: str = "clipforge"

_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}
_REDACT_KEYS = {"password", "token", "secret", "authorization", "api_key", "cookie", "jwt_secret"}


def bind_context(**values: Any) -> None:
    current = dict(_context.get() or {})
    current.update({k: v for k, v in values.items() if v is not None})
    _context.set(current)


def clear_context() -> None:
    _context.set({})


@contextlib.contextmanager
def log_context(**values: Any) -> Iterator[None]:
    token = _context.set({**(_context.get() or {}), **{k: v for k, v in values.items() if v is not None}})
    try:
        yield
    finally:
        _context.reset(token)


def _redact(key: str, value: Any) -> Any:
    if any(r in key.lower() for r in _REDACT_KEYS):
        return "[REDACTED]"
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "service": _service,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_context.get() or {})
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = _redact(key, value)
        if record.exc_info:
            # Stack traces stay in internal logs only, never in API responses.
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ctx = _context.get() or {}
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")}
        bits = " ".join(f"{k}={_redact(k, v)}" for k, v in {**ctx, **extras}.items())
        base = f"{datetime.now(UTC):%H:%M:%S} {record.levelname:<7} [{_service}] {record.name}: {record.getMessage()}"
        out = f"{base} {bits}".rstrip()
        if record.exc_info:
            out += "\n" + self.formatException(record.exc_info)
        return out


def configure_logging(service: str, level: str = "INFO", json_logs: bool = True) -> None:
    global _service
    _service = service
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else TextFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "multipart", "faster_whisper", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
