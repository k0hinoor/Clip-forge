"""Stable error model (TRD §30).

Every error carries a stable ``code``, whether it is ``retryable``, a
human-readable ``user_message`` and an internal ``diagnostic_id`` that links
the user-visible error to internal logs without leaking stack traces.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    # Stable pipeline / media codes (TRD §30)
    MEDIA_TOO_LARGE = "MEDIA_TOO_LARGE"
    MEDIA_TOO_LONG = "MEDIA_TOO_LONG"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    MEDIA_CORRUPTED = "MEDIA_CORRUPTED"
    TRANSCRIPTION_FAILED = "TRANSCRIPTION_FAILED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    INSUFFICIENT_MEMORY = "INSUFFICIENT_MEMORY"
    RENDER_FAILED = "RENDER_FAILED"
    OUTPUT_VALIDATION_FAILED = "OUTPUT_VALIDATION_FAILED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    STORAGE_FULL = "STORAGE_FULL"
    WORKER_OFFLINE = "WORKER_OFFLINE"
    JOB_CANCELLED = "JOB_CANCELLED"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    # Additional stable codes
    JOB_TIMEOUT = "JOB_TIMEOUT"
    SOURCE_EXPIRED = "SOURCE_EXPIRED"
    MEDIA_TOO_SHORT = "MEDIA_TOO_SHORT"
    NO_AUDIO = "NO_AUDIO"
    # API-level codes
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    CONFLICT = "CONFLICT"
    RATE_LIMITED = "RATE_LIMITED"
    INVALID_STATE_TRANSITION = "INVALID_STATE_TRANSITION"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    CSRF_FAILED = "CSRF_FAILED"
    REGISTRATION_DISABLED = "REGISTRATION_DISABLED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


@dataclass(frozen=True)
class ErrorSpec:
    http_status: int
    retryable: bool
    message: str


ERROR_SPECS: dict[ErrorCode, ErrorSpec] = {
    ErrorCode.MEDIA_TOO_LARGE: ErrorSpec(413, False, "The uploaded file is larger than the allowed limit."),
    ErrorCode.MEDIA_TOO_LONG: ErrorSpec(422, False, "The video is longer than the allowed duration."),
    ErrorCode.MEDIA_TOO_SHORT: ErrorSpec(422, False, "The video is too short to create clips from."),
    ErrorCode.UNSUPPORTED_FORMAT: ErrorSpec(415, False, "This file format is not supported. Use MP4, MOV, MKV, WebM or M4V."),
    ErrorCode.MEDIA_CORRUPTED: ErrorSpec(422, False, "The video file appears to be corrupted or unreadable."),
    ErrorCode.NO_AUDIO: ErrorSpec(422, False, "The video has no audio track."),
    ErrorCode.TRANSCRIPTION_FAILED: ErrorSpec(500, True, "Speech transcription failed. Please retry."),
    ErrorCode.MODEL_UNAVAILABLE: ErrorSpec(503, True, "The local AI model is unavailable. Retry or choose a smaller model."),
    ErrorCode.INSUFFICIENT_MEMORY: ErrorSpec(503, True, "Not enough memory to process this video right now. It will be retried."),
    ErrorCode.RENDER_FAILED: ErrorSpec(500, True, "Rendering the clip failed. Please retry."),
    ErrorCode.OUTPUT_VALIDATION_FAILED: ErrorSpec(500, True, "The rendered clip failed quality checks. Please retry."),
    ErrorCode.QUOTA_EXCEEDED: ErrorSpec(402, False, "You have reached your processing limit for this period."),
    ErrorCode.STORAGE_FULL: ErrorSpec(507, True, "The server is out of storage space. Please try again later."),
    ErrorCode.WORKER_OFFLINE: ErrorSpec(503, True, "The processing worker is offline. Your job will start when it is back."),
    ErrorCode.JOB_CANCELLED: ErrorSpec(409, False, "The job was cancelled."),
    ErrorCode.JOB_TIMEOUT: ErrorSpec(504, True, "Processing took longer than the allowed time."),
    ErrorCode.SOURCE_EXPIRED: ErrorSpec(410, False, "The source video has expired and was deleted. Upload it again to re-render."),
    ErrorCode.UNKNOWN_ERROR: ErrorSpec(500, True, "Something went wrong. Please retry."),
    ErrorCode.VALIDATION_ERROR: ErrorSpec(422, False, "The request is invalid."),
    ErrorCode.NOT_FOUND: ErrorSpec(404, False, "The requested resource was not found."),
    ErrorCode.UNAUTHORIZED: ErrorSpec(401, False, "Authentication is required."),
    ErrorCode.FORBIDDEN: ErrorSpec(403, False, "You do not have permission to perform this action."),
    ErrorCode.CONFLICT: ErrorSpec(409, False, "The request conflicts with the current state."),
    ErrorCode.RATE_LIMITED: ErrorSpec(429, True, "Too many requests. Please slow down."),
    ErrorCode.INVALID_STATE_TRANSITION: ErrorSpec(409, False, "This action is not allowed in the current state."),
    ErrorCode.IDEMPOTENCY_CONFLICT: ErrorSpec(409, False, "The Idempotency-Key was already used for a different request."),
    ErrorCode.CSRF_FAILED: ErrorSpec(403, False, "CSRF validation failed."),
    ErrorCode.REGISTRATION_DISABLED: ErrorSpec(403, False, "Registration is disabled on this server."),
    ErrorCode.NOT_IMPLEMENTED: ErrorSpec(501, False, "This feature is not enabled on this server."),
}

# Codes that must never be retried automatically (TRD §31).
NON_RETRYABLE_CODES = frozenset(code for code, spec in ERROR_SPECS.items() if not spec.retryable)


def new_diagnostic_id() -> str:
    return uuid.uuid4().hex[:16]


class AppError(Exception):
    """Base application error with a stable, user-safe representation."""

    def __init__(
        self,
        code: ErrorCode | str,
        message: str | None = None,
        *,
        retryable: bool | None = None,
        http_status: int | None = None,
        details: dict[str, Any] | None = None,
        internal: str | None = None,
        diagnostic_id: str | None = None,
    ) -> None:
        self.code = ErrorCode(code)
        spec = ERROR_SPECS[self.code]
        self.user_message = message or spec.message
        self.retryable = spec.retryable if retryable is None else retryable
        self.http_status = http_status or spec.http_status
        self.details = details or {}
        # ``internal`` is for logs only — never returned to clients.
        self.internal = internal
        self.diagnostic_id = diagnostic_id or new_diagnostic_id()
        super().__init__(f"{self.code}: {self.user_message}")

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "code": self.code.value,
            "message": self.user_message,
            "retryable": self.retryable,
            "diagnostic_id": self.diagnostic_id,
        }
        if self.details:
            body["details"] = self.details
        return body


def not_found(resource: str = "Resource") -> AppError:
    return AppError(ErrorCode.NOT_FOUND, f"{resource} not found.")
