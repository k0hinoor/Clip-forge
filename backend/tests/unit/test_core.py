from __future__ import annotations

import pytest

from clipforge.core.config import Settings
from clipforge.core.errors import ERROR_SPECS, AppError, ErrorCode
from clipforge.core.states import (
    PIPELINE_STAGES,
    JobStatus,
    assert_transition,
    can_transition,
    overall_progress,
)
from clipforge.core.versioning import STAGE_VERSIONS


def test_required_error_codes_exist():
    required = ["MEDIA_TOO_LARGE", "MEDIA_TOO_LONG", "UNSUPPORTED_FORMAT", "MEDIA_CORRUPTED", "TRANSCRIPTION_FAILED",
                "MODEL_UNAVAILABLE", "INSUFFICIENT_MEMORY", "RENDER_FAILED", "OUTPUT_VALIDATION_FAILED",
                "QUOTA_EXCEEDED", "STORAGE_FULL", "WORKER_OFFLINE", "JOB_CANCELLED", "UNKNOWN_ERROR"]
    for code in required:
        assert ErrorCode(code) in ERROR_SPECS


def test_app_error_shape_hides_internals():
    err = AppError(ErrorCode.RENDER_FAILED, internal="Traceback: secret path /etc/x")
    body = err.to_dict()
    assert body["code"] == "RENDER_FAILED" and body["retryable"] is True
    assert body["diagnostic_id"] and "secret" not in str(body)


def test_job_transitions():
    assert can_transition("QUEUED", "VALIDATING")
    assert can_transition("TRANSCRIBING", "SEGMENTING")
    assert can_transition("RENDERING", "FAILED")
    assert can_transition("FAILED", "QUEUED")  # retry
    assert not can_transition("COMPLETED", "RENDERING")
    assert not can_transition("CANCELLED", "COMPLETED")
    with pytest.raises(AppError) as exc:
        assert_transition("COMPLETED", "TRANSCRIBING")
    assert exc.value.code == ErrorCode.INVALID_STATE_TRANSITION


def test_every_status_is_reachable():
    statuses = {s.status for s in PIPELINE_STAGES}
    assert JobStatus.TRANSCRIBING in statuses and JobStatus.RENDERING in statuses


def test_progress_is_monotonic():
    values = []
    for stage in PIPELINE_STAGES:
        values += [overall_progress(stage.name, 0.0), overall_progress(stage.name, 1.0)]
    assert values == sorted(values)
    assert values[-1] <= 100


def test_stage_versions_cover_all_stages():
    assert {s.name for s in PIPELINE_STAGES} <= set(STAGE_VERSIONS)


def test_settings_validation(tmp_path):
    with pytest.raises(ValueError):
        Settings(APP_ENV="production", STORAGE_ROOT=tmp_path)  # insecure default secret
    with pytest.raises(ValueError):
        Settings(MIN_CLIP_SECONDS=30, TARGET_MIN_SECONDS=20, STORAGE_ROOT=tmp_path)
    s = Settings(DEPLOYMENT_MODE="cloud", STORAGE_ROOT=tmp_path)
    assert s.DEFAULT_PLAN == "free" and s.auth_required
    assert Settings(STORAGE_ROOT=tmp_path).DEFAULT_PLAN == "unlimited"


def test_list_settings_accept_csv_and_json(tmp_path, monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://a.dev, http://b.dev")
    monkeypatch.setenv("ADMIN_EMAILS", '["Boss@X.dev"]')
    monkeypatch.setenv("ALLOWED_EXTENSIONS", "mp4,MOV")
    s = Settings(STORAGE_ROOT=tmp_path)
    assert s.CORS_ORIGINS == ["http://a.dev", "http://b.dev"]
    assert s.ADMIN_EMAILS == ["boss@x.dev"]
    assert s.ALLOWED_EXTENSIONS == [".mp4", ".mov"]


def test_env_example_parses(tmp_path, monkeypatch):
    from pathlib import Path

    example = Path(__file__).resolve().parents[2] / ".env.example"
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(example.read_text())
    s = Settings()
    assert s.DEPLOYMENT_MODE == "local" and s.WHISPER_MODEL == "auto" and s.DATABASE_URL.startswith("sqlite")
