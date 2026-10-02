"""Shared fixtures: isolated settings, app client, synthetic test videos."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from clipforge.core.config import Settings, reset_settings_cache
from clipforge.db.session import dispose_engines

FIXTURES = Path(__file__).parent / "fixtures"


def ffmpeg_bin() -> str | None:
    return shutil.which(os.environ.get("FFMPEG_PATH", "ffmpeg"))


requires_ffmpeg = pytest.mark.skipif(ffmpeg_bin() is None or shutil.which(os.environ.get("FFPROBE_PATH", "ffprobe")) is None,
                                     reason="ffmpeg/ffprobe not installed")


def make_settings(root: Path, **overrides) -> Settings:
    base = dict(
        APP_ENV="test",
        DEPLOYMENT_MODE="local",
        STORAGE_ROOT=root,
        DATABASE_URL=f"sqlite:///{root / 'test.db'}",
        TRANSCRIPTION_PROVIDER="fake",
        LLM_PROVIDER="heuristic",
        VISION_PROVIDER="auto",
        VISION_SAMPLE_FPS=1.0,
        RENDER_PROFILES_FILE=FIXTURES / "render_profiles_small.yaml",
        RATE_LIMIT_ENABLED=False,
        WARMUP_MODELS=False,
        LOG_JSON=False,
        LOG_LEVEL="WARNING",
        WORKER_LEASE_SECONDS=30,
        RETRY_BACKOFF_BASE_SECONDS=0,
        MIN_FREE_DISK_BYTES=0,
        MIN_FREE_RAM_BYTES=0,
        SSE_POLL_SECONDS=0.05,
        CLEANUP_INTERVAL_SECONDS=10_000,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def settings(tmp_path: Path) -> Iterator[Settings]:
    reset_settings_cache()
    s = make_settings(tmp_path / "data")
    s.ensure_directories()
    yield s
    dispose_engines()


@pytest.fixture
def session_factory(settings: Settings):
    from clipforge.db.session import make_session_factory
    from clipforge.services.bootstrap import bootstrap

    factory = make_session_factory(settings)
    bootstrap(settings, factory)
    return factory


@pytest.fixture
def app(settings: Settings):
    from clipforge.api.app import create_app

    return create_app(settings)


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


def register(client, email: str = "user@clipforge.dev", password: str = "correct-horse-1") -> dict:
    r = client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert r.status_code == 201, r.text
    body = r.json()
    client.headers["X-CSRF-Token"] = body["csrf_token"]
    return body


def make_video(path: Path, seconds: float = 60, *, size: str = "640x360", audio: bool = True,
               face: bool = False) -> Path:
    """Synthesise a test video with lavfi (testsrc2 + tone)."""
    args = [ffmpeg_bin() or "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=15:duration={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=330:sample_rate=44100:duration={seconds}"]
    args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac", "-b:a", "64k", "-shortest"]
    args += [str(path)]
    subprocess.run(args, check=True, timeout=120)
    return path


@pytest.fixture(scope="session")
def video_60s(tmp_path_factory) -> Path:
    if ffmpeg_bin() is None:
        pytest.skip("ffmpeg not installed")
    return make_video(tmp_path_factory.mktemp("media") / "talk.mp4", 75)
