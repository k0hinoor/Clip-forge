"""Shared pytest fixtures.

Two things matter here:

* every test runs against a throwaway data directory (``CLIPFORGE_DATA_DIR``),
  configured *before* any ``clipforge`` module is imported, so nothing touches
  the developer's real projects;
* the end-to-end test needs a real video. Provide one with
  ``CLIPFORGE_TEST_MEDIA=/path/to/video.mp4`` (see ``tools/make_test_media.py``)
  - otherwise it is skipped with an explanation instead of inventing results.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


# --------------------------------------------------------------------------- #
# Isolation - must happen at import time, before test modules pull in clipforge
# --------------------------------------------------------------------------- #

_TEMP_ROOT = Path(tempfile.mkdtemp(prefix="clipforge-tests-"))
# Data dir first: clipforge.config reads it while building Env.
os.environ["CLIPFORGE_DATA_DIR"] = os.environ.get("CLIPFORGE_TEST_DATA_DIR", str(_TEMP_ROOT / "data"))
# Never start background worker threads inside tests unless a test asks for them.
os.environ["CLIPFORGE_EMBED_WORKER"] = "false"
os.environ.setdefault("CLIPFORGE_LOG_LEVEL", "INFO")


@pytest.fixture(scope="session", autouse=True)
def data_dir() -> Path:
    from clipforge.config import Env, ensure_dirs
    from clipforge.db import init_db

    ensure_dirs()
    init_db()
    return Env.DATA_DIR


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # pragma: no cover - teardown
    if os.environ.get("CLIPFORGE_KEEP_TEST_DATA"):
        print(f"\nCLIPFORGE test data kept at {_TEMP_ROOT}")
        return
    shutil.rmtree(_TEMP_ROOT, ignore_errors=True)


@pytest.fixture()
def settings():
    from clipforge.config import get_settings

    return get_settings()


@pytest.fixture()
def app_client():
    """A TestClient wired to a freshly created app (worker threads disabled)."""
    from fastapi.testclient import TestClient

    from clipforge.api.app import create_app

    application = create_app()
    with TestClient(application) as client:
        yield client


# --------------------------------------------------------------------------- #
# Media fixtures
# --------------------------------------------------------------------------- #


def _media_path() -> Path | None:
    candidate = os.environ.get("CLIPFORGE_TEST_MEDIA", "")
    if candidate:
        path = Path(candidate).expanduser()
        if path.exists():
            return path
    for fallback in (
        Path("/tmp/fix/source.mp4"),
        BACKEND / "tests" / "data" / "sample.mp4",
    ):
        if fallback.exists():
            return fallback
    return None


@pytest.fixture(scope="session")
def media_file() -> Path:
    path = _media_path()
    if path is None:
        pytest.skip(
            "No test video available. Run 'python tools/make_test_media.py --out /tmp/fix/source.mp4' "
            "or set CLIPFORGE_TEST_MEDIA to an existing video file."
        )
    return path


@pytest.fixture(scope="session")
def media_manifest() -> dict:
    """Ground-truth manifest for the generated fixture video (script + real timings)."""
    path = _media_path()
    if path is None:
        pytest.skip("No test video available (see media_file).")
    candidates = [path.parent / "audio" / "manifest.json", path.parent / "manifest.json"]
    for manifest_path in candidates:
        if manifest_path.exists():
            import json

            return json.loads(manifest_path.read_text())
    pytest.skip("Missing manifest.json next to the fixture video: rebuild it with tools/make_test_media.py.")


@pytest.fixture(scope="session")
def tone_media(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny real media file (tone + test pattern) built with ffmpeg for fast tests."""
    from clipforge.media.ffmpeg import ffmpeg_bin
    from clipforge.media.runner import run_command

    target = tmp_path_factory.mktemp("media") / "tone.mp4"
    if not target.exists():
        run_command(
            [
                ffmpeg_bin(),
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=0x203040:s=640x360:r=25:d=12",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=220:duration=12",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "30",
                "-c:a",
                "aac",
                "-shortest",
                str(target),
            ],
            timeout=180,
        )
    return target


@dataclass
class SilenceWav:
    path: Path
    duration: float


@pytest.fixture(scope="session")
def speech_wav(tmp_path_factory: pytest.TempPathFactory) -> SilenceWav:
    """A small 16 kHz mono WAV with two tones and a silence gap (for diarization maths)."""
    import numpy as np

    target = tmp_path_factory.mktemp("audio") / "speech.wav"
    rate = 16000
    seconds = 6.0
    t = np.arange(int(rate * seconds)) / rate
    signal = np.zeros_like(t)
    signal[: int(2.0 * rate)] = 0.6 * np.sin(2 * np.pi * 180 * t[: int(2.0 * rate)])
    signal[int(3.0 * rate) :] = 0.5 * np.sin(2 * np.pi * 320 * t[int(3.0 * rate) :])
    pcm = (signal * 32767).astype("<i2")
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return SilenceWav(path=target, duration=seconds)


@pytest.fixture(scope="session")
def fixture_srt(tmp_path_factory: pytest.TempPathFactory, media_manifest: dict) -> Path:
    """A real SRT built from the fixture's ground-truth segment timings.

    This stands in for "the user already owns captions": cue timings come from the
    generated speech itself, and word times inside a cue are distributed by
    character length - exactly what a subtitle-only transcript can offer. It keeps
    the end-to-end test deterministic and usable on machines without a downloaded
    Whisper model, while still exercising every stage after transcription.
    """
    target = tmp_path_factory.mktemp("transcripts") / "fixture.srt"

    def stamp(seconds: float) -> str:
        milliseconds = int(round(seconds * 1000))
        hours, remainder = divmod(milliseconds, 3600_000)
        minutes, rest = divmod(remainder, 60_000)
        secs, ms = divmod(rest, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

    lines: list[str] = []
    for index, segment in enumerate(media_manifest["segments"], start=1):
        start = float(segment["start"])
        end = start + float(segment["duration"])
        lines += [str(index), f"{stamp(start)} --> {stamp(end)}", segment["text"].strip(), ""]
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "e2e: full pipeline test that needs real media and runs ffmpeg")
    config.addinivalue_line("markers", "asr: requires a locally available faster-whisper model")


def pytest_report_header(config: pytest.Config) -> str:
    media = _media_path()
    try:
        from clipforge.ai.transcribe import whisperx_available
        from clipforge.system import ai_stack

        stack = ai_stack()
        whisper = f"faster-whisper={stack.get('faster_whisper')} whisperx={whisperx_available()}"
    except Exception:  # noqa: BLE001 - header must never break collection
        whisper = "clipforge not importable yet"
    return f"CLIPFORGE test media: {media if media else 'not configured (e2e skipped)'} · {whisper}"
