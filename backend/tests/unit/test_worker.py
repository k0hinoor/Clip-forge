from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from clipforge.core.errors import AppError, ErrorCode
from clipforge.core.timeutil import utcnow
from clipforge.db.models import Asset, Job, JobEvent, User
from clipforge.db.session import session_scope
from clipforge.queue.db_queue import DatabaseQueue
from clipforge.worker.models.manager import ModelAttempt, ModelManager
from clipforge.worker.pipeline.base import Manifest


def _job(factory, status="QUEUED", **kw) -> str:
    with session_scope(factory) as s:
        u = User(email=f"u{utcnow().timestamp()}@x.dev", password_hash=None, role="user", plan="unlimited", settings={})
        s.add(u)
        s.flush()
        a = Asset(user_id=u.id, original_filename="a.mp4", extension=".mp4", storage_key="k", status="READY",
                  duration_seconds=30)
        s.add(a)
        s.flush()
        j = Job(user_id=u.id, source_asset_id=a.id, status=status, settings={}, source_seconds=30, **kw)
        s.add(j)
        s.flush()
        return j.id


def test_claim_is_exclusive_and_lease_based(session_factory):
    q = DatabaseQueue(session_factory, 0.1)
    job_id = _job(session_factory)
    assert q.claim_job("w1", 30) == job_id
    assert q.claim_job("w2", 30) is None
    assert q.renew_job(job_id, "w1", 30)
    assert not q.renew_job(job_id, "w2", 30)
    q.release_job(job_id, "w1")
    with session_scope(session_factory) as s:
        assert s.get(Job, job_id).lock_owner is None


def test_backoff_respected(session_factory):
    q = DatabaseQueue(session_factory, 0.1)
    _job(session_factory, next_attempt_at=utcnow() + timedelta(minutes=5))
    assert q.claim_job("w1", 30) is None


def test_recover_abandoned_jobs(session_factory):
    q = DatabaseQueue(session_factory, 0.1)
    job_id = _job(session_factory, status="TRANSCRIBING", lock_owner="dead",
                  lock_expires_at=utcnow() - timedelta(seconds=5), current_stage="transcribe")
    live_id = _job(session_factory, status="RENDERING", lock_owner="alive",
                   lock_expires_at=utcnow() + timedelta(seconds=60))
    result = q.recover_abandoned()
    assert result["jobs"] == 1
    with session_scope(session_factory) as s:
        j = s.get(Job, job_id)
        assert j.status == "QUEUED" and j.lock_owner is None and j.current_stage == "transcribe"
        assert s.get(Job, live_id).status == "RENDERING"
        assert s.query(JobEvent).filter_by(job_id=job_id, event_type="job.recovered").count() == 1


def test_manifest_stage_validity(tmp_path: Path):
    m = Manifest.load("job1", tmp_path, None)
    f = tmp_path / "work" / "job1" / "audio.wav"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"abc")
    m.add_artifact("audio", f, stage="audio_extract")
    m.mark_stage("audio_extract", ["audio"], 10)
    m.save()
    m2 = Manifest.load("job1", tmp_path, None)
    assert m2.stage_done("audio_extract")
    f.write_bytes(b"tampered")  # checksum mismatch → stage must re-run
    assert not Manifest.load("job1", tmp_path, None).stage_done("audio_extract")
    f.unlink()
    assert not m2.stage_done("audio_extract")


def test_transcription_plan_fallback_order(settings):
    mm = ModelManager(settings)
    plan = mm.transcription_plan("large-v3")
    assert [a.model for a in plan][:4] == ["large-v3", "medium", "small", "base"]
    assert all(a.device == "cpu" for a in plan)  # no CUDA in CI


def test_transcribe_stage_falls_back(settings, session_factory, monkeypatch, tmp_path):
    """INSUFFICIENT_MEMORY on a big model → next model is tried (TRD §54)."""
    from clipforge.worker.pipeline import transcription as tmod
    from clipforge.worker.providers.transcription import TranscriptResult

    calls: list[str] = []

    class Flaky:
        name = "flaky"

        def transcribe(self, audio, *, model, **kw):
            calls.append(model)
            if model in ("large-v3", "medium"):
                raise AppError(ErrorCode.INSUFFICIENT_MEMORY)
            return TranscriptResult(language="en", duration=5, segments=[], provider="flaky", model=model,
                                    device="cpu")

    class FakeModels:
        transcriber = Flaky()

        def transcription_plan(self, requested=None):
            return [ModelAttempt(m, "cpu", "int8") for m in ("large-v3", "medium", "small", "base")]

    class Guard:
        import threading
        transcription_slots = threading.BoundedSemaphore(1)

    job_id = _job(session_factory)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    manifest = Manifest.load(job_id, tmp_path, None)
    manifest.add_artifact("audio", audio, stage="audio_extract")

    class Ctx:
        def __init__(self):
            self.manifest, self.media, self.job_settings = manifest, {"has_audio": True}, {}
            self.settings, self.models, self.guard = settings, FakeModels(), Guard()
            self.session_factory, self.job_id = session_factory, job_id

        def check_cancel(self):
            pass

        def is_cancelled(self):
            return False

        def report(self, *a, **k):
            pass

    out = tmod.TranscribeStage().execute(Ctx())
    assert calls == ["large-v3", "medium", "small"]
    assert out["model"] == "small"
    with session_scope(session_factory) as s:
        assert s.query(JobEvent).filter_by(job_id=job_id, event_type="model.fallback").count() == 2

    # provider not installed → no pointless fallback
    class Missing(Flaky):
        def transcribe(self, audio, *, model, **kw):
            calls.append(model)
            raise AppError(ErrorCode.MODEL_UNAVAILABLE, retryable=False)

    FakeModels.transcriber = Missing()
    calls.clear()
    with pytest.raises(AppError) as exc:
        tmod.TranscribeStage().execute(Ctx())
    assert exc.value.code == ErrorCode.MODEL_UNAVAILABLE and calls == ["large-v3"]


def test_parse_audio_levels(tmp_path):
    from clipforge.worker.pipeline.audio import parse_levels

    p = tmp_path / "levels.txt"
    p.write_text("frame:0 pts:0 pts_time:0\nlavfi.astats.Overall.RMS_level=-20.5\n"
                 "frame:1 pts:1 pts_time:1\nlavfi.astats.Overall.RMS_level=-inf\n")
    levels = parse_levels(p)
    assert len(levels) == 2 and levels[0] == pytest.approx(-20.5)
