"""End-to-end: upload → job → worker pipeline → clips → download → edit → re-render.

Uses a synthetic lavfi video and the deterministic fake transcription provider
so it runs anywhere ffmpeg is installed (no model downloads).
"""

from __future__ import annotations

import json
import subprocess

import pytest

from clipforge.queue import get_queue
from clipforge.storage import get_storage
from clipforge.worker.runner import Worker
from tests.conftest import register, requires_ffmpeg

pytestmark = [pytest.mark.integration, requires_ffmpeg]


def run_worker(app) -> Worker:
    st = app.state
    worker = Worker(st.settings, st.session_factory, get_storage(st.settings), get_queue(st.settings, st.session_factory),
                    worker_id="test-worker")
    worker.run_until_idle(max_iterations=20)
    return worker


def upload(client, path) -> dict:
    with open(path, "rb") as fh:
        r = client.post("/api/v1/uploads", files={"file": ("talk.mp4", fh, "video/mp4")})
    assert r.status_code == 201, r.text
    return r.json()


def ffprobe_json(path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def test_full_pipeline(client, app, video_60s, tmp_path):
    register(client)
    asset = upload(client, video_60s)
    assert asset["status"] == "READY"
    assert 74 <= asset["duration_seconds"] <= 76

    # duplicate detection returns the existing asset
    dup = upload(client, video_60s)
    assert dup["id"] == asset["id"] and dup["duplicate_of"] == asset["id"]

    r = client.post("/api/v1/jobs", json={"upload_id": asset["id"], "settings": {"num_clips": 2}},
                    headers={"Idempotency-Key": "job-key-123456"})
    assert r.status_code == 201, r.text
    job = r.json()
    assert job["status"] == "QUEUED"
    # idempotent replay
    again = client.post("/api/v1/jobs", json={"upload_id": asset["id"], "settings": {"num_clips": 2}},
                        headers={"Idempotency-Key": "job-key-123456"})
    assert again.status_code == 200 and again.json()["id"] == job["id"]

    run_worker(app)

    job = client.get(f"/api/v1/jobs/{job['id']}").json()
    assert job["status"] == "COMPLETED", job
    assert job["progress"] == 100
    assert job["versions"]["scoring_version"]
    assert job["versions"]["transcription_provider"] == "fake"

    events = client.get(f"/api/v1/jobs/{job['id']}/events").json()
    types = [e["event_type"] for e in events]
    assert "stage.completed" in types and "job.completed" in types
    stages = {e["stage"] for e in events if e["event_type"] == "stage.completed"}
    assert {"validate", "transcribe", "candidate_scoring", "render", "qa", "finalize"} <= stages

    clips = client.get(f"/api/v1/jobs/{job['id']}/clips").json()
    assert 1 <= len(clips) <= 2
    first = clips[0]
    assert first["status"] == "READY"
    assert 10 <= first["duration"] <= 90
    assert first["title"] and first["reasons"]
    assert set(first["features"]) >= {"hook", "completeness", "emotional_interest", "information_value",
                                      "narrative_completeness", "audio_quality", "visual_quality",
                                      "caption_suitability", "length_fit"}
    assert first["download_url"] and first["thumbnail_url"]
    assert first["render"]["filename"].startswith(f"clip_{job['id']}_01_")

    # download via signed URL (no auth needed) with Range support
    anon = client.__class__(app)
    url = first["download_url"]
    r = anon.get(url)
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    out = tmp_path / "clip.mp4"
    out.write_bytes(r.content)
    info = ffprobe_json(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert v["codec_name"] == "h264" and a["codec_name"] == "aac"
    assert (v["width"], v["height"]) == (360, 640)
    r = anon.get(url, headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100
    # tampered signature rejected
    assert anon.get(url.replace("sig=", "sig=x")).status_code == 403

    transcript = client.get(f"/api/v1/jobs/{job['id']}/transcript").json()
    assert transcript["segments"] and transcript["segments"][0]["words"]

    # manual correction triggers a re-render with the new aspect ratio
    r = client.patch(f"/api/v1/clips/{first['id']}", json={"aspect_ratio": "1:1", "caption_preset": "bold",
                                                           "title": "Edited title"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["render_required"] and body["render_queued"]
    assert body["clip"]["status"] == "RENDERING"
    run_worker(app)
    clip = client.get(f"/api/v1/clips/{first['id']}").json()
    assert clip["status"] == "READY", clip
    assert clip["title"] == "Edited title"
    assert (clip["render"]["width"], clip["render"]["height"]) == (480, 480)
    assert clip["render"]["id"] != first["render"]["id"]
    # old render file was removed; new one downloads
    assert anon.get(url).status_code == 404
    assert anon.get(clip["download_url"]).status_code == 200

    usage = client.get("/api/v1/me/usage").json()
    assert usage["usage"]["today_source_minutes"] > 1

    # SSE stream of a finished job ends with job.finished
    with client.stream("GET", f"/api/v1/jobs/{job['id']}/stream") as resp:
        text = "".join(resp.iter_text())
    assert "event: job.progress" in text and "event: job.finished" in text

    # delete job → clips gone
    assert client.delete(f"/api/v1/jobs/{job['id']}").status_code == 204
    assert client.get(f"/api/v1/jobs/{job['id']}").status_code == 404


def test_resume_skips_completed_stages(client, app, video_60s):
    """Crash recovery: a job re-queued mid-pipeline reuses valid artifacts (TRD §45)."""
    from clipforge.db.models import Job
    from clipforge.db.session import session_scope

    register(client)
    asset = upload(client, video_60s)
    job = client.post("/api/v1/jobs", json={"upload_id": asset["id"], "settings": {"num_clips": 1}}).json()
    run_worker(app)
    assert client.get(f"/api/v1/jobs/{job['id']}").json()["status"] == "COMPLETED"

    # Simulate a crash after rendering: drop the qa/finalize stage records and requeue.
    with session_scope(app.state.session_factory) as s:
        j = s.get(Job, job["id"])
        manifest = dict(j.manifest)
        manifest["stages"] = {k: v for k, v in manifest["stages"].items() if k not in ("render", "qa", "finalize")}
        j.manifest = manifest
        j.status = "QUEUED"
        j.completed_at = None
        j.progress = 70
    from clipforge.core.config import Settings  # noqa: F401  (settings unchanged)

    manifest_path = app.state.settings.STORAGE_ROOT / "work" / job["id"] / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    run_worker(app)
    j = client.get(f"/api/v1/jobs/{job['id']}").json()
    assert j["status"] == "COMPLETED", j
    events = client.get(f"/api/v1/jobs/{job['id']}/events").json()
    skipped = {e["stage"] for e in events if e["event_type"] == "stage.skipped"}
    assert "transcribe" in skipped and "candidate_scoring" in skipped


def test_cancel_queued_job(client, app, video_60s):
    register(client)
    asset = upload(client, video_60s)
    job = client.post("/api/v1/jobs", json={"upload_id": asset["id"]}).json()
    r = client.post(f"/api/v1/jobs/{job['id']}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "CANCELLED"
    run_worker(app)
    assert client.get(f"/api/v1/jobs/{job['id']}").json()["status"] == "CANCELLED"
    # retry a cancelled job → queued again
    r = client.post(f"/api/v1/jobs/{job['id']}/retry")
    assert r.status_code == 200 and r.json()["status"] == "QUEUED"


def test_rejects_corrupted_upload(client, tmp_path):
    register(client)
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"garbage" * 1000)
    with open(bad, "rb") as fh:
        r = client.post("/api/v1/uploads", files={"file": ("bad.mp4", fh, "video/mp4")})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "MEDIA_CORRUPTED"
    assert r.json()["error"]["diagnostic_id"]


def test_chunked_upload(client, video_60s):
    register(client)
    data = video_60s.read_bytes()
    init = client.post("/api/v1/uploads/init", json={"filename": "talk.mp4", "size_bytes": len(data)}).json()
    chunk = 256 * 1024
    for offset in range(0, len(data), chunk):
        r = client.put(f"/api/v1/uploads/{init['id']}/chunks?offset={offset}", content=data[offset:offset + chunk])
        assert r.status_code == 200, r.text
    # wrong offset is rejected
    assert client.put(f"/api/v1/uploads/{init['id']}/chunks?offset=5", content=b"x").status_code == 409
    import hashlib

    r = client.post(f"/api/v1/uploads/{init['id']}/complete",
                    json={"checksum_sha256": hashlib.sha256(data).hexdigest()})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "READY"
