"""End-to-end: real video → transcript → clip discovery → edit plan → rendered MP4.

The fixture is a 7½-minute episode built with a speech synthesiser over synthetic
video (``tools/make_test_media.py``), including a scripted "filler" block (ads and
admin) that the analyser must reject. Every stage below runs the production code
path with ffmpeg, so a passing run means the pipeline really works on media files.

Requires ``CLIPFORGE_TEST_MEDIA`` (or ``/tmp/fix/source.mp4``). Skipped otherwise.
"""

from __future__ import annotations

import json

import pytest

from clipforge.db import Candidate, Clip, Project, TranscriptSegment, session_scope

pytestmark = pytest.mark.e2e

MIN_WORDS = 400
"""The fixture speaks ~800 words; anything far below means the transcript broke."""


@pytest.fixture(scope="module")
def analysed_project(media_file, fixture_srt, tmp_path_factory):
    """Create a project from the fixture video + captions and run the full analysis."""
    from clipforge.config import get_settings, settings_store
    from clipforge.pipeline.analyze import analyze
    from clipforge.pipeline.context import NullReporter, ProjectPaths
    from clipforge.services import projects as project_service

    # Small, fast settings: preview-sized output, CPU encoder, no LLM, no vision.
    store = settings_store()
    store.update(
        {
            "whisper_model": "tiny",
            "llm_enabled": False,
            "vision_enabled": False,
            "output_width": 1080,
            "output_height": 1920,
            "render_preset": "veryfast",
            "min_clip_seconds": 35,
            "target_clip_seconds": 60,
            "max_clip_seconds": 75,
            "min_score": 62,
            "clip_mode": "max",
            "gameplay_enabled": False,
            "broll_enabled": False,
            "music_enabled": False,
        }
    )

    project = project_service.create_project(
        title="CLIPFORGE fixture episode",
        source_type="upload",
        source_path=media_file,
        options={},
    )
    project_id = project["id"]

    paths = ProjectPaths.for_snapshot(project).ensure()
    target = paths.transcript / "provided.srt"
    target.write_text(fixture_srt.read_text(encoding="utf-8"), encoding="utf-8")

    class Recorder(NullReporter):
        def __init__(self) -> None:
            self.stages: list[str] = []
            self.messages: list[str] = []

        def stage(self, key, message, *, fraction=0.0):  # noqa: D102
            self.stages.append(f"{key}:{message}")
            self.messages.append(message)

        def log(self, message):  # noqa: D102
            self.messages.append(message)

    reporter = Recorder()
    outcome = analyze(project_id, report=reporter)
    return {"project_id": project_id, "outcome": outcome, "reporter": reporter, "paths": paths, "settings": get_settings()}


def test_analysis_persisted_word_level_transcript(analysed_project):
    project_id = analysed_project["project_id"]
    with session_scope() as session:
        rows = (
            session.query(TranscriptSegment)
            .filter(TranscriptSegment.project_id == project_id)
            .order_by(TranscriptSegment.idx)
            .all()
        )
        assert rows, "the transcript must be stored per project"
        words = []
        for row in rows[:40]:
            words.extend(row.words)
        assert len(words) > 60, f"expected word-level records, found {len(words)}"
        for word in words[:20]:
            assert word["word"].strip()
            assert word["end"] >= word["start"] >= 0
            assert "speaker" in word
        project = session.get(Project, project_id)
        assert project.word_count >= MIN_WORDS, project.word_count
        assert project.segment_count >= 30
        assert project.language.startswith("en")
        assert project.status == "ready", (project.status, project.stage, project.error_message)


def test_transcript_text_matches_the_spoken_script(analysed_project):
    from clipforge.services.projects import project_transcript

    transcript = project_transcript(analysed_project["project_id"], with_words=True)
    text = " ".join(segment["text"] for segment in transcript["segments"]).lower()
    for phrase in ("nobody talks about what happens", "one million", "hustle culture", "your price is a story"):
        assert phrase in text, f"missing {phrase!r} in transcript"


def test_candidates_were_scored_and_explained(analysed_project):
    project_id = analysed_project["project_id"]
    with session_scope() as session:
        candidates = session.query(Candidate).filter(Candidate.project_id == project_id).all()
        assert candidates, "candidates must be recorded"
        scored = [candidate for candidate in candidates if candidate.score]
        assert scored, "candidates must carry real scores"
        assert any(candidate.status == "selected" for candidate in candidates)
        for candidate in scored[:20]:
            assert 0 <= candidate.score <= 100
            factors = json.loads(candidate.factors_json or "{}")
            assert factors, "each candidate keeps its factor breakdown"
            assert set(factors) >= {"hook", "emotion", "standalone", "payoff"}
            verdicts = json.loads(candidate.highlights_json or "[]")
            assert verdicts, "why-this-clip reasons must be stored"


def test_clip_count_is_content_driven(analysed_project):
    outcome = analysed_project["outcome"]
    assert outcome.clip_count >= 3, (
        f"a 7.5 minute episode with several strong stories should yield multiple clips, got {outcome.clip_count}"
    )
    assert outcome.candidate_count > outcome.clip_count, "the pool must be wider than the selection"
    stats = json.loads(json.dumps(outcome.stats))
    report = stats.get("scoring", {})
    assert report.get("input", 0) >= outcome.clip_count


def test_ads_and_admin_were_rejected(analysed_project):
    """The scripted sponsor/admin block must not become a clip."""
    with session_scope() as session:
        clips = session.query(Clip).filter(Clip.project_id == analysed_project["project_id"]).all()
        text = " ".join((clip.summary or "") + " " + (clip.hook or "") for clip in clips).lower()
    assert "sponsorship slots" not in text
    assert "newsletter is moving to tuesday" not in text


def test_clips_have_plans_assets_and_reasons(analysed_project):
    from clipforge.services.clips import clip_detail

    with session_scope() as session:
        clips = (
            session.query(Clip)
            .filter(Clip.project_id == analysed_project["project_id"])
            .order_by(Clip.score.desc())
            .all()
        )
        clip_ids = [clip.id for clip in clips]
        assert clip_ids

    detail = clip_detail(clip_ids[0])
    assert detail["score"] > 0
    assert detail["why"], "the UI must be able to show WHY THIS CLIP"
    assert detail["start"] < detail["end"]
    assert detail["duration"] >= 20
    plan = detail["plan"]
    assert plan["layout"] in {"split", "podcast", "broll", "gameplay", "cinematic", "blur"}
    assert plan["timeline"]["segments"], "timeline (silence removal) must be persisted"
    assert detail["captions"] is not None, "caption plan must be available"


def test_caption_preview_uses_the_real_words(analysed_project):
    from clipforge.services.clips import caption_preview

    with session_scope() as session:
        clip = session.query(Clip).filter(Clip.project_id == analysed_project["project_id"]).first()
        clip_id = clip.id
    preview = caption_preview(clip_id, preset="bold_creator")
    assert preview["lines"], preview
    spoken = " ".join(line.get("text", "") for line in preview["lines"]).lower()
    assert len(spoken.split()) >= 5


def test_render_produces_a_valid_vertical_mp4(analysed_project, tmp_path):
    """Render one clip for real and inspect the file with ffmpeg."""
    from clipforge.media.ffmpeg import probe_media
    from clipforge.pipeline.render import render_clip

    with session_scope() as session:
        clip = (
            session.query(Clip)
            .filter(Clip.project_id == analysed_project["project_id"])
            .order_by(Clip.score.desc())
            .first()
        )
        clip_id = clip.id

    result = render_clip(clip_id, quality="final", overrides={"output_width": 540, "output_height": 960, "render_preset": "veryfast"})
    output = result["output"]
    info = probe_media(output)
    assert info.has_video and info.has_audio
    assert (info.width, info.height) == (540, 960)
    assert info.fps == 30
    assert info.duration > 15
    assert result["size_bytes"] > 50_000
    assert result["encode_seconds"] > 0

    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        assert clip.status == "rendered"
        assert clip.output_path
        assert clip.progress >= 0.99


def test_clip_files_and_streaming_endpoints(analysed_project, app_client):
    project_id = analysed_project["project_id"]

    status = app_client.get(f"/api/projects/{project_id}/status")
    assert status.status_code == 200
    body = status.json()
    assert body["project"]["status"] == "ready"
    assert body["stages"], "the UI needs per-stage state"

    clips = app_client.get(f"/api/projects/{project_id}/clips")
    assert clips.status_code == 200
    payload = clips.json()
    assert payload["clips"]
    clip_id = payload["clips"][0]["id"]

    detail = app_client.get(f"/api/clips/{clip_id}")
    assert detail.status_code == 200
    assert detail.json()["clip"]["why"]

    command = app_client.get(f"/api/clips/{clip_id}/command")
    assert command.status_code == 200
    assert "ffmpeg" in command.json()["command"].lower() or "-filter_complex" in command.json()["command"]

    assets = app_client.get(f"/api/clips/{clip_id}/assets")
    assert assets.status_code == 200

    transcript = app_client.get(f"/api/projects/{project_id}/transcript?with_words=false&limit=50")
    assert transcript.status_code == 200
    assert transcript.json()["segments"]

    source = app_client.get(f"/api/projects/{project_id}/source", headers={"Range": "bytes=0-2047"})
    assert source.status_code in (200, 206)
    assert len(source.content) > 0


def test_plan_files_and_srt_exports_exist(analysed_project):
    from clipforge.services.clips import write_clip_srt

    with session_scope() as session:
        clip = session.query(Clip).filter(Clip.project_id == analysed_project["project_id"]).first()
        clip_id, index = clip.id, clip.index

    srt = write_clip_srt(clip_id)
    assert srt.exists() and "-->" in srt.read_text(encoding="utf-8")

    clip_dir = analysed_project["paths"].clip_dir(index)
    assert (clip_dir / "plan.json").exists(), "the edit plan must be written next to the clip"
    plan = json.loads((clip_dir / "plan.json").read_text(encoding="utf-8"))
    assert plan.get("timeline") and plan.get("captions")


def test_job_queue_runs_a_worker(analysed_project):
    """The queue must execute real work in a background worker thread."""
    import time

    from clipforge.jobs.manager import manager
    from clipforge.jobs import queue as job_queue

    with session_scope() as session:
        clip = (
            session.query(Clip)
            .filter(Clip.project_id == analysed_project["project_id"])
            .order_by(Clip.score.desc())
            .first()
        )
        clip_id = clip.id

    handle = manager().enqueue_preview(clip_id)
    assert handle["id"]
    manager().start(workers=1)
    try:
        deadline = time.time() + 420
        status = "queued"
        while time.time() < deadline:
            job = job_queue.get(handle["id"])
            status = job["status"]
            if status in {"finished", "failed", "cancelled"}:
                break
            time.sleep(1.0)
        assert status == "finished", job
    finally:
        manager().stop()


def test_queue_state_and_events_report_progress(analysed_project):
    from clipforge.jobs import queue as job_queue
    from clipforge.services.events import BUS

    state = job_queue.queue_state(analysed_project["project_id"])
    assert "counts" in state
    assert state["counts"]["total"] >= 1

    from clipforge.services import events as bus_module

    channel = BUS.subscribe()
    try:
        bus_module.publish("test.ping", {"ok": True}, project_id=analysed_project["project_id"])
        received = None
        for _ in range(60):
            try:
                received = channel.get(timeout=0.2)
                break
            except Exception:  # noqa: BLE001 - empty queue
                continue
        assert received is not None
        assert received.to_payload()["type"] == "test.ping"
    finally:
        BUS.unsubscribe(channel)
