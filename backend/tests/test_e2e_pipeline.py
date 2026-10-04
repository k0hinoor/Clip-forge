"""End-to-end: real video → transcript → clip discovery → edit plan → rendered MP4.

The fixture is a 7½-minute episode built with a speech synthesiser over synthetic
video (``tools/make_test_media.py``), including a scripted "filler" block (ads and
admin) that the analyser must reject. Every stage below runs the production code
path with ffmpeg, so a passing run means the pipeline really works on media files.

Requires ``CLIPFORGE_TEST_MEDIA`` (or ``/tmp/fix/source.mp4``). Skipped otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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
            "min_score": 70,        # the shipped default: the fixture must clear it on its own
            "clip_mode": "max",     # discover everything publishable, no arbitrary cap
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


def test_transcript_text_matches_the_spoken_script(analysed_project, media_manifest):
    """Every line the fixture speaks comes back from the transcript, in order.

    The expected strings come from the fixture's own manifest, so this test keeps
    working when the script is rewritten - it checks the plumbing, not the prose.
    """
    from clipforge.services.projects import project_transcript

    transcript = project_transcript(analysed_project["project_id"], with_words=True)
    body = " ".join(" ".join(segment["text"].split()) for segment in transcript["segments"]).lower()
    spoken = [" ".join(segment["text"].split()).lower() for segment in media_manifest["segments"]]

    assert spoken, "the fixture manifest must describe what was spoken"

    # Walk forward through the transcript: every line must appear, in the order it
    # was spoken, and a deliberately repeated line must still match its own turn.
    cursor = 0
    for line in spoken:
        found = body.find(line, cursor)
        assert found >= 0, f"missing {line[:70]!r} from the transcript"
        cursor = found + len(line)


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
    """The clip count is whatever the content clears, never a fixed number.

    A short synthetic episode legitimately yields only a handful of clips at the
    shipped threshold, so the interesting assertion is not "how many" but "driven
    by what": every persisted clip must clear the quality gate, the selection must
    be exactly reproducible from its own candidate pool, and opening the gate must
    admit more moments. No constant caps that list - only the episode length.
    """
    from clipforge.ai.candidates import select_by_mode
    from clipforge.ai.scoring import enforce_non_overlap

    outcome = analysed_project["outcome"]
    stats = json.loads(json.dumps(outcome.stats))
    report = (stats.get("stages") or {}).get("score") or {}
    min_score = float(report.get("min_score") or 0)

    assert outcome.candidate_count >= 100, "a five minute episode must yield a wide pool"
    assert outcome.clip_count >= 2, f"expected more than one publishable moment, got {outcome.clip_count}"
    assert report.get("selected") == outcome.clip_count
    assert report.get("input", 0) > outcome.clip_count, "the pool must be wider than the selection"
    assert min_score > 0 and report.get("mode") == "max"

    with session_scope() as session:
        clips = session.query(Clip).filter(Clip.project_id == analysed_project["project_id"]).all()
        rows = (
            session.query(Candidate)
            .filter(Candidate.project_id == analysed_project["project_id"])
            # "selected" rows are the same candidates after the selection pass, so the
            # pool the analyser ranked is the union of all three statuses.
            .filter(Candidate.status.in_(("kept", "duplicate", "selected")))
            .all()
        )
        assert clips, "the analysis must have persisted clips"
        for clip in clips:
            assert clip.score >= min_score * 0.85 - 0.1, (
                f"clip {clip.title!r} scored {clip.score}, below the gate of {min_score * 0.85:.1f}"
            )
        pool = [
            SimpleNamespace(
                score=row.score,
                start=row.start,
                end=row.end,
                duration=row.duration,
                status="candidate",
                reason="",
                features=SimpleNamespace(penalties=json.loads(row.penalties_json or "{}")),
            )
            for row in rows
        ]

    # Replaying the real selection rule over the real candidate pool.
    counts = {}
    chosen_at_min: list = []
    for threshold in (min_score, 55.0, 30.0):
        chosen, _dropped = enforce_non_overlap(
            select_by_mode(list(pool), min_score=threshold, mode="max", max_clips=0)
        )
        counts[threshold] = len(chosen)
        if threshold == min_score:
            chosen_at_min = chosen

    # The shipped analysis must be reproducible from its own inputs...
    assert counts[min_score] == outcome.clip_count, (
        f"the persisted pool reproduces {counts[min_score]} clips, the analysis reported {outcome.clip_count}"
    )
    # ...and opening the gate must admit more moments: the count follows the
    # content, it is never a constant. (Past a point the ceiling is the episode
    # itself - clips may not overlap, and each one is 35-75s long - which is the
    # only cap this design has.)
    assert counts[30.0] > counts[min_score], counts
    assert counts[55.0] >= counts[min_score], counts
    assert outcome.clip_count >= 2, f"expected more than one publishable moment, got {outcome.clip_count}"

    # Clips may touch, and may even share a little air time (the non-overlap rule
    # tolerates up to 35% of the shorter clip so a strong line is not lost), but
    # they may never be the same moment twice.
    worst = 0.0
    for index, first in enumerate(chosen_at_min):
        for second in chosen_at_min[index + 1 :]:
            shared = min(first.end, second.end) - max(first.start, second.start)
            if shared <= 0:
                continue
            worst = max(worst, shared / min(first.duration, second.duration))
    assert worst <= 0.36, f"two selected clips share {worst:.0%} of the shorter one"
    covered = sum(candidate.end - candidate.start for candidate in chosen_at_min)
    assert covered <= outcome.duration * 1.5, "clips must stay inside the episode"


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
        assert clip.file_path and Path(clip.file_path).exists(), "the rendered file must be recorded on the clip"
        assert clip.render_seconds > 0
        assert clip.progress >= 0.99


def test_render_preserves_timing_when_the_frame_rate_differs(analysed_project):
    """A 24 fps output of a 30 fps source must keep its full length.

    ``zoompan`` (the punch-in step) renumbers timestamps at whatever frame rate it
    is given, so it runs at the *source* cadence. If that ever regresses the video
    is squeezed onto a shorter clock and drifts away from its own audio - which is
    exactly what this test measures.
    """
    from clipforge.media.ffmpeg import probe_media
    from clipforge.media.framing import crop_plan_from_dict
    from clipforge.pipeline.edit import layout_payload
    from clipforge.pipeline.render import _preview_dims, render_clip

    with session_scope() as session:
        clip = (
            session.query(Clip)
            .filter(Clip.project_id == analysed_project["project_id"])
            .order_by(Clip.score.desc())
            .first()
        )
        clip_id = clip.id
        trim = json.loads(clip.trim_json or "{}")
        layout = json.loads(clip.layout_json or "{}")
        expected = float((trim.get("segments") or [{}])[-1].get("out_end", 0.0))

    assert expected > 10, "the fixture clip must be a real, silence-trimmed clip"
    assert layout_payload(layout).get("layout"), "the clip must carry its edit plan"
    assert crop_plan_from_dict(layout.get("crop")) is not None, "the plan must carry its reframing"

    result = render_clip(
        clip_id,
        quality="preview",
        overrides={"output_fps": 24, "output_width": 360, "output_height": 640, "render_preset": "ultrafast"},
    )

    expected_dims = _preview_dims(360, 640)
    info = probe_media(result["output"])
    assert info.fps == 24
    assert (info.width, info.height) == expected_dims
    assert abs(info.duration - expected) <= 0.6, (
        f"rendered {info.duration:.2f}s for a {expected:.2f}s timeline - the frame rate conversion compressed it"
    )
    # the punch-in step is part of that render, so the zoompan cadence is covered
    assert any("zoompan" in line or "Punch-ins" in line for line in result["plan"]), result["plan"]


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
    assert detail.json()["why"]

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
            if status in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(1.0)
        assert status == "succeeded", job
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
                event = channel.get(timeout=0.2)
            except Exception:  # noqa: BLE001 - empty queue
                continue
            # Subscribing replays the recent buffer (project.created, job.* ...),
            # so drain until the event we just published shows up.
            if event.to_payload()["type"] == "test.ping":
                received = event
                break
        assert received is not None, "the published event never reached the subscriber"
    finally:
        BUS.unsubscribe(channel)
