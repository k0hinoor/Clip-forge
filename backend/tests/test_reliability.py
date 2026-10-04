"""Regression tests for the reliability fixes (no media fixtures needed).

Covers settings merging, the asyncio event bus behind the SSE stream, panel
cropping, export file names, the job queue's state machine (claim limits,
restart recovery, retry rules) and clip bookkeeping.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from clipforge.config import AppSettings, SettingsStore, load_settings_leniently, merge_settings_patch
from clipforge.errors import ClipForgeError


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


def test_partial_caption_update_keeps_the_rest_of_the_theme():
    base = AppSettings(caption={"font": "Montserrat", "font_size": 60}).model_dump()
    merged = AppSettings.model_validate(merge_settings_patch(base, {"caption": {"highlight_color": "#00FF00"}}))

    assert merged.caption.highlight_color == "#00FF00"
    assert merged.caption.font == "Montserrat" and merged.caption.font_size == 60


def test_caption_preset_applies_its_look_but_explicit_keys_win():
    base = AppSettings().model_dump()
    merged = AppSettings.model_validate(merge_settings_patch(base, {"caption": {"preset": "minimal", "font_size": 50}}))

    assert merged.caption.preset == "minimal"
    assert merged.caption.uppercase is False, "the preset's own look is applied"
    assert merged.caption.font_size == 50, "an explicit key in the same patch wins"


def test_settings_store_rejects_unknown_and_invalid_values_with_readable_messages():
    store = SettingsStore()  # never persisted: both updates fail validation first
    with pytest.raises(ClipForgeError) as unknown:
        store.update({"no_such_setting": 1})
    assert "no_such_setting" in unknown.value.message

    with pytest.raises(ClipForgeError) as invalid:
        store.update({"min_score": 150})
    assert "min_score" in invalid.value.message and "validation error" not in invalid.value.message


def test_stored_settings_survive_unknown_and_invalid_entries():
    loaded = load_settings_leniently({"min_score": 55, "renamed_in_a_later_version": True, "output_fps": 7})

    assert loaded.min_score == 55, "valid values are kept"
    assert loaded.output_fps == AppSettings().output_fps, "an invalid value falls back to its default alone"


def test_broken_filename_templates_are_rejected():
    with pytest.raises(ValueError):
        AppSettings(export_filename_template="{nope}")
    assert AppSettings(export_filename_template="").export_filename_template  # blank = default


# --------------------------------------------------------------------------- #
# Event bus (SSE)
# --------------------------------------------------------------------------- #


def test_async_subscribers_receive_events_published_from_worker_threads():
    from clipforge.services.events import EventBus

    async def scenario() -> None:
        bus = EventBus()
        bus.publish("job.queued", {"n": 0}, project_id="prj_a")  # before subscribing: replayed
        subscription = bus.subscribe_async("prj_a")

        def worker() -> None:
            bus.publish("job.progress", {"progress": 0.5}, project_id="prj_a")
            bus.publish("job.progress", {"progress": 0.9}, project_id="prj_b")  # another project

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        replayed = await subscription.next(timeout=2)
        live = await subscription.next(timeout=2)
        assert replayed is not None and replayed.type == "job.queued"
        assert live is not None and live.to_payload()["progress"] == 0.5
        assert await subscription.next(timeout=0.2) is None, "events of other projects are filtered out"

        subscription.close()
        assert bus.subscriber_count() == 0

    asyncio.run(scenario())


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def test_crop_window_is_recut_for_the_panel_it_fills():
    from clipforge.media.framing import CropKeyframe, CropPlan, fit_crop_to_panel

    full_frame = CropPlan(1920, 1080, 1080, 1920, 608, 1080, mode="tracked", keyframes=[CropKeyframe(0.0, 0.3), CropKeyframe(4.0, 0.7)])
    panel = fit_crop_to_panel(full_frame, 1080, 1248)  # 65% split panel

    assert panel.crop_height == 1080
    assert abs(panel.crop_width / panel.crop_height - 1080 / 1248) < 0.01
    assert [keyframe.center_x for keyframe in panel.keyframes] == [0.3, 0.7], "subject tracking is preserved"

    vertical = fit_crop_to_panel(CropPlan(1080, 1920, 1920, 1080, 1080, 1920), 1920, 1080)
    assert vertical.crop_width == 1080 and vertical.crop_height == 608, "a tall source is cropped vertically"


def test_ducking_strength_follows_the_setting():
    from clipforge.media.compose import ducking_ratio

    assert ducking_ratio(-12) == pytest.approx(9.0)
    assert ducking_ratio(-1) == 2.0 and ducking_ratio(-60) == 20.0


def test_export_names_survive_broken_templates():
    from clipforge.pipeline.context import render_basename

    assert render_basename("{index:03d}-{title_slug}", index=4, title="Big idea!", project="Show") == "004-big-idea"
    assert render_basename("{oops}", index=2, title="Talk", project="My Show") == "my-show_02_talk"
    assert "/" not in render_basename("{title_slug}", index=1, title="../../etc/passwd", project="p")


# --------------------------------------------------------------------------- #
# Job queue state machine
# --------------------------------------------------------------------------- #


@pytest.fixture()
def empty_queue():
    from clipforge.db import Job, session_scope

    def wipe() -> None:
        with session_scope() as session:
            session.query(Job).delete()

    wipe()
    yield
    wipe()


def test_claims_respect_per_kind_concurrency_limits(empty_queue):
    from clipforge.jobs import queue

    first = queue.enqueue("render_clip", project_id="prj_q", clip_id="clp_1", priority=1)
    second = queue.enqueue("render_clip", project_id="prj_q", clip_id="clp_2", priority=2)
    analysis = queue.enqueue("analyze", project_id="prj_q2", priority=3)

    assert queue.claim_next("w1", max_renders=1)["id"] == first["id"]
    assert queue.claim_next("w2", max_renders=1)["id"] == analysis["id"], "the second render waits; other kinds may run"
    assert queue.claim_next("w3", max_renders=1) is None, "both kinds are at their limit"

    queue.finish(first["id"])
    assert queue.claim_next("w3", max_renders=1)["id"] == second["id"]


def test_interrupted_jobs_resume_once_then_fail(empty_queue):
    from clipforge.jobs import queue

    job = queue.enqueue("render_clip", project_id="prj_r", clip_id="clp_r")
    queue.claim_next("w1")  # attempt 1, then the process "dies"
    queue.recover_stale_jobs()
    resumed = queue.get(job["id"])
    assert resumed["status"] == "queued" and "restart" in resumed["message"]

    queue.claim_next("w1")  # attempt 2 dies too: no endless crash loop
    queue.recover_stale_jobs()
    failed = queue.get(job["id"])
    assert failed["status"] == "failed" and failed["error"]["code"] == "interrupted"


def test_only_failed_or_cancelled_jobs_can_be_retried(empty_queue):
    from clipforge.jobs import queue

    job = queue.enqueue("render_preview", project_id="prj_t", clip_id="clp_t")
    queue.claim_next("w1")
    queue.finish(job["id"])
    with pytest.raises(ClipForgeError) as conflict:
        queue.retry(job["id"])
    assert conflict.value.status_code == 409

    queue.fail(job["id"], code="render_failed", message="boom")
    assert queue.retry(job["id"])["status"] == "queued"
    assert queue.retry("job_missing") is None


def test_a_deleted_job_counts_as_cancelled():
    from clipforge.jobs import queue

    assert queue.is_cancelled("job_that_does_not_exist") is True


# --------------------------------------------------------------------------- #
# Clip bookkeeping
# --------------------------------------------------------------------------- #


def test_duplicate_and_delete_keep_the_clip_count_right(empty_queue):
    from clipforge.db import Clip, Project, session_scope
    from clipforge.services import clips as clip_service
    from clipforge.services.projects import create_project, delete_project

    project = create_project(url="https://example.com/talk.mp4", title="Bookkeeping", source_type="url")
    with session_scope() as session:
        session.add(Clip(project_id=project["id"], index=1, title="First", start=10.0, end=40.0, duration=30.0))
        session.get(Project, project["id"]).clip_count = 1
    with session_scope() as session:
        original = session.query(Clip).filter(Clip.project_id == project["id"]).one()
        original_id, original_index = original.id, original.index

    try:
        copy = clip_service.duplicate_clip(original_id)
        assert copy["index"] != original_index, "the copy gets its own clip folder"
        with session_scope() as session:
            assert session.get(Project, project["id"]).clip_count == 2

        clip_service.delete_clip(copy["id"])
        with session_scope() as session:
            assert session.get(Project, project["id"]).clip_count == 1
            assert session.get(Clip, original_id) is not None
    finally:
        delete_project(project["id"])
