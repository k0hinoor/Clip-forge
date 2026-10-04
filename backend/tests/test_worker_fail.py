"""A failed render job must flip the owning clip out of `queued`."""

from __future__ import annotations


def _make_clip(status: str = "queued") -> tuple[str, str]:
    from clipforge.db import Clip, Project, session_scope

    with session_scope() as session:
        project = Project(title="worker-fail test", source_type="url", source_url="http://example.invalid/v.mp4")
        session.add(project)
        session.flush()
        clip = Clip(project_id=project.id, title="clip", status=status)
        session.add(clip)
        session.flush()
        return project.id, clip.id


def test_failed_render_job_marks_clip_failed() -> None:
    from clipforge.db import Clip, session_scope
    from clipforge.errors import ClipForgeError, ErrorCode
    from clipforge.jobs.worker import _mark_clip_failed

    project_id, clip_id = _make_clip("queued")
    _mark_clip_failed(
        {"id": "job_x", "kind": "render_clip", "project_id": project_id, "clip_id": clip_id},
        ClipForgeError(code=ErrorCode.INTERNAL, message="boom"),
    )
    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        assert clip is not None
        assert clip.status == "failed"
        assert clip.error_message == "boom"


def test_non_render_jobs_leave_clip_alone() -> None:
    from clipforge.db import Clip, session_scope
    from clipforge.errors import ClipForgeError, ErrorCode
    from clipforge.jobs.worker import _mark_clip_failed

    project_id, clip_id = _make_clip("queued")
    _mark_clip_failed(
        {"id": "job_y", "kind": "analyze", "project_id": project_id, "clip_id": clip_id},
        ClipForgeError(code=ErrorCode.INTERNAL, message="boom"),
    )
    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        assert clip is not None
        assert clip.status == "queued"


def test_settle_project_resets_running_after_clip_job() -> None:
    from clipforge.db import Project, session_scope
    from clipforge.jobs.worker import _settle_project

    with session_scope() as session:
        project = Project(title="settle test", source_type="url", status="running", stage="thumbnail")
        session.add(project)
        session.flush()
        project_id = project.id

    _settle_project({"id": "job_z", "kind": "render_clip", "project_id": project_id})
    with session_scope() as session:
        project = session.get(Project, project_id)
        assert project is not None
        assert project.status == "ready"
        assert project.stage == "ready"


def test_settle_project_ignores_analyze_and_failed() -> None:
    from clipforge.db import Project, session_scope
    from clipforge.jobs.worker import _settle_project

    with session_scope() as session:
        a = Project(title="a", source_type="url", status="running", stage="x")
        b = Project(title="b", source_type="url", status="failed", stage="failed")
        session.add_all([a, b])
        session.flush()
        a_id, b_id = a.id, b.id

    _settle_project({"id": "job_a", "kind": "analyze", "project_id": a_id})
    _settle_project({"id": "job_b", "kind": "render_clip", "project_id": b_id})
    with session_scope() as session:
        assert session.get(Project, a_id).status == "running"
        assert session.get(Project, b_id).status == "failed"
