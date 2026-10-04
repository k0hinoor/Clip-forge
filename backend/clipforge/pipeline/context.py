"""Pipeline context: project folders, transcript persistence and progress reporting.

Every project owns a self-describing directory tree::

    data/projects/<slug>_<id>/
        source/      downloaded or uploaded media
        audio/       extracted 16 kHz mono wav
        transcript/  transcript.json + words.jsonl
        analysis/    segmentation, candidates, scoring report
        clips/       per-clip caption files and edit plans
        renders/     finished MP4s (+ previews, thumbnails, srt)
        metadata/    source metadata + ffprobe output

Keeping the artefacts on disk (as well as in SQLite) means the user can inspect,
back up or script against a project without going through the API.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .. import __version__
from ..ai.segment import Sentence
from ..ai.transcribe import Utterance, Word
from ..config import Env, get_settings
from ..db import Project, TranscriptSegment, session_scope, slugify
from ..logging_setup import get_logger
from ..media.download import safe_upload_path

log = get_logger("clipforge.worker")

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


# --------------------------------------------------------------------------- #
# Progress reporting
# --------------------------------------------------------------------------- #


class ProgressReporter:
    """Interface the pipeline uses to publish progress (implemented by the job runner)."""

    def stage(self, key: str, message: str, *, fraction: float = 0.0) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def sub(self, fraction: float, message: str = "") -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def log(self, message: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def cancelled(self) -> bool:  # pragma: no cover - interface
        return False


class NullReporter(ProgressReporter):
    """Used in tests and one-off scripts."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.stage_key = ""

    def stage(self, key: str, message: str, *, fraction: float = 0.0) -> None:
        self.stage_key = key
        self.messages.append(f"[{key}] {message}")

    def sub(self, fraction: float, message: str = "") -> None:
        if message:
            self.messages.append(f"  {message}")

    def log(self, message: str) -> None:
        self.messages.append(message)


# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #


@dataclass
class ProjectPaths:
    root: Path
    source: Path
    audio: Path
    transcript: Path
    analysis: Path
    clips: Path
    renders: Path
    metadata: Path

    @classmethod
    def for_project(cls, project: Project) -> "ProjectPaths":
        return cls.for_snapshot({"id": project.id, "title": project.title})

    @classmethod
    def for_snapshot(cls, snapshot: dict[str, Any]) -> "ProjectPaths":
        base = Env.DATA_DIR / "projects" / f"{slugify(str(snapshot.get('title') or 'project'), max_length=40)}_{snapshot.get('id')}"
        return cls(
            root=base,
            source=base / "source",
            audio=base / "audio",
            transcript=base / "transcript",
            analysis=base / "analysis",
            clips=base / "clips",
            renders=base / "renders",
            metadata=base / "metadata",
        )

    def ensure(self) -> "ProjectPaths":
        for folder in (self.root, self.source, self.audio, self.transcript, self.analysis, self.clips, self.renders, self.metadata):
            folder.mkdir(parents=True, exist_ok=True)
        return self

    def to_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "source": str(self.source),
            "audio": str(self.audio),
            "transcript": str(self.transcript),
            "analysis": str(self.analysis),
            "clips": str(self.clips),
            "renders": str(self.renders),
            "metadata": str(self.metadata),
        }

    def clip_dir(self, index: int) -> Path:
        folder = self.clips / f"clip_{index:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def render_path(self, index: int, title: str, *, extension: str = ".mp4") -> Path:
        settings = get_settings()
        template = settings.export_filename_template or "{project_slug}_{index:02d}_{title_slug}"
        name = template.format(
            project_slug=slugify(self.root.name, max_length=40),
            index=index,
            title_slug=slugify(title, max_length=48),
        )
        name = SAFE_NAME.sub("-", name).strip("-") or f"clip_{index:02d}"
        self.renders.mkdir(parents=True, exist_ok=True)
        target = self.renders / f"{name}{extension}"
        if target.exists():
            target = self.renders / f"{name}_{int(target.stat().st_mtime)}{extension}"
        return target

    def safe_child(self, folder: Path, filename: str) -> Path:
        """Resolve a filename inside one of our folders (never outside)."""
        candidate = safe_upload_path(folder, filename)
        return candidate

    def free_space_gb(self) -> float:
        from ..system import disk_report

        return float(disk_report(self.root).get("free_gb", 0.0))


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #


def save_transcript(project_id: str, utterances: Sequence[Utterance], *, language: str, engine: str, model: str) -> int:
    """Persist every utterance and every word of the transcript."""
    with session_scope() as session:
        session.query(TranscriptSegment).filter(TranscriptSegment.project_id == project_id).delete()
        for index, utterance in enumerate(utterances):
            session.add(
                TranscriptSegment(
                    project_id=project_id,
                    idx=index,
                    start=utterance.start,
                    end=utterance.end,
                    text=utterance.text.strip(),
                    speaker=utterance.speaker or "SPEAKER_01",
                    language=language,
                    confidence=utterance.confidence,
                    words_json=json.dumps([word.to_dict() for word in utterance.words]),
                )
            )
        project = session.get(Project, project_id)
        if project is not None:
            project.segment_count = len(utterances)
            project.word_count = sum(len(utterance.words) for utterance in utterances)
            try:
                stats = json.loads(project.stats_json or "{}")
            except (TypeError, ValueError):
                stats = {}
            stats["transcript"] = {"engine": engine, "model": model, "language": language}
            stats["speakers"] = sorted({utterance.speaker for utterance in utterances if utterance.speaker})
            project.stats_json = json.dumps(stats)
    return len(utterances)


def load_sentences(project_id: str) -> list[Sentence]:
    """Rebuild pipeline sentences from the stored transcript."""
    from ..ai.segment import build_sentences

    with session_scope() as session:
        rows = (
            session.query(TranscriptSegment)
            .filter(TranscriptSegment.project_id == project_id)
            .order_by(TranscriptSegment.idx)
            .all()
        )
        utterances: list[Utterance] = []
        for row in rows:
            words = [
                Word(
                    word=item.get("word", ""),
                    start=float(item.get("start", 0.0)),
                    end=float(item.get("end", 0.0)),
                    confidence=float(item.get("confidence", 0.0)),
                    speaker=item.get("speaker", row.speaker),
                )
                for item in row.words
                if item.get("word")
            ]
            utterances.append(
                Utterance(
                    text=row.text,
                    start=row.start,
                    end=row.end,
                    speaker=row.speaker,
                    confidence=row.confidence,
                    words=words,
                )
            )
    return build_sentences(utterances)


def write_transcript_files(paths: ProjectPaths, sentences: Sequence[Sentence], *, language: str, engine: str, model: str) -> None:
    payload = {
        "clipforge_version": __version__,
        "language": language,
        "engine": engine,
        "model": model,
        "sentences": [sentence.to_dict(with_words=True) for sentence in sentences],
    }
    (paths.transcript / "transcript.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    with (paths.transcript / "words.jsonl").open("w", encoding="utf-8") as handle:
        for sentence in sentences:
            for word in sentence.words:
                handle.write(json.dumps({**word.to_dict(), "sentence": sentence.index}, ensure_ascii=False) + "\n")


def write_analysis_files(paths: ProjectPaths, name: str, payload: Any) -> Path:
    target = paths.analysis / f"{name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return target


def cached_source_path(video_id: str, extension: str = ".mp4") -> Path:
    folder = Env.DATA_DIR / "cache" / "sources"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{SAFE_NAME.sub('_', video_id)}{extension}"


def link_or_copy(source: Path, destination: Path) -> Path:
    """Hard-link when possible (same volume), otherwise copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    try:
        import os

        os.link(source, destination)
        return destination
    except OSError:
        shutil.copy2(source, destination)
        return destination


def find_media(folder: Path) -> Path | None:
    extensions = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".mp3", ".m4a", ".wav"}
    if not folder.exists():
        return None
    candidates = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in extensions]
    if not candidates:
        return None
    candidates.sort(key=lambda path: (-path.stat().st_size))
    return candidates[0]


@dataclass
class StageTiming:
    stage: str
    seconds: float
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def format_seconds(value: float) -> str:
    minutes, seconds = divmod(int(max(value, 0)), 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {seconds:02d}s"


__all__ = [
    "NullReporter",
    "ProgressReporter",
    "ProjectPaths",
    "StageTiming",
    "cached_source_path",
    "find_media",
    "format_seconds",
    "link_or_copy",
    "load_sentences",
    "save_transcript",
    "write_analysis_files",
    "write_transcript_files",
]
