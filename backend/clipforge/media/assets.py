"""Asset library: gameplay, B-roll and music.

CLIPFORGE ships no third-party footage and scrapes nothing: the library is built
from files the user imports (their own recordings, or anything they are licensed
to use). Categories mirror the folder layout under ``assets/gameplay/`` so the
library can also be filled by simply dropping files in the right folder.

Selection is deterministic per clip (seeded by the clip id), keeps roughly the
requested ratio between calm and fast footage, and avoids reusing the same clip
back to back inside one project.
"""

from __future__ import annotations

import hashlib
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..config import AppSettings, Env, get_settings
from ..db import Asset, session_scope
from ..errors import ClipForgeError, ErrorCode, invalid_input
from ..logging_setup import get_logger

log = get_logger("clipforge.render")

GAMEPLAY_CATEGORIES = (
    "subway_surfer", "temple_run", "minecraft", "parkour", "racing",
    "satisfying", "simulation", "general",
)
BROLL_CATEGORIES = ("broll", "nature", "city", "tech", "abstract", "general")
MUSIC_MOODS = ("emotional", "motivational", "cinematic", "energetic", "dark", "chill")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".flac"}

# Which gameplay feel suits which clip category.
CATEGORY_PACING: dict[str, str] = {
    "funny": "satisfying",
    "story": "parkour",
    "emotional": "satisfying",
    "controversial": "racing",
    "argument": "racing",
    "inspirational": "parkour",
    "advice": "simulation",
    "lesson": "simulation",
    "information": "simulation",
    "surprising": "temple_run",
    "revelation": "temple_run",
    "curiosity": "parkour",
    "experience": "satisfying",
    "qa": "simulation",
}


def assets_root() -> Path:
    root = Env.DATA_DIR / "assets"
    root.mkdir(parents=True, exist_ok=True)
    return root


def default_assets_dir() -> Path:
    """The repo's ``assets/`` folder, used as a seed location for imports."""
    root = Env.DATA_DIR / "assets" / "library"
    root.mkdir(parents=True, exist_ok=True)
    return root


def seed_folders() -> list[str]:
    """Create the standard category folders (both in the data dir and repo)."""
    created: list[str] = []
    for kind, categories in (("gameplay", GAMEPLAY_CATEGORIES), ("broll", BROLL_CATEGORIES), ("music", MUSIC_MOODS)):
        for category in categories:
            folder = assets_root() / kind / category
            folder.mkdir(parents=True, exist_ok=True)
            created.append(str(folder))
    return created


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #


def _sanitize(filename: str) -> str:
    stem = Path(filename).name
    keep = [char if (char.isalnum() or char in "._- ") else "_" for char in stem]
    cleaned = "".join(keep).strip().replace(" ", "_")
    return cleaned or "asset"


def import_asset(
    source: Path,
    *,
    kind: str = "gameplay",
    category: str = "general",
    name: str = "",
    tags: Sequence[str] = (),
    copy: bool = True,
    move: bool = False,
    filename: str = "",
    probe: bool = True,
) -> dict[str, Any]:
    """Import a file into the library and register it in the database.

    ``copy`` copies the file into the library folder (``False`` registers it in
    place); ``move`` moves it there instead (used for finished uploads), and
    ``filename`` names the library copy when the source has a temporary name.
    """
    if kind not in {"gameplay", "broll", "music"}:
        raise invalid_input(f"Unknown asset kind: {kind}")
    allowed = IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
    if source.suffix.lower() not in allowed:
        raise ClipForgeError(
            code=ErrorCode.ASSET_FAILED,
            message=f"{source.suffix or 'That file type'} cannot be imported.",
            hint="Supported: " + ", ".join(sorted(allowed)),
            status_code=422,
        )
    category = _safe_category(kind, category)
    target_dir = assets_root() / kind / category
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / _sanitize(Path(filename).name if filename else source.name)
    if target.exists() and target.resolve() != source.resolve():
        stem, suffix = target.stem, target.suffix
        index = 2
        while target.exists():
            target = target_dir / f"{stem}_{index}{suffix}"
            index += 1

    if move:
        shutil.move(str(source), str(target))
    elif copy:
        shutil.copy2(source, target)
    else:
        target = source

    duration = width = height = 0
    fps = 0.0
    if probe and target.suffix.lower() in VIDEO_EXTENSIONS and kind != "music":
        try:
            from .ffmpeg import probe_media

            info = probe_media(target)
            duration, width, height, fps = info.duration, info.width, info.height, info.fps
        except ClipForgeError as exc:
            log.warning("could not probe imported asset %s: %s", target.name, exc.message)
    elif probe and kind == "music":
        try:
            from .ffmpeg import probe_media

            duration = probe_media(target).duration
        except ClipForgeError:
            duration = 0.0

    with session_scope() as session:
        asset = Asset(
            kind=kind,
            category=category,
            name=name or target.stem.replace("_", " "),
            filename=target.name,
            path=str(target),
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            size_bytes=target.stat().st_size if target.exists() else 0,
            tags_json=__import__("json").dumps(list(tags)),
        )
        session.add(asset)
        session.flush()
        payload = asset.to_dict()
    log.info("imported %s asset '%s' (%s)", kind, target.name, category)
    return payload


def _safe_category(kind: str, category: str) -> str:
    options = GAMEPLAY_CATEGORIES if kind == "gameplay" else BROLL_CATEGORIES if kind == "broll" else MUSIC_MOODS
    cleaned = (category or "general").strip().lower().replace(" ", "_")
    if cleaned in options:
        return cleaned
    return "general" if "general" in options else options[0]


def scan_library_folders() -> list[dict[str, Any]]:
    """Register any file dropped straight into the asset folders."""
    imported: list[dict[str, Any]] = []
    allowed = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS | AUDIO_EXTENSIONS
    for kind in ("gameplay", "broll", "music"):
        root = assets_root() / kind
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in allowed:
                continue
            with session_scope() as session:
                from sqlalchemy import select

                exists = session.execute(select(Asset.id).where(Asset.path == str(path))).first()
            if exists:
                continue
            try:
                payload = import_asset(path, kind=kind, category=path.parent.name, copy=False)
                imported.append(payload)
            except ClipForgeError as exc:
                log.warning("skipping %s: %s", path.name, exc.message)
    if imported:
        log.info("library scan registered %d new assets", len(imported))
    return imported


# --------------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------------- #


def list_assets(kind: str | None = None, category: str | None = None, *, enabled_only: bool = False) -> list[dict[str, Any]]:
    from sqlalchemy import select

    with session_scope() as session:
        query = select(Asset)
        if kind:
            query = query.where(Asset.kind == kind)
        if category:
            query = query.where(Asset.category == category)
        if enabled_only:
            query = query.where(Asset.enabled.is_(True))
        rows = session.execute(query.order_by(Asset.created_at.desc())).scalars().all()
        return [row.to_dict() for row in rows]


def library_summary() -> dict[str, Any]:
    assets = list_assets()
    summary: dict[str, dict[str, int]] = {"gameplay": {}, "broll": {}, "music": {}}
    for asset in assets:
        summary[asset["kind"]][asset["category"]] = summary[asset["kind"]].get(asset["category"], 0) + 1
    counts = {kind: sum(categories.values()) for kind, categories in summary.items()}
    return {"counts": counts, "by_category": summary, "total_seconds": round(sum(a["duration"] for a in assets), 1)}


def delete_asset(asset_id: str, *, remove_file: bool = True) -> bool:

    with session_scope() as session:
        asset = session.get(Asset, asset_id)
        if asset is None:
            return False
        path = Path(asset.path)
        session.delete(asset)
    if remove_file and path.exists() and assets_root() in path.parents:
        try:
            path.unlink()
        except OSError as exc:
            log.warning("could not remove asset file %s: %s", path, exc)
    return True


def set_asset_flags(asset_id: str, *, enabled: bool | None = None, favorite: bool | None = None, category: str | None = None) -> dict[str, Any] | None:
    with session_scope() as session:
        asset = session.get(Asset, asset_id)
        if asset is None:
            return None
        if enabled is not None:
            asset.enabled = enabled
        if favorite is not None:
            asset.favorite = favorite
        if category is not None:
            asset.category = _safe_category(asset.kind, category)
        session.flush()
        return asset.to_dict()


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


@dataclass
class AssetChoice:
    path: Path
    asset_id: str
    name: str
    category: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.asset_id, "name": self.name, "category": self.category, "path": str(self.path), "reason": self.reason}


def _seed_for(clip_key: str) -> int:
    digest = hashlib.sha256(clip_key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def pick_gameplay(
    *,
    clip_key: str,
    category: str,
    duration: float,
    settings: AppSettings | None = None,
    recent_ids: Sequence[str] = (),
    rng_seed: int | None = None,
) -> AssetChoice | None:
    """Choose a gameplay clip for one clip (mode- and category-aware)."""
    settings = settings or get_settings()
    if not settings.gameplay_enabled or settings.gameplay_mode == "off":
        return None

    assets = [asset for asset in list_assets("gameplay") if asset["enabled"]]
    if not assets:
        return None

    desired_category = settings.gameplay_category
    reason = "random gameplay"
    if settings.gameplay_mode == "auto":
        desired_category = CATEGORY_PACING.get(category, settings.gameplay_category)
        reason = f"auto-selected gameplay for a {category} moment"
    elif settings.gameplay_mode == "random" or settings.gameplay_random:
        desired_category = ""
        reason = "random gameplay"

    pool = [asset for asset in assets if not desired_category or asset["category"] == desired_category]
    if not pool:
        pool = assets
        reason += " (no asset in the preferred category, using the whole library)"

    favorites = [asset for asset in pool if asset["favorite"]]
    if favorites:
        pool = favorites + [asset for asset in pool if not asset["favorite"]]
        reason += " · favourites first"

    fresh = [asset for asset in pool if asset["id"] not in set(recent_ids)]
    if fresh:
        pool = fresh

    # Prefer assets long enough to cover the clip, then the largest remaining.
    sufficient = [asset for asset in pool if (asset["duration"] or 0) >= duration * 0.9]
    if sufficient:
        pool = sufficient

    rng = random.Random(rng_seed if rng_seed is not None else _seed_for(clip_key))
    choice = rng.choice(pool)
    if choice["duration"] and choice["duration"] < duration:
        reason += f" · looped ({choice['duration']:.0f}s source for a {duration:.0f}s clip)"

    return AssetChoice(
        path=Path(choice["path"]),
        asset_id=choice["id"],
        name=choice["name"],
        category=choice["category"],
        reason=reason,
    )


def pick_broll(
    *,
    clip_key: str,
    settings: AppSettings | None = None,
    recent_ids: Sequence[str] = (),
) -> AssetChoice | None:
    settings = settings or get_settings()
    if not settings.broll_enabled:
        return None
    assets = [asset for asset in list_assets("broll") if asset["enabled"]]
    if not assets:
        return None
    pool = [asset for asset in assets if asset["category"] == settings.broll_category] or assets
    fresh = [asset for asset in pool if asset["id"] not in set(recent_ids)] or pool
    rng = random.Random(_seed_for(clip_key))
    choice = rng.choice(fresh)
    return AssetChoice(path=Path(choice["path"]), asset_id=choice["id"], name=choice["name"], category=choice["category"], reason="b-roll for this topic")


def pick_music(
    *,
    clip_key: str,
    settings: AppSettings | None = None,
    recent_ids: Sequence[str] = (),
    category: str = "",
) -> AssetChoice | None:
    settings = settings or get_settings()
    if not settings.music_enabled:
        return None
    assets = [asset for asset in list_assets("music") if asset["enabled"]]
    if not assets:
        return None
    mood = settings.music_mood
    if category in {"emotional", "inspirational"} and mood not in {"emotional", "cinematic"}:
        mood = "emotional"
    pool = [asset for asset in assets if asset["category"] == mood] or assets
    fresh = [asset for asset in pool if asset["id"] not in set(recent_ids)] or pool
    rng = random.Random(_seed_for(clip_key))
    choice = rng.choice(fresh)
    return AssetChoice(path=Path(choice["path"]), asset_id=choice["id"], name=choice["name"], category=choice["category"], reason=f"{mood} track")


def resolve_asset_path(asset_id: str) -> Path | None:
    with session_scope() as session:
        asset = session.get(Asset, asset_id)
        if asset is None:
            return None
        path = Path(asset.path)
    return path if path.exists() else None


def ensure_seed_assets() -> int:
    """Create the folder skeleton, then register anything already inside it."""
    seed_folders()
    return len(scan_library_folders())


def copy_example_structure() -> list[str]:
    """Return the recommended folder tree (used by the Assets screen)."""
    lines: list[str] = []
    for kind, categories in (("gameplay", GAMEPLAY_CATEGORIES), ("broll", BROLL_CATEGORIES), ("music", MUSIC_MOODS)):
        lines.append(f"{kind}/")
        for category in categories:
            lines.append(f"  {category}/")
    return lines


__all__ = [
    "BROLL_CATEGORIES",
    "GAMEPLAY_CATEGORIES",
    "MUSIC_MOODS",
    "AssetChoice",
    "assets_root",
    "copy_example_structure",
    "delete_asset",
    "ensure_seed_assets",
    "import_asset",
    "library_summary",
    "list_assets",
    "pick_broll",
    "pick_gameplay",
    "pick_music",
    "resolve_asset_path",
    "scan_library_folders",
    "seed_folders",
    "set_asset_flags",
]
