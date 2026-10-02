"""Startup bootstrap: migrations + seeding config-driven presets."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from clipforge.core.config import BACKEND_ROOT, Settings
from clipforge.core.logging import get_logger
from clipforge.core.presets import render_profiles_config
from clipforge.db.models import RenderProfile, SystemSetting
from clipforge.db.session import session_scope

log = get_logger(__name__)


def run_migrations(settings: Settings) -> None:
    """Apply Alembic migrations up to head (works from a fresh install)."""
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(BACKEND_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(Path(BACKEND_ROOT) / "migrations"))
    cfg.set_main_option("sqlalchemy.url", settings.DATABASE_URL.replace("%", "%%"))
    cfg.attributes["configure_logger"] = False
    command.upgrade(cfg, "head")


def seed_render_profiles(session: Session, settings: Settings) -> int:
    """Insert render profiles from config if missing (DB copy is authoritative)."""
    created = 0
    for name, cfg in render_profiles_config(settings).items():
        if session.scalar(select(RenderProfile).where(RenderProfile.name == name)):
            continue
        session.add(RenderProfile(
            name=name,
            aspect_ratio=str(cfg["aspect_ratio"]),
            width=int(cfg["width"]),
            height=int(cfg["height"]),
            fps=cfg.get("fps"),
            video_codec=str(cfg.get("video_codec", "h264")),
            audio_codec=str(cfg.get("audio_codec", "aac")),
            crf=int(cfg.get("crf", 20)),
            preset=str(cfg.get("preset", "veryfast")),
            audio_bitrate_kbps=int(cfg.get("audio_bitrate_kbps", 160)),
            version=str(cfg.get("version", "1")),
            is_default=bool(cfg.get("default", False)),
            is_active=True,
            config={},
        ))
        created += 1
    return created


def bootstrap(settings: Settings, session_factory: sessionmaker[Session]) -> None:
    settings.ensure_directories()
    if settings.AUTO_MIGRATE:
        run_migrations(settings)
    with session_scope(session_factory) as s:
        n = seed_render_profiles(s, settings)
        if s.get(SystemSetting, "deployment") is None:
            s.add(SystemSetting(key="deployment", value={"mode": settings.DEPLOYMENT_MODE}))
    if n:
        log.info("seeded render profiles", extra={"count": n})
