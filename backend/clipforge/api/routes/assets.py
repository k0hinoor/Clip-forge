"""Asset routes: the user's gameplay / B-roll / music library."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Body, File, Form, Query, Request, UploadFile

from ...config import Env, get_settings
from ...errors import ClipForgeError, ErrorCode
from ...logging_setup import get_logger
from ...media import assets as asset_service
from ...media.download import safe_upload_path
from ...services import events
from ..media import require_file, stream_file
from ..schemas import AssetImportRequest, AssetUpdate

router = APIRouter(prefix="/assets", tags=["assets"])
log = get_logger(__name__)


@router.get("")
def list_assets(
    kind: str = Query("", pattern="^(|gameplay|broll|music)$"),
    category: str = Query(""),
    enabled_only: bool = Query(False),
):
    return {
        "assets": asset_service.list_assets(kind or None, category or None, enabled_only=enabled_only),
        "library": asset_service.library_summary(),
    }


@router.get("/gameplay")
def get_gameplay(category: str = Query(""), enabled_only: bool = Query(True)):
    """Gameplay library (the spec's ``GET /api/assets/gameplay``)."""
    return {
        "assets": asset_service.list_assets("gameplay", category or None, enabled_only=enabled_only),
        "categories": list(asset_service.GAMEPLAY_CATEGORIES),
        "library": asset_service.library_summary(),
    }


@router.get("/broll")
def get_broll(category: str = Query(""), enabled_only: bool = Query(True)):
    return {
        "assets": asset_service.list_assets("broll", category or None, enabled_only=enabled_only),
        "categories": list(asset_service.BROLL_CATEGORIES),
    }


@router.get("/music")
def get_music(category: str = Query(""), enabled_only: bool = Query(True)):
    return {
        "assets": asset_service.list_assets("music", category or None, enabled_only=enabled_only),
        "moods": list(asset_service.MUSIC_MOODS),
    }


@router.post("/upload", status_code=201)
async def upload_asset(
    file: UploadFile = File(...),
    kind: str = Form("gameplay"),
    category: str = Form("general"),
    name: str = Form(""),
):
    """Import a gameplay / B-roll / music file into the library."""
    cache = Env.DATA_DIR / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "asset.mp4").suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(cache)) as handle:
        temp_path = Path(handle.name)
        while chunk := await file.read(1024 * 1024 * 4):
            handle.write(chunk)
    try:
        asset = asset_service.import_asset(temp_path, kind=kind, category=category, name=name or Path(file.filename or "asset").stem)
    finally:
        temp_path.unlink(missing_ok=True)
        await file.close()
    events.publish("asset.imported", {"asset": asset})
    return {"asset": asset, "library": asset_service.library_summary()}


@router.post("/import-path", status_code=201)
def import_from_path(payload: AssetImportRequest):
    """Import a file that already exists on this machine (no upload needed)."""
    source = Path(payload.path).expanduser()
    if not source.exists():
        raise ClipForgeError(
            code=ErrorCode.NOT_FOUND,
            message=f"No file at {source}",
            hint="Check the path, or upload the file instead.",
            status_code=404,
        )
    asset = asset_service.import_asset(source, kind=payload.kind, category=payload.category, name=payload.name, copy_file=payload.copy_file)
    events.publish("asset.imported", {"asset": asset})
    return {"asset": asset, "library": asset_service.library_summary()}


@router.get("/folders")
def get_folders():
    asset_service.seed_folders()
    return {
        "root": str(asset_service.assets_root()),
        "structure": asset_service.copy_example_structure(),
        "gameplay": str(asset_service.assets_root() / "gameplay"),
        "broll": str(asset_service.assets_root() / "broll"),
        "music": str(asset_service.assets_root() / "music"),
        "note": "Drop files into these folders and press Scan - or use the upload button.",
    }


@router.post("/scan")
def scan():
    imported = asset_service.ensure_seed_assets()
    return {"imported": imported, "library": asset_service.library_summary()}


@router.patch("/{asset_id}")
def update_asset(asset_id: str, payload: AssetUpdate):
    updated = asset_service.set_asset_flags(
        asset_id,
        enabled=payload.enabled,
        favorite=payload.favorite,
        category=payload.category,
    )
    if updated is None:
        raise ClipForgeError(code=ErrorCode.NOT_FOUND, message="Asset not found.", status_code=404)
    return updated


@router.delete("/{asset_id}")
def delete_asset(asset_id: str, remove_file: bool = Query(True)):
    if not asset_service.delete_asset(asset_id, remove_file=remove_file):
        raise ClipForgeError(code=ErrorCode.NOT_FOUND, message="Asset not found.", status_code=404)
    return {"deleted": asset_id}


@router.get("/{asset_id}/file")
def stream_asset(asset_id: str, request: Request):
    from ...db import Asset, session_scope

    with session_scope() as session:
        asset = session.get(Asset, asset_id)
        if asset is None or not asset.path:
            raise ClipForgeError(code=ErrorCode.NOT_FOUND, message="Asset not found.", status_code=404)
        path = Path(asset.path)
    return stream_file(require_file(path), request, cache_seconds=3600)


__all__ = ["router"]
