"""Upload routes (TRD §28.2, PRD §6.1).

Three ways to upload — all streamed to storage, never buffered in memory:

* ``POST /uploads`` multipart form (``file`` field) — simple browser uploads
* ``POST /uploads?filename=x.mp4`` with a raw body (``Content-Type: application/octet-stream``)
* resumable: ``POST /uploads/init`` → ``PUT /uploads/{id}/chunks?offset=N`` … → ``POST /uploads/{id}/complete``
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.orm import Session

from clipforge.api.deps import current_user, get_db, get_settings_dep, get_storage_dep, upload_rate_limit
from clipforge.api.schemas import AssetOut, UploadCompleteRequest, UploadInitRequest
from clipforge.api.serializers import asset_out
from clipforge.core.config import Settings
from clipforge.core.errors import AppError, ErrorCode, not_found
from clipforge.db.models import Asset, User
from clipforge.services import idempotency
from clipforge.services.uploads import UploadService
from clipforge.storage import StorageProvider

router = APIRouter(prefix="/uploads", tags=["uploads"])


def _get_asset(db: Session, user: User, upload_id: str) -> Asset:
    asset = db.scalar(select(Asset).where(Asset.id == upload_id, Asset.user_id == user.id,
                                          Asset.deleted_at.is_(None)))
    if asset is None:
        raise not_found("Upload")
    return asset


async def _multipart_file_stream(request: Request) -> tuple[str | None, AsyncIterator[bytes]]:
    """Stream the first file part of a multipart body without spooling to disk."""
    from python_multipart.multipart import MultipartParser, parse_options_header

    content_type, params = parse_options_header(request.headers.get("content-type", ""))
    boundary = params.get(b"boundary")
    if not boundary:
        raise AppError(ErrorCode.VALIDATION_ERROR, "Missing multipart boundary.")

    import asyncio

    state: dict[str, Any] = {"filename": None, "in_file": False, "done_file": False, "headers": {},
                             "header_field": b"", "header_value": b""}
    header_ready = asyncio.Event()
    pending: list[bytes] = []

    def on_part_begin() -> None:
        state["headers"] = {}

    def on_header_field(data: bytes, start: int, end: int) -> None:
        state["header_field"] += data[start:end]

    def on_header_value(data: bytes, start: int, end: int) -> None:
        state["header_value"] += data[start:end]

    def on_header_end() -> None:
        state["headers"][state["header_field"].lower()] = state["header_value"]
        state["header_field"] = state["header_value"] = b""

    def on_headers_finished() -> None:
        disp, opts = parse_options_header(state["headers"].get(b"content-disposition", b""))
        if opts.get(b"filename") is not None and not state["done_file"] and state["filename"] is None:
            state["filename"] = opts[b"filename"].decode("utf-8", "replace")
            state["in_file"] = True
            header_ready.set()

    def on_part_data(data: bytes, start: int, end: int) -> None:
        if state["in_file"]:
            pending.append(bytes(data[start:end]))

    def on_part_end() -> None:
        if state["in_file"]:
            state["in_file"] = False
            state["done_file"] = True

    parser = MultipartParser(boundary, {
        "on_part_begin": on_part_begin, "on_header_field": on_header_field, "on_header_value": on_header_value,
        "on_header_end": on_header_end, "on_headers_finished": on_headers_finished, "on_part_data": on_part_data,
        "on_part_end": on_part_end})

    body = request.stream()

    async def pump_until_header() -> None:
        async for chunk in body:
            parser.write(chunk)
            if header_ready.is_set():
                return
        raise AppError(ErrorCode.VALIDATION_ERROR, "No file part found in the upload.")

    await pump_until_header()

    async def gen() -> AsyncIterator[bytes]:
        while pending:
            yield pending.pop(0)
        if state["done_file"]:
            return
        async for chunk in body:
            parser.write(chunk)
            while pending:
                yield pending.pop(0)
            if state["done_file"]:
                break

    return state["filename"], gen()


async def _finish(db: Session, svc: UploadService, asset: Asset, checksum: str | None,
                  user: User, key: str | None, req_hash: str) -> dict[str, Any]:
    try:
        final, duplicate = await run_in_threadpool(svc.finalize, asset, checksum)
    except AppError:
        db.commit()  # persist the REJECTED status + analytics before reporting the error
        raise
    if key:
        idempotency.store(db, user.id, "uploads.create", key, req_hash, "asset", final.id)
    db.commit()
    return asset_out(final, duplicate_of=final.id if duplicate else None)


@router.post("", response_model=AssetOut, status_code=201, dependencies=[Depends(upload_rate_limit)])
async def create_upload(request: Request, filename: str | None = Query(default=None, max_length=255),
                        project_id: str | None = Query(default=None, max_length=36),
                        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
                        user: User = Depends(current_user), db: Session = Depends(get_db),
                        settings: Settings = Depends(get_settings_dep),
                        storage: StorageProvider = Depends(get_storage_dep)) -> dict[str, Any]:
    key = idempotency.validate_key(idempotency_key)
    req_hash = idempotency.request_hash({"filename": filename, "length": request.headers.get("content-length")})
    existing = idempotency.lookup(db, user.id, "uploads.create", key, req_hash)
    if existing:
        return asset_out(_get_asset(db, user, existing))
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > settings.MAX_UPLOAD_BYTES + 1024 * 1024:
        raise AppError(ErrorCode.MEDIA_TOO_LARGE)
    svc = UploadService(db, settings, storage)
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("multipart/form-data"):
        part_name, stream = await _multipart_file_stream(request)
        asset = svc.create_asset(user, part_name or filename, project_id=project_id)
    else:
        if not filename:
            raise AppError(ErrorCode.VALIDATION_ERROR, "Provide ?filename= for raw uploads.")
        expected = int(length) if length and length.isdigit() else None
        asset = svc.create_asset(user, filename, expected_size=expected, project_id=project_id)
        stream = request.stream()
    db.commit()
    try:
        _, checksum = await svc.receive(asset, stream)
    except AppError:
        db.commit()
        raise
    except Exception:
        svc.reject(asset, AppError(ErrorCode.MEDIA_CORRUPTED))
        db.commit()
        raise
    return await _finish(db, svc, asset, checksum, user, key, req_hash)


@router.post("/init", response_model=AssetOut, status_code=201, dependencies=[Depends(upload_rate_limit)])
def init_upload(body: UploadInitRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
                user: User = Depends(current_user), db: Session = Depends(get_db),
                settings: Settings = Depends(get_settings_dep),
                storage: StorageProvider = Depends(get_storage_dep)) -> dict[str, Any]:
    key = idempotency.validate_key(idempotency_key)
    req_hash = idempotency.request_hash(body.model_dump())
    existing = idempotency.lookup(db, user.id, "uploads.init", key, req_hash)
    if existing:
        return asset_out(_get_asset(db, user, existing))
    asset = UploadService(db, settings, storage).create_asset(user, body.filename, expected_size=body.size_bytes,
                                                              project_id=body.project_id)
    if key:
        idempotency.store(db, user.id, "uploads.init", key, req_hash, "asset", asset.id)
    db.commit()
    return asset_out(asset)


@router.put("/{upload_id}/chunks", response_model=AssetOut)
async def upload_chunk(upload_id: str, request: Request, offset: int = Query(ge=0),
                       user: User = Depends(current_user), db: Session = Depends(get_db),
                       settings: Settings = Depends(get_settings_dep),
                       storage: StorageProvider = Depends(get_storage_dep)) -> dict[str, Any]:
    asset = _get_asset(db, user, upload_id)
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > settings.UPLOAD_CHUNK_MAX_BYTES:
        raise AppError(ErrorCode.VALIDATION_ERROR, "Chunk is too large.",
                       details={"max_chunk_bytes": settings.UPLOAD_CHUNK_MAX_BYTES})
    svc = UploadService(db, settings, storage)
    try:
        await svc.receive(asset, request.stream(), offset=offset, chunked=True)
    except AppError:
        db.commit()
        raise
    db.commit()
    return asset_out(asset)


@router.post("/{upload_id}/complete", response_model=AssetOut)
async def complete_upload(upload_id: str, body: UploadCompleteRequest | None = None,
                          user: User = Depends(current_user), db: Session = Depends(get_db),
                          settings: Settings = Depends(get_settings_dep),
                          storage: StorageProvider = Depends(get_storage_dep)) -> dict[str, Any]:
    asset = _get_asset(db, user, upload_id)
    svc = UploadService(db, settings, storage)
    expected = body.checksum_sha256.lower() if body and body.checksum_sha256 else None
    if expected and asset.status == "UPLOADING":
        actual = await run_in_threadpool(storage.checksum, asset.storage_key)
        if actual != expected:
            raise AppError(ErrorCode.MEDIA_CORRUPTED, "Checksum mismatch: the upload was corrupted in transit.")
    return await _finish(db, svc, asset, expected, user, None, "")


@router.get("/{upload_id}", response_model=AssetOut)
def get_upload(upload_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    return asset_out(_get_asset(db, user, upload_id))


@router.get("", response_model=list[AssetOut])
def list_uploads(limit: int = Query(default=20, ge=1, le=100), offset: int = Query(default=0, ge=0),
                 user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = db.scalars(select(Asset).where(Asset.user_id == user.id, Asset.deleted_at.is_(None))
                      .order_by(Asset.created_at.desc()).limit(limit).offset(offset)).all()
    return [asset_out(a) for a in rows]


@router.delete("/{upload_id}", status_code=204)
def delete_upload(upload_id: str, user: User = Depends(current_user), db: Session = Depends(get_db),
                  settings: Settings = Depends(get_settings_dep),
                  storage: StorageProvider = Depends(get_storage_dep)) -> None:
    UploadService(db, settings, storage).delete(_get_asset(db, user, upload_id))
    db.commit()
