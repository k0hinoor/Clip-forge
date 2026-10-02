"""Current-user routes: usage/quota and default settings (TRD §28)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from clipforge.api.deps import client_ip, current_user, get_db, get_settings_dep
from clipforge.api.schemas import UserSettingsPatch
from clipforge.api.serializers import user_out
from clipforge.core.config import Settings
from clipforge.core.errors import AppError, ErrorCode
from clipforge.core.presets import caption_presets
from clipforge.db.models import User
from clipforge.services.audit import audit
from clipforge.services.quotas import EntitlementService, UsageService

router = APIRouter(prefix="/me", tags=["me"])


@router.get("")
def me(user: User = Depends(current_user)) -> dict[str, Any]:
    return {"user": user_out(user), "settings": user.settings or {}}


@router.get("/usage")
def usage(user: User = Depends(current_user), db: Session = Depends(get_db),
          settings: Settings = Depends(get_settings_dep)) -> dict[str, Any]:
    ent = EntitlementService(db, settings).entitlement(user)
    return {"plan": ent.to_dict(), "usage": UsageService(db).summary(user.id)}


@router.get("/settings")
def get_user_settings(user: User = Depends(current_user)) -> dict[str, Any]:
    return user.settings or {}


@router.patch("/settings")
def patch_user_settings(body: UserSettingsPatch, request: Request, user: User = Depends(current_user),
                        db: Session = Depends(get_db), settings: Settings = Depends(get_settings_dep)) -> dict[str, Any]:
    patch = body.model_dump(exclude_unset=True)
    if "default_caption_preset" in patch and patch["default_caption_preset"] not in caption_presets(settings):
        raise AppError(ErrorCode.VALIDATION_ERROR, "Unknown caption preset.")
    if "display_name" in patch:
        user.display_name = patch.pop("display_name") or None
    user.settings = {**(user.settings or {}), **patch}
    audit(db, "user.settings_updated", actor_user_id=user.id, target_type="user", target_id=user.id,
          ip_address=client_ip(request), metadata={"fields": sorted(patch)})
    db.commit()
    return user.settings
