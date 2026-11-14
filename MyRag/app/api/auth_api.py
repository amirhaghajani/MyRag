"""Auth status endpoint used by the login screen."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..auth import is_authenticated
from ..config import get_settings

router = APIRouter(tags=["auth"])


@router.get("/api/auth/status")
def auth_status(request: Request) -> dict:
    settings = get_settings()
    return {
        "enabled": settings.auth_enabled,
        "authenticated": is_authenticated(request.headers.get("authorization")),
    }