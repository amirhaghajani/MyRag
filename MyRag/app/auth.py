"""HTTP Basic auth driven by AUTH_USERNAME / AUTH_PASSWORD."""

from __future__ import annotations

import base64
import binascii
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import get_settings

# Checked without credentials so the UI can render its login screen.
OPEN_PATHS = {"/api/health", "/api/auth/status"}


def parse_basic_auth(header: str | None) -> tuple[str, str] | None:
    """Return (username, password) from an Authorization header, or None."""
    if not header:
        return None
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "basic" or not value:
        return None
    try:
        decoded = base64.b64decode(value, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return None
    username, sep, password = decoded.partition(":")
    if not sep:
        return None
    return username, password


def is_authenticated(header: str | None) -> bool:
    settings = get_settings()
    if not settings.auth_enabled:
        return True
    creds = parse_basic_auth(header)
    if creds is None:
        return False
    username, password = creds
    return secrets.compare_digest(username, settings.auth_username) and secrets.compare_digest(
        password, settings.auth_password
    )


class BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        if not settings.auth_enabled:
            return await call_next(request)
        if request.method == "OPTIONS" or request.url.path in OPEN_PATHS:
            return await call_next(request)
        if is_authenticated(request.headers.get("authorization")):
            return await call_next(request)
        return JSONResponse(
            status_code=401,
            content={"detail": "Authentication required"},
            headers={"WWW-Authenticate": 'Basic realm="MyRag"'},
        )