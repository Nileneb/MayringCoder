"""FastAPI auth dependency — workspace_id resolution via Sanctum tokens."""
from __future__ import annotations

import os

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.api.sanctum_auth import validate_sanctum_token_full

_bearer = HTTPBearer(auto_error=False)
_MCP_AUTH_TOKEN = os.getenv("MAYRING_MCP_AUTH_TOKEN", "")


async def get_workspace(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """Validate Bearer token and return workspace_id."""
    if not creds:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token")

    if _MCP_AUTH_TOKEN and creds.credentials == _MCP_AUTH_TOKEN:
        return "system"

    info = validate_sanctum_token_full(creds.credentials)
    if not info:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    if not info.mayring_active:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="MayringCoder Memory requires an active subscription (€5/month). "
                   "Subscribe at https://app.linn.games/einstellungen/mayring-abo",
        )

    return info.workspace_id
