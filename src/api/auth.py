"""FastAPI auth dependency — RS256 JWT or MCP_SERVICE_TOKEN."""
from __future__ import annotations

import hmac
import os

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.api.jwt_auth import TokenInfo, validate_jwt_token

_bearer = HTTPBearer(auto_error=False)

# Service-to-service token: loaded once at startup.
# On the server this is set in .env.production; on laptops it's empty,
# so users always need a proper RS256 JWT.
_SERVICE_TOKEN = os.getenv("MCP_SERVICE_TOKEN", "")


async def get_token_info(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> TokenInfo:
    """Validate Bearer token — accepts RS256 JWT (users) or MCP_SERVICE_TOKEN (server daemons).

    Service token → workspace_id 'system', scope '*'.
    """
    if not creds:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
        )
    token = creds.credentials
    if _SERVICE_TOKEN and hmac.compare_digest(
        token.encode() if isinstance(token, str) else token,
        _SERVICE_TOKEN.encode() if isinstance(_SERVICE_TOKEN, str) else _SERVICE_TOKEN,
    ):
        return TokenInfo(workspace_id="system", scopes=("*",))
    info = validate_jwt_token(token)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return info


async def get_workspace(info: TokenInfo = Depends(get_token_info)) -> str:
    """Return workspace_id from validated JWT."""
    return info.workspace_id
