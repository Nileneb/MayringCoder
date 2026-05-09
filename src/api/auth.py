"""FastAPI auth dependency — RS256 JWT or MCP_SERVICE_TOKEN."""
from __future__ import annotations

import hmac
import os

from fastapi import Depends, HTTPException, Header, status
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

    Service-Token: scope='*', workspace_id='system' als Maintenance-
    Default. Wer in einen anderen Workspace schreiben will, muss
    explizit den X-Workspace-Id-Header oder einen body.workspace_id
    mitgeben — siehe get_workspace().
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


async def get_workspace(
    info: TokenInfo = Depends(get_token_info),
    x_workspace_id: str | None = Header(default=None, alias="X-Workspace-Id"),
) -> str:
    """Workspace-Resolution mit Multi-Tenant-Guarantees:

    - User-JWT: workspace_id ist deterministisch aus email-slug; ein
      X-Workspace-Id-Header wird IGNORIERT (User darf nicht in fremde
      Workspaces schreiben).
    - Service-Token (scope='*'): X-Workspace-Id-Header ist erlaubt
      und überschreibt den Default 'system'. So können post-deploy-
      ingest, smoke etc. explizit in den Tenant-Workspace eines
      bestimmten Users schreiben (X-Workspace-Id: bene), ohne dass
      der einzige globale Service-Token alle anderen Workspaces
      vergiftet.
    """
    if "*" in info.scopes and x_workspace_id:
        candidate = x_workspace_id.strip()
        if candidate:
            return candidate
    return info.workspace_id
