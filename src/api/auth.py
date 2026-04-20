"""FastAPI auth dependency — RS256 JWT only."""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.api.jwt_auth import TokenInfo, validate_jwt_token

_bearer = HTTPBearer(auto_error=False)


async def get_token_info(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> TokenInfo:
    """Validate Bearer JWT and return TokenInfo (workspace_id + scopes)."""
    if not creds:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
        )
    info = validate_jwt_token(creds.credentials)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return info


async def get_workspace(info: TokenInfo = Depends(get_token_info)) -> str:
    """Return workspace_id from validated JWT."""
    return info.workspace_id
