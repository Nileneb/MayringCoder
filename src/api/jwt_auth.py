"""RS256 JWT validation — only auth method for MayringCoder.

Public key is held locally; app.linn.games holds the matching private key and
issues tokens via app/Services/JwtIssuer.php (commit 8dfa1c0). Required
claims: exp, iss, aud, workspace_id, scope ∋ "mcp:memory".

Required .env:
    JWT_PUBLIC_KEY_PATH   Path to RS256 public key PEM file
    JWT_ISSUER            Expected "iss" claim (e.g. "https://app.linn.games")
    JWT_AUDIENCE          Expected "aud" claim (e.g. "mayringcoder")

A token with scope containing "admin" gets cross-workspace (system) access.
Every token must carry scope "mcp:memory" — a JWT issued for another service
(e.g. paper-search) is rejected here even if iss/aud match.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class TokenInfo:
    workspace_id: str
    scopes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_admin(self) -> bool:
        return "admin" in self.scopes


@lru_cache(maxsize=1)
def _public_key() -> str | None:
    path = os.getenv("JWT_PUBLIC_KEY_PATH", "").strip()
    if not path:
        return None
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def _issuer() -> str:
    return os.getenv("JWT_ISSUER", "https://app.linn.games")


def _audience() -> str:
    return os.getenv("JWT_AUDIENCE", "mayringcoder")


def validate_jwt_token(token: str) -> TokenInfo | None:
    """Return TokenInfo for a valid RS256 JWT, or None.

    Rejects: missing/expired/invalid-sig/wrong-iss/wrong-aud/no-workspace_id.
    """
    if not token:
        return None
    public_key = _public_key()
    if not public_key:
        return None

    try:
        import jwt  # PyJWT
    except ImportError:
        return None

    try:
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            issuer=_issuer(),
            audience=_audience(),
            options={"require": ["exp", "iss", "aud"]},
        )
    except jwt.InvalidTokenError:
        return None

    workspace_id = payload.get("workspace_id")
    if workspace_id is None:
        return None
    workspace_id = str(workspace_id).strip()
    if not workspace_id:
        return None

    raw_scopes = payload.get("scope", [])
    if isinstance(raw_scopes, str):
        scopes = tuple(s for s in raw_scopes.split() if s)
    elif isinstance(raw_scopes, list):
        scopes = tuple(str(s) for s in raw_scopes if s)
    else:
        scopes = ()

    # Every MayringCoder JWT must carry the mcp:memory scope, so tokens minted
    # for a different service (e.g. paper-search) don't silently work here.
    if "mcp:memory" not in scopes:
        return None

    return TokenInfo(workspace_id=workspace_id, scopes=scopes)


def reset_public_key_cache() -> None:
    _public_key.cache_clear()
