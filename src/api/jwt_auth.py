"""RS256 JWT validation — only auth method for MayringCoder.

Public key is held locally; app.linn.games holds the matching private key and
issues tokens. Claims validated: exp, iss, aud, workspace_id (required).

Required .env:
    JWT_PUBLIC_KEY_PATH   Path to RS256 public key PEM file
    JWT_ISSUER            Expected "iss" claim (e.g. "https://app.linn.games")
    JWT_AUDIENCE          Expected "aud" claim (e.g. "mayringcoder")

A token with `scope: ["admin"]` gets cross-workspace (system) access. A token
without a `workspace_id` claim is rejected outright — no silent fallback.
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
    if not isinstance(workspace_id, str) or not workspace_id.strip():
        return None

    raw_scopes = payload.get("scope", [])
    if isinstance(raw_scopes, str):
        scopes = tuple(s for s in raw_scopes.split() if s)
    elif isinstance(raw_scopes, list):
        scopes = tuple(str(s) for s in raw_scopes if s)
    else:
        scopes = ()

    return TokenInfo(workspace_id=workspace_id.strip(), scopes=scopes)


def reset_public_key_cache() -> None:
    _public_key.cache_clear()
