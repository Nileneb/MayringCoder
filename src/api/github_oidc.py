"""GitHub Actions OIDC verification for secretless repo-watch auth (#1).

WHY: repo-watch (mayring-watch.yml → POST /repo-events) previously needed a shared
MCP_SERVICE_TOKEN secret in every watched repo. Distributing a workspace-spanning
admin token into PUBLIC repos (core/pi-agent are public) is a credential-exfiltration
risk. GitHub Actions can instead mint a short-lived OIDC JWT per run; we verify it
against GitHub's JWKS and gate on `repository_owner` — zero distributed secrets, any
owner-repo can join just by adding the caller workflow.

This grants ONLY repo-events authorization (used as a local dependency there), NOT
the global admin scope — an OIDC token must never unlock other privileged endpoints.
"""
from __future__ import annotations

import os
import threading

import jwt

_GH_ISS = "https://token.actions.githubusercontent.com"
_JWKS_URL = f"{_GH_ISS}/.well-known/jwks"

# Audience the workflow requests (`...&audience=<aud>`); MUST match strictly.
_AUD = os.getenv("GITHUB_OIDC_AUDIENCE", "mcp.linn.games")
# Only repos under this owner may authenticate via OIDC.
_OWNER = os.getenv("GITHUB_OIDC_OWNER", "Nileneb")
_OWNER_ID = os.getenv("GITHUB_OIDC_OWNER_ID", "")  # optional, defends against owner-rename

_jwk_client: "jwt.PyJWKClient | None" = None
_jwk_lock = threading.Lock()


def _client() -> "jwt.PyJWKClient":
    global _jwk_client
    if _jwk_client is None:
        with _jwk_lock:
            if _jwk_client is None:
                # PyJWKClient caches fetched signing keys internally.
                _jwk_client = jwt.PyJWKClient(_JWKS_URL)
    return _jwk_client


def verify_github_oidc(token: str) -> dict | None:
    """Return verified claims for a GitHub Actions OIDC token from the allowed owner,
    else None. Never raises — a non-OIDC token (e.g. the static service token) simply
    returns None WITHOUT a network call (the unverified-iss peek fails fast first).
    """
    if not token or token.count(".") != 2:
        return None
    try:
        unverified = jwt.decode(token, options={"verify_signature": False})
    except Exception:
        return None
    if unverified.get("iss") != _GH_ISS:
        return None  # not a GitHub OIDC token → no JWKS fetch

    try:
        signing_key = _client().get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=_AUD,
            issuer=_GH_ISS,
            options={"require": ["exp", "iat", "aud", "iss"]},
        )
    except Exception:
        return None

    # Owner gate: only our org's repos. repository_owner_id is rename-stable.
    if claims.get("repository_owner") != _OWNER:
        if not (_OWNER_ID and str(claims.get("repository_owner_id")) == _OWNER_ID):
            return None
    return claims
