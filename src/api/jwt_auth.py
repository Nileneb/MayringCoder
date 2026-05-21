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
from typing import NamedTuple


class Membership(NamedTuple):
    """Single workspace-membership entry from the JWT `memberships[]` claim.

    Issued by app.linn.games JwtIssuer. type is 'personal' or 'organization';
    role is 'owner' | 'editor' | 'viewer'. See docs/v2-workspaces-spec.md.
    """
    id: str
    type: str
    role: str


@dataclass(frozen=True)
class TokenInfo:
    workspace_id: str
    scopes: tuple[str, ...] = field(default_factory=tuple)
    # Provider-Claims für per-User BYO-LLM-Routing (app.linn.games JwtIssuer).
    # llm_provider: "platform" | "anthropic-byo" | "openai-compatible"
    llm_provider: str = "platform"
    llm_model: str | None = None
    llm_endpoint: str | None = None   # nur für openai-compatible gesetzt
    llm_requires_key: bool = False    # True → Worker muss Key via Callback holen
    sub: str | None = None            # User-ID, wird für Key-Cache genutzt
    iat: int | None = None            # Issued-at, Teil des Key-Cache-Keys
    # WHY(v2-workspaces): replaced single `org_id` with `memberships[]` so a
    # user can be in multiple orgs at once. Backward-compat: an old JWT without
    # memberships still works — `org_ids` then returns ().
    memberships: tuple[Membership, ...] = ()
    # WHY(backward-compat): legacy JWTs still ship a single `org_id` claim.
    # We preserve it on TokenInfo so existing callsites that read .org_id keep
    # working until they migrate to .org_ids.
    org_id: str | None = None
    # The app.linn.games *active* workspace (the `workspace_id` claim, incl.
    # the #195 switch). Distinct from `workspace_id` above, which stays the
    # per-user email-slug home bucket. active_workspace_kind is derived from
    # the matching memberships[] entry ('personal' | 'organization').
    active_workspace_id: str | None = None
    active_workspace_kind: str = "personal"

    @property
    def is_admin(self) -> bool:
        return "admin" in self.scopes

    @property
    def uses_custom_provider(self) -> bool:
        return self.llm_provider != "platform"

    @property
    def org_ids(self) -> tuple[str, ...]:
        """All organization-workspace ids the caller is a member of.

        Derived from the V2 `memberships[]` claim. For legacy JWTs that only
        carry a single `org_id`, falls back to that value as a 1-tuple so
        callers don't need a second branch. Returns () when neither is set.
        """
        ids = tuple(m.id for m in self.memberships if m.type == "organization")
        if ids:
            return ids
        return (self.org_id,) if self.org_id else ()


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

    # workspace_id (home bucket) wird seit 2026-05-09 aus dem email-Claim als
    # Slug abgeleitet ('bene@linn.games' → 'bene'). Das ist der per-user
    # Tenant-Key für SQLite-Isolation und bleibt absichtlich stabil (kein
    # Re-Key). Der `workspace_id`-Claim von app.linn.games (die *aktive*
    # Workspace-UUID, inkl. #195-Switch) ist eine ANDERE Achse — er wird
    # weiter unten als active_workspace_id konsumiert und steuert via
    # mcp_auth.resolve_write_visibility, ob ein Write in einen geteilten
    # Org-Bucket (visibility='org') geht.
    # WHY(v2-stufe4-contract): JWT-Vertrag V2 erfordert version-claim.
    # V1-tokens (kein version-Field) werden vorerst akzeptiert für
    # backward-compat. V2-tokens (version=2) sind preferred.
    contract_version = payload.get("version")
    if contract_version is not None and contract_version != 2:
        return None  # unbekannte JWT-version → reject

    sub_raw = payload.get("sub")
    if sub_raw is None or not str(sub_raw).strip():
        return None
    sub_str = str(sub_raw).strip()

    # email-Claim ist PFLICHT — kein user-N-Fallback. JwtIssuer in
    # app.linn.games schickt 'email' seit 2026-04-18. Token ohne email
    # = invalid (z.B. battlefield-JWT mit anderer audience landet hier
    # eh nicht, weil aud-check vorher).
    from src.identity.workspace_resolver import email_to_slug
    workspace_id = email_to_slug(payload.get("email") or "")
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

    llm_provider = str(payload.get("llm_provider") or "platform").strip() or "platform"
    llm_model = payload.get("llm_model")
    llm_endpoint = payload.get("llm_endpoint")
    raw_requires_key = payload.get("llm_requires_key", False)
    iat_raw = payload.get("iat")
    org_id_raw = payload.get("org_id")

    # V2 memberships claim — issued by app.linn.games JwtIssuer alongside
    # workspace_id. Missing/malformed → empty tuple (legacy-JWT path).
    memberships_raw = payload.get("memberships")
    memberships: tuple[Membership, ...] = ()
    if isinstance(memberships_raw, list):
        parsed: list[Membership] = []
        for m in memberships_raw:
            if not isinstance(m, dict):
                continue
            mid = m.get("id")
            if not mid:
                continue
            parsed.append(Membership(
                id=str(mid),
                type=str(m.get("type") or "personal"),
                role=str(m.get("role") or "viewer"),
            ))
        memberships = tuple(parsed)

    # The active app.linn.games workspace (its UUID), kept separate from the
    # email-slug home bucket above. kind from the matching membership entry —
    # decides whether a write lands in a shared org bucket (mcp_auth.
    # resolve_write_visibility). Absent claim → no active workspace.
    active_ws_raw = payload.get("workspace_id")
    active_workspace_id = str(active_ws_raw).strip() if active_ws_raw else None
    active_workspace_kind = "personal"
    if active_workspace_id:
        for m in memberships:
            if m.id == active_workspace_id:
                active_workspace_kind = m.type
                break

    return TokenInfo(
        workspace_id=workspace_id,
        scopes=scopes,
        active_workspace_id=active_workspace_id,
        active_workspace_kind=active_workspace_kind,
        llm_provider=llm_provider,
        llm_model=str(llm_model) if isinstance(llm_model, str) and llm_model else None,
        llm_endpoint=str(llm_endpoint) if isinstance(llm_endpoint, str) and llm_endpoint else None,
        llm_requires_key=bool(raw_requires_key),
        sub=sub_str,
        iat=int(iat_raw) if isinstance(iat_raw, (int, float)) else None,
        memberships=memberships,
        org_id=str(org_id_raw) if org_id_raw else None,
    )


def reset_public_key_cache() -> None:
    _public_key.cache_clear()
