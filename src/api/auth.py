"""FastAPI auth dependency — RS256 JWT or MCP_SERVICE_TOKEN."""
from __future__ import annotations

import hmac
import os

from fastapi import Depends, HTTPException, Header, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.api.jwt_auth import Membership, TokenInfo, validate_jwt_token

_bearer = HTTPBearer(auto_error=False)

# Service-to-service token: loaded once at startup.
# On the server this is set in .env.production; on laptops it's empty,
# so users always need a proper RS256 JWT.
_SERVICE_TOKEN = os.getenv("MCP_SERVICE_TOKEN", "")


def _is_privileged(info: TokenInfo) -> bool:
    """Service token (scope '*') or admin JWT — the only callers allowed to act-as."""
    return info.is_admin or "*" in info.scopes


def _coerce_header(val: object) -> str | None:
    """Return the value only if it's a plain str; swallow FastAPI sentinel objects.

    WHY(test-compat): when get_token_info is called directly (not via FastAPI
    DI), Header(default=None) sentinel objects are passed for unset params.
    """
    return val if isinstance(val, str) else None


def _maybe_act_as(
    info: TokenInfo,
    sub: object,
    orgs: object,
    workspace: object,
    role: object = None,
) -> TokenInfo:
    """Admin/service-only test harness: simulate another caller so the prod
    smoke can prove cross-tenant isolation.

    WHY(v2-org-acceptance): the 9 acceptance smokes need a SECOND identity
    (User B / org-member / non-member); the service token only carries one.
    STRICTLY gated: only a privileged token, only when MAYRING_ALLOW_ACT_AS is
    set. The synthetic identity DROPS privilege (scope=('mcp:memory',)) — else
    an admin/service caller bypasses _scope_filter and every isolation test
    trivially passes. Non-privileged callers' act-as headers are ignored (never
    escalate).
    """
    sub = _coerce_header(sub)
    orgs = _coerce_header(orgs)
    workspace = _coerce_header(workspace)
    role = _coerce_header(role)
    if not (sub or orgs or workspace or role):
        return info
    if os.getenv("MAYRING_ALLOW_ACT_AS", "").lower() not in ("1", "true", "yes"):
        return info
    if not _is_privileged(info):
        return info
    org_tuple = tuple(o.strip() for o in (orgs or "").split(",") if o.strip())
    memberships = tuple(
        Membership(id=o, type="organization", role="viewer") for o in org_tuple
    )
    ws = workspace or info.workspace_id
    # WHY(tenancy phase B): let the harness simulate a member with a ROLE for its
    # active workspace, so role-gated paths (write/share/manage_*) are testable.
    # Still on the privilege-dropped synthetic identity — no escalation (the gate
    # above already requires a privileged caller + MAYRING_ALLOW_ACT_AS).
    if role:
        memberships = (Membership(id=ws, type="organization", role=role),) + memberships
    return TokenInfo(
        workspace_id=ws,
        scopes=("mcp:memory",),
        sub=sub or info.sub,
        memberships=memberships,
        active_workspace_id=ws,
    )


async def get_token_info(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    x_act_as_sub: str | None = Header(default=None, alias="X-Act-As-Sub"),
    x_act_as_orgs: str | None = Header(default=None, alias="X-Act-As-Orgs"),
    x_act_as_workspace: str | None = Header(default=None, alias="X-Act-As-Workspace"),
    x_act_as_role: str | None = Header(default=None, alias="X-Act-As-Role"),
) -> TokenInfo:
    """Validate Bearer token — accepts RS256 JWT (users) or MCP_SERVICE_TOKEN.

    Service-Token: scope='*', workspace_id='system'. X-Workspace-Id / body
    override per get_workspace(). X-Act-As-* lets a privileged caller simulate
    another identity (admin-only test harness, see _maybe_act_as).
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
        info: TokenInfo = TokenInfo(workspace_id="system", scopes=("*",))
    else:
        validated = validate_jwt_token(token)
        if not validated:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
            )
        info = validated
    return _maybe_act_as(info, x_act_as_sub, x_act_as_orgs, x_act_as_workspace, x_act_as_role)


async def get_workspace(
    info: TokenInfo = Depends(get_token_info),
    x_workspace_id: str | None = Header(default=None, alias="X-Workspace-Id"),
) -> str:
    """Workspace-Resolution mit Multi-Tenant-Guarantees.

    Delegiert an `resolve_workspace_from_token` — die EINZIGE quelle
    der wahrheit für workspace_id (V2 Stufe 1.1 SoT, audit Section 6
    Regel 4). Vor V2 gab es 4+ resolution-pfade die divergieren konnten;
    jetzt sind alle FastAPI- und MCP-Pfade auf diesen einen call
    konsolidiert.
    """
    from mayring_core.identity.workspace_resolver import resolve_workspace_from_token
    # conn → Alias-Auflösung (workspace-repoint): alte Tokens (019d6933) lösen
    # transparent auf die kanonische 019e14d6 auf, kein Reissue nötig.
    try:
        from src.api.dependencies import get_conn
        conn = get_conn()
    except Exception:  # noqa: BLE001 — ohne DB kein Alias, aber Auth darf nicht brechen
        conn = None
    return resolve_workspace_from_token(info, override_header=x_workspace_id, conn=conn)
