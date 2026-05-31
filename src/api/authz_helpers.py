"""Bridge the JWT TokenInfo to the core authz model (tenancy Phase B)."""
from __future__ import annotations

import os

import mayring_core.identity.workspace_resolver as _wr
from mayring_core.identity import authz
from src.api.jwt_auth import TokenInfo


def _get_db_conn():
    from src.api.dependencies import get_conn
    return get_conn()


def _public_ws_id() -> str:
    return os.getenv("MAYRING_PUBLIC_WORKSPACE_ID", "")


def caller_role(info: TokenInfo, workspace_id: str) -> str:
    """Normalized role of the caller IN workspace_id.

    Resolution order:
      1. Workspace OWNER is always admin (even without a memberships entry).
      2. JWT memberships entry for workspace_id → normalize_role(m.role).
      3. No match → 'user' (least privilege).
    """
    if info.sub:
        try:
            conn = _get_db_conn()
            if _wr.workspace_owner(conn, workspace_id) == info.sub:
                return "admin"
        except Exception:
            pass
    for m in info.memberships:
        if m.id == workspace_id:
            return authz.normalize_role(m.role)
    return "user"


def caller_can(info: TokenInfo, workspace_id: str, permission: str, *,
               overrides: dict | None = None) -> bool:
    return authz.can(
        caller_role(info, workspace_id),
        permission,
        is_super_admin=bool(info.is_admin),
        is_public_ws=bool(_public_ws_id()) and workspace_id == _public_ws_id(),
        overrides=overrides,
    )
