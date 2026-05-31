"""Per-workspace role→permission matrix (tenancy Phase B). Read+edit, gated by
the manage_members permission (super-admin only on the public workspace)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from mayring_core.identity import authz
from mayring_core.memory.store import get_role_permissions, set_role_permission
from src.api.auth import get_token_info, get_workspace
from src.api.authz_helpers import caller_can
from src.api.dependencies import get_conn as _get_conn
from src.api.jwt_auth import TokenInfo

router = APIRouter()


class RolePermissionUpdate(BaseModel):
    role: str
    permission: str
    allowed: bool


def _effective_matrix(overrides: dict[tuple[str, str], bool]) -> dict[str, dict[str, bool]]:
    out: dict[str, dict[str, bool]] = {}
    for role in authz.ROLES:
        out[role] = {perm: authz.can(role, perm, overrides=overrides) for perm in authz.PERMISSIONS}
    return out


@router.get("/stats/workspaces/{ws}/role-permissions")
async def get_role_permissions_endpoint(ws: str, workspace_id: str = Depends(get_workspace), info: TokenInfo = Depends(get_token_info)) -> dict:
    overrides = get_role_permissions(_get_conn(), ws)
    if not caller_can(info, ws, "manage_members", overrides=overrides):
        raise HTTPException(status_code=403, detail="manage_members required")
    return {"workspace_id": ws, "matrix": _effective_matrix(overrides), "locked": {"admin": list(authz.ADMIN_LOCKED_PERMISSIONS)}}


@router.put("/stats/workspaces/{ws}/role-permissions")
async def put_role_permission_endpoint(ws: str, body: RolePermissionUpdate, workspace_id: str = Depends(get_workspace), info: TokenInfo = Depends(get_token_info)) -> dict:
    overrides = get_role_permissions(_get_conn(), ws)
    if not caller_can(info, ws, "manage_members", overrides=overrides):
        raise HTTPException(status_code=403, detail="manage_members required")
    if body.role not in authz.ROLES:
        raise HTTPException(status_code=422, detail=f"role must be one of {authz.ROLES}")
    if body.permission not in authz.PERMISSIONS:
        raise HTTPException(status_code=422, detail=f"permission must be one of {authz.PERMISSIONS}")
    if body.role == "admin" and body.permission in authz.ADMIN_LOCKED_PERMISSIONS and not body.allowed:
        raise HTTPException(status_code=422, detail=f"admin.{body.permission} cannot be disabled (anti-lockout)")
    set_role_permission(_get_conn(), ws, body.role, body.permission, body.allowed)
    return {"workspace_id": ws, "role": body.role, "permission": body.permission, "allowed": body.allowed}
