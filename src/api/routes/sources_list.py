"""GET /stats/sources — list a workspace's sources with visibility, for the
Share-UI (tenancy Phase C). Read-only; workspace-scoped (under /stats/ → already
in the nginx allowlist). NOT a memory search — a flat catalog for the dashboard."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.auth import get_token_info, get_workspace
from src.api.dependencies import get_conn as _get_conn
from src.api.jwt_auth import TokenInfo

router = APIRouter()


@router.get("/stats/sources")
async def list_sources_endpoint(
    limit: int = 200,
    workspace_id: str = Depends(get_workspace),
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    rows = _get_conn().execute(
        "SELECT source_id, source_type, visibility, org_id, scope_key, captured_at "
        "FROM sources WHERE workspace_id = ? ORDER BY captured_at DESC LIMIT ?",
        (workspace_id, int(limit)),
    ).fetchall()
    out = []
    for r in rows:
        g = (lambda k, i: r[k] if hasattr(r, "keys") else r[i])
        out.append({
            "source_id": g("source_id", 0), "source_type": g("source_type", 1),
            "visibility": g("visibility", 2) or "private", "org_id": g("org_id", 3),
            "scope_key": g("scope_key", 4), "captured_at": g("captured_at", 5),
        })
    return {"workspace_id": workspace_id, "sources": out}
