"""POST /stats/workspaces/claim — adopt an unclaimed:<device> bucket into the caller's
workspace. Only `unclaimed:`-prefixed buckets are claimable (a device that wrote without
a valid token); claiming any real/infra workspace is refused. Repoints rows + Chroma and
registers an alias so the device's old writes resolve to the user's workspace."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.api.auth import get_workspace
from src.api.dependencies import get_conn as _get_conn
from src.api.workspace_repoint import repoint_workspace

router = APIRouter()


class ClaimRequest(BaseModel):
    workspace_id: str   # the unclaimed:<device> bucket to adopt


@router.post("/stats/workspaces/claim")
def claim_workspace(
    req: ClaimRequest,
    target: str = Depends(get_workspace),
) -> dict:
    src = (req.workspace_id or "").strip()
    if not src.startswith("unclaimed:"):
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "only unclaimed:<device> buckets can be claimed")
    if src == target:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "cannot claim into itself")
    conn = _get_conn()
    try:
        from mayring_core.memory.store import get_chroma_collection
        chroma = get_chroma_collection("memory_chunks")
    except Exception:  # noqa: BLE001 — ohne Chroma trotzdem die SQLite-Rows umhängen
        chroma = None
    counts = repoint_workspace(conn, src, target, chroma=chroma)
    return {"ok": True, "claimed": src, "into": target, "moved": counts}
