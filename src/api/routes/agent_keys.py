"""Per-agent API keys — mint / list / revoke. Each key is bound to one agent
(label), workspace-scoped, shown in plaintext exactly once at mint. The /a2a auth
gate (server.py /auth/verify) accepts these via the X-API-Key header.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api import agent_keys as store
from src.api.auth import get_workspace

router = APIRouter()


@router.get("/stats/agent-keys")
async def list_agent_keys(workspace_id: str = Depends(get_workspace)) -> dict:
    return {"keys": store.list_keys(workspace_id)}


class MintRequest(BaseModel):
    label: str = "agent"


@router.post("/stats/agent-keys")
async def mint_agent_key(
    req: MintRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Mint a key for one agent. Returns the plaintext ONCE — it's not recoverable."""
    plaintext, rec = store.mint(workspace_id, req.label)
    return {"ok": True, "api_key": plaintext, "key": rec}


@router.delete("/stats/agent-keys/{key_id}")
async def revoke_agent_key(
    key_id: str,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    if not store.revoke(workspace_id, key_id):
        raise HTTPException(status_code=404, detail="key not found in workspace")
    return {"ok": True, "revoked": key_id}
