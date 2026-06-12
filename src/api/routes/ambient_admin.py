"""Admin route to (re)generate ambient memory snapshots.

WHY(2026-05-30 audit): `generate_ambient_snapshot()` was only ever called from
the CLI — no cron/scheduler in prod — so `build_context()` always returned ""
and the entire ambient memory layer was a no-op in production. This endpoint runs
the same code path so a self-hosted cron can keep snapshots fresh.
Admin/service-token gated.
"""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, HTTPException

from src.api.auth import get_token_info
from src.api.dependencies import get_conn as _conn
from src.api.jwt_auth import TokenInfo

router = APIRouter()
logger = logging.getLogger(__name__)


def _is_admin(info: TokenInfo) -> bool:
    return bool(getattr(info, "is_admin", False)) or "*" in (info.scopes or ())


@router.post("/stats/ambient-refresh")
async def trigger_ambient_refresh(
    info: TokenInfo = Depends(get_token_info),
    repo_slug: str = "",
    model: str | None = None,
    workspace_id: str | None = None,
) -> dict:
    """Generate an ambient snapshot for `repo_slug` (empty = workspace-global).

    Admin-only (service token or scope='admin'). Synchronous (one LLM call),
    run off the event loop. Returns the snapshot length.
    """
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")

    if model is None:
        from mayring_core.model_router import ModelRouter
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        model = ModelRouter(ollama_url).resolve("text") or "qwen2.5-coder:7b"
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ws = workspace_id or info.workspace_id

    import asyncio

    from mayring_core.memory.ambient import generate_ambient_snapshot

    def _run() -> str | None:
        return generate_ambient_snapshot(_conn(), ollama_url, model, repo_slug, ws)

    try:
        snapshot = await asyncio.to_thread(_run)
    except Exception as exc:  # surface, never hide
        logger.exception("ambient-refresh failed for repo=%s", repo_slug)
        raise HTTPException(status_code=500, detail=f"ambient-refresh failed: {exc}") from exc

    return {
        "repo_slug": repo_slug or "(global)",
        "workspace_id": ws,
        "chars": len(snapshot or ""),
        "generated": bool(snapshot),
    }
