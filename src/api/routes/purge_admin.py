"""Admin route: purge ALL data for one workspace (smoke self-clean + ops tool).

WHY(#253): smoke runs leak ephemeral `<prefix>-<ts>` workspaces; the HTTP-only
smoke harness self-cleans them via this endpoint. PROTECTED_WORKSPACES refuses
the real ones (422). Admin/service-token gated.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.admin_purge_workspace import purge_smoke_projects, purge_workspace
from src.api.auth import get_token_info
from src.api.dependencies import get_chroma as _chroma
from src.api.dependencies import get_conn as _conn
from src.api.jwt_auth import TokenInfo

router = APIRouter()
logger = logging.getLogger(__name__)


class PurgeRequest(BaseModel):
    workspace_id: str


def _is_admin(info: TokenInfo) -> bool:
    return bool(getattr(info, "is_admin", False)) or "*" in (info.scopes or ())


@router.post("/stats/admin/purge-workspace")
async def purge_workspace_route(
    body: PurgeRequest, info: TokenInfo = Depends(get_token_info)
) -> dict:
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")

    def _run() -> dict:
        return purge_workspace(_conn(), _chroma(), body.workspace_id)

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _run)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    logger.info("purged workspace %s: %s", body.workspace_id, result["rows"])
    return result


@router.post("/stats/admin/purge-smoke-projects")
async def purge_smoke_projects_route(info: TokenInfo = Depends(get_token_info)) -> dict:
    """Delete smoke-suite throwaway PROJECTS (+ their chunk_project_links and
    leftover smoke project-groups) across ALL workspaces.

    WHY(2026-06-10): the C3 check creates one `smoke/repo-c3-<ts>` project per
    run via /projects/route and nothing ever deleted them — ~90 junk rows piled
    up in the dashboard Projekte list. Workspace-purge doesn't cover them (they
    live in 'system', which is purge-protected). Pattern-gated: only
    source_ref containing 'smoke/repo-' resp. group names 'smoke-*' — never a
    real project (mirrors the NOT_SMOKE guard in routes/projects.py).
    """
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")

    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: purge_smoke_projects(_conn()))
    logger.info("purged smoke projects: %s", result)
    return result
