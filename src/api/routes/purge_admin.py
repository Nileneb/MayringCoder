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


class PurgeSourceTypeRequest(BaseModel):
    source_type: str


class DeactivateSourcePrefixRequest(BaseModel):
    prefix: str


# WHY(corpus-noise 2026-06-21): source_types that are pure operational noise — never
# code or recall — and may be bulk-deactivated. log_event = internal app logger lines
# (e.g. "vector stage: chroma returned 4 hits") that were ingested as searchable chunks
# and polluted code retrieval (2193 of them). Safelist-gated so a typo can never nuke
# repo_file / note / conversation_summary.
_PURGEABLE_NOISE_TYPES = {"log_event"}

# WHY(reference-doc-noise 2026-06-21): bulk external reference dumps (framework docs)
# stored as plain chunks DROWN the user's own code in retrieval — 3495 unity-docs:*
# chunks (~33% of the corpus) out-scored every repo's code on any 3D/WebGL query.
# Deactivation is reversible (is_active=0). Prefix-safelisted so only known reference
# corpora can be bulk-deactivated. Long-term these belong in a scoped reference layer
# (see docs/superpowers spec) — this endpoint is the interim + the management hook.
_DEACTIVATABLE_REFERENCE_PREFIXES = {"unity-docs:"}


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


@router.post("/stats/admin/purge-source-type")
async def purge_source_type_route(
    body: PurgeSourceTypeRequest, info: TokenInfo = Depends(get_token_info)
) -> dict:
    """Deactivate (is_active=0) every active chunk of a pure-noise source_type
    across ALL workspaces. Safelist-gated to _PURGEABLE_NOISE_TYPES (422 otherwise)
    so it can never touch repo_file/note/etc. Retrieval filters is_active=1, so the
    chunks vanish from results immediately; run /stats/admin/reconcile-chroma after
    to drop the now-dead vectors from the index."""
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    if body.source_type not in _PURGEABLE_NOISE_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"source_type {body.source_type!r} not in purgeable noise safelist "
                   f"{sorted(_PURGEABLE_NOISE_TYPES)}")

    def _run() -> dict:
        from mayring_core.memory.store import deactivate_chunks_by_source
        conn = _conn()
        rows = conn.execute(
            "SELECT DISTINCT ch.source_id FROM chunks ch "
            "JOIN sources s ON ch.source_id = s.source_id "
            "WHERE ch.is_active = 1 AND s.source_type = ?",
            (body.source_type,),
        ).fetchall()
        deactivated = sum(deactivate_chunks_by_source(conn, r[0]) for r in rows)
        return {"source_type": body.source_type,
                "sources": len(rows), "deactivated": deactivated}

    result = await asyncio.get_event_loop().run_in_executor(None, _run)
    logger.info("purged noise source_type %s: %s", body.source_type, result)
    return result


@router.post("/stats/admin/deactivate-source-prefix")
async def deactivate_source_prefix_route(
    body: DeactivateSourcePrefixRequest, info: TokenInfo = Depends(get_token_info)
) -> dict:
    """Deactivate (is_active=0) every active chunk whose source_id starts with a
    known reference-doc prefix (e.g. 'unity-docs:'). REVERSIBLE — flips is_active,
    does not delete. Safelist-gated to _DEACTIVATABLE_REFERENCE_PREFIXES (422 else)
    so it can never touch repo:/conversation:/note authored content broadly.

    WHY: bulk framework-doc dumps stored as plain chunks drown the user's own code
    in retrieval. Until a scoped reference layer exists, this lifts them out of the
    default candidate pool. Run /admin/reconcile-chroma?dry_run=false after to drop
    the dead vectors. Restore = re-ingest, or flip is_active back."""
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    if body.prefix not in _DEACTIVATABLE_REFERENCE_PREFIXES:
        raise HTTPException(
            status_code=422,
            detail=f"prefix {body.prefix!r} not in reference safelist "
                   f"{sorted(_DEACTIVATABLE_REFERENCE_PREFIXES)}")

    def _run() -> dict:
        from mayring_core.memory.store import deactivate_chunks_by_source
        conn = _conn()
        rows = conn.execute(
            "SELECT DISTINCT source_id FROM chunks "
            "WHERE is_active = 1 AND source_id LIKE ?",
            (body.prefix + "%",),
        ).fetchall()
        deactivated = sum(deactivate_chunks_by_source(conn, r[0]) for r in rows)
        return {"prefix": body.prefix, "sources": len(rows), "deactivated": deactivated}

    result = await asyncio.get_event_loop().run_in_executor(None, _run)
    logger.info("deactivated reference prefix %s: %s", body.prefix, result)
    return result
