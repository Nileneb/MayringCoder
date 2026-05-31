"""Admin route: backfill visibility/org_id/user_id into existing Chroma metadata.

WHY(tenancy phase A 2026-05-31): the new Chroma `where` clause (build_chroma_where)
filters vector candidates by visibility/user_id/org_id metadata. Chunks ingested
BEFORE phase A lack those fields, so the vector stage drops them (memory_search_vector
/ visibility_isolation smokes red) until this one-off, idempotent backfill stamps
them from SQLite `sources`. Admin/service-token gated. Safe to re-run.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException

from src.api.admin_backfill_chroma_visibility import backfill_visibility_metadata
from src.api.auth import get_token_info
from src.api.dependencies import get_chroma as _chroma
from src.api.dependencies import get_conn as _conn
from src.api.jwt_auth import TokenInfo

router = APIRouter()
logger = logging.getLogger(__name__)


def _is_admin(info: TokenInfo) -> bool:
    return bool(getattr(info, "is_admin", False)) or "*" in (info.scopes or ())


@router.post("/stats/admin/backfill-chroma-visibility")
async def backfill_chroma_visibility(info: TokenInfo = Depends(get_token_info)) -> dict:
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    collection = _chroma()
    if collection is None:
        raise HTTPException(status_code=503, detail="chroma collection unavailable")

    def _run() -> int:
        return backfill_visibility_metadata(_conn(), collection)

    n = await asyncio.get_event_loop().run_in_executor(None, _run)
    logger.info("chroma visibility backfill stamped %d chunks", n)
    return {"backfilled": n}
