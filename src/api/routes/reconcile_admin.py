"""POST /admin/reconcile-chroma — Diff SQLite chunks vs ChromaDB.

V2 Stufe 5.3 (audit Section 7): bisher gab es nur den
`chroma_candidate_mismatch`-Diagnose-Counter, aber keinen Reconcile-Job.
Wenn ein Ingest-Pfad nur in einem der beiden Stores landete (z.B. Chroma-
Insert OK aber sqlite-commit-fail wegen DB-locked), driftet das System
schleichend.

Diese Route listet die Differenz und kann optional `dry_run=False`
bekommen, dann werden orphane Chroma-Einträge gelöscht und sqlite-Chunks
ohne Vektor zur Re-Embed-Queue (`ingestion_log` event_type='reembed-pending')
markiert.

Authz: nur Service-Token oder admin-scope-JWT.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.api.auth import get_token_info
from src.api.dependencies import get_chroma as _get_chroma, get_conn as _get_conn
from src.api.jwt_auth import TokenInfo

router = APIRouter()
_log = logging.getLogger(__name__)


def _is_admin(info: TokenInfo) -> bool:
    return "*" in info.scopes or "admin" in info.scopes


def _reconcile_chroma_sqlite(
    conn: Any,
    chroma: Any,
    workspace_id: str | None = None,
) -> dict:
    """Return a diff between active sqlite chunk_ids and Chroma vector ids.

    Args:
        conn: open DBAdapter, or None for trivial-test mode.
        chroma: ChromaDB collection, or None for trivial-test mode.
        workspace_id: if set, restrict comparison to this workspace.

    Returns:
        {
          missing_in_chroma: list[chunk_id],   # in sqlite, not in chroma
          missing_in_sqlite: list[chunk_id],   # in chroma, not in sqlite
          total_sqlite: int,
          total_chroma: int,
        }
    """
    out: dict[str, Any] = {
        "missing_in_chroma": [],
        "missing_in_sqlite": [],
        "total_sqlite": 0,
        "total_chroma": 0,
    }
    if conn is None or chroma is None:
        return out

    sql = "SELECT chunk_id FROM chunks WHERE is_active = 1"
    params: list = []
    if workspace_id:
        sql += " AND workspace_id = ?"
        params.append(workspace_id)
    sqlite_ids = {r[0] for r in conn.execute(sql, params).fetchall()}
    out["total_sqlite"] = len(sqlite_ids)

    try:
        # Chroma's get() ohne ids gibt alle zurück (paginiert). limit=None.
        chroma_payload = chroma.get(include=[])
        chroma_ids = set(chroma_payload.get("ids") or [])
    except (RuntimeError, ValueError, AttributeError) as e:
        # WHY(v2-stufe5.3): konkrete Chroma-Fehler typisieren — anderes laut.
        _log.warning("reconcile: chroma.get() failed: %s", e)
        chroma_ids = set()
    out["total_chroma"] = len(chroma_ids)

    out["missing_in_chroma"] = sorted(sqlite_ids - chroma_ids)[:500]
    out["missing_in_sqlite"] = sorted(chroma_ids - sqlite_ids)[:500]
    return out


@router.post("/admin/reconcile-chroma")
async def reconcile_chroma(
    workspace_id: str | None = None,
    dry_run: bool = True,
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """Run a Chroma↔SQLite reconcile pass.

    Returns a diff. With `dry_run=False` (admin-only) writes:
      - For each missing_in_sqlite id: chroma.delete(id) — orphan vector cleanup.
      - For each missing_in_chroma id: ingestion_log event_type='reembed-pending'
        so a follow-up worker can re-embed.

    Authz: scope='*' (service token) or scope='admin'.
    """
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin-scope required")

    conn = _get_conn()
    chroma = _get_chroma()
    diff = _reconcile_chroma_sqlite(conn, chroma, workspace_id=workspace_id)

    if not dry_run:
        # Orphan-Vector-Delete in Chroma:
        if diff["missing_in_sqlite"]:
            try:
                chroma.delete(ids=diff["missing_in_sqlite"])
            except (RuntimeError, ValueError) as e:
                _log.warning("reconcile chroma.delete failed: %s", e)
        # Re-embed-pending-events:
        now = datetime.now(timezone.utc).isoformat()
        for cid in diff["missing_in_chroma"]:
            conn.execute(
                "INSERT INTO ingestion_log (source_id, event_type, payload, created_at) "
                "VALUES (?, ?, ?, ?)",
                (cid, "reembed-pending", json.dumps({"workspace_id": workspace_id}), now),
            )
        conn.commit()
        diff["applied"] = True
    else:
        diff["applied"] = False
    diff["workspace_id"] = workspace_id or "<all>"
    return diff
