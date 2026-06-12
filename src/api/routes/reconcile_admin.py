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
        # WHY(#drift 2026-06-12, Datenverlust-Bug): bei gesetztem workspace_id MUSS
        # Chroma auf denselben Workspace gescopet werden. Ein unscoped get() holt ALLE
        # ids → missing_in_sqlite = chroma_ids - sqlite_ids enthielte dann jeden FREMDEN
        # Workspace-Vektor, den dry_run=False löschen würde. Collection-Metadaten tragen
        # workspace_id (siehe _desired_metadata), also where={"workspace_id": …} abziehen.
        if workspace_id:
            chroma_payload = chroma.get(where={"workspace_id": workspace_id}, include=[])
        else:
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


def _desired_metadata(row: dict) -> dict:
    """Canonical Chroma metadata for a chunk — mirrors the ingest writer
    (ingestion/core.py upsert). Chroma forbids None → empty-string the optionals."""
    return {
        "workspace_id": row["workspace_id"] or "default",
        "source_id": row["source_id"],
        "chunk_level": row["chunk_level"] or "file",
        "category_labels": row["category_labels"] or "",
        "category_source": row["category_source"] or "",
        "category_confidence": float(row["category_confidence"] or 0.0),
        "is_active": 1,
        "visibility": row["visibility"] or "private",
        "org_id": row["org_id"] or "",
        "user_id": row["user_id"] or "",
    }


# The tenancy axis the vector where-clause filters on (build_chroma_where).
# A drift in any of these silently drops the chunk from the vector candidate pool.
_TENANCY_KEYS = ("visibility", "user_id", "org_id")


def _repair_chroma_metadata(
    conn: Any,
    chroma: Any,
    *,
    workspace_id: str | None,
    source_type: str | None,
    dry_run: bool,
    batch: int = 200,
) -> dict:
    """Re-write Chroma metadata from the SQLite source-of-truth WITHOUT re-embedding.

    WHY(bge-m3-migration 2026-06-08): the migration wrote repo_file chunks into the
    bge collection without the visibility/user_id axis. The vectors are correct and
    present (reconcile: not missing) — only the metadata is wrong, so build_chroma_where
    drops them from the vector pool (score_vector=0.000) while the SQL/keyword lens
    (which reads the sources table) still finds them. Re-embedding 1831 sources over
    HTTP would burn hours of GPU for nothing; chroma.update(ids, metadatas=...) fixes
    the metadata in place at zero embedding cost. Only ids whose tenancy axis actually
    drifted are touched, so the pass is idempotent and reports the true repair count.
    """
    out: dict[str, Any] = {"scanned": 0, "in_chroma": 0, "mismatched": 0,
                           "updated": 0, "missing_in_chroma": 0, "dry_run": dry_run}
    if conn is None or chroma is None:
        return out

    sql = (
        "SELECT c.chunk_id, c.workspace_id, c.source_id, c.chunk_level, "
        "c.category_labels, c.category_source, c.category_confidence, "
        "s.visibility, s.org_id, s.user_id "
        "FROM chunks c LEFT JOIN sources s ON c.source_id = s.source_id "
        "WHERE c.is_active = 1"
    )
    params: list = []
    if workspace_id:
        sql += " AND c.workspace_id = ?"
        params.append(workspace_id)
    if source_type:
        sql += " AND s.source_type = ?"
        params.append(source_type)

    cols = ["chunk_id", "workspace_id", "source_id", "chunk_level",
            "category_labels", "category_source", "category_confidence",
            "visibility", "org_id", "user_id"]
    rows = [dict(zip(cols, r)) for r in conn.execute(sql, params).fetchall()]
    out["scanned"] = len(rows)
    by_id = {r["chunk_id"]: r for r in rows}

    ids = list(by_id)
    for i in range(0, len(ids), batch):
        chunk_ids = ids[i:i + batch]
        try:
            cur = chroma.get(ids=chunk_ids, include=["metadatas"])
        except (RuntimeError, ValueError, AttributeError) as e:
            _log.warning("repair: chroma.get batch failed: %s", e)
            continue
        cur_ids = cur.get("ids") or []
        cur_metas = cur.get("metadatas") or []
        present = {cid: (cur_metas[j] or {}) for j, cid in enumerate(cur_ids)}
        out["in_chroma"] += len(cur_ids)
        out["missing_in_chroma"] += len(chunk_ids) - len(cur_ids)

        upd_ids: list[str] = []
        upd_metas: list[dict] = []
        for cid in chunk_ids:
            if cid not in present:
                continue  # no vector → repair can't help; reconcile handles those
            want = _desired_metadata(by_id[cid])
            have = present[cid]
            if any(str(have.get(k, "")) != str(want[k]) for k in _TENANCY_KEYS):
                upd_ids.append(cid)
                upd_metas.append(want)
        out["mismatched"] += len(upd_ids)
        if upd_ids and not dry_run:
            try:
                chroma.update(ids=upd_ids, metadatas=upd_metas)
                out["updated"] += len(upd_ids)
            except (RuntimeError, ValueError) as e:
                _log.warning("repair: chroma.update batch failed: %s", e)
    return out


def _assign_owner_to_orphans(
    conn: Any, *, workspace_id: str, owner: str, dry_run: bool,
) -> dict:
    """Stamp an owner on ownerless private data in ONE workspace.

    WHY(owner-attribution 2026-06-08): private data with no user_id is unreachable —
    build_chroma_where requires `user_id = caller`, so a private chunk with user_id=''
    matches nobody (dead, ungoverned data). In a single-user personal workspace every
    ownerless row provably belongs to that one owner, so we backfill it. Scoped to the
    given workspace_id ONLY — never a blanket cross-tenant stamp. Only visibility=
    'private' rows are touched (the sources CHECK constraint guarantees visibility is
    always one of private/org/public); public/org data keeps its semantics."""
    where = ("workspace_id = ? AND visibility = 'private' "
             "AND (user_id IS NULL OR user_id = '')")
    n_src = conn.execute(
        f"SELECT COUNT(*) FROM sources WHERE {where}", (workspace_id,)).fetchone()[0]
    n_chk = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE workspace_id = ? "
        "AND (user_id IS NULL OR user_id = '')", (workspace_id,)).fetchone()[0]
    if not dry_run and (n_src or n_chk):
        conn.execute(
            f"UPDATE sources SET user_id = ? WHERE {where}", (owner, workspace_id))
        conn.execute(
            "UPDATE chunks SET user_id = ? WHERE workspace_id = ? "
            "AND (user_id IS NULL OR user_id = '')", (owner, workspace_id))
        conn.commit()
    return {"orphan_sources": n_src, "orphan_chunks": n_chk, "owner": owner}


def _repair_all_workspaces(
    conn: Any,
    chroma: Any,
    *,
    source_type: str | None,
    dry_run: bool,
    batch: int,
) -> dict:
    """Full-scope repair as a per-workspace iteration instead of one monolithic scan.

    WHY(#drift 2026-06-12, full-scope-crash): a single unscoped scan over ~8.5k chunks
    reproducibly killed the uvicorn child (502, no traceback) — likely a memory/pointer
    blow-up holding the whole by_id map + Chroma payloads at once. Iterating the distinct
    workspace_ids keeps each scan+update round bounded; results are summed so the response
    shape is identical to a scoped call. Workspace-scoped callers never hit this path.
    """
    ws_ids = [r[0] for r in conn.execute(
        "SELECT DISTINCT workspace_id FROM chunks WHERE is_active = 1").fetchall()]
    agg: dict[str, Any] = {"scanned": 0, "in_chroma": 0, "mismatched": 0,
                           "updated": 0, "missing_in_chroma": 0, "dry_run": dry_run,
                           "workspaces_iterated": len(ws_ids)}
    for ws in ws_ids:
        part = _repair_chroma_metadata(
            conn, chroma, workspace_id=ws, source_type=source_type,
            dry_run=dry_run, batch=batch)
        for k in ("scanned", "in_chroma", "mismatched", "updated", "missing_in_chroma"):
            agg[k] += part[k]
    return agg


@router.post("/admin/repair-chroma-metadata")
def repair_chroma_metadata(
    workspace_id: str | None = None,
    source_type: str | None = None,
    assign_owner: str | None = None,
    dry_run: bool = True,
    batch: int = 200,
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """Repair drifted Chroma tenancy metadata (visibility/user_id/org_id) from the
    SQLite source-of-truth, in place, WITHOUT re-embedding. Fixes chunks that the
    bge-m3 migration wrote without the visibility axis (score_vector=0.000 despite a
    present vector). `dry_run=False` writes; restrict scope with workspace_id and/or
    source_type (e.g. 'repo_file'). `assign_owner=<sub>` first stamps that owner on
    ownerless private data in the workspace (single-user-workspace backfill) so the
    chroma sync below picks up a real user_id — requires workspace_id. Authz: service
    token or admin scope."""
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin-scope required")
    batch = max(1, min(int(batch), 500))
    conn = _get_conn()
    result: dict = {}
    if assign_owner:
        if not workspace_id:
            raise HTTPException(status_code=400,
                                detail="assign_owner requires workspace_id (no blanket stamp)")
        result["owner_assignment"] = _assign_owner_to_orphans(
            conn, workspace_id=workspace_id, owner=assign_owner, dry_run=dry_run)
    chroma = _get_chroma()
    if workspace_id:
        result.update(_repair_chroma_metadata(
            conn, chroma,
            workspace_id=workspace_id, source_type=source_type,
            dry_run=dry_run, batch=batch,
        ))
    else:
        result.update(_repair_all_workspaces(
            conn, chroma, source_type=source_type, dry_run=dry_run, batch=batch,
        ))
    if not dry_run and result.get("updated"):
        from mayring_core.memory.retrieval import invalidate_query_cache
        invalidate_query_cache()
    result["workspace_id"] = workspace_id or "<all>"
    result["source_type"] = source_type or "<all>"
    return result


def _process_reembed_queue(
    conn: Any,
    chroma: Any,
    *,
    embed_fn: Any,
    ollama_url: str,
    dry_run: bool,
    limit: int,
) -> dict:
    """Drain the reembed-pending queue: re-embed each pending chunk into the LIVE
    Chroma collection and log a reembed-done event.

    WHY(#drift 2026-06-12): reconcile writes 'reembed-pending' events (source_id column
    holds the CHUNK id) but nothing ever consumed them — the queue was a feature built
    but never wired. Idempotency: a chunk is pending iff it has a 'reembed-pending' event
    with no later 'reembed-done' event for the same id. Each chunk is isolated (one
    failure does not abort the run). Inactive/deleted chunks are marked done + counted
    as skipped_missing so they leave the queue.
    """
    out: dict[str, Any] = {"queued": 0, "processed": 0, "failed": 0,
                           "skipped_missing": 0, "dry_run": dry_run}
    if conn is None or chroma is None:
        return out

    pending = [r[0] for r in conn.execute(
        "SELECT DISTINCT p.source_id FROM ingestion_log p "
        "WHERE p.event_type = 'reembed-pending' "
        "AND NOT EXISTS (SELECT 1 FROM ingestion_log d "
        "  WHERE d.source_id = p.source_id AND d.event_type = 'reembed-done') "
        "ORDER BY p.source_id LIMIT ?",
        [int(limit)],
    ).fetchall()]
    out["queued"] = len(pending)
    if dry_run or not pending:
        return out

    now = datetime.now(timezone.utc).isoformat()

    def _log_done(cid: str, status: str) -> None:
        conn.execute(
            "INSERT INTO ingestion_log (source_id, event_type, payload, created_at) "
            "VALUES (?, ?, ?, ?)",
            (cid, "reembed-done", json.dumps({"status": status}), now),
        )

    cols = ["chunk_id", "text", "workspace_id", "source_id", "chunk_level",
            "category_labels", "category_source", "category_confidence",
            "visibility", "org_id", "user_id"]
    select = (
        "SELECT c.chunk_id, c.text, c.workspace_id, c.source_id, c.chunk_level, "
        "c.category_labels, c.category_source, c.category_confidence, "
        "s.visibility, s.org_id, s.user_id "
        "FROM chunks c LEFT JOIN sources s ON c.source_id = s.source_id "
        "WHERE c.chunk_id = ? AND c.is_active = 1"
    )
    for cid in pending:
        row = conn.execute(select, [cid]).fetchone()
        if row is None:
            # chunk gone or deactivated — pull it off the queue, can't re-embed
            _log_done(cid, "skipped_missing")
            out["skipped_missing"] += 1
            conn.commit()
            continue
        rec = dict(zip(cols, row))
        try:
            emb = embed_fn([rec["text"][:500]], ollama_url)[0]
            chroma.upsert(
                ids=[cid],
                documents=[rec["text"][:500]],
                embeddings=[emb],
                metadatas=[_desired_metadata(rec)],
            )
            _log_done(cid, "ok")
            out["processed"] += 1
        except Exception as e:  # isolate per-chunk; surface in counter, never abort
            _log.warning("reembed failed for %s: %s", cid[:12], e)
            out["failed"] += 1
        # WHY(write-lock-starvation 2026-06-13): EIN End-Commit hielt den SQLite-
        # Write-Lock über ALLE Embed-HTTP-Calls (Live-Lauf: 98s für 313 Chunks) —
        # parallele /repo-events-Webhooks starben nach busy_timeout=10s mit
        # "database is locked", deren CI-Events gingen verloren (best-effort-Sender).
        # Per-Chunk-Commit hält den Lock nur Millisekunden.
        conn.commit()
    conn.commit()
    return out


@router.post("/admin/process-reembed-queue")
def process_reembed_queue(
    limit: int = 100,
    dry_run: bool = True,
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """Consume the reembed-pending queue written by /admin/reconcile-chroma.

    For each pending chunk (a 'reembed-pending' ingestion_log event with no matching
    'reembed-done'): load its text, embed it (bge-m3 via ModelRouter — no OLLAMA_MODEL
    env-bleed), upsert into the LIVE Chroma collection with canonical metadata, and log
    'reembed-done'. Inactive/deleted chunks are marked done + skipped. Per-chunk errors
    are isolated (counted, not fatal). `dry_run=True` (default) only reports the queue
    depth. Authz: service token or admin scope."""
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin-scope required")
    limit = max(1, int(limit))

    import os
    from mayring_core.providers import embed_texts as _embed_texts

    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    result = _process_reembed_queue(
        _get_conn(), _get_chroma(),
        embed_fn=_embed_texts, ollama_url=ollama_url,
        dry_run=dry_run, limit=limit,
    )
    if not dry_run and result.get("processed"):
        from mayring_core.memory.retrieval import invalidate_query_cache
        invalidate_query_cache()
    return result


@router.post("/admin/reconcile-chroma")
def reconcile_chroma(
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
