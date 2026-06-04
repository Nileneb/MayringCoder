"""Admin: collapse the multi-codebook table into the ONE domain-independent codebook.

The canonical Mayring method is domain-INDEPENDENT — there is exactly ONE codebook,
not per-domain profiles (generic/laravel/python/sozialforschung/…). The code was made
codebook-agnostic (mayring_reduce/process/categorize_chunks write to _the_codebook_id =
'universal'); this endpoint consolidates the EXISTING category rows: every category in a
non-universal codebook is MOVED into universal, or MERGED into the same-named universal
category (chunk_categories/parent refs repointed, evidence summed, Chroma embedding +
row dropped), then the now-empty extra codebooks are deleted.

Mounted under /stats/ so the production nginx whitelist already covers it.
Admin-gated. ``dry_run=true`` reports the plan without writing.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.api.auth import get_token_info
from src.api.dependencies import get_conn as _get_conn
from src.api.jwt_auth import TokenInfo

router = APIRouter()
_log = logging.getLogger(__name__)

_UNIVERSAL_SLUG = "universal"


def _is_admin(info: TokenInfo) -> bool:
    return "*" in info.scopes or "admin" in info.scopes


def _universal_id(conn: Any) -> int:
    row = conn.execute("SELECT id FROM codebooks WHERE slug=?", (_UNIVERSAL_SLUG,)).fetchone()
    if row is None:
        raise HTTPException(status_code=409,
                            detail=f"no '{_UNIVERSAL_SLUG}' codebook — nothing to collapse into")
    return int(row[0])


def collapse_codebooks(conn: Any, chroma: Any = None, *, dry_run: bool = True) -> dict:
    """Move/merge every non-universal category into the universal codebook, then drop the
    extra codebooks. Idempotent: a second run finds no non-universal codebooks → all-zero.

    dry_run=True: only classify (move vs merge vs delete-codebook), write nothing.
    dry_run=False: SQLite work in one transaction (rollback on error), then best-effort
    Chroma embedding cleanup (orphan embeddings are harmless — cat_match reads active
    categories by embedding_id, and the rows are gone)."""
    univ = _universal_id(conn)
    others = [int(r[0]) for r in conn.execute(
        "SELECT id FROM codebooks WHERE id != ? ORDER BY id", (univ,)).fetchall()]
    plan = {"universal_id": univ, "other_codebooks": others,
            "moved": 0, "merged": 0, "chunk_links_repointed": 0,
            "chunk_links_deduped": 0, "codebooks_deleted": 0, "dry_run": dry_run}

    # Classify every non-universal category (move if name free in universal, else merge).
    cats = conn.execute(
        "SELECT id, name, evidence_count, embedding_id FROM codebook_categories "
        f"WHERE codebook_id != {univ} ORDER BY id").fetchall()
    if dry_run:
        seen_universal_names = {r[0] for r in conn.execute(
            "SELECT name FROM codebook_categories WHERE codebook_id=?", (univ,)).fetchall()}
        moved_names: set[str] = set()
        for _cid, name, _ev, _emb in cats:
            if name in seen_universal_names or name in moved_names:
                plan["merged"] += 1
            else:
                plan["moved"] += 1
                moved_names.add(name)
        plan["codebooks_deleted"] = len(others)
        plan["categories_total_after"] = conn.execute(
            "SELECT COUNT(*) FROM codebook_categories").fetchone()[0] - plan["merged"]
        return plan

    backup_path = _backup_db()
    plan["backup"] = str(backup_path) if backup_path else None
    dropped_embeddings: list[str] = []
    try:
        for cid, name, evidence, emb_id in cats:
            cid = int(cid)
            u = conn.execute(
                "SELECT id FROM codebook_categories WHERE codebook_id=? AND name=?",
                (univ, name)).fetchone()
            if u is None:
                # MOVE: name free in universal → just re-home the row (id/links unchanged).
                conn.execute("UPDATE codebook_categories SET codebook_id=? WHERE id=?",
                             (univ, cid))
                plan["moved"] += 1
                continue
            # MERGE cid → u: repoint links, sum evidence, repoint parent refs, drop row.
            u_id = int(u[0])
            before = conn.execute(
                "SELECT COUNT(*) FROM chunk_categories WHERE category_id=?", (cid,)).fetchone()[0]
            conn.execute("UPDATE OR IGNORE chunk_categories SET category_id=? WHERE category_id=?",
                         (u_id, cid))
            leftover = conn.execute(
                "SELECT COUNT(*) FROM chunk_categories WHERE category_id=?", (cid,)).fetchone()[0]
            conn.execute("DELETE FROM chunk_categories WHERE category_id=?", (cid,))
            plan["chunk_links_repointed"] += before - leftover
            plan["chunk_links_deduped"] += leftover
            conn.execute("UPDATE codebook_categories SET parent_id=? WHERE parent_id=?",
                         (u_id, cid))
            conn.execute("UPDATE codebook_proposals SET parent_hint_id=? WHERE parent_hint_id=?",
                         (u_id, cid))
            conn.execute(
                "UPDATE codebook_categories SET evidence_count = evidence_count + ? WHERE id=?",
                (int(evidence or 0), u_id))
            conn.execute("DELETE FROM codebook_categories WHERE id=?", (cid,))  # cascades proposals
            if emb_id:
                dropped_embeddings.append(emb_id)
            plan["merged"] += 1
        # Extra codebooks are now category-free → drop them.
        for ocid in others:
            conn.execute("DELETE FROM codebooks WHERE id=?", (ocid,))
            plan["codebooks_deleted"] += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    if chroma is not None and dropped_embeddings:
        try:
            chroma.delete(ids=dropped_embeddings)
        except Exception as exc:  # WHY: orphan embeddings are harmless (rows gone) — log, don't fail
            _log.warning("collapse-codebooks: chroma cleanup of %d embeddings failed: %s",
                         len(dropped_embeddings), exc)
    plan["categories_total_after"] = conn.execute(
        "SELECT COUNT(*) FROM codebook_categories").fetchone()[0]
    return plan


def _backup_db():
    """Online SQLite backup of memory.db before the destructive write. Best-effort:
    if the memory store isn't a local SQLite file (path missing), skip + log."""
    try:
        from mayring_core.memory.store import MEMORY_DB_PATH
        if not MEMORY_DB_PATH.exists():
            _log.warning("collapse-codebooks: MEMORY_DB_PATH %s missing — no backup", MEMORY_DB_PATH)
            return None
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dst = MEMORY_DB_PATH.with_name(f"memory.db.bak-collapse-{ts}")
        src = sqlite3.connect(str(MEMORY_DB_PATH))
        try:
            out = sqlite3.connect(str(dst))
            try:
                src.backup(out)
            finally:
                out.close()
        finally:
            src.close()
        _log.info("collapse-codebooks: backup written to %s", dst)
        return dst
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"backup failed, aborting: {exc}")


@router.post("/stats/admin/collapse-codebooks")
async def collapse_codebooks_endpoint(
    dry_run: bool = True,
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """Collapse all non-universal codebooks into 'universal'. dry_run=true (default) only
    reports the plan. Pass dry_run=false to execute (backs up memory.db first)."""
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    conn = _get_conn()
    chroma = None
    if not dry_run:
        from mayring_core.memory.store import get_chroma_collection
        chroma = get_chroma_collection("codebook_categories")
    return collapse_codebooks(conn, chroma, dry_run=dry_run)
