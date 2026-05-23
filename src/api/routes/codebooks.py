"""Codebook API (#workspace-uuid-sot v2.0 Phase 1.3).

DB-as-SoT codebook endpoints — reads the SQLite tables seeded by
tools/import_codebooks_to_db.py. Consumed by the v2 SessionStart hook
(GET /codebooks/{slug} + /categories) and the Pi-Agent mayring_process
(POST /proposals). Single-workspace → no workspace filter; auth via the
standard token dependency.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.api.auth import get_workspace
from src.api.dependencies import get_conn as _get_conn

router = APIRouter(tags=["codebooks"])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _category_row(r) -> dict:
    return {
        "id": r[0], "codebook_id": r[1], "name": r[2], "igio_axis": r[3],
        "parent_id": r[4], "description": r[5], "status": r[6], "source": r[7],
        "evidence_count": r[8], "embedding_id": r[9], "risk_level": r[10],
        "languages": json.loads(r[11] or "[]"), "patterns": json.loads(r[12] or "[]"),
    }


_CAT_COLS = ("id, codebook_id, name, igio_axis, parent_id, description, status, "
             "source, evidence_count, embedding_id, risk_level, languages, patterns")


class ProposalRequest(BaseModel):
    category_name: str
    pi_job_id: str = ""
    chunk_id: str | None = None
    paraphrase: str = ""
    parent_hint_id: int | None = None
    igio_axis: str | None = None


@router.get("/codebooks")
async def list_codebooks(_ws: str = Depends(get_workspace)) -> dict:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, slug, description, version, auto_promote_threshold FROM codebooks "
        "ORDER BY slug").fetchall()
    return {"codebooks": [
        {"id": r[0], "slug": r[1], "description": r[2], "version": r[3],
         "auto_promote_threshold": r[4]} for r in rows]}


@router.get("/codebooks/{slug}")
async def get_codebook(slug: str, _ws: str = Depends(get_workspace)) -> dict:
    conn = _get_conn()
    r = conn.execute(
        "SELECT id, slug, description, version, auto_promote_threshold FROM codebooks "
        "WHERE slug = ?", (slug,)).fetchone()
    if r is None:
        raise HTTPException(status_code=404, detail=f"codebook {slug!r} not found")
    n = conn.execute(
        "SELECT count(*) FROM codebook_categories WHERE codebook_id=? AND status='active'",
        (r[0],)).fetchone()[0]
    return {"id": r[0], "slug": r[1], "description": r[2], "version": r[3],
            "auto_promote_threshold": r[4], "active_categories": n}


@router.get("/codebooks/{codebook_id}/categories")
async def list_categories(
    codebook_id: int,
    status: str = Query(default="active"),
    _ws: str = Depends(get_workspace),
) -> dict:
    conn = _get_conn()
    rows = conn.execute(
        f"SELECT {_CAT_COLS} FROM codebook_categories "
        "WHERE codebook_id = ? AND status = ? ORDER BY evidence_count DESC, name",
        (codebook_id, status)).fetchall()
    return {"categories": [_category_row(r) for r in rows], "count": len(rows)}


@router.post("/codebooks/{codebook_id}/proposals")
async def create_proposal(
    codebook_id: int, req: ProposalRequest, _ws: str = Depends(get_workspace),
) -> dict:
    """Pi-Agent proposes a (possibly new) category. Embedding-dedup + auto-promote
    run via the promote endpoint / cron. New category → status='proposed'."""
    conn = _get_conn()
    now = _now()
    cat = conn.execute(
        "SELECT id, evidence_count FROM codebook_categories WHERE codebook_id=? AND name=?",
        (codebook_id, req.category_name)).fetchone()
    if cat is None:
        # WHY(#270): induzierte Kategorie startet als 'proposed' (parent_hint PFLICHT
        # bei induktiv — der Caller liefert ihn), bis evidence sie auto-promotet.
        conn.execute(
            "INSERT INTO codebook_categories(codebook_id, name, igio_axis, parent_id, "
            "description, status, source, evidence_count, embedding_id) "
            "VALUES (?,?,?,?,?, 'proposed','induced', 1, '')",
            (codebook_id, req.category_name, req.igio_axis, req.parent_hint_id,
             req.paraphrase[:200]))
        cat_id = conn.execute("SELECT id FROM codebook_categories WHERE codebook_id=? "
                              "AND name=?", (codebook_id, req.category_name)).fetchone()[0]
    else:
        cat_id = cat[0]
        conn.execute("UPDATE codebook_categories SET evidence_count = evidence_count + 1 "
                     "WHERE id=?", (cat_id,))
    conn.execute(
        "INSERT INTO codebook_proposals(category_id, pi_job_id, chunk_id, paraphrase, "
        "parent_hint_id, proposed_at) VALUES (?,?,?,?,?,?)",
        (cat_id, req.pi_job_id, req.chunk_id, req.paraphrase, req.parent_hint_id, now))
    conn.commit()
    return {"category_id": cat_id, "status": "recorded"}


@router.post("/codebooks/{codebook_id}/proposals/{category_id}/promote")
async def promote_category(
    codebook_id: int, category_id: int, _ws: str = Depends(get_workspace),
) -> dict:
    conn = _get_conn()
    conn.execute("UPDATE codebook_categories SET status='active', promoted_at=? "
                 "WHERE id=? AND codebook_id=?", (_now(), category_id, codebook_id))
    conn.execute("UPDATE codebook_proposals SET decision='promote', reviewed_by='api' "
                 "WHERE category_id=? AND decision IS NULL", (category_id,))
    conn.commit()
    return {"category_id": category_id, "status": "active"}
