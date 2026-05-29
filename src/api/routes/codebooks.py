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
        "project_id": r[13],
    }


_CAT_COLS = ("id, codebook_id, name, igio_axis, parent_id, description, status, "
             "source, evidence_count, embedding_id, risk_level, languages, patterns, "
             "project_id")


class ProposalRequest(BaseModel):
    category_name: str
    pi_job_id: str = ""
    chunk_id: str | None = None
    paraphrase: str = ""
    parent_hint_id: int | None = None
    igio_axis: str | None = None
    project_id: str | None = None


class ProcessRequest(BaseModel):
    text: str = ""
    task: str = ""
    chunk_id: str | None = None
    pi_job_id: str = ""
    codebook_version: int = 1
    project_id: str | None = None


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


def record_proposal(
    conn, codebook_id: int, category_name: str, *,
    paraphrase: str = "", parent_hint_id: int | None = None,
    igio_axis: str | None = None, pi_job_id: str = "",
    chunk_id: str | None = None, embedding_id: str = "",
    project_id: str | None = None,
) -> int:
    """Create-or-evidence a category + record the proposal row. Returns category_id.

    Shared by the /proposals endpoint and mayring_process (DRY). Does NOT commit —
    the caller owns the transaction so the mixed-method pipeline can batch its writes.
    """
    now = _now()
    cat = conn.execute(
        "SELECT id FROM codebook_categories WHERE codebook_id=? AND name=?",
        (codebook_id, category_name)).fetchone()
    if cat is None:
        # WHY(#270): induzierte Kategorie startet als 'proposed' (parent_hint PFLICHT
        # bei induktiv — der Caller liefert ihn), bis evidence sie auto-promotet.
        conn.execute(
            "INSERT INTO codebook_categories(codebook_id, name, igio_axis, parent_id, "
            "description, status, source, evidence_count, embedding_id, project_id) "
            "VALUES (?,?,?,?,?, 'proposed','induced', 1, ?, ?)",
            (codebook_id, category_name, igio_axis, parent_hint_id,
             paraphrase[:200], embedding_id, project_id))
        cat_id = conn.execute("SELECT id FROM codebook_categories WHERE codebook_id=? "
                              "AND name=?", (codebook_id, category_name)).fetchone()[0]
    else:
        cat_id = cat[0]
        conn.execute("UPDATE codebook_categories SET evidence_count = evidence_count + 1 "
                     "WHERE id=?", (cat_id,))
    conn.execute(
        "INSERT INTO codebook_proposals(category_id, pi_job_id, chunk_id, paraphrase, "
        "parent_hint_id, proposed_at) VALUES (?,?,?,?,?,?)",
        (cat_id, pi_job_id, chunk_id, paraphrase, parent_hint_id, now))
    return cat_id


@router.post("/codebooks/{codebook_id}/proposals")
async def create_proposal(
    codebook_id: int, req: ProposalRequest, _ws: str = Depends(get_workspace),
) -> dict:
    """Pi-Agent proposes a (possibly new) category. Embedding-dedup + auto-promote
    run via mayring_process / the promote endpoint. New category → status='proposed'."""
    conn = _get_conn()
    cat_id = record_proposal(
        conn, codebook_id, req.category_name, paraphrase=req.paraphrase,
        parent_hint_id=req.parent_hint_id, igio_axis=req.igio_axis,
        pi_job_id=req.pi_job_id, chunk_id=req.chunk_id, project_id=req.project_id)
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


@router.post("/codebooks/{codebook_id}/proposals/{category_id}/reject")
async def reject_category(
    codebook_id: int, category_id: int, _ws: str = Depends(get_workspace),
) -> dict:
    """Verwirft eine PROPOSED-Kategorie (Gegenstück zu promote): löscht ihre chunk_categories-
    Links, die Kategorie selbst und ihr Chroma-Embedding; markiert offene Proposals 'reject'.
    Safety: aktive Kategorien sind NICHT löschbar (400) — die bedienen cat_match."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT status, embedding_id FROM codebook_categories WHERE id=? AND codebook_id=?",
        (category_id, codebook_id)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"category {category_id} not found")
    if row[0] == "active":
        raise HTTPException(status_code=400, detail="cannot reject an active category")
    conn.execute("DELETE FROM codebook_proposals WHERE category_id=?", (category_id,))
    conn.execute("DELETE FROM chunk_categories WHERE category_id=?", (category_id,))
    conn.execute("DELETE FROM codebook_categories WHERE id=? AND codebook_id=?",
                 (category_id, codebook_id))
    conn.commit()
    emb_id = row[1]
    if emb_id:
        try:
            from mayring_core.memory.store import get_chroma_collection
            get_chroma_collection("codebook_categories").delete(ids=[emb_id])
        except Exception as exc:  # noqa: BLE001 — Embedding-Cleanup darf den Delete nicht versenken
            import logging
            logging.getLogger(__name__).warning(
                "reject: chroma embedding %s nicht entfernt: %s", emb_id, exc)
    return {"category_id": category_id, "status": "rejected"}


@router.post("/codebooks/{codebook_id}/process")
async def process_text(
    codebook_id: int, req: ProcessRequest, _ws: str = Depends(get_workspace),
) -> dict:
    """Phase 3 mixed-method, fail-closed categorization. Wires the real Ollama
    embed/LLM providers + the codebook_categories Chroma collection into the pure
    mayring_process pipeline. ValueError (empty text/task, no active categories) → 400."""
    import os

    from mayring_core import providers
    from mayring_core.memory.ingestion.mayring_process import mayring_process
    from mayring_core.memory.store import get_chroma_collection
    from mayring_core.model_router import ModelRouter

    conn = _get_conn()
    if conn.execute("SELECT 1 FROM codebooks WHERE id=?", (codebook_id,)).fetchone() is None:
        raise HTTPException(status_code=404, detail=f"codebook {codebook_id} not found")

    # Ollama via Proxy ohne Port (CLAUDE.md-Invariante); Modell aus ModelRouter, nicht env.
    ollama_url = os.environ.get("OLLAMA_URL", "https://three.linn.games")
    model = ModelRouter(ollama_url=ollama_url).resolve("text") or "mistral:7b-instruct"

    def _embed_one(t: str) -> list[float]:
        out = providers.embed_texts([t], ollama_url)
        return (out[0] if out else []) or []

    def _llm(prompt: str) -> str:
        # temperature=0 + fixer seed: die Reduktion soll deterministisch sein (gleicher Text
        # → gleiches Label), damit Dedup/Evidenz-Akkumulation greift statt zu variieren.
        return providers.generate_text(prompt=prompt, ollama_url=ollama_url,
                                       model=model, label="mayring_process",
                                       options={"temperature": 0.0, "seed": 7})

    try:
        res = mayring_process(
            req.text, req.task, codebook_id, conn=conn,
            chroma_categories=get_chroma_collection("codebook_categories"),
            embed_fn=_embed_one, llm_fn=_llm, chunk_id=req.chunk_id,
            pi_job_id=req.pi_job_id, codebook_version=req.codebook_version,
            active_project_id=req.project_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "category_id": res.category_id, "category_name": res.category_name,
        "decision": res.decision, "confidence": res.confidence,
        "igio_axis": res.igio_axis, "proposed": res.proposed,
    }
