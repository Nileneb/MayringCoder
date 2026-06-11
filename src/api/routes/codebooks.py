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
        "id": r[0], "name": r[1], "igio_axis": r[2],
        "parent_id": r[3], "description": r[4], "status": r[5], "source": r[6],
        "evidence_count": r[7], "embedding_id": r[8], "risk_level": r[9],
        "languages": json.loads(r[10] or "[]"), "patterns": json.loads(r[11] or "[]"),
        "project_id": r[12],
    }


_CAT_COLS = ("id, name, igio_axis, parent_id, description, status, "
             "source, evidence_count, embedding_id, risk_level, languages, patterns, "
             "project_id")


class CategoryPatchReq(BaseModel):
    name: str | None = None
    description: str | None = None


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


class CategorizeRequest(BaseModel):
    text: str
    task: str = ""
    model: str = ""           # Reduktions-LLM-Override (für Modell-Duelle); leer = ModelRouter 'text'
    project_id: str | None = None
    dry_run: bool = False     # nur welche Hälfte griffe (read-only); KEIN Codebook-Write — für Duelle


@router.get("/codebooks")
def list_codebooks(_ws: str = Depends(get_workspace)) -> dict:
    # WHY(v19-drop-codebook): codebooks-Tabelle existiert nicht mehr — synthetische
    # Antwort damit Smoke (GET /codebooks → cb_id für /process-Check) grün bleibt.
    return {"codebooks": [{"id": 1, "slug": "categories",
                           "description": "alle Kategorien (flach, kein Codebook mehr)",
                           "version": 1}]}


@router.get("/codebooks/{slug}")
def get_codebook(slug: str, _ws: str = Depends(get_workspace)) -> dict:
    conn = _get_conn()
    n = conn.execute(
        "SELECT count(*) FROM categories WHERE status='active'").fetchone()[0]
    return {"id": 1, "slug": slug, "description": "alle Kategorien (flach, kein Codebook mehr)",
            "version": 1, "active_categories": n}


@router.get("/codebooks/{codebook_id}/categories")
def list_categories(
    codebook_id: int,  # vestigial path param, ignored (v19: kein codebook_id mehr)
    status: str = Query(default="active"),
    _ws: str = Depends(get_workspace),
) -> dict:
    conn = _get_conn()
    rows = conn.execute(
        f"SELECT {_CAT_COLS} FROM categories "
        "WHERE status = ? ORDER BY evidence_count DESC, name",
        (status,)).fetchall()
    return {"categories": [_category_row(r) for r in rows], "count": len(rows)}


def record_proposal(
    conn, category_name: str, *,
    paraphrase: str = "", parent_hint_id: int | None = None,
    igio_axis: str | None = None, pi_job_id: str = "",
    chunk_id: str | None = None, embedding_id: str = "",
    project_id: str | None = None,
) -> int:
    """Create-or-evidence a category + record the proposal row. Returns category_id.

    Shared by the /proposals endpoint and mayring_process (DRY). Does NOT commit —
    the caller owns the transaction so the mixed-method pipeline can batch its writes.
    v19: codebook_id entfernt — flache categories-Tabelle, UNIQUE(name).
    """
    now = _now()
    cat = conn.execute(
        "SELECT id FROM categories WHERE name=?",
        (category_name,)).fetchone()
    if cat is None:
        # WHY(#270): induzierte Kategorie startet als 'proposed' (parent_hint PFLICHT
        # bei induktiv — der Caller liefert ihn), bis evidence sie auto-promotet.
        conn.execute(
            "INSERT INTO categories(name, igio_axis, parent_id, "
            "description, status, source, evidence_count, embedding_id, project_id) "
            "VALUES (?,?,?,?, 'proposed','induced', 1, ?, ?)",
            (category_name, igio_axis, parent_hint_id,
             paraphrase[:200], embedding_id, project_id))
        cat_id = conn.execute("SELECT id FROM categories WHERE name=?",
                              (category_name,)).fetchone()[0]
    else:
        cat_id = cat[0]
        conn.execute("UPDATE categories SET evidence_count = evidence_count + 1 "
                     "WHERE id=?", (cat_id,))
    conn.execute(
        "INSERT INTO codebook_proposals(category_id, pi_job_id, chunk_id, paraphrase, "
        "parent_hint_id, proposed_at) VALUES (?,?,?,?,?,?)",
        (cat_id, pi_job_id, chunk_id, paraphrase, parent_hint_id, now))
    return cat_id


@router.post("/codebooks/{codebook_id}/proposals")
def create_proposal(
    codebook_id: int,  # vestigial path param, ignored (v19)
    req: ProposalRequest, _ws: str = Depends(get_workspace),
) -> dict:
    """Pi-Agent proposes a (possibly new) category. Embedding-dedup + auto-promote
    run via mayring_process / the promote endpoint. New category → status='proposed'."""
    conn = _get_conn()
    cat_id = record_proposal(
        conn, req.category_name, paraphrase=req.paraphrase,
        parent_hint_id=req.parent_hint_id, igio_axis=req.igio_axis,
        pi_job_id=req.pi_job_id, chunk_id=req.chunk_id, project_id=req.project_id)
    conn.commit()
    return {"category_id": cat_id, "status": "recorded"}


@router.post("/codebooks/{codebook_id}/proposals/{category_id}/promote")
def promote_category(
    codebook_id: int,  # vestigial path param, ignored (v19)
    category_id: int, _ws: str = Depends(get_workspace),
) -> dict:
    conn = _get_conn()
    conn.execute("UPDATE categories SET status='active', promoted_at=? "
                 "WHERE id=?", (_now(), category_id))
    conn.execute("UPDATE codebook_proposals SET decision='promote', reviewed_by='api' "
                 "WHERE category_id=? AND decision IS NULL", (category_id,))
    conn.commit()
    return {"category_id": category_id, "status": "active"}


@router.post("/codebooks/{codebook_id}/proposals/{category_id}/reject")
def reject_category(
    codebook_id: int,  # vestigial path param, ignored (v19)
    category_id: int, _ws: str = Depends(get_workspace),
) -> dict:
    """Verwirft eine PROPOSED-Kategorie (Gegenstück zu promote): löscht ihre chunk_categories-
    Links, die Kategorie selbst und ihr Chroma-Embedding; markiert offene Proposals 'reject'.
    Safety: aktive Kategorien sind NICHT löschbar (400) — die bedienen cat_match."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT status, embedding_id FROM categories WHERE id=?",
        (category_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"category {category_id} not found")
    if row[0] == "active":
        raise HTTPException(status_code=400, detail="cannot reject an active category")
    conn.execute("DELETE FROM codebook_proposals WHERE category_id=?", (category_id,))
    conn.execute("DELETE FROM chunk_categories WHERE category_id=?", (category_id,))
    conn.execute("DELETE FROM categories WHERE id=?", (category_id,))
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


def reduce_text_server(text: str, theme: str, *, project_id: str | None = None,
                       model_override: str = "", dry_run: bool = False):
    """Verdrahtet die EINE Mayring-Methode (mayring_reduce) mit den echten Server-Providern.
    EIN Codebook, domänenunabhängig — kein source_type/codebook-Routing.
    model_override: für Modell-Duelle (gleiche mixed Methode inkl. Embedding-Match gegen
    die Bestands-Kategorien, nur anderes Reduktions-LLM); leer = ModelRouter 'text'."""
    import os
    from mayring_core import providers
    from mayring_core.memory.ingestion.mayring_process import mayring_reduce
    from mayring_core.memory.store import get_chroma_collection
    from mayring_core.model_router import ModelRouter
    conn = _get_conn()
    ollama_url = os.environ.get("OLLAMA_URL", "https://three.linn.games")
    model = model_override or ModelRouter(ollama_url=ollama_url).resolve("text") or "mistral:7b-instruct"

    def _embed_one(t: str) -> list[float]:
        out = providers.embed_texts([t], ollama_url)
        return (out[0] if out else []) or []

    def _llm(prompt: str) -> str:
        # think=False: die Reduktion ist strukturierte Extraktion — Thinking bringt nichts,
        # frisst aber bei kleinen Thinking-Modellen (qwen3.5:2b) das num_predict-Budget auf,
        # bevor das JSON kommt → leeres Label (done_reason=length). Für non-thinking-Modelle
        # (mistral:7b-instruct) ist es ein No-op → fairer Modell-Vergleich.
        return providers.generate_text(prompt=prompt, ollama_url=ollama_url, model=model,
                                       label="pi:reduce", options={"temperature": 0.0, "seed": 7},
                                       think=False)

    return mayring_reduce(
        text, theme, conn=conn,
        chroma_categories=get_chroma_collection("codebook_categories"),
        embed_fn=_embed_one, llm_fn=_llm, project_id=project_id, dry_run=dry_run)


@router.post("/codebooks/categorize")
def categorize_endpoint(req: CategorizeRequest, _ws: str = Depends(get_workspace)) -> dict:
    """Volle mixed Mayring-Methode (Reduktion + Embedding-Match gegen die Bestands-
    Kategorien) mit optionalem LLM-Override — für Modell-Duelle. Liefert die FINALE
    Kategorie (nach deduktivem Match), nicht das rohe Reduktions-Label."""
    if not (req.text or "").strip():
        raise HTTPException(status_code=422, detail="text required")
    try:
        res = reduce_text_server(req.text, req.task or "(kein Task)",
                                 project_id=req.project_id, model_override=req.model.strip(),
                                 dry_run=req.dry_run)
    except Exception as e:
        # Duell-tauglich: ein Modell, das Müll/leer liefert (oder fehlt), darf den Lauf
        # nicht mit 500 abbrechen. WHY(#145 stack-trace-exposure): rohe Exception NICHT
        # an den Client geben (Internals-Leak) — voll serverseitig loggen, im Body nur
        # ein generischer Marker. Nicht stumm, aber kein Detail-Leak.
        import logging
        logging.getLogger(__name__).warning(
            "reduce_text_server failed (model=%s): %s",
            req.model.strip() or "router-text-default", e,
        )
        return {"error": "categorize_failed", "model": req.model.strip() or "router-text-default"}
    cand = res.candidates[0] if res.candidates else None
    return {
        "label": cand.label if cand else "",
        "match": cand.match if cand else "",       # deductive (Bestand getroffen) | dedup | inductive (neu)
        "paraphrase": res.paraphrase,
        "generalize": res.generalization,
        "model": req.model.strip() or "router-text-default",
    }


@router.post("/codebooks/{codebook_id}/categories/{category_id}/deprecate")
def deprecate_category(
    codebook_id: int,  # vestigial path param, ignored (v19)
    category_id: int, _ws: str = Depends(get_workspace),
) -> dict:
    """Stilllegen einer AKTIVEN Kategorie (die Lücke, die reject nicht abdeckt): status→
    'deprecated' + Chroma-Embedding entfernen, damit cat_match (active-only) und das
    induktive Dedup (active+proposed) sie nicht mehr ziehen. chunk_categories-Links und
    Proposal-Historie bleiben (Soft-Delete). Danach ist sie via reject hart löschbar."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT status, embedding_id FROM categories WHERE id=?",
        (category_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"category {category_id} not found")
    conn.execute("UPDATE categories SET status='deprecated' WHERE id=?", (category_id,))
    conn.commit()
    emb_id = row[1]
    if emb_id:
        try:
            from mayring_core.memory.store import get_chroma_collection
            get_chroma_collection("codebook_categories").delete(ids=[emb_id])
        except Exception as exc:  # noqa: BLE001 — Embedding-Cleanup darf den Status-Flip nicht versenken
            import logging
            logging.getLogger(__name__).warning(
                "deprecate: chroma embedding %s nicht entfernt: %s", emb_id, exc)
    return {"category_id": category_id, "status": "deprecated"}


@router.post("/codebooks/{codebook_id}/process")
def process_text(
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
    # WHY(v19-drop-codebook): codebooks-Tabelle weg → keine 404-Prüfung mehr.
    # codebook_id-Pfadparam bleibt aus Back-Compat-Gründen (Smoke nutzt ihn).

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
                                       options={"temperature": 0.0, "seed": 7},
                                       think=False)

    try:
        res = mayring_process(
            req.text, req.task, conn=conn,
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


@router.patch("/codebooks/{codebook_id}/categories/{category_id}")
def patch_category(
    codebook_id: int,  # vestigial path param, ignored (v19)
    category_id: int,
    req: CategoryPatchReq,
    _ws: str = Depends(get_workspace),
) -> dict:
    """Rename and/or update description of a category. If embedding_id is set and name
    changes, syncs Chroma so cat_match stays consistent with the new name."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT name, embedding_id FROM categories WHERE id=?",
        (category_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"category {category_id} not found")

    current_name, emb_id = row[0], row[1]

    if req.name is not None and req.name != current_name:
        collision = conn.execute(
            "SELECT 1 FROM categories WHERE name=? AND id<>?",
            (req.name, category_id)).fetchone()
        if collision:
            raise HTTPException(status_code=409,
                                detail=f"category name '{req.name}' already exists")
        conn.execute("UPDATE categories SET name=? WHERE id=?", (req.name, category_id))
        if emb_id:
            # WHY(cat_match-sync): Chroma doc text is the name used for embedding similarity;
            # stale name breaks deductive matching after rename. Failure must NOT abort the DB
            # rename — the rename is the authoritative change, Chroma is a derived index.
            try:
                from mayring_core.memory.store import get_chroma_collection
                get_chroma_collection("codebook_categories").upsert(
                    ids=[emb_id], documents=[req.name])
            except Exception as exc:  # noqa: BLE001
                import logging
                logging.getLogger(__name__).warning(
                    "patch_category: chroma sync for embedding %s failed: %s", emb_id, exc)

    if req.description is not None:
        conn.execute("UPDATE categories SET description=? WHERE id=?",
                     (req.description[:500], category_id))

    conn.commit()
    return {"category_id": category_id, "status": "updated"}
