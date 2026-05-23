# Phase 3 `mayring_process` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]`.

**Goal:** Eine fail-closed, mixed-method `mayring_process`-Pipeline server-seitig in
MayringCoder, exponiert als `POST /codebooks/{id}/process`.

**Architecture:** Reines Modul `mayring_process.py` (injizierbare embed_fn/llm_fn/chroma/
conn) + dünne FastAPI-Route. Deduktiv (Cosine gegen `codebook_categories`-Embeddings) →
Merge nach Score → induktiv (LLM + Pflicht-parent_hint) → Embedding-Dedup → Proposal/
chunk_categories-Write. Reuse der codebooks-API-Write-Logik via `record_proposal`.

**Tech Stack:** Python, FastAPI, SQLite (`mayring_core.memory.store`), ChromaDB,
Ollama (`providers.embed_texts`/`generate_text`, via `https://three.linn.games`).

---

### Task 1: Modul-Skelett + fail-closed + ProcessResult

**Files:**
- Create: `core/mayring_core/memory/ingestion/mayring_process.py`
- Test: `tests/test_mayring_process.py`

- [ ] Test: `mayring_process("", "task", 1, ...)` → `ValueError`; leeres task → `ValueError`;
      Codebook ohne aktive Kategorien → `ValueError`. `ProcessResult` ist ein dataclass
      mit `category_id, category_name, decision, confidence, igio_axis, proposed`.
- [ ] Run: `pytest tests/test_mayring_process.py -k failclosed -v` → FAIL (kein Modul).
- [ ] Implement: Konstanten `_DEDUCTIVE_MIN=0.78`, `_HYBRID_MIN=0.55`, `_DEDUP_MIN=0.92`;
      `@dataclass ProcessResult`; `_load_active_categories(conn, codebook_id)`; Guards.
- [ ] Run → PASS. Commit.

### Task 2: deduktive Cosine

- [ ] Test: mit `_FakeChroma` (numpy-array-Embeddings) + 2 Kategorien → höchster Cosine
      gewinnt; `_deductive_match` gibt `(category_row, score)`. numpy-Regression
      (`np.array`, kein `or []`).
- [ ] Run → FAIL. Implement `_cosine` (zero-safe) + `_deductive_match` (fetch via
      `chroma.get(ids=embedding_ids, include=['embeddings'])`, `[] if None`). Run → PASS. Commit.

### Task 3: Merge deduktiv ≥0.78

- [ ] Test: stub embed_fn so dass top-score ≥0.78 → `decision="deductive"`,
      `category_id` = existierende aktive, `proposed=False`, kein LLM-Call (llm_fn raises).
- [ ] Run → FAIL. Implement Merge-Zweig 1. Run → PASS. Commit.

### Task 4: hybrid 0.55..0.78 + record_proposal-Refactor

**Files:** Modify `src/api/routes/codebooks.py` (extrahiere `record_proposal`).

- [ ] Test: top-score 0.6 → `decision="hybrid"`, existierende Kategorie zugeordnet UND
      `codebook_proposals`-Row angelegt + `evidence_count` erhöht.
- [ ] Run → FAIL. Implement: `record_proposal(conn, codebook_id, name, *, paraphrase,
      parent_hint_id, igio_axis, pi_job_id, chunk_id, embedding_id="")` in codebooks.py;
      `create_proposal`-Endpoint ruft sie; mayring_process hybrid-Zweig ruft sie. Run → PASS. Commit.

### Task 5: induktiv <0.55 + Pflicht-parent_hint + Embedding-Dedup

- [ ] Test a: top-score 0.3, llm_fn→"neues_label" → `decision="inductive"`, neue
      `status='proposed'`-Kategorie, `parent_id` = nächste deduktive Kategorie-id (nicht None).
- [ ] Test b: induzierter Label-Embed cosine >0.92 zu existierender → KEINE neue Kategorie,
      `evidence_count` der existierenden +1, `decision="inductive-dedup"`.
- [ ] Run → FAIL. Implement induktiver Zweig: `llm_fn(prompt)` → Label; `parent_hint_id` =
      `_deductive_match`-Top-id (PFLICHT); `_dedup_against_existing(embed_fn(label), conn,
      chroma, codebook_id)` → evidence++ oder `record_proposal`. Run → PASS. Commit.

### Task 6: chunk_categories-Write

- [ ] Test: mit `chunk_id="c1"` → Row in `chunk_categories` mit korrekter `source`
      (deductive/inductive/hybrid-merge) + `confidence`. Ohne `chunk_id` → keine Row.
- [ ] Run → FAIL. Implement `_link_chunk(conn, chunk_id, category_id, version, confidence,
      source)` (INSERT OR REPLACE). Run → PASS. Commit.

### Task 7: API-Endpoint `POST /codebooks/{id}/process`

**Files:** Modify `src/api/routes/codebooks.py`; Test `tests/test_codebooks_process_route.py`.

- [ ] Test: TestClient POST mit leerem text → 400; gültig (gemockte embed/llm/chroma) →
      200 + ProcessResult-Felder.
- [ ] Run → FAIL. Implement `ProcessRequest` + Route: wired `embed_fn`/`llm_fn`/
      `get_chroma_collection("codebook_categories")`; `ValueError`→`HTTPException(400)`;
      404 bei fehlendem Codebook. Run → PASS. Commit.

### Task 8: Deploy + Live-Verify + Smoke-Check

- [ ] Schema unverändert (alle Tabellen Phase 1) → kein Version-Bump nötig.
- [ ] Push master → Build&Push → Deploy. Warten bis live.
- [ ] Live-Verify: `POST /codebooks/3/process` mit echtem Python-Code-Text → deduktive
      Zuordnung mit score>0.5; mit leerem task → 400.
- [ ] Smoke: Check `mayring_process_fail_closed` (400 ohne task) zu
      `tools/smoke_test_production.py` hinzufügen. Push. Commit.
