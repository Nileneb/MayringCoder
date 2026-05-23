# Phase 3 — `mayring_process` (mixed-method, fail-closed) Design

**Datum:** 2026-05-24
**Status:** Approved (User-Entscheid 2026-05-24: zentral in MayringCoder)
**Tracking:** v2.0-Plan Phase 3, MayringCoder #270

## Entscheidung & Begründung

Der Plan-Text platzierte `mayring_process` in `mayring-pi-agent/pi.py`. **Kritischer
Befund:** der Pi-Agent erreicht MayringCoders Chroma/SQLite ausschließlich über HTTP
`/memory/search` (`_cloud_search`) — er hat **keinen** Zugriff auf die
`codebook_categories`-Embedding-Collection, die die deduktive Cosine-Stufe braucht.
Zudem läuft Kategorisierung bereits server-seitig (`categorization.py::mayring_categorize`).

**→ Zentralisierung in MayringCoder** (User-Entscheid): deduktiv + induktiv + Merge +
Dedup + Proposal-Write leben dort, wo Embeddings + DB liegen. Eine Pipeline, kein
Netz-Hop mitten in der deterministischen Logik. Reuse von codebooks-API +
`get_chroma_collection("codebook_categories")`. Pi-Agent-`pi_*`-Tools + der
Ingestion-Pfad werden später Thin-Wrapper darauf (Phase 3.2, nicht-blockierend).

## Datenlage (verifiziert auf Prod 2026-05-24)

- Codebooks: generic(1), laravel(2), python(3, 24 Kategorien), sozialforschung(4, 8).
- Kategorie-Embeddings in Chroma-Collection `codebook_categories`, `embedding_id =
  "cb:<slug>:<name>"`. `chunk_categories`-Tabelle existiert (Phase 1).
- **`igio_axis` ist überall NULL** (Importer-Gap). `mayring_process` funktioniert ohne;
  Achse wird best-effort inferiert (`O`=ergebnis, `I`=limitation, `G`=goal, `V`=...),
  sonst NULL. Voller Fix = separater Importer-Pass (Phase 4/5).

## Architektur

**Modul** `core/mayring_core/memory/ingestion/mayring_process.py` — reine, injizierbare
Funktionen (kein FastAPI-Import), damit Route UND Ingestion-Pfad sie teilen.

```
def mayring_process(text, task, codebook_id, *, conn, chroma_categories,
                    embed_fn, llm_fn, chunk_id=None, pi_job_id="",
                    codebook_version=1) -> ProcessResult
```

**Fail-closed (NICHT-VERHANDELBAR):** `ValueError` (→ HTTP 400) wenn `text` leer,
`task` leer, oder das Codebook keine aktiven Kategorien hat. Kein stiller Default,
kein „uncategorized"-Fallback. Genau das Anti-Pattern, das #270 adressiert.

### Schwellen (Modul-Konstanten)
- `_DEDUCTIVE_MIN = 0.78` — cosine ≥ → harte deduktive Zuordnung, kein LLM.
- `_HYBRID_MIN   = 0.55` — 0.55..0.78 → Zuordnung **+** Proposal (Evidenz).
- `_DEDUP_MIN    = 0.92` — induzierter Label cosine > → Evidenz statt Neuanlage.

### Ablauf
1. Aktive Kategorien laden (`id, name, embedding_id, igio_axis, parent_id`).
2. **3a deduktiv:** `embed_fn(text)` → Cosine gegen die via
   `chroma_categories.get(ids=embedding_ids, include=['embeddings'])` geholten
   Kategorie-Vektoren (numpy-safe: `[] if x is None else x`). Top-Kategorie + Score.
3. **3c Merge (deterministisch, score-getrieben):**
   - `score ≥ 0.78` → `decision="deductive"`, existierende aktive Kategorie. Kein LLM.
   - `0.55 ≤ score < 0.78` → `decision="hybrid"`: Kategorie zuordnen **und** Proposal
     anlegen (evidence++ auf dieselbe Kategorie). `chunk_categories.source='hybrid-merge'`.
   - `score < 0.55` → **3b induktiv:** `llm_fn` leitet Label ab; **`parent_hint_id`
     PFLICHT** = die nächstliegende deduktive Kategorie-id (nie freier Wurzel-Knoten).
     Dann **Embedding-Dedup:** `embed_fn(label)` → Cosine > 0.92 gegen alle
     aktiven+proposed Embeddings → falls Dup: evidence++ auf existierende (kein Neu);
     sonst neue `status='proposed'`-Kategorie via shared `record_proposal`.
4. `chunk_categories`-Link schreiben (chunk_id, category_id, codebook_version,
   confidence=score, source ∈ {deductive,inductive,hybrid-merge}) — nur wenn `chunk_id`.
5. `ProcessResult{category_id, category_name, decision, confidence, igio_axis, proposed}`.

### Shared Write-Helper
`create_proposal`-Endpoint-Logik wird zu `record_proposal(conn, codebook_id, name, *,
paraphrase, parent_hint_id, igio_axis, pi_job_id, chunk_id)` refaktoriert (DRY): von
Endpoint UND `mayring_process` benutzt. Setzt `embedding_id` jetzt korrekt (Dedup-fix).

## API

`POST /codebooks/{codebook_id}/process` in `src/api/routes/codebooks.py`:
- `ProcessRequest{text, task, chunk_id?, pi_job_id?, codebook_version?}`.
- 400 bei leerem text/task (Pydantic + expliziter Guard → ValueError-Mapping).
- 404 wenn Codebook fehlt; 400 wenn keine aktiven Kategorien.
- Verdrahtet `embed_fn=providers.embed_texts`-wrapper, `llm_fn=providers.generate_text`-
  wrapper (Ollama via `https://three.linn.games`, KEIN Port), `chroma=get_chroma_collection`.
- Antwort = `ProcessResult`-dict.

## Tests (TDD)
Reine Funktionen mit `_FakeChroma` (numpy-array-Embeddings!) + In-Memory-SQLite +
Stub-`embed_fn`/`llm_fn`. Fälle: fail-closed (leer text/task/keine Kategorien),
deduktiv ≥0.78, hybrid 0.55..0.78 (Proposal angelegt), induktiv <0.55 (LLM, parent_hint
gesetzt), Dedup >0.92 (evidence++ statt neu), chunk_categories-Write, numpy-Regression.

## Out of Scope (Phase 3 Kern)
- Thin-Wrapper der alten `pi_*`-MCP-Tools (Phase 3.2).
- Auto-Promote-Cron (existiert als `/promote`-Endpoint; Cron = Phase 4).
- igio_axis-Backfill-Importer (Phase 4/5).
- UI (Phase 5).
