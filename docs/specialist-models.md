# Spezialisten-Models — Datenstrategie & Export-Pipelines (#260)

Hauptmodell bleibt `mistral:7b-instruct`. Für den Gaming-Layer-Device-Pool
(Android/iPhone) sind kleine, task-spezifische Expertenmodelle interessant.
Damit „wir trainieren jetzt ein Spezialisten-Model" nicht „von vorne
Daten sammeln" heißt, sind die Datenquellen + Paar-Formate + Export-Scripts hier
festgenagelt.

**Scope dieses Dokuments / Issue #260:** nur die Daten-Pipeline. Kein Training.

Alle Exporter folgen dem Muster von `tools/export_retrieval_dataset.py`:
`export(...) -> int` + `main()`, plain `sqlite3`, JSONL nach
`cache/finetuning/<name>.jsonl`. **Kein Train/Test-Split im Exporter** — der
Split passiert downstream im Trainer (analog `tools/train_reranker.py:120`,
`train_test_split(..., test_size=0.2, random_state=42, stratify=...)`).

---

## 1. Mayring-Categorizer

Text + Task → Mayring-Kategorien mit Belegen.

| | |
|---|---|
| **Tool** | `tools/export_mayring_categorizer_dataset.py` |
| **Quelle** | `chunks` (memory.db) LEFT JOIN `wiki_category_evidence` (wiki_v2.db) |
| **Filter** | `category_source='hybrid'` AND `category_labels != ''` AND `is_active=1` |
| **Format** | `{"input": {"text", "task"}, "output": {"kategorien": [str], "belege": [{kategorie, span, excerpt, reasoning}]}, "workspace_id", "chunk_id"}` |
| **Lokal verfügbar** | ✅ ~728 Zeilen (hybrid chunks); `belege: []` bis Backfill |
| **Min-Rows** | ~500 für ein erstes LoRA |

**Backfill für Belege:** `wiki_category_evidence` ist leer bis
`pi_mark_categories(persist=True)` über die Hybrid-Chunks gelaufen ist. Ohne
Belege ist der Datensatz ein schwächeres (label-only) Set — als Quick-Start
nutzbar, für span-grounded Codierung erst nach dem Backfill vollwertig.

```bash
python tools/export_mayring_categorizer_dataset.py \
    --db cache/memory.db --wiki-db cache/wiki_v2.db \
    --out cache/finetuning/mayring_categorizer_dataset.jsonl
```

---

## 2. Judge-Relevance

(Query, Chunk-Text) → Relevanz-Score [0, 1].

| | |
|---|---|
| **Tool** | `tools/export_judge_relevance_dataset.py` |
| **Quelle** | `chunk_feedback` (signal 1..5) JOIN `chunks` (text) |
| **Mapping** | `score = (avg_rating - 1) / 4`; legacy `positive`→4, `negative`→2; `neutral`/sonstige → übersprungen |
| **Query** | aus `chunk_feedback.metadata.query_context` |
| **Format** | `{"input": {"query", "chunk_text"}, "output": float, "workspace_id", "chunk_id", "n_ratings"}` |
| **Lokal verfügbar** | ⚠ ~683 Rating-Zeilen, aber **ohne `query_context`** → `query: ""` |
| **Min-Rows** | ~300 mit Query |

**⚠ Daten-Prerequisite (Blocker für nutzbares Set):** lokal hat keine
`chunk_feedback`-Row einen `query_context` (alle aus dem Auto-Memory-Context-
Generator, nicht aus echter Suche). Ein judge-relevance-Model braucht aber das
(Query, Chunk)-Paar. Backfill nötig: `metadata.query_context` beim Rating
mitschreiben (Stop-Hook / `tools/replay_feedback.py`). Bis dahin liefert
`--require-query` 0 Zeilen — bewusst, statt unbrauchbare Leerquery-Paare.

```bash
# alle Ratings (Leerquery erlaubt, für Inspektion):
python tools/export_judge_relevance_dataset.py --db cache/memory.db \
    --out cache/finetuning/judge_relevance_dataset.jsonl
# nur trainierbare Zeilen (mit Query):
python tools/export_judge_relevance_dataset.py --db cache/memory.db \
    --out cache/finetuning/judge_relevance_dataset.jsonl --require-query
```

---

## 3. Forschungsfrage-Quality-Scorer

Forschungsfrage → `{score 0-100, warnings, pico}`.

| | |
|---|---|
| **Tool** | `tools/export_forschungsfrage_quality_dataset.py` |
| **Quelle** | **app.linn.games (Laravel PostgreSQL)** — `PhaseAgentResult.result_data.qualitaets_bewertung` + `GameState`-evals + `P1Warnsignal`/`P1Komponente` |
| **Format** | `{"input": {"forschungsfrage"}, "output": {"score", "warnings": [...], "pico": {...}}, "workspace_id"}` |
| **Lokal verfügbar** | ❌ 0 Zeilen — Daten liegen nicht in diesem Repo |
| **Min-Rows** | ~200 |

**Scope-Hinweis:** Diese Quelle liegt komplett in der app.linn.games-PG, nicht
in `cache/memory.db`. Das Tool liest eine logische Quelle mit Spalten
`(forschungsfrage, score, warnings, pico, workspace_id)` (warnings/pico = JSON):

* `--dsn postgres://…` — Postgres-DSN auf eine app.linn.games-View, die
  `qualitaets_bewertung` in diese Spalten flacht. **Das exakte Spalten-Mapping
  ist auf der Laravel-Seite zu bestätigen** (separate Aufgabe, app.linn.games-
  Scope) — z.B. eine View `forschungsfrage_quality` über
  `P1`-PhaseAgentResults.
* `--db path.sqlite` — eine `forschungsfrage_quality`-Tabelle gleicher Form
  (von den Tests genutzt).

```bash
python tools/export_forschungsfrage_quality_dataset.py \
    --dsn "$APP_LINN_GAMES_DSN" \
    --out cache/finetuning/forschungsfrage_quality_dataset.jsonl
```

---

## Trainings-Hinweis (out of scope für #260)

Vermutlich LoRA/QLoRA auf einem kleinen Base (qwen2.5-0.5b/1.5b o.ä.).
Split + Training gehören in ein separates `tools/train_<task>.py` (Vorlage:
`tools/train_reranker.py`). Erst relevant wenn pro Task genug saubere Paare da
sind (siehe Min-Rows + Backfills oben).
