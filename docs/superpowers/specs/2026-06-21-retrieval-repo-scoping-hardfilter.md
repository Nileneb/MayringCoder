# Spec: Repo-Scoping Hard-Filter im Retrieval (Kandidaten-Auswahl)

**Datum:** 2026-06-21 · **Repo:** mayring-core (PR) · **Status:** einfache Spec für frische Session

## Problem (gemessen 2026-06-21)
Eine `repo_slug`-gescopte Suche (`/memory/search {repo_slug: "app.linn.games"}`) liefert trotzdem Chunks aus *fremden* Repos (z.B. MayringCoder `tools/smoke_test_production.py`). Wurzel:
- Die **Vector-Stage filtert NICHT nach Repo** — `build_chroma_where()` (retrieval.py) scopt nur `visibility`/`user_id`/`org_id`. Die Top-K Vektor-Kandidaten kommen also aus ALLEN Repos.
- Der `project_match`-Boost (C3, ~0.08, post-hoc) kann nichts retten: die Repo-Chunks sind oft gar nicht erst im Kandidaten-Pool. `score_project_match` war in den Messungen 0.0.

## Ziel
Wenn `repo_slug`/`project_id` gesetzt ist, sollen die Chunks DIESES Repos den Kandidaten-Pool dominieren — fremde Repos nur noch als schwache Ergänzung (oder gar nicht).

## Ansatz (einer wählen in der Session)
1. **Repo in Chroma-Metadata + Where-Filter** (sauberste): `repo` (canonical) in die Chroma-Metadata jedes Chunks schreiben (Insert-Pfad + Backfill der bestehenden), dann in `build_chroma_where()` bei gesetztem repo `{"repo": <canonical>}` ergänzen. Harter Filter, schnell.
2. **Scoped zweite Vektor-Query**: globale Top-K wie bisher + eine ZWEITE Chroma-Query `where {"chunk_id": {"$in": repo_chunk_ids}}` (chunk_ids aus `_scope_filter(repo=...)`), beide mergen. Garantiert Repo-Präsenz ohne Metadata-Backfill. Achtung: `$in` mit vielen ids — cappen/batchen.
3. **project_match als harter Re-Rank statt 0.08**: wenn repo-scoped, Repo-Chunks im Reranker stark priorisieren — hilft nur wenn sie im Pool sind (also + größeres Top-K).

**Empfehlung:** Ansatz 1 (repo in Metadata) — strukturell richtig, deckt sich mit dem Reference-Layer-`source_class`-Backfill ([[2026-06-21-reference-doc-layer]]), beides in einem Metadata-Migrationslauf.

## Falle
- `repo_slug` MUSS canonicalisiert werden (`canonical_repo_ref`), sonst kein Match (C3-Falle, [[project_c3_project_scoped_memory]]).
- NICHT die Cross-Tenant-Visibility brechen (`build_chroma_where` bleibt die visibility-Disjunktion; repo-Filter ist ZUSÄTZLICH).

## Akzeptanz
`{repo_slug:"app.linn.games", query:"3D experience tier fallback"}` → top-5 sind app.linn.games-Chunks (resources/js/experience surfaced), 0 fremde Repos. Test: hermetisch gegen seed-DB + ein Live-Smoke.

## Verwandt
[[project_swe_benchmark_slice_2026_06_21]] (Diagnose), [[project_reranker_degeneration_2026_06_20]] (Ranking ist orthogonal — das hier ist Kandidaten-AUSWAHL).
