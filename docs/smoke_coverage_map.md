# Smoke Coverage Map — Closed-Issue Acceptance ↔ Live-Verification

Diese Datei ist die Single Source of Truth dafür, **wo jedes geschlossene
Issue in der Production-Smoke-Suite verifiziert wird**. Jeder Eintrag
hat genau eine Coverage-Methode:

- **API**: `tools/smoke_test_production.py` Check-Name (verifiziert live gegen prod)
- **Pytest**: bestimmtes Test-File in `tests/`
- **Code-only**: kein HTTP-Surface — Code-Removal/Refactor/Doku
- **Reopened**: Acceptance nicht erfüllt; Issue wieder offen

Wenn ein neuer Smoke-Check geschrieben oder ein Issue geschlossen wird,
**hier eintragen**. Ein Meta-Check (`check_coverage_map_complete`)
verifiziert, dass jedes closed Issue genau einen Eintrag hat.

## Aktive Smoke-Checks (gegen prod)

| # | Issue | Coverage |
|---|---|---|
| 184a | Rationale-Edges (WHY-marker → wiki_edges → /memory/search co-injection) | Pytest `tests/test_rationale_parser.py` (9 tests inkl. relative-path, vendor-skip, multiline-far-target — Code-review-fix in d848853) + `tests/test_memory_retrieval.py::test_search_attaches_rationale_edges` + `test_compress_for_prompt_renders_rationale_block` (Red-Green TDD verifiziert) — Spec `docs/superpowers/specs/2026-05-09-rationale-edges-design.md`, Plan `docs/superpowers/plans/2026-05-09-rationale-edges.md` |
| 187 | Phantom-Features (pt + re) im Reranker-Trainer | Pytest `tests/test_reranker_v2_sanity.py::test_negative_pt_or_re_weight_rejected` + memory_service.py logs pt + re per chunk in stage_scores + tools/export_retrieval_dataset.py FEATURES_OUT erweitert + tools/train_reranker.py FEATURES analog (commit 8316d93) |
| 185 | Path traversal CodeQL #129/#130 | Pytest `tests/test_predictive.py::test_load_keyword_index_rejects_path_traversal` (Red-Green TDD verifiziert) + CodeQL Re-Scan 0 alerts (commits 7a1b729, 2887894) |
| 184 | Cross-DB-Join → predicted-topic-boost | Pytest `tests/test_predictive.py::test_search_boosts_chunks_matching_predicted_topics` + RetrievalRecord.score_predicted_topic field als Reranker-Trainings-Feature exposed (commit 46e9c2e) |
| 180 | Reranker-Training liefert degeneriertes Modell | Pytest `tests/test_train_reranker.py::test_rejects_negative_vector_weight` + `test_reranker_v2_sanity.py` Sanity-Gate (commits 817bc2e, a6e2691) |
| 138 | Memory-MCP feedback unreliable | API: `feedback_slug_resolution`, `stop_hook_e2e` |
| 137 | Ingest NEW/CHANGED/UNCHANGED | API: `ingest_state_field` |
| 107 | Local Pi-Agent | API: `pi_tasks_schema` |
| 106 | Modell-Upgrade Qwen 3.5 | API: `memory_search_vector` (model resolves) |
| 105 | conversation_watcher systemd | **Obsolet** durch Plugin-Hooks; siehe Comment |
| 104 | Shared Memory Visibility | API: `visibility_isolation` |
| 101 | LLM-Kategorisierung Logging | API: `categorization_logging` |
| 91 | Image-Routing | API: `image_routing_supported` |
| 90 | Task-Feedback-Matrix | API: `task_feedback_matrix` |
| 89 | Legacy Wiki entfernen | Code-only: dateien gelöscht (`wiki.py` etc.) |
| 88 | OLLAMA_MODEL env-Bleed | Code-only: `src/model_router.py` |
| 87 | Training-Data-Generator | **REOPENED** — `training_merge_endpoint` failt live |
| 86 | Mayring Plugin + Feedback Loop | API: `feedback_count_delta`, `stop_hook_e2e` |
| 85 | GPU-Entlastung Batch-Delay | API: `jobs_progress_observability` |
| 84 | Pipeline-Observability | API: `jobs_progress_observability` |
| 83 | TurbulenceChecker | API: `turbulence_endpoint` |
| 78 | Wiki-v2 P8 (History/Diff/Team) | API: `wiki_p8_history` |
| 77 | Wiki-v2 P7 (API + Web-UI) | API: `wiki_p7_endpoints` |
| 76 | Wiki-v2 P6 (Second-Opinion) | Code-only CLI flag (siehe #139 für Pi-Tool) |
| 75 | Wiki-v2 P5 (Context Injection) | API: `wiki_context_injector_used` |
| 74 | Wiki-v2 P4 (Watcher-Hooks) | API: `wiki_p7_endpoints` (Trigger-Pfad) |
| 73 | Wiki-v2 P3 (Cluster-Engine) | API: `wiki_graph_clusters` |
| 72 | Wiki-v2 P2 (Edge-Erkennung) | API: `wiki_graph_clusters` (edges in response) |
| 71 | Wiki-v2 P1 (Graph-Schema) | API: `wiki_graph_clusters` |
| 70 | Wiki-v2 EPIC | API: alle `wiki_*` Smoke-Checks |
| 68 | SQLite-Parallelität | Pytest: `tests/test_memory_store.py` |
| 67 | DB-Adapter Abstraktion | Code-only: `src/memory/db_adapter.py` |
| 66 | Architektur-Konsolidierung | Code-only: refactor |
| 65 | Production split Docker | Infra: `docker-compose.mayring.yml` |
| 64 | LARAVEL_INTERNAL_URL fix | Code-only: env config |
| 61 | Docker Compose Wildwuchs | Infra: docker-compose review |
| 60 | JWT-Schlüssel ≥32 Bytes | Code-only: key generation script |
| 55 | Ambient Context v2.0 | API: `wiki_context_injector_used`, `dashboard_endpoints` (`activations`) |
| 54 | Ambient Context v1.0 | API: same as #55 |
| 53 | Verknüpfungswiki | API: `wiki_graph_clusters` (edges) |
| 52 | Conversation-Watcher v1 | API: `micro_batch_indexes` (replaced) |
| 38 | MCP auth + workspace Chroma | API: `workspace_scoping`, `jwt_invalid_signature` |
| 37 | Docker-Compose Pipeline | Infra |
| 36 | Housekeeping `=4.0` | Code-only: file removed |
| 35 | Dokumentation Modelle | Docs |
| 34 | Memory-Roadmap | Meta |
| 33 | E2E-Tests Web-UI | Pytest: `tests/test_web_ui.py` |
| 32 | Chunking 3000 Limit | Pytest: `tests/test_memory_ingest.py` |
| 31 | BLIP-Captions Bilder | Code-only: `src/memory/image_ingest.py` |
| 30 | Multi-view + GPU Benchmark | API: `dashboard_endpoints` |
| 29 | Memory Batch Ingestion | API: `feedback_log_movement` |
| 28 | Architektur-Entscheidungen | Meta |
| 27 | MCP-basierte lokale Memory | API: `pi_tasks_schema` |
| 26 | Prompt Hardening | Code-only: `prompts/` files |
| 25 | Training-Pipeline | **REOPENED** via #87 (`training_merge_endpoint`) |
| 24 | Categorizer Kategorien | API: `categorization_logging` (mistral runs) |
| 23 | Similarity-Score 1.0 | Pytest: `tests/test_categorizer.py` |
| 22 | Dead Code | Code-only: removed |
| 21 | ChromaDB functions[] | API: `memory_search_vector` (matches > 0) |
| 20 | Full-Scan-Modus | Code-only: `--full` CLI flag |
| 19 | Leere Redundanz-Labels | Pytest: `tests/test_categorizer.py` |
| 18 | RAG: reactive Queries | API: `memory_search_vector` |
| 17 | Pipeline Feed-Forward | Code-only: orchestration |
| 16 | turbulence model param | Code-only: CLI |
| 15 | Zwei-Modell-Pipeline (Duel) | Code-only: `/duel` route exists |
| 14 | report.py refactor | Code-only |
| 13 | extractor.py refactor | Code-only |
| 12 | turbulence_analyzer split | Code-only |
| 11 | Embedding-Index | API: `memory_search_vector` |
| 10 | Refactoring history.py | Code-only |
| 9 | `--run-id` cache-isolation | Code-only: CLI |
| 8 | `--codebook` parameter | Code-only: CLI |
| 7 | LICENSE-Widerspruch | Code-only: doc fix |
| 6 | Run-Historie | Code-only: `cache/runs/` |
| 5 | Exclude-Patterns | Pytest: `tests/test_categorizer.py` |
| 4 | Projektkontext | API: `wiki_context_injector_used` |
| 3 | Export CSV/JSON | Code-only: `/reports` |
| 1 | Pipeline verbessern | Code-only: `--full` flag |

## Test-Infrastructure-Bugs (alle live in `pytest`)

| # | Pytest-File | Status |
|---|---|---|
| 100 | `tests/test_context_improvements.py` | ✓ live grün |
| 99 | `tests/test_embedder.py` | ✓ live grün |
| 98 | `tests/test_rag_enrichment.py` | ✓ live grün |
| 97 | `tests/test_second_opinion.py` + `tests/test_extractor.py` | ✓ live grün |
| 96 | `tests/test_v2_ops_endpoints.py` | ✓ live grün |
| 95 | `tests/test_web_ui*.py` | ✓ live grün |
| 94 | `tests/test_jwt_auth.py` + landing page | ✓ live grün |
| 93 | `tests/test_analyzer_hardening.py` | ✓ live grün |

Live verifiziert via `pytest tests/test_*` 158/158 passed @ 2026-05-08.

## Reopened (Acceptance not met)

| # | Status | Smoke-Check |
|---|---|---|
| 92 | Retraining mayringqwen | offen, kein automatisierter Test (manueller Trainings-Run) |

## Production Bugs (tracker issues)

| # | Description | Coverage |
|---|---|---|
| 162 | Wiki cluster engine produces shells without members | API: `wiki_cluster_depth` + RED-GREEN regression test `tests/test_wiki_v2_store.py::test_upsert_node_preserves_cluster_id_on_re_ingest` (commit a6daa78) |

## Currently Open (parallel work)

| # | Description |
|---|---|
| 141 | IGIO classifier rerun auf bestehende chunks (auto-close via igio-coverage-watch when ratio ≥ 0.5) |

## Recently Closed (today, 2026-05-08)

| # | Description | Coverage |
|---|---|---|
| 87 | Training-Data-Generator pipelines | API: `training_merge_endpoint` (200/400/401/422) + Pytest `tests/test_memory_context_generator.py` (8 tests) + workflow `training-data-pipeline.yml` daily 06:30 UTC |
| 139 | Second-Opinion als Pi-Agent-Tool | API: `pi_second_opinion_endpoint` + MCP-Tool registriert in `local_mcp.py` + `mcp.py` |
| 140 | ModelRouter zur Runtime konfigurierbar | API: `model_router_runtime` + `model_identity` + reset endpoint |

## Recently Closed (today, 2026-05-09)

| # | Description | Coverage |
|---|---|---|
| 182 | IGIO-Backfill hält SQLite-Lock zu lang | Pytest `tests/test_igio_backfill.py` chunked-commit + smoke `db_wal_journal_active` (Lock-acquire-Probe unter 1s) |
| 183 | Pi-Agent: Job-Verteilung & In-Process-Queue stabilisieren | API: `pi_tasks_schema` + `/pi-jobs/stats` (job_class p50/p95) + Pytest `tests/test_pi_queue.py` (3-lane priority routing, with_lanes(), classify_pi_job) |
| 192 | Ollama-API-Skalierung: parallel-jobs + Pi-Worker-Pool | host: OLLAMA_NUM_PARALLEL=4 + MAX_LOADED_MODELS=3 (systemd override) · code: PiQueue 3-lane defaults 4/4/2 (PR #219) · resilience: cloud-fallback OLLAMA_CLOUD_API_KEY |

## Recently Closed (today, 2026-05-11)

| # | Description | Coverage |
|---|---|---|
| 209 | Reranker-Training: rating-basiert statt binary | `tools/export_retrieval_dataset.py::_label_map` rating 1..5 → (label, sample_weight); `tools/train_reranker.py` `clf.fit(sample_weight=…)`; Pytest `tests/test_export_retrieval_dataset.py` (6 rating-weighted tests) — PR #216 |
| 210 | Codebook-editor + style-auswahl | HINFÄLLIG durch task-based categorization (#215): kategorien beziehen sich auf den task/forschungsfrage — deduktiv = subkategorien des tasks, induktiv = kreative leistung via `prompts/mayring_induktiv.md`. `pi_categorize` nutzt die kanonischen `prompts/mayring_{deduktiv,induktiv,hybrid}.md` (PR #229). Kein UI-editor mehr nötig. |
| 211 | Pi-Agent als first-class tool-replacement | `claude-plugin/agents/` (subagent-def, später dropped) + 3 spezialisierte MCP-tools `pi_categorize`/`pi_judge_relevance`/`pi_summarize_for_memory` (`src/api/mcp_agent_tools.py`) + CLAUDE.md decision-table; Pytest `tests/test_pi_specialized_tools.py` (14 tests) — PR #218/#223/#229 |
| 213 | Log-Endpoints ohne SSH (Phase A+B+C) | MayringCoder: `GET /admin/logs?service=...&since=...&grep=...` (admin-scope, rate-limit 5/min, secret-redaction) + MCP-tool `live_logs`; Pytest `tests/test_admin_logs.py` (20 tests) — PR #218. app.linn.games: `GET /api/mayring/logs` proxy — PR #345. Phase D (UI-tab) → app.linn.games separat |
