# Memory-System Roadmap

Stand: 2026-04-10 — Referenz: `ARCHITECTURE.md` (Target-Architecture)

## Phase 1 — Grundfunktionen (abgeschlossen)

- [x] MCP Tool-Verträge: put, get, search_memory, invalidate, list_by_source, explain, reindex, feedback
- [x] SQLite Memory-DB (`cache/memory.db`) mit sources, chunks, chunk_feedback, ingestion_log
- [x] ChromaDB Embedding-Retrieval (`cache/memory_chroma/`)
- [x] Structural Chunking: Python (AST), JS/TS (Regex), Markdown (Headings), YAML/JSON (Top-Level-Keys)
- [x] Multi-View Indexing für GitHub Issues (fact/impl/decision/entities/full)
- [x] 4-Stufen Hybrid-Search: Scope Filter → Symbolic → Vector → Rerank
- [x] Exact Dedup via text_hash
- [x] KV-Cache (in-process)
- [x] Conversation-Summary Ingestion
- [x] HTTP/SSE Transport mit X-Auth-Token
- [x] Docker-Compose Profile (stdio/http)

## Phase 2 — Semantische Anreicherung (offen)

- [ ] Qwen-basierte induktive Kategorisierung (automatisch, nicht nur per `--memory-categorize`)
- [ ] Cross-Encoder Re-Ranker als 5. Stufe nach dem Hybrid-Reranking
- [ ] Codebook-Auto-Erkennung verfeinern: mehr source_types, bessere Heuristiken
- [ ] Near-Dedup: normalisierter Text-Hash für leicht abweichende Duplikate
- [ ] Chunk-Summaries via LLM (für `compress_for_prompt` bei langen Chunks)
- [ ] Vision-RAG: Bild-Captioning via Qwen2.5-VL → Text-Chunks (Issue #31)

## Phase 3 — Training & Feedback-Loop (offen)

- [ ] Feedback-gewichtetes Re-Ranking: positive/negative Signale beeinflussen score_final
- [ ] Training-Data-Export: Chunk-Paare (Query → relevanter Chunk) als Fine-Tuning-Datensatz
- [ ] Active Learning: Chunks mit niedrigem Quality-Score zur manuellen Review vorschlagen
- [ ] Embedding-Modell-Vergleich: Benchmark verschiedener Modelle auf realen Queries

## Phase 4 — Governance & Skalierung (offen)

- [ ] Retention Policies: automatisches Invalidieren alter Chunks (TTL-basiert)
- [ ] Workspace-Isolation: JWT-Auth + separate Chroma-Collections pro Workspace (Issue #38)
- [ ] Audit-Log: Wer hat wann welche Memory-Operation ausgeführt
- [ ] Rate-Limiting pro Workspace (429 bei Überschreitung)
- [ ] Schema-Versioning: explizite Migrationsskripte statt ALTER TABLE in init

## Architekturentscheidungen

| Entscheidung | Wahl | Begründung |
|---|---|---|
| Embedding-Modell | nomic-embed-text | Gute Balance aus Qualität und Geschwindigkeit, lokal via Ollama |
| Re-Ranker | Weighted Linear (0.45v+0.25s+0.15r+0.15a) | Einfach, erklärbar, keine Extra-Dependency |
| Dedup-Strategie | Exact (sha256) | Near-Dedup erst wenn Exact nicht reicht |
| Chroma-Isolation | Separate Collections pro Workspace | Kein Leaking möglich |
| Auth-Mechanismus | JWT (HS256) | Standard, leichtgewichtig, Workspace-ID als Claim |
