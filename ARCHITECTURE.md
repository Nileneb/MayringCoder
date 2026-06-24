# MayringCoder — Architektur & Ökosystem

LLM-gestützter Code-Analyzer (Mayring Qualitative Inhaltsanalyse) + persistente
Memory-Schicht (MCP). Das System ist auf **vier Repos** aufgeteilt (Refactor #267,
Ports & Adapters) — diese Datei ist die maßgebliche Karte über alle vier.

## Die vier Repos

| Repo | Rolle | Eingebunden als |
|---|---|---|
| **MayringCoder** (master) | **Host / Orchestrator.** `src/` (api, analysis, workflows, wiki_v2, training), Entrypoints, Docker, Web UI. Verdrahtet die Core-Ports mit echten Implementierungen. | dieses Repo |
| **mayring-core** (main) | **Core / Ports.** Reine, host-freie Bausteine: `config`, `ollama_client`, `model_router`, `model_selector`, **`providers` (DI-Boundary)**, `memory/`, `identity/`, `llm/`. Importiert **nie** `src.*`. | Git-Submodule `vendor/mayring-core` |
| **mayring-pi-agent** (main) | **Pi-Agent Microservice.** Tool-calling-Loop mit Memory-Zugriff, Vision. | Git-Submodule `vendor/mayring-pi-agent` |
| **mayring-claude-plugin** (main) | **Claude-Code-Plugin.** Hooks (SessionStart/Stop/PostCompact), Skills, MCP-stdio-Bridge. | eigenständig, **nicht** vendored |

`mayring-core` und `mayring-pi-agent` werden in den jeweiligen Sibling-Repos
(`~/Desktop/mayring-core`, `~/Desktop/mayring-pi-agent`) entwickelt, gepusht und
dann hier per Submodule-Pointer-Bump nachgezogen. `pip install -e vendor/*`
macht sie als `mayring_core` / `mayring_pi_agent` importierbar.

## Schichten (im Host `src/`)

```
┌─ Layer 1 · HTTP/MCP Boundary ── src/api/ ───────────────────────────┐
│  server.py (FastAPI :8080) · mcp.py (FastMCP :8090) · web_ui.py      │
│  (Gradio :7860) · routes/ · auth/JWT/OIDC                            │
├─ Layer 2 · Business Logic & Adapters ───────────────────────────────┤
│  analysis/ (LLM+RAG, Provider-Impls) · workflows/ (Batch-Orchestr.)  │
│  wiki_v2/ (Knowledge-Graph) · training/ (Coaching-Daten)             │
│  provider_setup.py ── registriert Host-Impls in den Core-Ports       │
│  embed_facade.py ── Schicht 3: verifiziertes Embedding (#365)        │
├─ Layer 3 · Core / Ports ── mayring_core (vendor/mayring-core) ───────┤
│  providers (DI) · memory/ · identity/ · ollama_client · config       │
├─ Layer 4 · Extern ──────────────────────────────────────────────────┤
│  Ollama · ChromaDB · SQLite · mayring-pi-agent (:8091)               │
└─────────────────────────────────────────────────────────────────────┘
```

## Die DI-Grenze (so bleibt Core host-frei)

`mayring_core.providers` ist die Dependency-Injection-Boundary. Core ruft
abstrakte Accessoren; der Host registriert die echten Funktionen **einmal beim
Boot** über `src/provider_setup.py::setup_providers()`:

| Port | Host-Implementierung |
|---|---|
| `embed_texts` | `src/analysis/context_rag.py::_embed_texts` (cached + batched) |
| `generate_text` | `src/analysis/analyzer.py::_ollama_generate` (Ollama-Streaming) |
| `vision_*` | `mayring_pi_agent.vision` |
| `record_proposal` | `src/api/routes/codebooks.py::record_proposal` |

So importiert `mayring_core` **nie** `src.*`. Standalone-Core-Installs ohne
`register_*` nutzen Thin-Defaults auf `ollama_client` (embed/generate); Vision
und `record_proposal` haben keinen sicheren Default → `ProviderNotConfigured` bis
registriert.

## Entrypoints

| Kommando | Was | Wann |
|---|---|---|
| `python -m src.main` | **Prod:** startet FastAPI + MCP + Pi-Agent + Web UI parallel. `docker/Dockerfile` `CMD`. | Deploy |
| `python -m src.cli` | **CLI/Direkt-Executor** der Analyse-Pipeline. Wird auch als Subprozess aus `src/api/job_queue.py` gestartet. | lokal / Job-Dispatch |
| `bash run.sh` | 3-Stufen-Dev-Pipeline (overview → turbulence → analyze) über `python -m src.pipeline`. | lokale Analyse |

`src/pipeline.py` ist ein bewusster Backward-Compat-Shim → leitet zu `src/cli.py`.

## Deploy

Prod-Stack lebt im Repo **app.linn.games** (`docker-compose.mayring.yml`,
`deploy-mayring`-Workflow, nginx-Proxies), läuft auf u-server. Ollama-Routing
intern über den GPU-Host (`192.168.178.11:11434`), nie auf u-server (keine GPU).
Details: `DEPLOYMENT.md`.
