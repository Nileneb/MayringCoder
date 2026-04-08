# MCP Memory Tool Contracts

Local MCP server for Claude Code. Transport: stdio (Standard für Claude Code) oder http (Remote/App-Zugriff)

Wire names follow the pattern `mcp__memory__<tool_name>`.

## HTTP-Modus

Für Remote-Zugriff (z. B. von app.linn.games oder unterwegs):

```bash
# Starten
MCP_TRANSPORT=http MCP_AUTH_TOKEN=<secret> docker compose --profile http up -d
```

**Auth-Header:** `X-Auth-Token: <secret>` (kein Bearer — proxy-kompatibel)

**Endpunkt:** `POST http://<host>:8000/mcp`

**Env-Vars:**
- `MCP_TRANSPORT=http` — aktiviert HTTP-Modus
- `MCP_AUTH_TOKEN=` — Shared Secret (leer = kein Auth, nur lokal)
- `MCP_HTTP_PORT=8000` — Port (Standard: 8000)
- `MCP_HTTP_HOST=0.0.0.0` — Bind-Adresse

---

## Server Configuration

Add to Claude Code MCP settings:

```json
{
    "mcpServers": {
        "memory": {
            "command": "/path/to/MayringCoder/.venv/bin/python",
            "args": ["-m", "src.mcp_server"],
            "cwd": "/path/to/MayringCoder"
        }
    }
}
```

---

## Tool: `put` (wire: `mcp__memory__put`)

Ingest content into persistent memory. Chunks the content structurally, optionally categorizes via Mayring LLM, deduplicates, and writes to SQLite + ChromaDB.

### Input

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `source` | object | ✅ | — | Source metadata (see Source Object below) |
| `content` | string | ✅ | — | Raw text to ingest |
| `scope` | string | | `"repo"` | Scope label for the memory key |
| `tags` | string[] | | `null` | Extra category tags |
| `categorize` | bool | | `false` | Run Mayring LLM categorization |
| `log` | bool | | `false` | Write JSONL log entry |

**Source Object:**

| Field | Type | Description |
|---|---|---|
| `source_id` | string | Canonical ID (auto-generated if omitted) |
| `source_type` | string | `"repo_file"` \| `"doc"` \| `"note"` |
| `repo` | string | `"owner/name"` |
| `path` | string | Relative file path |
| `branch` | string | Git branch (default: `"main"`) |
| `commit` | string | Git commit hash |
| `content_hash` | string | SHA256 of raw content |

### Output

```json
{
    "source_id": "repo:owner/name:src/auth.py",
    "chunk_ids": ["chk_9f3a1b2c...", "chk_4d8e2f1a..."],
    "indexed": true,
    "deduped": 0,
    "superseded": 0
}
```

### Errors

```json
{"error": "description of error"}
```

---

## Tool: `get` (wire: `mcp__memory__get`)

Retrieve a specific memory chunk by ID. Checks the in-process KV cache first, then SQLite.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `chunk_id` | string | ✅ | Chunk ID (e.g. `"chk_9f3a1b2c..."`) |

### Output

```json
{
    "chunk_id": "chk_9f3a1b2c...",
    "source_id": "repo:owner/name:src/auth.py",
    "chunk_level": "function",
    "ordinal": 2,
    "text": "def authenticate(user, password): ...",
    "text_hash": "sha256:abc...",
    "summary": "",
    "category_labels": ["auth", "api"],
    "is_active": true,
    "created_at": "2026-04-08T12:00:00+00:00"
}
```

### Errors

```json
{"error": "not found", "chunk_id": "chk_..."}
```

---

## Tool: `search_memory` (wire: `mcp__memory__search_memory`)

4-stage hybrid memory search: SQLite scope filter → symbolic token matching → ChromaDB vector retrieval → weighted re-ranking.

Re-ranking formula: `0.45 × vector + 0.25 × symbolic + 0.15 × recency + 0.15 × source_affinity`

### Input

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | string | ✅ | — | Natural language search query |
| `repo` | string | | `null` | Filter to this repository |
| `categories` | string[] | | `null` | Filter by any of these category labels |
| `source_type` | string | | `null` | Filter by source type |
| `top_k` | int | | `8` | Maximum results |
| `include_text` | bool | | `true` | Include chunk text in results |
| `source_affinity` | string | | `null` | source_id to boost in ranking |
| `char_budget` | int | | `6000` | Max chars for `prompt_context` |

### Output

```json
{
    "results": [
        {
            "chunk_id": "chk_9f3a1b2c...",
            "score_vector": 0.82,
            "score_symbolic": 0.67,
            "score_recency": 0.95,
            "score_source_affinity": 0.0,
            "score_final": 0.76,
            "reasons": ["embedding_similarity", "token_overlap"],
            "source_id": "repo:owner/name:src/auth.py",
            "text": "...",
            "summary": "",
            "category_labels": ["auth"]
        }
    ],
    "prompt_context": "## Memory Context\n\n- [auth] repo:owner/name:src/auth.py\n  def authenticate..."
}
```

---

## Tool: `invalidate` (wire: `mcp__memory__invalidate`)

Deactivate all memory chunks for a source. Use when a file is deleted or no longer relevant.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `source_id` | string | ✅ | Source to invalidate |

### Output

```json
{
    "source_id": "repo:owner/name:src/auth.py",
    "deactivated_count": 3
}
```

---

## Tool: `list_by_source` (wire: `mcp__memory__list_by_source`)

List all memory chunks for a given source.

### Input

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `source_id` | string | ✅ | — | Source to list |
| `active_only` | bool | | `true` | Only return active chunks |

### Output

```json
{
    "source_id": "repo:owner/name:src/auth.py",
    "chunks": [ { "chunk_id": "...", ... } ],
    "count": 3
}
```

---

## Tool: `explain` (wire: `mcp__memory__explain`)

Explain a memory chunk: its canonical key, origin, category, and versioning status.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `chunk_id` | string | ✅ | Chunk to explain |

### Output

```json
{
    "chunk_id": "chk_9f3a1b2c...",
    "memory_key": "memory:repo:auth:owner-name-src-auth.py:9f3a1b2c",
    "source_id": "repo:owner/name:src/auth.py",
    "category_labels": ["auth", "api"],
    "chunk_level": "function",
    "ordinal": 2,
    "created_at": "2026-04-08T12:00:00+00:00",
    "is_active": true,
    "superseded_by": null,
    "quality_score": 0.0,
    "source": {
        "source_id": "repo:owner/name:src/auth.py",
        "repo": "owner/name",
        "path": "src/auth.py",
        "branch": "main",
        "commit": "abc123"
    }
}
```

---

## Tool: `reindex` (wire: `mcp__memory__reindex`)

Re-embed and re-upsert chunks to ChromaDB. Use after embedding model changes or ChromaDB corruption.

### Input

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `source_id` | string | | `null` | Reindex only this source; `null` = all active chunks |

### Output

```json
{
    "reindexed_count": 42,
    "errors": 0
}
```

---

## Tool: `feedback` (wire: `mcp__memory__feedback`)

Record a usage signal for a chunk. Stored in `chunk_feedback` table for future training or re-ranking.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `chunk_id` | string | ✅ | Chunk that was used |
| `signal` | string | ✅ | `"positive"` \| `"negative"` \| `"neutral"` |
| `metadata` | object | | Optional context (e.g. `{"query": "..."}`) |

### Output

```json
{
    "chunk_id": "chk_9f3a1b2c...",
    "recorded": true
}
```

---

## Memory Key Format

```
memory:{scope}:{category}:{source_fingerprint}:{chunk_hash_prefix}
```

Example:
```
memory:repo:auth:owner-name-src-auth.py:9f3a1b2c
```

## Storage Layout

| Store | Path | Purpose |
|---|---|---|
| SQLite | `cache/memory.db` | Chunk metadata, versioning, feedback, ingestion log |
| ChromaDB | `cache/memory_chroma/` | Embeddings + semantic retrieval |
| JSONL log | `cache/<slug>_memory_log.jsonl` | Opt-in training data |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `""` | Model for categorization (optional) |
