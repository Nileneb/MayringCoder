"""Shared memory search and ingest logic used by server.py and mcp.py."""
from __future__ import annotations

import time as _time
from collections import deque
from typing import Any

from src.memory.ingest import ingest
from src.memory.retrieval import compress_for_prompt, search
from src.memory.schema import Source

# Brain visualization: recent search activations (ring buffer, shared in-process)
_RECENT_ACTIVATIONS: deque[dict] = deque(maxlen=200)


def run_search(
    query: str,
    conn: Any,
    chroma: Any,
    ollama_url: str,
    opts: dict[str, Any],
    char_budget: int = 6000,
    session_compacted: bool = False,
) -> dict[str, Any]:
    """Run hybrid search and compress results. Returns {results, prompt_context}."""
    results = search(
        query=query,
        conn=conn,
        chroma_collection=chroma,
        ollama_url=ollama_url,
        opts=opts,
        session_compacted=session_compacted,
    )
    workspace_id = opts.get("workspace_id", "default")
    _RECENT_ACTIVATIONS.append({
        "workspace_id": workspace_id,
        "query": query,
        "source_ids": [r.source_id for r in results],
        "ts": _time.time(),
    })
    response = {
        "results": [r.to_dict() for r in results],
        "prompt_context": compress_for_prompt(results, char_budget),
        "diagnostics": {
            "vector_stage": opts.get("_vector_diag", "unknown"),
            "candidates": len(results),
        },
    }

    # Inject-effizienz tracking: every search that produces hits also
    # produces a row in context_feedback_log so the "Memory-Effizienz (24h)"
    # card on the dashboard counts hook-injections too. Without this only
    # the legacy MCP-tool path (mcp_memory_tools.py) wrote rows here, so
    # the counter froze the moment everything moved to the hook-path.
    if results:
        try:
            import json as _json
            from datetime import datetime, timezone
            _ids = _json.dumps([r.chunk_id for r in results])
            conn.execute(
                "INSERT INTO context_feedback_log"
                " (trigger_ids,context_text,was_referenced,led_to_retrieval,relevance_score,captured_at)"
                " VALUES (?,?,0,0,0.0,?)",
                (_ids, response["prompt_context"][:2000],
                 datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        except Exception:
            pass  # non-critical; never block the search result

    return response


def run_ingest(
    source_dict: dict[str, Any],
    content: str,
    conn: Any,
    chroma: Any,
    ollama_url: str,
    model: str,
    opts: dict[str, Any],
    workspace_id: str = "default",
) -> dict[str, Any]:
    """Create Source from dict and ingest into memory. Returns ingest result dict."""
    src = Source(
        source_id=source_dict.get("source_id") or Source.make_id(
            source_dict.get("repo", ""), source_dict.get("path", "")
        ),
        source_type=source_dict.get("source_type", "repo_file"),
        repo=source_dict.get("repo", ""),
        path=source_dict.get("path", ""),
        branch=source_dict.get("branch", "main"),
        commit=source_dict.get("commit", ""),
        content_hash=source_dict.get("content_hash", ""),
        visibility=source_dict.get("visibility") or "private",
        org_id=source_dict.get("org_id"),
        user_id=source_dict.get("user_id"),
    )
    return ingest(
        source=src,
        content=content,
        conn=conn,
        chroma_collection=chroma,
        ollama_url=ollama_url,
        model=model,
        opts=opts,
        workspace_id=workspace_id,
    )
