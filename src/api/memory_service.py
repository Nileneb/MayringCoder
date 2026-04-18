"""Shared memory search and ingest logic used by server.py and mcp.py."""
from __future__ import annotations

from typing import Any

from src.memory.ingest import ingest
from src.memory.retrieval import compress_for_prompt, search
from src.memory.schema import Source


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
    return {
        "results": [r.to_dict() for r in results],
        "prompt_context": compress_for_prompt(results, char_budget),
    }


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
