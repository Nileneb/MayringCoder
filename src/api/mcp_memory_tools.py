"""Memory tools registered onto the FastMCP instance."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.api.mcp_auth import _enforce_tenant
from src.api.dependencies import get_conn as _get_conn, get_chroma as _get_chroma
from src.api.memory_service import run_search as _run_search
from src.memory.retrieval import invalidate_query_cache
from src.memory.store import (
    add_feedback,
    deactivate_chunks_by_source,
    log_ingestion_event,
)


def register_memory_tools(mcp: FastMCP) -> None:

    @mcp.tool()
    def search_memory(
        query: str,
        repo: str | None = None,
        categories: list[str] | None = None,
        source_type: str | None = None,
        top_k: int = 8,
        include_text: bool = True,
        source_affinity: str | None = None,
        char_budget: int = 6000,
        compacted: bool = False,
        workspace_id: str | None = None,
    ) -> dict:
        """Hybrid 4-stage memory search (scope filter → symbolic → vector → rerank).

        Args:
            query: Natural language search query
            repo: Filter by repository (e.g. "owner/name")
            categories: Filter by any of these Mayring category labels
            source_type: Filter by source type (e.g. "repo_file")
            top_k: Maximum number of results (default 8)
            include_text: Include chunk text in results (default True)
            source_affinity: source_id to boost in affinity scoring
            char_budget: Max chars for prompt_context output
            compacted: Set True after /compact to boost conversation_summary chunks
            workspace_id: Tenant namespace filter (None = no filter)

        Returns:
            {results: list[RetrievalRecord], prompt_context: str}
        """
        try:
            ws = _enforce_tenant(workspace_id)
            opts = {
                "repo": repo,
                "categories": categories,
                "source_type": source_type,
                "top_k": top_k,
                "include_text": include_text,
                "source_affinity": source_affinity,
                "workspace_id": ws,
            }
            result = _run_search(query, _get_conn(), _get_chroma(), None,
                                 opts, char_budget, session_compacted=compacted)
            try:
                import json as _json
                from datetime import datetime, timezone
                _ids = _json.dumps([r.get("chunk_id", "") for r in result.get("results", [])])
                _conn = _get_conn()
                _conn.execute(
                    "INSERT INTO context_feedback_log"
                    " (trigger_ids,context_text,was_referenced,led_to_retrieval,relevance_score,captured_at)"
                    " VALUES (?,?,0,0,0.0,?)",
                    (_ids, result.get("prompt_context", "")[:2000],
                     datetime.now(timezone.utc).isoformat()),
                )
                _conn.commit()
            except Exception:
                pass  # non-critical; never block the search result
            return result
        except Exception as exc:
            return {"error": str(exc), "results": [], "prompt_context": ""}

    @mcp.tool()
    def invalidate(source_id: str) -> dict:
        """Deactivate all memory chunks for a source (use when source is deleted/outdated).

        Returns:
            {source_id, deactivated_count}
        """
        try:
            count = deactivate_chunks_by_source(_get_conn(), source_id)
            log_ingestion_event(_get_conn(), source_id, "invalidated", {"count": count})
            invalidate_query_cache()
            return {"source_id": source_id, "deactivated_count": count}
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def feedback(
        chunk_id: str,
        signal: str,
        metadata: dict | None = None,
    ) -> dict:
        """Record usage feedback for a memory chunk (training signal).

        Args:
            chunk_id: The chunk that was used
            signal: "positive" | "negative" | "1"–"5" (star rating)
            metadata: Optional context (e.g. {"query": "...", "task": "..."})

        Returns:
            {chunk_id, recorded: True}
        """
        try:
            add_feedback(_get_conn(), chunk_id, signal, metadata or {})
            invalidate_query_cache()
            return {"chunk_id": chunk_id, "recorded": True}
        except Exception as exc:
            return {"error": str(exc)}
