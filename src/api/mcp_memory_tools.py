"""Memory tools registered onto the FastMCP instance."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.api.mcp_auth import _OLLAMA_URL, _MODEL, _enforce_tenant
from src.api.dependencies import get_conn as _get_conn, get_chroma as _get_chroma
from src.api.memory_service import run_ingest as _run_ingest, run_search as _run_search
from src.memory.retrieval import invalidate_query_cache
from src.memory.schema import make_memory_key, source_fingerprint
from src.memory.store import (
    add_feedback,
    deactivate_chunks_by_source,
    get_chunk,
    get_chunks_by_source,
    get_source,
    kv_get,
    log_ingestion_event,
)


def register_memory_tools(mcp: FastMCP) -> None:

    @mcp.tool()
    def get(chunk_id: str) -> dict:
        """Retrieve a specific memory chunk by ID.

        Checks the in-process KV cache first, then SQLite.

        Returns:
            Chunk dict or {"error": "not found"}
        """
        cached = kv_get(chunk_id)
        if cached is not None:
            return cached
        chunk = get_chunk(_get_conn(), chunk_id)
        if chunk is None:
            return {"error": "not found", "chunk_id": chunk_id}
        return chunk.to_dict()

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
            return _run_search(query, _get_conn(), _get_chroma(), _OLLAMA_URL,
                               opts, char_budget, session_compacted=compacted)
        except Exception as exc:
            return {"error": str(exc), "results": [], "prompt_context": ""}

    @mcp.tool()
    def invalidate(source_id: str) -> dict:
        """Deactivate all memory chunks for a source.

        Use when a source file has been deleted or is no longer relevant.

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
    def list_by_source(source_id: str, active_only: bool = True) -> dict:
        """List all memory chunks for a given source.

        Returns:
            {source_id, chunks: list[Chunk.to_dict()], count}
        """
        try:
            chunks = get_chunks_by_source(_get_conn(), source_id, active_only=active_only)
            return {
                "source_id": source_id,
                "chunks": [c.to_dict() for c in chunks],
                "count": len(chunks),
            }
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def explain(chunk_id: str) -> dict:
        """Explain a memory chunk: its origin, key, category, and version.

        Returns:
            {chunk_id, memory_key, source_id, category_labels,
             chunk_level, created_at, is_active, superseded_by, source}
        """
        try:
            chunk = get_chunk(_get_conn(), chunk_id)
            if chunk is None:
                return {"error": "not found", "chunk_id": chunk_id}

            cats = chunk.category_labels[0] if chunk.category_labels else "uncategorized"
            fp = source_fingerprint(chunk.source_id)
            hash_prefix = chunk.text_hash.replace("sha256:", "")[:8]
            memory_key = make_memory_key("repo", cats, fp, hash_prefix)

            source = get_source(_get_conn(), chunk.source_id)
            source_info = source.to_dict() if source else {}

            return {
                "chunk_id": chunk_id,
                "memory_key": memory_key,
                "source_id": chunk.source_id,
                "category_labels": chunk.category_labels,
                "chunk_level": chunk.chunk_level,
                "ordinal": chunk.ordinal,
                "created_at": chunk.created_at,
                "is_active": chunk.is_active,
                "superseded_by": chunk.superseded_by,
                "quality_score": chunk.quality_score,
                "source": source_info,
            }
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def reindex(source_id: str | None = None) -> dict:
        """Re-embed and re-upsert chunks to ChromaDB.

        If source_id is None, reindexes ALL active chunks (can be slow).

        Returns:
            {reindexed_count, errors}
        """
        try:
            from src.analysis.context import _embed_texts

            chroma = _get_chroma()
            conn = _get_conn()

            if source_id:
                chunks = get_chunks_by_source(conn, source_id, active_only=True)
            else:
                rows = conn.execute(
                    "SELECT chunk_id FROM chunks WHERE is_active = 1"
                ).fetchall()
                chunk_ids = [r[0] for r in rows]
                chunks = [c for cid in chunk_ids if (c := get_chunk(conn, cid)) is not None]

            reindexed = 0
            errors = 0

            for chunk in chunks:
                try:
                    emb = _embed_texts([chunk.text[:500]], _OLLAMA_URL)[0]
                    if chroma is not None:
                        _ws_row = conn.execute(
                            "SELECT workspace_id FROM chunks WHERE chunk_id = ?", (chunk.chunk_id,)
                        ).fetchone()
                        _ws_id = _ws_row[0] if _ws_row else "default"
                        chroma.upsert(
                            ids=[chunk.chunk_id],
                            documents=[chunk.text[:500]],
                            embeddings=[emb],
                            metadatas=[{
                                "workspace_id": _ws_id,
                                "source_id": chunk.source_id,
                                "chunk_level": chunk.chunk_level,
                                "category_labels": ",".join(chunk.category_labels),
                                "is_active": 1,
                            }],
                        )
                    reindexed += 1
                except Exception:
                    errors += 1

            invalidate_query_cache()
            return {"reindexed_count": reindexed, "errors": errors}
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
            signal: "positive" | "negative" | "neutral"
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
