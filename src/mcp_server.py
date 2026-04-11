"""MCP Memory Server for Claude Code.

Provides persistent, local memory via 8 MCP tools over stdio transport.

Add to Claude Code MCP settings (.claude/settings.json or user settings):

Option 1: Direct Python venv (development):
{
    "mcpServers": {
        "memory": {
            "command": "/path/to/.venv/bin/python",
            "args": ["-m", "src.mcp_server"],
            "cwd": "/path/to/MayringCoder"
        }
    }
}

Option 2: Docker (production):
{
    "mcpServers": {
        "memory": {
            "command": "docker",
            "args": ["run", "-i", "--rm",
                     "-v", "cache:/app/cache",
                     "--env-file", "/path/to/.env",
                     "mayrингcoder-mcp"]
        }
    }
}

Build Docker image: docker build -t mayrингcoder-mcp .
Or with compose:  docker-compose up -d mcp-memory

Tools exposed (wire name = mcp__memory__<name>):
    put              — ingest content into memory
    get              — retrieve chunk by ID
    search           — hybrid 4-stage memory search
    invalidate       — deactivate all chunks for a source
    list_by_source   — list chunks for a source
    explain          — explain a chunk (key, scores, reasons)
    reindex          — re-embed and re-upsert chunks to ChromaDB
    feedback         — record usage signal for a chunk
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load .env from project root
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

# ---------------------------------------------------------------------------
# HTTP transport configuration (read before FastMCP is instantiated)
# ---------------------------------------------------------------------------

_TRANSPORT    = os.getenv("MCP_TRANSPORT", "stdio")
_AUTH_ENABLED = os.getenv("MCP_AUTH_ENABLED", "false").lower() in ("true", "1", "yes")
_AUTH_SECRET  = os.getenv("MCP_AUTH_SECRET", "")
_AUTH_TOKEN   = os.getenv("MCP_AUTH_TOKEN", "")  # legacy backward compat
_HTTP_PORT    = int(os.getenv("MCP_HTTP_PORT", "8000"))
_HTTP_HOST    = os.getenv("MCP_HTTP_HOST", "0.0.0.0")


class _XAuthMiddleware:
    """Rejects HTTP requests missing a valid X-Auth-Token header.
    Skips auth if MCP_AUTH_TOKEN is empty (local/dev mode).
    Kept for backward compatibility — new code should use _JWTAuthMiddleware.
    """

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http" and _AUTH_TOKEN:
            headers = dict(scope.get("headers", []))
            token = headers.get(b"x-auth-token", b"").decode()
            if token != _AUTH_TOKEN:
                body = b"Unauthorized"
                await send({
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        [b"content-type", b"text/plain; charset=utf-8"],
                        [b"content-length", str(len(body)).encode()],
                    ],
                })
                await send({"type": "http.response.body", "body": body})
                return
        await self._app(scope, receive, send)


class _JWTAuthMiddleware:
    """JWT authentication middleware for MCP HTTP transport.

    Supports three modes:
    - JWT (MCP_AUTH_ENABLED=true + MCP_AUTH_SECRET set): validates HS256 JWT,
      extracts workspace_id from payload.
    - Legacy static (MCP_AUTH_ENABLED=true + MCP_AUTH_TOKEN set): compares
      X-Auth-Token header value, sets workspace_id="default".
    - Disabled (MCP_AUTH_ENABLED=false, default): passes all requests through
      with workspace_id="default".

    Token can be passed as:
      X-Auth-Token: <token>
      Authorization: Bearer <token>
    """

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        if not _AUTH_ENABLED:
            scope["workspace_id"] = "default"
            await self._app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))

        # Extract token: X-Auth-Token first, then Authorization: Bearer
        token: str = ""
        raw = headers.get(b"x-auth-token", b"").decode().strip()
        if raw:
            token = raw
        else:
            auth_header = headers.get(b"authorization", b"").decode().strip()
            if auth_header.lower().startswith("bearer "):
                token = auth_header[7:].strip()

        if not token:
            await self._send_401(send, "Missing authentication token")
            return

        if _AUTH_SECRET:
            # JWT validation (lazy import to avoid ImportError when not used)
            import jwt  # type: ignore[import]
            try:
                payload = jwt.decode(token, _AUTH_SECRET, algorithms=["HS256"])
                scope["workspace_id"] = payload.get("workspace_id", "default")
            except jwt.ExpiredSignatureError:
                await self._send_401(send, "Token expired")
                return
            except jwt.InvalidTokenError:
                await self._send_401(send, "Invalid token")
                return
        elif _AUTH_TOKEN:
            # Legacy static token comparison
            if token != _AUTH_TOKEN:
                await self._send_401(send, "Unauthorized")
                return
            scope["workspace_id"] = "default"
        else:
            await self._send_401(send, "No auth secret configured")
            return

        await self._app(scope, receive, send)

    @staticmethod
    async def _send_401(send: Any, message: str) -> None:
        body = message.encode()
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                [b"content-type", b"text/plain; charset=utf-8"],
                [b"content-length", str(len(body)).encode()],
            ],
        })
        await send({"type": "http.response.body", "body": body})


from src.memory_ingest import get_or_create_chroma_collection, ingest
from src.memory_retrieval import compress_for_prompt, invalidate_query_cache, search
from src.memory_schema import Chunk, Source, make_memory_key, source_fingerprint
from src.memory_store import (
    add_feedback,
    deactivate_chunks_by_source,
    get_active_chunk_count,
    get_chunk,
    get_chunks_by_source,
    get_source,
    get_source_count,
    init_memory_db,
    kv_get,
    log_ingestion_event,
)

mcp = FastMCP(
    "memory",
    host=_HTTP_HOST,
    port=_HTTP_PORT,
)

_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
_MODEL = os.environ.get("OLLAMA_MODEL", "")

# ---------------------------------------------------------------------------
# Lazy singletons — initialized on first tool call to avoid startup latency
# ---------------------------------------------------------------------------

_conn = None
_chroma = None


def _get_conn():
    global _conn
    if _conn is None:
        _conn = init_memory_db()
    return _conn


def _get_chroma():
    global _chroma
    if _chroma is None:
        _chroma = get_or_create_chroma_collection()
    return _chroma


# ---------------------------------------------------------------------------
# Tool 1: memory.put
# ---------------------------------------------------------------------------

@mcp.tool()
def put(
    source: dict,
    content: str,
    scope: str = "repo",
    tags: list[str] | None = None,
    categorize: bool = False,
    log: bool = False,
    workspace_id: str = "default",
) -> dict:
    """Ingest content into persistent memory.

    Args:
        source: Dict with fields: source_id, source_type, repo, path,
                branch (opt), commit (opt), content_hash (opt)
        content: Raw text content to ingest and chunk
        scope: Scope label (default: "repo")
        tags: Optional extra tags stored as category_labels on all chunks
        categorize: Run Mayring LLM categorization (requires Ollama)
        log: Write JSONL log entry
        workspace_id: Tenant namespace (default: "default")

    Returns:
        {source_id, chunk_ids, indexed, deduped, superseded}
    """
    try:
        src = Source(
            source_id=source.get("source_id") or Source.make_id(
                source.get("repo", ""), source.get("path", "")
            ),
            source_type=source.get("source_type", "repo_file"),
            repo=source.get("repo", ""),
            path=source.get("path", ""),
            branch=source.get("branch", "main"),
            commit=source.get("commit", ""),
            content_hash=source.get("content_hash", ""),
        )
        opts = {"categorize": categorize, "log": log}
        result = ingest(
            source=src,
            content=content,
            conn=_get_conn(),
            chroma_collection=_get_chroma(),
            ollama_url=_OLLAMA_URL,
            model=_MODEL,
            opts=opts,
            workspace_id=workspace_id,
        )
        invalidate_query_cache()
        return result
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 2: memory.get
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tool 3: memory.search
# ---------------------------------------------------------------------------

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
        workspace_id: Tenant namespace filter (None = no filter, searches all workspaces)

    Returns:
        {results: list[RetrievalRecord], prompt_context: str}
    """
    try:
        opts = {
            "repo": repo,
            "categories": categories,
            "source_type": source_type,
            "top_k": top_k,
            "include_text": include_text,
            "source_affinity": source_affinity,
            "workspace_id": workspace_id,
        }
        results = search(
            query=query,
            conn=_get_conn(),
            chroma_collection=_get_chroma(),
            ollama_url=_OLLAMA_URL,
            opts=opts,
            session_compacted=compacted,
        )
        prompt_context = compress_for_prompt(results, char_budget)
        return {
            "results": [r.to_dict() for r in results],
            "prompt_context": prompt_context,
        }
    except Exception as exc:
        return {"error": str(exc), "results": [], "prompt_context": ""}


# ---------------------------------------------------------------------------
# Tool 4: memory.invalidate
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tool 5: memory.list_by_source
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tool 6: memory.explain
# ---------------------------------------------------------------------------

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

        # Build memory key
        cats = chunk.category_labels[0] if chunk.category_labels else "uncategorized"
        fp = source_fingerprint(chunk.source_id)
        hash_prefix = chunk.text_hash.replace("sha256:", "")[:8]
        memory_key = make_memory_key("repo", cats, fp, hash_prefix)

        # Load source info
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


# ---------------------------------------------------------------------------
# Tool 7: memory.reindex
# ---------------------------------------------------------------------------

@mcp.tool()
def reindex(source_id: str | None = None) -> dict:
    """Re-embed and re-upsert chunks to ChromaDB.

    If source_id is None, reindexes ALL active chunks (can be slow).

    Returns:
        {reindexed_count, errors}
    """
    try:
        from src.context import _embed_texts

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
                    # Lookup workspace_id for this chunk
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


# ---------------------------------------------------------------------------
# Tool 8: memory.feedback
# ---------------------------------------------------------------------------

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
        return {"chunk_id": chunk_id, "recorded": True}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if _TRANSPORT == "http":
        import uvicorn

        # FastMCP exposes its internal ASGI app via streamable_http_app()
        _asgi_app = mcp.streamable_http_app()
        _wrapped = _JWTAuthMiddleware(_asgi_app)
        if _AUTH_ENABLED:
            _auth_mode = "JWT" if _AUTH_SECRET else "static"
        else:
            _auth_mode = "disabled"
        print(
            f"[mcp-memory] HTTP mode on {_HTTP_HOST}:{_HTTP_PORT}"
            f" | auth={_auth_mode}"
        )
        uvicorn.run(_wrapped, host=_HTTP_HOST, port=_HTTP_PORT)
    else:
        mcp.run()  # stdio transport (default for Claude Code)
