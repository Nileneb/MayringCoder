"""MCP Memory Server for Claude Code.

Provides persistent, local memory via MCP tools over stdio/HTTP transport.

Add to Claude Code MCP settings (.claude/settings.json or user settings):

Option 1: Direct Python venv (development):
{
    "mcpServers": {
        "memory": {
            "command": "/path/to/.venv/bin/python",
            "args": ["-m", "src.api.mcp"],
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
    put                  — ingest content into memory
    get                  — retrieve chunk by ID
    search_memory        — hybrid 4-stage memory search
    invalidate           — deactivate all chunks for a source
    list_by_source       — list chunks for a source
    explain              — explain a chunk (key, scores, reasons)
    reindex              — re-embed and re-upsert chunks to ChromaDB
    feedback             — record usage signal for a chunk
    pi_task              — memory-augmented reasoning (Pi-Agent mit Tool-Calling)
    conversation_ingest  — store conversation turns as summarized memory
    analyze              — start full code analysis pipeline (async, fire-and-forget)
"""

from __future__ import annotations

import base64
import contextvars
import hashlib
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load .env from project root
_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_ROOT / ".env")

# ---------------------------------------------------------------------------
# HTTP transport configuration (read before FastMCP is instantiated)
# ---------------------------------------------------------------------------

_TRANSPORT       = os.getenv("MCP_TRANSPORT", "stdio")
_AUTH_ENABLED    = os.getenv("MCP_AUTH_ENABLED", "false").lower() in ("true", "1", "yes")
_HTTP_PORT       = int(os.getenv("MCP_HTTP_PORT", "8000"))
_HTTP_HOST       = os.getenv("MCP_HTTP_HOST", "0.0.0.0")
# Shared secret for service-to-service calls (app.linn.games → MayringCoder),
# used only by POST /authorize/register-code. NOT a user auth token.
_SERVICE_TOKEN   = os.getenv("MCP_SERVICE_TOKEN", "")

from src.api.jwt_auth import TokenInfo, validate_jwt_token

# Per-request context: TokenInfo derived from Bearer JWT in HTTP mode.
# None → stdio mode → tools fall back to their caller-supplied default.
_TOKEN_CTX: contextvars.ContextVar["TokenInfo | None"] = contextvars.ContextVar(
    "token_info", default=None
)

# Raw JWT-String wird für den Key-Callback gebraucht (signiert identifiziert
# den User gegenüber app.linn.games). Nur in HTTP-Mode gesetzt.
_RAW_JWT_CTX: contextvars.ContextVar["str | None"] = contextvars.ContextVar(
    "raw_jwt", default=None
)


def _current_token_info() -> "TokenInfo | None":
    return _TOKEN_CTX.get(None)


def _current_raw_jwt() -> "str | None":
    return _RAW_JWT_CTX.get(None)


def _effective_workspace_id(caller_default: str = "default") -> str:
    """Return workspace_id from JWT context (HTTP) or caller default (stdio)."""
    info = _TOKEN_CTX.get(None)
    if info is None:
        return caller_default or "default"
    return info.workspace_id


def _enforce_tenant(requested: str | None) -> str | None:
    """For workspace-scoped tools: resolve the effective workspace_id for a request.

    - admin JWT (scope: ["admin"]): may request any workspace or None (cross-workspace).
    - tenant JWT: locked to its own workspace_id; returns that regardless of request.
    - stdio mode: falls through to requested (or None).
    """
    info = _TOKEN_CTX.get(None)
    if info is None:
        return requested
    if info.is_admin:
        return requested
    return info.workspace_id


class _JWTAuthMiddleware:
    """RS256 JWT auth for MCP HTTP transport.

    Token via: Authorization: Bearer <jwt>  or  X-Auth-Token: <jwt>
    Admin access: JWT claim `scope: ["admin"]`.
    """

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        if not _AUTH_ENABLED:
            _TOKEN_CTX.set(None)
            _RAW_JWT_CTX.set(None)
            await self._app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
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

        info = validate_jwt_token(token)
        if info is None:
            await self._send_401(send, "Invalid or expired token")
            return

        _TOKEN_CTX.set(info)
        _RAW_JWT_CTX.set(token)
        scope["workspace_id"] = info.workspace_id
        await self._app(scope, receive, send)

    @staticmethod
    async def _send_401(send: Any, message: str) -> None:
        body = message.encode()
        metadata_url = f"{_OAUTH_BASE_URL}/.well-known/oauth-authorization-server"
        www_auth = f'Bearer realm="{_OAUTH_BASE_URL}", resource_metadata="{metadata_url}"'
        await send({
            "type": "http.response.start", "status": 401,
            "headers": [
                [b"content-type", b"text/plain; charset=utf-8"],
                [b"content-length", str(len(body)).encode()],
                [b"www-authenticate", www_auth.encode()],
            ],
        })
        await send({"type": "http.response.body", "body": body})


from src.memory.retrieval import invalidate_query_cache
from src.memory.schema import Chunk, Source, make_memory_key, source_fingerprint
from src.api.memory_service import run_ingest as _run_ingest, run_search as _run_search
from src.memory.store import (
    add_feedback,
    deactivate_chunks_by_source,
    get_active_chunk_count,
    get_chunk,
    get_chunks_by_source,
    get_source,
    get_source_count,
    kv_get,
    log_ingestion_event,
)
from src.api.dependencies import get_conn as _get_conn, get_chroma as _get_chroma

mcp = FastMCP(
    "memory",
    host=_HTTP_HOST,
    port=_HTTP_PORT,
)

_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
_MODEL = os.environ.get("OLLAMA_MODEL", "")

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
        # Tenant tokens are locked to their own workspace_id; admin tokens may override.
        ws = _enforce_tenant(workspace_id) or workspace_id
        result = _run_ingest(
            source, content, _get_conn(), _get_chroma(),
            _OLLAMA_URL, _MODEL, {"categorize": categorize, "log": log},
            ws,
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
        # Tenant tokens: workspace_id is forced to the tenant's own workspace.
        # Admin tokens: workspace_id stays as requested (None → cross-workspace).
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
        result = _run_search(query, _get_conn(), _get_chroma(), _OLLAMA_URL,
                             opts, char_budget, session_compacted=compacted)
        # Auto-log neutral usage signal for every returned chunk so the feedback
        # table gets populated even without explicit feedback() calls.
        try:
            conn = _get_conn()
            for r in result.get("results", []):
                cid = r.get("chunk_id") if isinstance(r, dict) else getattr(r, "chunk_id", None)
                if cid:
                    add_feedback(conn, cid, "neutral", {"auto": True, "query": query[:80]})
        except Exception:
            pass
        return result
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
        invalidate_query_cache()
        return {"chunk_id": chunk_id, "recorded": True}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 9: memory.pi_task
# ---------------------------------------------------------------------------

@mcp.tool()
def pi_task(
    task: str,
    repo_slug: str | None = None,
    system_prompt: str | None = None,
    timeout: float = 180.0,
    workspace_id: str | None = None,
) -> dict:
    """Run a free-form task using the Pi-Agent with memory-augmented reasoning.

    The Pi-Agent searches workspace memory for relevant context, injects it
    into the system prompt (ambient context), and can issue up to 5
    search_memory tool calls during reasoning. Requires Ollama to be reachable.

    Args:
        task: Free-form task or question, e.g. "Welche Forschungsmethoden sind bekannt?"
        repo_slug: Repository slug to scope memory retrieval (e.g. "app.linn.games")
        system_prompt: Custom system prompt (default: Pi-Agent task prompt)
        timeout: Per-request HTTP timeout in seconds (default: 180)
        workspace_id: Tenant namespace for memory scope (default: from JWT)

    Returns:
        {result: str, workspace_id: str} or {error: str}
    """
    try:
        from src.agents.pi import run_task_with_memory
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        result = run_task_with_memory(
            task=task,
            ollama_url=_OLLAMA_URL,
            model=_MODEL,
            repo_slug=repo_slug,
            system_prompt=system_prompt,
            timeout=timeout,
        )
        return {"result": result, "workspace_id": ws}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 10: memory.conversation_ingest
# ---------------------------------------------------------------------------

@mcp.tool()
def conversation_ingest(
    turns: list[dict],
    session_id: str,
    workspace_slug: str = "default",
    presumarized: str | None = None,
) -> dict:
    """Ingest a batch of conversation turns into workspace memory.

    Summarizes (via Ollama) and stores as a conversation_summary source.
    Dedup: same session_id + same content is skipped automatically.

    Args:
        turns: List of {"role": "user"|"assistant", "content": str, "timestamp": str}
        session_id: Unique session identifier (used for dedup)
        workspace_slug: Workspace namespace (default: "default")
        presumarized: Pre-written summary to skip Ollama summarization

    Returns:
        {workspace_id, source_id, chunk_ids, indexed, deduped, skipped}
    """
    try:
        import hashlib
        from tools.ingest_conversations import _summarize as _summarize_turns

        if not turns:
            return {"error": "turns must not be empty"}

        ws = _enforce_tenant(workspace_slug) or workspace_slug
        batch_key = f"{session_id}:{len(turns)}:{turns[-1].get('timestamp', '')}"
        content_hash = "sha256:" + hashlib.sha256(batch_key.encode()).hexdigest()[:16]
        source_id = f"conversation:{workspace_slug}:{session_id[:16]}"

        summary = presumarized or _summarize_turns(turns, "", _OLLAMA_URL, _MODEL)

        if not summary or not summary.strip():
            return {"workspace_id": ws, "source_id": source_id,
                    "chunk_ids": [], "indexed": False, "skipped": True}

        first_ts = (turns[0].get("timestamp") or "")[:10]
        source_dict = {
            "source_id": source_id,
            "source_type": "conversation_summary",
            "repo": workspace_slug,
            "path": f"conversations/{session_id[:16]}",
            "content_hash": content_hash,
        }
        result = _run_ingest(
            source_dict, summary, _get_conn(), _get_chroma(),
            _OLLAMA_URL, _MODEL, {"categorize": False}, ws,
        )
        return {"workspace_id": ws, **result}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 11: memory.analyze (async — use job_status to poll)
# ---------------------------------------------------------------------------

@mcp.tool()
def analyze(
    repo: str,
    full: bool = False,
    no_pi: bool = False,
    budget: int | None = None,
    workspace_id: str | None = None,
) -> dict:
    """Start a full code analysis pipeline for a repository (async).

    Runs src.pipeline as a subprocess. Returns immediately with a pid.
    Analysis output is written to reports/. No job_id — fire and forget.

    Args:
        repo: Repository URL or local path (e.g. "https://github.com/org/repo")
        full: Force full re-analysis (ignore incremental cache)
        no_pi: Disable Pi-Agent (faster, no memory-augmented analysis)
        budget: Max files to analyse (None = no limit)
        workspace_id: Tenant namespace (default: from JWT)

    Returns:
        {status: "started", pid: int, repo: str} or {error: str}
    """
    try:
        import subprocess
        import sys

        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        cmd = [sys.executable, "-m", "src.pipeline", "--repo", repo, "--workspace-id", ws]
        if full:
            cmd.append("--full")
        if no_pi:
            cmd.append("--no-pi")
        if budget is not None:
            cmd.extend(["--budget", str(budget)])

        proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).parent.parent.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return {"status": "started", "pid": proc.pid, "repo": repo, "workspace_id": ws}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def duel(
    task: str,
    model_a: str,
    model_b: str,
    repo_slug: str | None = None,
    judge: bool = True,
    judge_model: str | None = None,
    no_memory_baseline: bool = False,
    timeout: float = 180.0,
    workspace_id: str | None = None,
) -> dict:
    """Run the same task on two models and get an automatic verdict.

    Both models run through the Pi-Agent with full memory injection.
    An auto-judge LLM rates both answers (0-10) and picks a winner.

    Args:
        task: The task/question both models should answer
        model_a: First model name (e.g. "gemma4:e4b")
        model_b: Second model name (e.g. "qwen3:2b")
        repo_slug: Optional repo scope for memory search
        judge: Whether to auto-judge both answers (default True)
        judge_model: Model for judging (default: server OLLAMA_MODEL)
        no_memory_baseline: Also run both models WITHOUT memory for comparison
        timeout: Per-model timeout in seconds
        workspace_id: Tenant namespace (default: from JWT)

    Returns:
        Job dict with job_id — poll /jobs/{id} for result_a, result_b, verdict
    """
    import httpx
    ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
    _api = os.getenv("MAYRING_API_URL", "http://localhost:8090").rstrip("/")
    _jwt = _current_raw_jwt()
    headers = {"Authorization": f"Bearer {_jwt}"} if _jwt else {}
    try:
        resp = httpx.post(
            f"{_api}/duel",
            json={
                "task": task,
                "model_a": model_a,
                "model_b": model_b,
                "repo_slug": repo_slug,
                "judge": judge,
                "judge_model": judge_model,
                "no_memory_baseline": no_memory_baseline,
                "timeout": timeout,
            },
            headers=headers,
            timeout=30.0,
        )
        resp.raise_for_status()
        return {**resp.json(), "workspace_id": ws}
    except Exception as exc:
        return {"error": str(exc), "workspace_id": ws}


@mcp.tool()
def benchmark_tasks(
    model_a: str,
    model_b: str,
    category: str | None = None,
    repo_slug: str | None = None,
    judge_model: str | None = None,
    timeout: float = 180.0,
    workspace_id: str | None = None,
) -> dict:
    """Run the predefined task suite on two models and get a quality comparison.

    Tasks are defined in benchmarks/task_suite.yaml (categories: context_injection,
    pico, code_review, conversation_summary). Each task is scored by keyword hits
    and an auto-judge LLM.

    Args:
        model_a: First model name
        model_b: Second model name
        category: Filter by category (None = all tasks)
        repo_slug: Optional repo scope for memory search
        judge_model: Model used for judging (default: server OLLAMA_MODEL)
        timeout: Per-task per-model timeout in seconds
        workspace_id: Tenant namespace (default: from JWT)

    Returns:
        Report with wins_a, wins_b, avg_score_a, avg_score_b, per-task results
    """
    import httpx
    ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
    _api = os.getenv("MAYRING_API_URL", "http://localhost:8090").rstrip("/")
    _jwt = _current_raw_jwt()
    headers = {"Authorization": f"Bearer {_jwt}"} if _jwt else {}
    try:
        resp = httpx.post(
            f"{_api}/benchmark/tasks",
            json={
                "model_a": model_a,
                "model_b": model_b,
                "category": category,
                "repo_slug": repo_slug,
                "judge_model": judge_model,
                "timeout": timeout,
            },
            headers=headers,
            timeout=600.0,
        )
        resp.raise_for_status()
        return {**resp.json(), "workspace_id": ws}
    except Exception as exc:
        return {"error": str(exc), "workspace_id": ws}


# ---------------------------------------------------------------------------
# OAuth 2.0 / PKCE — for Claude Web MCP connector
# ---------------------------------------------------------------------------

_OAUTH_BASE_URL = os.getenv("MCP_OAUTH_BASE_URL", "https://mcp.linn.games")

# In-memory auth-code store (TTL = 5 min, cleared on exchange)
_auth_codes: dict[str, dict[str, Any]] = {}


def _pkce_verify(verifier: str, challenge: str, method: str) -> bool:
    if method == "S256":
        digest = hashlib.sha256(verifier.encode()).digest()
        expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        return secrets.compare_digest(expected, challenge)
    return secrets.compare_digest(verifier, challenge)


async def _oauth_metadata(request: Any) -> Any:
    from starlette.responses import JSONResponse
    base = _OAUTH_BASE_URL
    return JSONResponse({
        "issuer": base,
        "authorization_endpoint": f"{base}/authorize",
        "token_endpoint": f"{base}/token",
        "registration_endpoint": f"{base}/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
    })


async def _oauth_authorize(request: Any) -> Any:
    """GET /authorize — redirect to app.linn.games for session-based auth."""
    from starlette.requests import Request
    from starlette.responses import RedirectResponse
    req: Request = request

    params = dict(req.query_params)
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    return RedirectResponse(
        f"https://app.linn.games/mcp/authorize?{qs}",
        status_code=302,
    )


async def _oauth_register_code(request: Any) -> Any:
    """POST /authorize/register-code — internal endpoint for app.linn.games to register auth codes."""
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    req: Request = request

    # Verify shared service-to-service token (app.linn.games → MayringCoder)
    auth_header = req.headers.get("authorization", "")
    token = auth_header[7:].strip() if auth_header.lower().startswith("bearer ") else ""
    if not _SERVICE_TOKEN or not secrets.compare_digest(token, _SERVICE_TOKEN):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    code = str(body.get("code", "")).strip()
    if not code:
        return JSONResponse({"error": "code required"}, status_code=400)

    _auth_codes[code] = {
        "token":                 str(body.get("token", "")),
        "workspace_id":          str(body.get("workspace_id", "default")),
        "code_challenge":        str(body.get("code_challenge", "")),
        "code_challenge_method": str(body.get("code_challenge_method", "S256")),
        "redirect_uri":          str(body.get("redirect_uri", "")),
        "state":                 str(body.get("state", "")),
        "expires_at":            time.time() + 300,
    }
    return JSONResponse({"ok": True})


async def _oauth_token(request: Any) -> Any:
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    req: Request = request

    ct = req.headers.get("content-type", "")
    if "application/json" in ct:
        body = await req.json()
    else:
        form = await req.form()
        body = dict(form)

    grant_type = body.get("grant_type", "")
    if grant_type != "authorization_code":
        return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

    code = str(body.get("code", ""))
    code_verifier = str(body.get("code_verifier", ""))
    redirect_uri = str(body.get("redirect_uri", ""))

    entry = _auth_codes.pop(code, None)
    if not entry:
        return JSONResponse({"error": "invalid_grant", "error_description": "Unknown code"}, 400)
    if time.time() > entry["expires_at"]:
        return JSONResponse({"error": "invalid_grant", "error_description": "Code expired"}, 400)
    if redirect_uri and redirect_uri != entry["redirect_uri"]:
        return JSONResponse({"error": "invalid_grant", "error_description": "redirect_uri mismatch"}, 400)
    if code_verifier and not _pkce_verify(code_verifier, entry["code_challenge"], entry["code_challenge_method"]):
        return JSONResponse({"error": "invalid_grant", "error_description": "PKCE verification failed"}, 400)

    return JSONResponse({
        "access_token": entry["token"],
        "token_type": "bearer",
        "workspace_id": entry["workspace_id"],
    })


async def _oauth_register(request: Any) -> Any:
    from starlette.responses import JSONResponse
    client_id = secrets.token_urlsafe(16)
    try:
        body = await request.json()
    except Exception:
        body = {}
    return JSONResponse({"client_id": client_id, "client_secret": None, **body}, status_code=201)


# ---------------------------------------------------------------------------
# Path normalizer — Claude Web sends POST / after OAuth, FastMCP expects /mcp
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Public landing page (GET /)
# ---------------------------------------------------------------------------

_LANDING_HTML_PATH = Path(__file__).parent / "templates" / "landing.html"


async def _landing_page(request: Any) -> Any:
    from starlette.responses import HTMLResponse
    try:
        html = _LANDING_HTML_PATH.read_text(encoding="utf-8")
    except OSError:
        return HTMLResponse("<h1>MayringCoder</h1>", status_code=200)
    return HTMLResponse(
        html,
        status_code=200,
        headers={"Cache-Control": "public, max-age=300"},
    )


class _PathNormMiddleware:
    """Rewrite / and /sse → /mcp so the streamable_http_app Route('/mcp') matches.

    GET / is routed to the landing page before this middleware runs (via an
    explicit Starlette Route), so only non-GET requests to / end up rewritten
    here — that's what Claude Web's MCP POST calls need.
    """

    _REWRITE = frozenset(("/", "/sse", ""))

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if (
            scope.get("type") == "http"
            and scope.get("path", "/") in self._REWRITE
            and scope.get("method", "") != "GET"
        ):
            scope = {**scope, "path": "/mcp", "raw_path": b"/mcp"}
        await self._app(scope, receive, send)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if _TRANSPORT in ("http", "sse"):
        import uvicorn
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route

        _mcp_http_app = mcp.streamable_http_app()
        _mcp_asgi = _PathNormMiddleware(_mcp_http_app)
        _inner = _JWTAuthMiddleware(_mcp_asgi)

        _auth_label = "rs256-jwt" if _AUTH_ENABLED else "disabled"
        print(
            f"[mcp-memory] HTTP/streamable on {_HTTP_HOST}:{_HTTP_PORT}"
            f" | auth={_auth_label}"
            f" | oauth={_OAUTH_BASE_URL}"
        )

        app = Starlette(
            routes=[
                Route("/.well-known/oauth-authorization-server", _oauth_metadata),
                Route("/.well-known/oauth-protected-resource", _oauth_metadata),
                Route("/.well-known/oauth-protected-resource/sse", _oauth_metadata),
                Route("/register", _oauth_register, methods=["POST"]),
                Route("/authorize", _oauth_authorize, methods=["GET"]),
                Route("/authorize/register-code", _oauth_register_code, methods=["POST"]),
                Route("/token", _oauth_token, methods=["POST"]),
                Route("/", _landing_page, methods=["GET"]),
                Mount("/", app=_inner),
            ],
            lifespan=_mcp_http_app.router.lifespan_context,
        )
        uvicorn.run(app, host=_HTTP_HOST, port=_HTTP_PORT)
    else:
        mcp.run()  # stdio transport (local dev)


if __name__ == "__main__":
    main()
