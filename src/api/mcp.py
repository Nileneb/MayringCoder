"""MCP Memory Server for Claude Code.

Provides persistent, local memory via 8 MCP tools over stdio transport.

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

_TRANSPORT    = os.getenv("MCP_TRANSPORT", "stdio")
_AUTH_ENABLED = os.getenv("MCP_AUTH_ENABLED", "false").lower() in ("true", "1", "yes")
_AUTH_MODE    = os.getenv("MCP_AUTH_MODE", "auto")   # "sanctum" | "jwt" | "static" | "none" | "auto"
_AUTH_SECRET  = os.getenv("MCP_AUTH_SECRET", "")
_AUTH_TOKEN   = os.getenv("MCP_AUTH_TOKEN", "")      # legacy static token
_HTTP_PORT    = int(os.getenv("MCP_HTTP_PORT", "8000"))
_HTTP_HOST    = os.getenv("MCP_HTTP_HOST", "0.0.0.0")

# Per-request context: workspace_id derived from Bearer token in HTTP mode.
# Empty string → stdio mode → tools fall back to their caller-supplied default.
_WORKSPACE_CTX: contextvars.ContextVar[str] = contextvars.ContextVar("workspace_id", default="")


def _effective_workspace_id(caller_default: str = "default") -> str:
    """Return workspace_id from token context (HTTP) or caller default (stdio)."""
    ctx = _WORKSPACE_CTX.get("")
    return ctx if ctx else (caller_default or "default")


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
    """Unified auth middleware for MCP HTTP transport.

    Auth modes (MCP_AUTH_MODE env var):
    - "sanctum"  — Laravel Sanctum token "{id}|{plaintext}" validated against
                   app.linn.games PostgreSQL. workspace_id from workspaces table.
    - "jwt"      — HS256 JWT with MCP_AUTH_SECRET; workspace_id from payload.
    - "static"   — compare against MCP_AUTH_TOKEN; workspace_id="default".
    - "none"     — no auth (dev/local only).
    - "auto"     — detect by token format: Sanctum if contains "|", else JWT/static.

    Token via: Authorization: Bearer <token>  or  X-Auth-Token: <token>
    """

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        if not _AUTH_ENABLED or _AUTH_MODE == "none":
            _WORKSPACE_CTX.set("")
            scope["workspace_id"] = "default"
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

        workspace_id = await self._validate_token(token, send)
        if workspace_id is None:
            return  # _validate_token already sent 401

        _WORKSPACE_CTX.set(workspace_id)
        scope["workspace_id"] = workspace_id
        await self._app(scope, receive, send)

    async def _validate_token(self, token: str, send: Any) -> str | None:
        """Validate token and return workspace_id, or None after sending 401."""
        use_sanctum = (
            _AUTH_MODE == "sanctum"
            or (_AUTH_MODE == "auto" and "|" in token)
        )

        if use_sanctum:
            from src.api.sanctum_auth import validate_sanctum_token_full
            info = validate_sanctum_token_full(token)
            if not info:
                await self._send_401(send, "Invalid or expired token")
                return None
            if not info.mayring_active:
                await self._send_402(send,
                    "MayringCoder Memory requires an active subscription. "
                    "Subscribe at https://app.linn.games/einstellungen/mayring-abo"
                )
                return None
            return info.workspace_id

        if _AUTH_MODE in ("jwt", "auto") and _AUTH_SECRET:
            import jwt  # type: ignore[import]
            try:
                payload = jwt.decode(token, _AUTH_SECRET, algorithms=["HS256"])
                return payload.get("workspace_id", "default")
            except jwt.ExpiredSignatureError:
                await self._send_401(send, "Token expired")
                return None
            except jwt.InvalidTokenError:
                await self._send_401(send, "Invalid token")
                return None

        if _AUTH_TOKEN:
            if token != _AUTH_TOKEN:
                await self._send_401(send, "Unauthorized")
                return None
            return "default"

        await self._send_401(send, "No auth method configured")
        return None

    @staticmethod
    async def _send_401(send: Any, message: str) -> None:
        body = message.encode()
        await send({
            "type": "http.response.start", "status": 401,
            "headers": [[b"content-type", b"text/plain; charset=utf-8"],
                        [b"content-length", str(len(body)).encode()]],
        })
        await send({"type": "http.response.body", "body": body})

    @staticmethod
    async def _send_402(send: Any, message: str) -> None:
        body = message.encode()
        await send({
            "type": "http.response.start", "status": 402,
            "headers": [[b"content-type", b"text/plain; charset=utf-8"],
                        [b"content-length", str(len(body)).encode()]],
        })
        await send({"type": "http.response.body", "body": body})


from src.memory.ingest import get_or_create_chroma_collection, ingest
from src.memory.retrieval import compress_for_prompt, invalidate_query_cache, search
from src.memory.schema import Chunk, Source, make_memory_key, source_fingerprint
from src.memory.store import (
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
            workspace_id=_effective_workspace_id(workspace_id),
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
            "workspace_id": _effective_workspace_id(workspace_id or "default"),
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
        return {"chunk_id": chunk_id, "recorded": True}
    except Exception as exc:
        return {"error": str(exc)}


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


_AUTHORIZE_HTML = """\
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MayringCoder — Anmelden</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     background:#f8fafc;display:flex;align-items:center;justify-content:center;
     min-height:100vh;margin:0;padding:16px}}
.card{{background:#fff;border-radius:12px;box-shadow:0 2px 16px rgba(0,0,0,.08);
       padding:32px;width:100%;max-width:420px}}
h2{{margin:0 0 8px;font-size:1.25rem;color:#1e293b}}
p{{margin:0 0 20px;color:#64748b;font-size:.95rem;line-height:1.5}}
a{{color:#2563eb}}
label{{display:block;font-size:.875rem;font-weight:500;color:#374151;margin-bottom:6px}}
input[type=password]{{width:100%;padding:10px 12px;border:1px solid #d1d5db;
                      border-radius:8px;font-size:1rem;outline:none;
                      transition:border-color .15s}}
input[type=password]:focus{{border-color:#2563eb;box-shadow:0 0 0 3px rgba(37,99,235,.12)}}
.hint{{margin:6px 0 20px;font-size:.8rem;color:#94a3b8}}
button{{width:100%;background:#2563eb;color:#fff;border:none;padding:11px;
        border-radius:8px;font-size:1rem;font-weight:500;cursor:pointer;
        transition:background .15s}}
button:hover{{background:#1d4ed8}}
.err{{color:#dc2626;font-size:.875rem;margin-bottom:12px}}
</style>
</head>
<body>
<div class="card">
  <h2>MayringCoder Memory</h2>
  <p>Sanctum-Token von
     <a href="https://app.linn.games/einstellungen" target="_blank">app.linn.games/einstellungen</a>
     eingeben, um Claude Web mit deinem Memory zu verbinden.</p>
  {error}
  <form method="post" action="/authorize">
    <label for="token">API-Token</label>
    <input type="password" id="token" name="token"
           placeholder="z.B. 42|abc123..." autocomplete="off" required>
    <p class="hint">Das Token findest du unter Einstellungen → API-Token.</p>
    {hidden}
    <button type="submit">Verbinden</button>
  </form>
</div>
</body>
</html>"""


async def _oauth_authorize(request: Any) -> Any:
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, RedirectResponse
    req: Request = request

    if req.method == "GET":
        params = dict(req.query_params)
        hidden = "".join(
            f'<input type="hidden" name="{k}" value="{v}">'
            for k, v in params.items()
        )
        return HTMLResponse(_AUTHORIZE_HTML.format(error="", hidden=hidden))

    # POST — validate token + issue auth code
    form = await req.form()
    token = str(form.get("token", "")).strip()
    redirect_uri = str(form.get("redirect_uri", ""))
    state = str(form.get("state", ""))
    code_challenge = str(form.get("code_challenge", ""))
    code_challenge_method = str(form.get("code_challenge_method", "S256"))
    client_id = str(form.get("client_id", ""))

    # Rebuild hidden fields for error re-render
    hidden = "".join(
        f'<input type="hidden" name="{k}" value="{v}">'
        for k, v in {
            "redirect_uri": redirect_uri, "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
            "client_id": client_id,
        }.items()
    )

    if not token:
        err = '<p class="err">Bitte Token eingeben.</p>'
        return HTMLResponse(_AUTHORIZE_HTML.format(error=err, hidden=hidden), status_code=400)

    # Validate via Sanctum
    try:
        from src.api.sanctum_auth import validate_sanctum_token_full
        info = validate_sanctum_token_full(token)
    except Exception:
        info = None

    if not info:
        err = '<p class="err">Token ungültig oder abgelaufen.</p>'
        return HTMLResponse(_AUTHORIZE_HTML.format(error=err, hidden=hidden), status_code=401)
    if not info.mayring_active:
        err = ('<p class="err">Kein aktives Mayring-Abo. '
               '<a href="https://app.linn.games/einstellungen/mayring-abo">Jetzt aktivieren</a>.</p>')
        return HTMLResponse(_AUTHORIZE_HTML.format(error=err, hidden=hidden), status_code=402)

    code = secrets.token_urlsafe(32)
    _auth_codes[code] = {
        "token": token,
        "workspace_id": info.workspace_id,
        "code_challenge": code_challenge,
        "code_challenge_method": code_challenge_method,
        "redirect_uri": redirect_uri,
        "state": state,
        "expires_at": time.time() + 300,
    }

    callback = f"{redirect_uri}?code={code}&state={state}"
    return RedirectResponse(callback, status_code=302)


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
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if _TRANSPORT in ("http", "sse"):
        import uvicorn
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route

        _inner = _JWTAuthMiddleware(mcp.sse_app())

        _auth_label = (
            "sanctum" if (_AUTH_ENABLED and _AUTH_MODE == "sanctum") else
            "auto-detect" if (_AUTH_ENABLED and _AUTH_MODE == "auto") else
            "jwt" if (_AUTH_ENABLED and _AUTH_SECRET) else
            "static" if (_AUTH_ENABLED and _AUTH_TOKEN) else
            "disabled"
        )
        print(
            f"[mcp-memory] HTTP/SSE on {_HTTP_HOST}:{_HTTP_PORT}"
            f" | auth={_auth_label}"
            f" | oauth={_OAUTH_BASE_URL}"
        )

        app = Starlette(routes=[
            Route("/.well-known/oauth-authorization-server", _oauth_metadata),
            Route("/.well-known/oauth-protected-resource", _oauth_metadata),
            Route("/.well-known/oauth-protected-resource/sse", _oauth_metadata),
            Route("/register", _oauth_register, methods=["POST"]),
            Route("/authorize", _oauth_authorize, methods=["GET", "POST"]),
            Route("/token", _oauth_token, methods=["POST"]),
            Mount("/", app=_inner),
        ])
        uvicorn.run(app, host=_HTTP_HOST, port=_HTTP_PORT)
    else:
        mcp.run()  # stdio transport (local dev)


if __name__ == "__main__":
    main()
