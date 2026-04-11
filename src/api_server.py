"""MayringCoder Multi-Tenant API Server.

FastAPI HTTP layer with GitHub OAuth onboarding + API-key auth.

Start:
    .venv/bin/python -m uvicorn src.api_server:app --port 8080

Onboarding (users):
    1. Visit  GET /login         → GitHub OAuth
    2. Lands on GET /auth/callback → workspace created, API key shown once
    3. Paste key into Claude Code MCP config

Endpoints (authenticated via Bearer <api_key>):
    POST /analyze          — submit analysis job
    POST /memory/search    — search workspace memory
    POST /memory/put       — ingest into workspace memory
    GET  /reports          — list reports
    GET  /health           — health check (no auth)
    GET  /login            — start GitHub OAuth flow (no auth)
    GET  /auth/callback    — GitHub OAuth callback (no auth)

Required .env:
    GITHUB_CLIENT_ID=...
    GITHUB_CLIENT_SECRET=...
    GITHUB_REDIRECT_URI=https://your-server/auth/callback
    APP_BASE_URL=https://your-server
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

try:
    from fastapi import Depends, FastAPI, HTTPException, Request, status
    from fastapi.responses import HTMLResponse, RedirectResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel
except ImportError:
    raise ImportError("Missing dependency: pip install fastapi uvicorn")

from src.memory_ingest import get_or_create_chroma_collection, ingest
from src.memory_retrieval import compress_for_prompt, search
from src.memory_schema import Source
from src.memory_store import init_memory_db

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_API_KEYS_PATH = _ROOT / "cache" / "api_keys.json"
_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "")

_GH_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")
_GH_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")
_GH_REDIRECT_URI = os.getenv("GITHUB_REDIRECT_URI", "")
_APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8080")

# In-memory CSRF state store (process-scoped, fine for 5-10 users)
_OAUTH_STATES: dict[str, float] = {}   # state → created_at (epoch)

app = FastAPI(title="MayringCoder API", version="1.0.0")
_bearer = HTTPBearer(auto_error=False)  # auto_error=False so /login doesn't demand Bearer

# Lazy singletons
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
# API Key management
# ---------------------------------------------------------------------------

def _load_keys() -> dict[str, dict]:
    if not _API_KEYS_PATH.exists():
        return {}
    try:
        return json.loads(_API_KEYS_PATH.read_text())
    except Exception:
        return {}


def _save_keys(keys: dict) -> None:
    _API_KEYS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _API_KEYS_PATH.write_text(json.dumps(keys, indent=2))


def _hash_key(raw_key: str) -> str:
    return "sha256:" + hashlib.sha256(raw_key.encode()).hexdigest()


def lookup_workspace(raw_key: str) -> str | None:
    """Return workspace_id for a valid API key, or None if invalid."""
    keys = _load_keys()
    hashed = _hash_key(raw_key)
    entry = keys.get(hashed)
    return entry["workspace_id"] if entry else None


def create_api_key(workspace_id: str) -> str:
    """Generate a new API key for workspace_id, persist it, return raw key."""
    raw_key = "mk_" + secrets.token_urlsafe(32)
    hashed = _hash_key(raw_key)
    keys = _load_keys()
    keys[hashed] = {
        "workspace_id": workspace_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_keys(keys)
    return raw_key


def revoke_api_keys(workspace_id: str) -> int:
    """Remove all API keys for workspace_id. Returns count removed."""
    keys = _load_keys()
    to_remove = [h for h, v in keys.items() if v["workspace_id"] == workspace_id]
    for h in to_remove:
        del keys[h]
    _save_keys(keys)
    return len(to_remove)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def get_workspace(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    if not creds:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key")
    ws = lookup_workspace(creds.credentials)
    if not ws:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return ws


# ---------------------------------------------------------------------------
# GitHub OAuth helpers
# ---------------------------------------------------------------------------

def _clean_old_states() -> None:
    """Remove CSRF states older than 10 minutes."""
    import time
    now = time.time()
    stale = [s for s, t in _OAUTH_STATES.items() if now - t > 600]
    for s in stale:
        _OAUTH_STATES.pop(s, None)


async def _github_exchange_code(code: str) -> str:
    """Exchange OAuth code for GitHub access token."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://github.com/login/oauth/access_token",
            data={
                "client_id": _GH_CLIENT_ID,
                "client_secret": _GH_CLIENT_SECRET,
                "code": code,
                "redirect_uri": _GH_REDIRECT_URI,
            },
            headers={"Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        token = data.get("access_token", "")
        if not token:
            raise ValueError(f"GitHub returned no token: {data.get('error_description', data)}")
        return token


async def _github_get_user(token: str) -> dict:
    """Fetch GitHub user profile with the given access token."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()


def _onboarding_page(username: str, api_key: str) -> str:
    """Return HTML page shown after successful OAuth login."""
    safe_user = html.escape(username)
    safe_key = html.escape(api_key)
    base = html.escape(_APP_BASE_URL)
    mcp_config = html.escape(json.dumps({
        "mcpServers": {
            "mayring-memory": {
                "url": f"{_APP_BASE_URL}/mcp",
                "headers": {"Authorization": f"Bearer {api_key}"},
            }
        }
    }, indent=2))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MayringCoder — Welcome {safe_user}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 640px; margin: 60px auto; padding: 0 20px; color: #222; }}
    h1 {{ font-size: 1.4rem; }}
    .key {{ background: #f4f4f4; border: 1px solid #ddd; border-radius: 6px;
             padding: 14px; font-family: monospace; font-size: 0.95rem;
             word-break: break-all; margin: 10px 0; }}
    .warn {{ color: #b45309; font-size: 0.875rem; margin-top: 6px; }}
    pre {{ background: #f8f8f8; border: 1px solid #e0e0e0; border-radius: 6px;
           padding: 14px; font-size: 0.82rem; overflow-x: auto; }}
    button {{ margin-top: 8px; padding: 6px 14px; cursor: pointer; border-radius: 4px;
               border: 1px solid #ccc; background: #fff; }}
    a {{ color: #1d4ed8; }}
  </style>
</head>
<body>
  <h1>Welcome, {safe_user} 👋</h1>
  <p>Your MayringCoder workspace <strong>{safe_user}</strong> is ready.</p>

  <h2>Your API Key</h2>
  <div class="key" id="apikey">{safe_key}</div>
  <p class="warn">⚠ Copy this now — it will not be shown again.</p>
  <button onclick="navigator.clipboard.writeText(document.getElementById('apikey').textContent)">Copy key</button>

  <h2>Claude Code — MCP Config</h2>
  <p>Add this to your <code>.claude/settings.json</code> (or user settings):</p>
  <pre id="mcp">{mcp_config}</pre>
  <button onclick="navigator.clipboard.writeText(document.getElementById('mcp').textContent)">Copy config</button>

  <h2>Quick test</h2>
  <pre>curl -s {base}/health
curl -s {base}/reports \\
  -H "Authorization: Bearer {safe_key}"</pre>

  <p style="margin-top:40px;font-size:0.8rem;color:#888">
    Need to rotate your key? Visit <a href="{base}/login">login again</a> — a new key will be issued.
  </p>
</body>
</html>"""


# ---------------------------------------------------------------------------
# OAuth endpoints
# ---------------------------------------------------------------------------

@app.get("/login", response_class=RedirectResponse, include_in_schema=False)
async def login() -> RedirectResponse:
    """Redirect to GitHub OAuth. No auth required."""
    if not _GH_CLIENT_ID:
        raise HTTPException(status_code=500, detail="GITHUB_CLIENT_ID not configured")
    import time
    _clean_old_states()
    state = secrets.token_urlsafe(16)
    _OAUTH_STATES[state] = time.time()
    params = urlencode({
        "client_id": _GH_CLIENT_ID,
        "redirect_uri": _GH_REDIRECT_URI,
        "scope": "read:user",
        "state": state,
    })
    return RedirectResponse(f"https://github.com/login/oauth/authorize?{params}")


@app.get("/auth/callback", response_class=HTMLResponse, include_in_schema=False)
async def auth_callback(code: str = "", state: str = "", error: str = "") -> HTMLResponse:
    """GitHub OAuth callback. Creates workspace + issues API key."""
    if error:
        return HTMLResponse(f"<p>GitHub OAuth error: {html.escape(error)}</p>", status_code=400)

    if not state or state not in _OAUTH_STATES:
        return HTMLResponse("<p>Invalid or expired OAuth state. <a href='/login'>Try again</a>.</p>", status_code=400)
    _OAUTH_STATES.pop(state, None)

    try:
        token = await _github_exchange_code(code)
        user = await _github_get_user(token)
    except Exception as exc:
        return HTMLResponse(f"<p>GitHub API error: {html.escape(str(exc))}</p>", status_code=502)

    username = user.get("login", "")
    if not username:
        return HTMLResponse("<p>Could not read GitHub username.</p>", status_code=502)

    # Workspace = github username (lowercase, safe chars only)
    workspace_id = username.lower()[:40]

    api_key = create_api_key(workspace_id)
    return HTMLResponse(_onboarding_page(username, api_key))


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    repo: str
    full: bool = False
    adversarial: bool = False
    pi: bool = False
    budget: int | None = None


class MemorySearchRequest(BaseModel):
    query: str
    repo: str | None = None
    source_type: str | None = None
    top_k: int = 8
    char_budget: int = 6000


class MemoryPutRequest(BaseModel):
    source_id: str | None = None
    source_type: str = "repo_file"
    repo: str = ""
    path: str = ""
    content: str
    categorize: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "1.0.0"}


@app.post("/analyze")
async def trigger_analysis(
    request: AnalyzeRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Submit a code analysis job. Runs checker.py in a subprocess.

    Returns immediately with job metadata; analysis output goes to reports/.
    """
    checker = str(_ROOT / "checker.py")
    python = str(_ROOT / ".venv" / "bin" / "python")
    if not Path(python).exists():
        python = "python"

    cmd = [python, checker, "--repo", request.repo, "--workspace-id", workspace_id]
    if request.full:
        cmd.append("--full")
    if request.adversarial:
        cmd.append("--adversarial")
    if request.pi:
        cmd.append("--pi")
    if request.budget is not None:
        cmd.extend(["--budget", str(request.budget)])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        # Don't await — return job info immediately
        return {
            "status": "started",
            "workspace_id": workspace_id,
            "repo": request.repo,
            "pid": proc.pid,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="checker.py or python not found")


@app.post("/memory/search")
async def memory_search(
    request: MemorySearchRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Search workspace memory."""
    opts: dict[str, Any] = {
        "top_k": request.top_k,
        "workspace_id": workspace_id,
    }
    if request.repo:
        opts["repo"] = request.repo
    if request.source_type:
        opts["source_type"] = request.source_type

    try:
        results = search(
            query=request.query,
            conn=_get_conn(),
            chroma_collection=_get_chroma(),
            ollama_url=_OLLAMA_URL,
            opts=opts,
        )
        prompt_context = compress_for_prompt(results, request.char_budget)
        return {
            "workspace_id": workspace_id,
            "results": [r.to_dict() for r in results],
            "prompt_context": prompt_context,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/memory/put")
async def memory_put(
    request: MemoryPutRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Ingest content into workspace memory."""
    try:
        src = Source(
            source_id=request.source_id or Source.make_id(request.repo, request.path),
            source_type=request.source_type,
            repo=request.repo,
            path=request.path,
        )
        result = ingest(
            source=src,
            content=request.content,
            conn=_get_conn(),
            chroma_collection=_get_chroma(),
            ollama_url=_OLLAMA_URL,
            model=_OLLAMA_MODEL,
            opts={"categorize": request.categorize},
            workspace_id=workspace_id,
        )
        return {"workspace_id": workspace_id, **result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/reports")
async def list_reports(
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """List analysis reports for this workspace."""
    reports_dir = _ROOT / "reports" / workspace_id
    if not reports_dir.exists():
        # Also check legacy flat reports dir
        reports_dir = _ROOT / "reports"
    reports = []
    if reports_dir.exists():
        for f in sorted(reports_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
            reports.append({"name": f.name, "size": f.stat().st_size})
    return {"workspace_id": workspace_id, "reports": reports, "count": len(reports)}
