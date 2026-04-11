"""MayringCoder Multi-Tenant API Server.

FastAPI HTTP layer for multi-tenant access.
Handles API-key auth, workspace routing, analysis job submission.

Start:
    .venv/bin/python -m uvicorn src.api_server:app --port 8080

Endpoints:
    POST /analyze          — submit analysis job (async subprocess)
    POST /memory/search    — search memory for workspace
    POST /memory/put       — ingest content into workspace memory
    GET  /reports          — list reports for workspace
    GET  /health           — health check (no auth)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

try:
    from fastapi import Depends, FastAPI, HTTPException, status
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

app = FastAPI(title="MayringCoder API", version="1.0.0")
_bearer = HTTPBearer(auto_error=True)

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
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    ws = lookup_workspace(creds.credentials)
    if not ws:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return ws


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
