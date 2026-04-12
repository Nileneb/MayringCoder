"""MayringCoder Multi-Tenant API Server.

FastAPI HTTP layer with Laravel Sanctum token auth.

Auth: Bearer tokens from app.linn.games (Sanctum format: "{id}|{plaintext}")
are validated directly against the shared MySQL DB.

Start:
    .venv/bin/python -m uvicorn src.api_server:app --port 8080

Endpoints (authenticated via Bearer <sanctum_token>):
    POST /analyze          — submit analysis job
    POST /memory/search    — search workspace memory
    POST /memory/put       — ingest into workspace memory
    GET  /reports          — list reports
    GET  /health           — health check (no auth)

Required .env:
    LARAVEL_DB_HOST=mysql      (default: mysql — Docker service name)
    LARAVEL_DB_PORT=3306
    LARAVEL_DB_USER=...
    LARAVEL_DB_PASSWORD=...
    LARAVEL_DB_NAME=...
    APP_BASE_URL=https://your-server
"""

from __future__ import annotations

import asyncio
import os
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
from src.sanctum_auth import validate_sanctum_token_full

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "")
_MCP_AUTH_TOKEN = os.getenv("MAYRING_MCP_AUTH_TOKEN", "")

app = FastAPI(title="MayringCoder API", version="1.0.0")
_bearer = HTTPBearer(auto_error=False)

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
# Auth dependency
# ---------------------------------------------------------------------------

async def get_workspace(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    if not creds:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token")

    # Static service-to-service token bypass (for internal Laravel MayringMcpClient)
    # This path is always allowed — no subscription check needed for the research platform itself
    if _MCP_AUTH_TOKEN and creds.credentials == _MCP_AUTH_TOKEN:
        return "system"

    # External Sanctum token (Claude Desktop / Claude Web users)
    info = validate_sanctum_token_full(creds.credentials)
    if not info:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    if not info.mayring_active:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="MayringCoder Memory requires an active subscription (€5/month). "
                   "Subscribe at https://app.linn.games/einstellungen/mayring-abo",
        )

    return info.workspace_id


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    repo: str
    full: bool = False
    adversarial: bool = False
    no_pi: bool = False
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
    if request.no_pi:
        cmd.append("--no-pi")
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


@app.post("/search")
async def search_alias(
    request: MemorySearchRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Alias for /memory/search — used by Laravel MayringMcpClient."""
    return await memory_search(request, workspace_id)


@app.post("/ingest")
async def ingest_alias(
    request: MemoryPutRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Alias for /memory/put — used by Laravel MayringMcpClient."""
    return await memory_put(request, workspace_id)


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
