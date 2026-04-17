"""MayringCoder Multi-Tenant API Server.

FastAPI HTTP layer with Laravel Sanctum token auth.

Auth: Bearer tokens from app.linn.games (Sanctum format: "{id}|{plaintext}")
are validated directly against the shared MySQL DB.

Start:
    .venv/bin/python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8080

Endpoints (authenticated via Bearer <sanctum_token>):
    POST /analyze          — submit analysis job (returns pid)
    POST /overview         — overview map of a repo (returns job_id)
    POST /turbulence       — turbulence analysis (returns job_id)
    POST /benchmark        — retrieval benchmark (returns job_id)
    POST /issues/ingest    — GitHub issues → memory (returns job_id)
    POST /populate         — repo source files → memory (returns job_id)
    GET  /jobs/{job_id}    — poll job status and output
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
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_ROOT / ".env")

try:
    from fastapi import Depends, FastAPI, HTTPException, status
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel
except ImportError:
    raise ImportError("Missing dependency: pip install fastapi uvicorn")

from src.memory.ingest import get_or_create_chroma_collection, ingest
from src.memory.retrieval import compress_for_prompt, search
from src.memory.schema import Source
from src.memory.store import init_memory_db
from src.api.sanctum_auth import validate_sanctum_token_full

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
        from src.memory.store import get_chroma_collection as get_collection
        _chroma = get_collection()
    return _chroma


# ---------------------------------------------------------------------------
# Job tracking (in-memory, sufficient for single-instance deployment)
# ---------------------------------------------------------------------------

_JOBS: dict[str, dict] = {}  # job_id → {status, output, workspace_id, started_at}


def _make_job(workspace_id: str) -> str:
    job_id = str(uuid.uuid4())[:8]
    _JOBS[job_id] = {
        "job_id": job_id,
        "status": "started",
        "output": "",
        "workspace_id": workspace_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    return job_id


def _python_exe() -> str:
    p = str(_ROOT / ".venv" / "bin" / "python")
    return p if Path(p).exists() else "python"


async def _run_checker_job(job_id: str, checker_args: list[str], workspace_id: str) -> None:
    try:
        proc = await asyncio.create_subprocess_exec(
            _python_exe(), "-m", "src.pipeline", *checker_args,
            "--workspace-id", workspace_id,
            cwd=str(_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        _JOBS[job_id]["status"] = "done" if proc.returncode == 0 else "error"
        _JOBS[job_id]["output"] = stdout.decode(errors="replace")
    except Exception as exc:
        _JOBS[job_id]["status"] = "error"
        _JOBS[job_id]["output"] = str(exc)


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

class PiTaskRequest(BaseModel):
    task: str
    repo_slug: str | None = None
    system_prompt: str | None = None
    timeout: float = 180.0


class AnalyzeRequest(BaseModel):
    repo: str
    full: bool = False
    adversarial: bool = False
    no_pi: bool = False
    budget: int | None = None


class RepoRequest(BaseModel):
    repo: str


class TurbulenceRequest(BaseModel):
    repo: str
    llm: bool = False


class BenchmarkRequest(BaseModel):
    top_k: int = 5
    repo: str | None = None


class IssuesIngestRequest(BaseModel):
    repo: str
    state: str = "open"
    force_reingest: bool = False


class PopulateRequest(BaseModel):
    repo: str
    force_reingest: bool = False


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


@app.post("/pi-task")
async def pi_task(
    request: PiTaskRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Run a task via the Pi-agent (memory-augmented reasoning)."""
    from src.agents.pi import run_task_with_memory
    _repo_slug = request.repo_slug or os.getenv("PI_REPO_SLUG", "")
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_task_with_memory(
                task=request.task,
                ollama_url=_OLLAMA_URL,
                model=_OLLAMA_MODEL,
                repo_slug=_repo_slug,
                system_prompt=request.system_prompt,
                timeout=request.timeout,
            ),
        )
        return {"workspace_id": workspace_id, "content": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/analyze")
async def trigger_analysis(
    request: AnalyzeRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Submit a code analysis job. Runs src.pipeline in a subprocess.

    Returns immediately with job metadata; analysis output goes to reports/.
    """
    cmd = [_python_exe(), "-m", "src.pipeline", "--repo", request.repo, "--workspace-id", workspace_id]
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
        raise HTTPException(status_code=500, detail="src.pipeline or python not found")


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


@app.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Poll status of a background job."""
    job = _JOBS.get(job_id)
    if not job or job["workspace_id"] != workspace_id:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/overview")
async def trigger_overview(
    request: RepoRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Start an overview map (function/class catalog) for a repository."""
    job_id = _make_job(workspace_id)
    asyncio.create_task(_run_checker_job(
        job_id,
        ["--repo", request.repo, "--mode", "overview", "--no-limit"],
        workspace_id,
    ))
    return {"job_id": job_id, "status": "started", "repo": request.repo}


@app.post("/turbulence")
async def trigger_turbulence(
    request: TurbulenceRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Start a turbulence (hot-zone) analysis for a repository."""
    job_id = _make_job(workspace_id)
    args = ["--repo", request.repo, "--mode", "turbulence"]
    if request.llm:
        args.append("--llm")
    asyncio.create_task(_run_checker_job(job_id, args, workspace_id))
    return {"job_id": job_id, "status": "started", "repo": request.repo}


@app.post("/benchmark")
async def trigger_benchmark(
    request: BenchmarkRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Run the retrieval benchmark and return MRR/Recall metrics."""
    job_id = _make_job(workspace_id)

    async def _run_benchmark(job_id: str) -> None:
        try:
            args = [
                _python_exe(), "-m", "src.benchmark_retrieval",
                "--top-k", str(request.top_k),
            ]
            if request.repo:
                args += ["--repo", request.repo]
            proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(_ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
            _JOBS[job_id]["status"] = "done" if proc.returncode == 0 else "error"
            _JOBS[job_id]["output"] = stdout.decode(errors="replace")
        except Exception as exc:
            _JOBS[job_id]["status"] = "error"
            _JOBS[job_id]["output"] = str(exc)

    asyncio.create_task(_run_benchmark(job_id))
    return {"job_id": job_id, "status": "started"}


@app.post("/issues/ingest")
async def trigger_issues_ingest(
    request: IssuesIngestRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Ingest GitHub issues from a repository into workspace memory."""
    job_id = _make_job(workspace_id)
    args = [
        "--ingest-issues", request.repo,
        "--issues-state", request.state,
        "--multiview",
    ]
    if request.force_reingest:
        args.append("--force-reingest")
    asyncio.create_task(_run_checker_job(job_id, args, workspace_id))
    return {"job_id": job_id, "status": "started", "repo": request.repo}


@app.post("/populate")
async def trigger_populate(
    request: PopulateRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Ingest repository source files into workspace memory."""
    job_id = _make_job(workspace_id)
    args = ["--repo", request.repo, "--populate-memory", "--multiview"]
    if request.force_reingest:
        args.append("--force-reingest")
    asyncio.create_task(_run_checker_job(job_id, args, workspace_id))
    return {"job_id": job_id, "status": "started", "repo": request.repo}


@app.get("/reports")
async def list_reports(
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """List analysis reports for this workspace."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", workspace_id):
        raise HTTPException(status_code=400, detail="Invalid workspace_id")
    safe_workspace_id = re.sub(r"[^A-Za-z0-9_-]", "", workspace_id)
    if not safe_workspace_id or safe_workspace_id != workspace_id:
        raise HTTPException(status_code=400, detail="Invalid workspace_id")
    base_reports_dir = (_ROOT / "reports").resolve()

    reports = []
    if base_reports_dir.exists():
        workspace_prefix = f"{safe_workspace_id}_"
        candidate_dirs = [base_reports_dir]
        candidate_dirs.extend(
            p
            for p in base_reports_dir.iterdir()
            if p.is_dir() and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", p.name)
        )
        for candidate_dir in candidate_dirs:
            markdown_files = [
                p
                for p in candidate_dir.iterdir()
                if p.is_file()
                and re.fullmatch(r"[A-Za-z0-9_.-]+\.md", p.name)
                and p.name.startswith(workspace_prefix)
            ]
            for f in sorted(markdown_files, key=lambda p: p.stat().st_mtime, reverse=True):
                reports.append({"name": f.name, "size": f.stat().st_size})
    return {"workspace_id": safe_workspace_id, "reports": reports, "count": len(reports)}


def main() -> None:
    import uvicorn
    import os
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("API_PORT", "8080")))


if __name__ == "__main__":
    main()
