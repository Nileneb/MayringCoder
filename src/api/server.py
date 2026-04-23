"""MayringCoder Multi-Tenant API Server.

FastAPI HTTP layer. Auth: **RS256-JWT** (2026-04 umgestellt von Sanctum).

JWTs werden von app.linn.games' JwtIssuer ausgestellt und tragen
``workspace_id``, ``sub``, ``scope`` sowie BYO-Provider-Claims. Dieser
Server validiert sie **offline** gegen den Public-Key unter
JWT_PUBLIC_KEY_PATH — keine Laravel-DB-Roundtrip mehr nötig.

Start:
    .venv/bin/python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8090

Endpoints (alle außer /health benötigen Bearer <RS256-JWT>):
    POST /analyze          — submit analysis job (returns pid)
    POST /overview         — overview map of a repo (returns job_id)
    POST /turbulence       — turbulence analysis (returns job_id)
    POST /benchmark        — retrieval benchmark (returns job_id)
    POST /issues/ingest    — GitHub issues → memory (returns job_id)
    POST /populate         — repo source files → memory (returns job_id,
                              fires v2-chain after success)
    POST /papers/ingest    — ArXiv papers → memory (returns job_id)
    POST /duel             — run same task on two models + auto-judge (returns job_id)
    POST /benchmark/tasks  — run task suite on two models, score + compare
    POST /wiki/generate    — rebuild wiki index + clusters (returns job_id)
    POST /ambient/snapshot — regenerate ambient snapshot (returns job_id)
    POST /predictive/rebuild-transitions — Markov matrix (returns job_id)
    GET  /jobs/{job_id}    — poll job status; v2-chain child job IDs
                              unter ``v2_jobs`` dictionary
    POST /memory/search          — search workspace memory
    POST /memory/put             — ingest content into workspace memory
    GET  /memory/chunk/{id}      — retrieve chunk by ID
    POST /memory/invalidate      — deactivate all chunks for a source
    GET  /memory/chunks/{src_id} — list chunks for a source
    GET  /memory/explain/{id}    — explain chunk (key, scores, origin)
    POST /memory/reindex         — re-embed + re-upsert to ChromaDB
    POST /memory/feedback        — record usage signal for a chunk
    GET  /reports          — list reports
    GET  /health           — health check (no auth)

Required env:
    JWT_PUBLIC_KEY_PATH   — path to RS256 public key (PEM)
    JWT_ISSUER            — expected `iss` claim (default: https://app.linn.games)
    JWT_AUDIENCE          — expected `aud` claim (default: mayringcoder)
    OLLAMA_URL            — Ollama endpoint (default three.linn.games)
    OLLAMA_MODEL          — default model (e.g. gemma4:e4b)
    MCP_SERVICE_TOKEN     — shared secret for the /authorize/register-code
                             service-to-service handshake (MCP-HTTP, not this
                             server)
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_ROOT / ".env")

try:
    from fastapi import Depends, FastAPI, HTTPException, status
    from pydantic import BaseModel
except ImportError:
    raise ImportError("Missing dependency: pip install fastapi uvicorn")

from src.api.memory_service import run_ingest as _run_ingest, run_search as _run_search
from src.api.dependencies import get_conn as _get_conn, get_chroma as _get_chroma
from src.api.auth import get_workspace
from src.api.training import router as _training_router
from src.api.job_queue import get_job as _get_job, make_job as _make_job, python_exe as _python_exe, run_checker_job as _run_checker_job, _JOBS
from src.model_router import ModelRouter as _ModelRouter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "")
_router = _ModelRouter(_OLLAMA_URL)

app = FastAPI(title="MayringCoder API", version="1.0.0")
app.include_router(_training_router)

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


class PaperIngestRequest(BaseModel):
    papers_dir: str = "/data/papers"
    repo: str = ""
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


class DuelRequest(BaseModel):
    task: str
    model_a: str
    model_b: str
    repo_slug: str | None = None
    system_prompt: str | None = None
    timeout: float = 180.0
    judge: bool = True
    judge_model: str | None = None
    no_memory_baseline: bool = False


class MemoryInvalidateRequest(BaseModel):
    source_id: str


class MemoryReindexRequest(BaseModel):
    source_id: str | None = None


class MemoryFeedbackRequest(BaseModel):
    chunk_id: str
    signal: str
    metadata: dict | None = None


class ConversationTurnModel(BaseModel):
    role: str
    content: str
    timestamp: str | None = None


class ConversationMicroBatchRequest(BaseModel):
    turns: list[ConversationTurnModel]
    session_id: str
    workspace_slug: str = "default"
    # Optional: caller can pre-summarize (local-watcher mode). When absent,
    # the server produces the summary via its own Ollama.
    presumarized: str | None = None


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

    # Note: _JOBS is intentionally not used here — analyze endpoint returns pid directly,
    # not via job queue (unlike /overview, /turbulence, /benchmark which use job_id)


@app.post("/memory/search")
async def memory_search(
    request: MemorySearchRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Search workspace memory."""
    try:
        opts: dict[str, Any] = {"top_k": request.top_k, "workspace_id": workspace_id}
        if request.repo:
            opts["repo"] = request.repo
        if request.source_type:
            opts["source_type"] = request.source_type
        result = _run_search(request.query, _get_conn(), _get_chroma(), _OLLAMA_URL,
                             opts, request.char_budget)
        return {"workspace_id": workspace_id, **result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/memory/put")
async def memory_put(
    request: MemoryPutRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Ingest content into workspace memory."""
    try:
        source_dict = {"source_id": request.source_id, "source_type": request.source_type,
                       "repo": request.repo, "path": request.path}
        result = _run_ingest(source_dict, request.content, _get_conn(), _get_chroma(),
                             _OLLAMA_URL, _OLLAMA_MODEL, {"categorize": request.categorize},
                             workspace_id)
        return {"workspace_id": workspace_id, **result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/conversation/micro-batch")
async def conversation_micro_batch(
    request: ConversationMicroBatchRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Accept a batch of raw Claude turns from a remote conversation watcher,
    produce a summary on the server side (so the user doesn't need their own
    Ollama), and ingest as a ``conversation_summary`` source.

    This is the endpoint the per-user `tools/conversation_watcher.py` in
    RemoteHttpSink-Modus calls. Dedup: the source_id is deterministic
    (``conversation:<workspace_slug>:<session_id>``); when the same content
    is re-posted, ingest() detects it via content_hash and skips.
    """
    try:
        import hashlib
        from tools.ingest_conversations import _summarize as _summarize_turns

        turns_dicts = [t.model_dump() for t in request.turns]
        if not turns_dicts:
            raise HTTPException(status_code=400, detail="turns must not be empty")

        first_ts = turns_dicts[0].get("timestamp", "")[:10]
        batch_key = f"{request.session_id}:{len(turns_dicts)}:{turns_dicts[-1].get('timestamp', '')}"
        content_hash = "sha256:" + hashlib.sha256(batch_key.encode()).hexdigest()[:16]
        source_id = f"conversation:{request.workspace_slug}:{request.session_id[:16]}"

        summary = (
            request.presumarized
            or _summarize_turns(turns_dicts, "", _OLLAMA_URL, _OLLAMA_MODEL)
        )
        content = (
            f"# Session {first_ts or 'unbekannt'} | {request.workspace_slug}\n\n"
            f"{summary}\n"
        )
        source_dict = {
            "source_id": source_id,
            "source_type": "conversation_summary",
            "repo": request.workspace_slug,
            "path": f"{request.workspace_slug}/incremental",
            "branch": "local",
            "commit": "",
            "content_hash": content_hash,
        }
        result = _run_ingest(
            source_dict, content, _get_conn(), _get_chroma(),
            _OLLAMA_URL, _OLLAMA_MODEL,
            {"categorize": True, "codebook": "social", "mode": "hybrid"},
            workspace_id,
        )
        return {"workspace_id": workspace_id, "source_id": source_id, **result}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/memory/chunk/{chunk_id}")
async def memory_get_chunk(
    chunk_id: str,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    from src.memory.store import kv_get, get_chunk
    cached = kv_get(chunk_id)
    if cached is not None:
        return {"workspace_id": workspace_id, "chunk": cached}
    chunk = get_chunk(_get_conn(), chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail="chunk not found")
    return {"workspace_id": workspace_id, "chunk": chunk.to_dict()}


@app.post("/memory/invalidate")
async def memory_invalidate(
    request: MemoryInvalidateRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    from src.memory.store import deactivate_chunks_by_source, log_ingestion_event
    from src.memory.retrieval import invalidate_query_cache
    conn = _get_conn()
    count = deactivate_chunks_by_source(conn, request.source_id)
    log_ingestion_event(conn, request.source_id, "invalidated", {"count": count})
    invalidate_query_cache()
    return {"workspace_id": workspace_id, "source_id": request.source_id, "deactivated_count": count}


@app.get("/memory/chunks/{source_id}")
async def memory_list_by_source(
    source_id: str,
    active_only: bool = True,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    from src.memory.store import get_chunks_by_source
    chunks = get_chunks_by_source(_get_conn(), source_id, active_only=active_only)
    return {
        "workspace_id": workspace_id,
        "source_id": source_id,
        "count": len(chunks),
        "chunks": [c.to_dict() for c in chunks],
    }


@app.get("/memory/explain/{chunk_id}")
async def memory_explain(
    chunk_id: str,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    from src.memory.store import get_chunk, get_source
    from src.memory.schema import make_memory_key, source_fingerprint
    chunk = get_chunk(_get_conn(), chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail="chunk not found")
    cats = chunk.category_labels[0] if chunk.category_labels else "uncategorized"
    fp = source_fingerprint(chunk.source_id)
    hash_prefix = chunk.text_hash.replace("sha256:", "")[:8]
    memory_key = make_memory_key("repo", cats, fp, hash_prefix)
    source = get_source(_get_conn(), chunk.source_id)
    return {
        "workspace_id": workspace_id,
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
        "source": source.to_dict() if source else {},
    }


@app.post("/memory/reindex")
async def memory_reindex(
    request: MemoryReindexRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    try:
        from src.analysis.context import _embed_texts
        from src.memory.store import get_chunks_by_source, get_chunk
        from src.memory.retrieval import invalidate_query_cache

        chroma = _get_chroma()
        conn = _get_conn()

        if request.source_id:
            chunks = get_chunks_by_source(conn, request.source_id, active_only=True)
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
        return {"workspace_id": workspace_id, "reindexed_count": reindexed, "errors": errors}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/memory/feedback")
async def memory_feedback(
    request: MemoryFeedbackRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    if request.signal not in ("positive", "negative", "neutral"):
        raise HTTPException(status_code=400, detail="signal must be positive|negative|neutral")
    from src.memory.store import add_feedback
    add_feedback(_get_conn(), request.chunk_id, request.signal, request.metadata or {})
    return {"workspace_id": workspace_id, "chunk_id": request.chunk_id, "recorded": True}


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
    job = _get_job(job_id)
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
            job = _get_job(job_id)
            if job:
                job["status"] = "done" if proc.returncode == 0 else "error"
                job["output"] = stdout.decode(errors="replace")
        except Exception as exc:
            job = _get_job(job_id)
            if job:
                job["status"] = "error"
                job["output"] = str(exc)

    asyncio.create_task(_run_benchmark(job_id))
    return {"job_id": job_id, "status": "started"}


async def _run_with_v2_postingest(
    job_id: str,
    args: list[str],
    workspace_id: str,
    repo: str,
) -> None:
    """Run a pipeline job and, on success, fire the v2.0 refresh chain.

    Chain layout (discovered during the first prod smoke-test):

      overview ─▶ wiki         (sequential — wiki aborts with
                                "Kein Overview-Cache für <slug>"
                                when no previous --mode overview ran)
      ambient               (parallel — reads chunks directly)
      predictive            (parallel — reads topic_transitions)

    v2.0 jobs are tracked under ``_JOBS[job_id]["v2_jobs"]`` so callers can
    poll their status without a separate lookup.
    """
    await _run_checker_job(job_id, args, workspace_id)
    if _JOBS.get(job_id, {}).get("status") != "done" or not repo:
        return

    v2_jobs: dict[str, str] = {}

    # Ambient + Predictive don't depend on the overview cache → fire now.
    for flag, label in (
        ("--generate-ambient", "ambient"),
        ("--rebuild-transitions", "predictive"),
    ):
        vid = _make_job(workspace_id)
        v2_jobs[label] = vid
        asyncio.create_task(_run_checker_job(vid, ["--repo", repo, flag], workspace_id))

    # Overview → Wiki chain (sequential).
    overview_id = _make_job(workspace_id)
    wiki_id = _make_job(workspace_id)
    v2_jobs["overview"] = overview_id
    v2_jobs["wiki"] = wiki_id
    _JOBS[job_id]["v2_jobs"] = v2_jobs

    async def _overview_then_wiki() -> None:
        await _run_checker_job(overview_id, ["--repo", repo, "--mode", "overview"], workspace_id)
        if _JOBS.get(overview_id, {}).get("status") == "done":
            await _run_checker_job(wiki_id, ["--repo", repo, "--generate-wiki"], workspace_id)
        else:
            _JOBS[wiki_id]["status"] = "error"
            _JOBS[wiki_id]["output"] = "skipped — overview job failed"

    asyncio.create_task(_overview_then_wiki())


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
    asyncio.create_task(_run_with_v2_postingest(job_id, args, workspace_id, request.repo))
    return {"job_id": job_id, "status": "started", "repo": request.repo}


@app.post("/populate")
async def trigger_populate(
    request: PopulateRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Ingest repository source files into workspace memory.

    On success, also refreshes the v2.0 layer for this repo
    (wiki index, ambient snapshot, predictive transitions) as background jobs.
    The child job ids are returned under ``v2_jobs`` when polled via GET /jobs/{id}.
    """
    job_id = _make_job(workspace_id)
    # --memory-categorize: ohne diesen Flag setzt run_populate_memory() in
    # _opts ein `categorize=False`, wodurch ingest() mayring_categorize
    # komplett überspringt und alle chunks ohne category_labels landen.
    # Das war der Befund aus dem ersten Prod-Smoke (d5023372): 1128 Chunks,
    # 0 mit Labels. Standard-Weg aus dem UI muss Mayring aktiv haben.
    args = ["--repo", request.repo, "--populate-memory", "--multiview",
            "--memory-categorize"]
    if request.force_reingest:
        args.append("--force-reingest")
    asyncio.create_task(_run_with_v2_postingest(job_id, args, workspace_id, request.repo))
    return {"job_id": job_id, "status": "started", "repo": request.repo}


# ---------------------------------------------------------------------------
# v2.0 layer — can also be triggered manually without a preceding ingest.
# Each endpoint spawns a background job so the running mayring-api stays up.
# ---------------------------------------------------------------------------

class WikiGenerateRequest(BaseModel):
    repo: str
    wiki_type: str = "code"


class AmbientSnapshotRequest(BaseModel):
    repo: str


class PredictiveRebuildRequest(BaseModel):
    repo: str | None = None


@app.post("/wiki/generate")
async def trigger_wiki_generate(
    request: WikiGenerateRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Rebuild the wiki index (_wiki_index.json + _wiki_clusters_emb.json)."""
    job_id = _make_job(workspace_id)
    args = ["--repo", request.repo, "--generate-wiki", "--wiki-type", request.wiki_type]
    asyncio.create_task(_run_checker_job(job_id, args, workspace_id))
    return {"job_id": job_id, "status": "started", "repo": request.repo}


@app.post("/ambient/snapshot")
async def trigger_ambient_snapshot(
    request: AmbientSnapshotRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Generate a fresh ambient-context snapshot for the given repo."""
    job_id = _make_job(workspace_id)
    args = ["--repo", request.repo, "--generate-ambient"]
    asyncio.create_task(_run_checker_job(job_id, args, workspace_id))
    return {"job_id": job_id, "status": "started", "repo": request.repo}


@app.post("/predictive/rebuild-transitions")
async def trigger_predictive_rebuild(
    request: PredictiveRebuildRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Rebuild the Markov transition matrix feeding the predictive layer."""
    job_id = _make_job(workspace_id)
    args = ["--rebuild-transitions"]
    if request.repo:
        args = ["--repo", request.repo] + args
    asyncio.create_task(_run_checker_job(job_id, args, workspace_id))
    return {"job_id": job_id, "status": "started", "repo": request.repo or ""}


@app.post("/papers/ingest")
async def trigger_paper_ingest(
    request: PaperIngestRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Fetch ArXiv papers by ID and ingest into workspace memory."""
    job_id = _make_job(workspace_id)
    args = ["--ingest-papers", request.papers_dir]
    if request.repo:
        args += ["--repo", request.repo]
    if request.force_reingest:
        args.append("--force-reingest")
    asyncio.create_task(_run_checker_job(job_id, args, workspace_id))
    return {"job_id": job_id, "status": "started", "papers_dir": request.papers_dir}


@app.post("/duel")
async def trigger_duel(
    request: DuelRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Run the same task on two models sequentially. Poll /jobs/{id} for results."""
    if not request.task.strip():
        raise HTTPException(status_code=400, detail="task required")
    if not request.model_a or not request.model_b:
        raise HTTPException(status_code=400, detail="model_a and model_b required")
    job_id = _make_job(workspace_id)
    asyncio.create_task(_run_duel(job_id, request, workspace_id))
    return {"job_id": job_id, "status": "started", "model_a": request.model_a, "model_b": request.model_b}


async def _run_duel(job_id: str, request: DuelRequest, workspace_id: str) -> None:
    """Sequential execution of Pi-task on both models, optional judge + baseline."""
    import json as _json
    import time
    from src.agents.pi import run_task_with_memory
    _repo_slug = request.repo_slug or os.getenv("PI_REPO_SLUG", "")
    loop = asyncio.get_event_loop()

    def _run_one(model: str, disable_memory: bool = False) -> tuple[str, float]:
        t0 = time.monotonic()
        try:
            out = run_task_with_memory(
                task=request.task,
                ollama_url=_OLLAMA_URL,
                model=model,
                repo_slug=_repo_slug,
                system_prompt=request.system_prompt,
                timeout=request.timeout,
                disable_memory=disable_memory,
            )
        except Exception as exc:
            out = f"[Fehler] {exc}"
        return out, round((time.monotonic() - t0) * 1000, 1)

    def _judge(task: str, answer_a: str, answer_b: str, judge_model: str) -> dict:
        from src import ollama_client as _oc
        _prompt = (
            f"Aufgabe: {task}\n\n"
            f"Antwort A:\n{answer_a[:1500]}\n\n"
            f"Antwort B:\n{answer_b[:1500]}\n\n"
            "Bewerte beide Antworten sachlich. Antworte NUR mit validem JSON (kein Markdown, keine Erklärung):\n"
            '{"winner":"A","score_a":8,"score_b":6,"reasoning":"..."}'
        )
        try:
            raw = _oc.generate(
                _OLLAMA_URL, judge_model, _prompt,
                num_predict=512, keep_alive="0",
            )
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.strip("`").lstrip("json").strip()
            return _json.loads(raw)
        except Exception as exc:
            return {"winner": "tie", "score_a": 0, "score_b": 0, "reasoning": f"Judge-Fehler: {exc}"}

    job = _JOBS.get(job_id)
    if not job:
        return
    job["model_a"] = request.model_a
    job["model_b"] = request.model_b

    # --- Memory-enabled runs ---
    job["progress"] = "running_a"
    result_a, time_a = await loop.run_in_executor(None, _run_one, request.model_a, False)
    job["result_a"] = result_a
    job["time_a_ms"] = time_a

    job["progress"] = "running_b"
    result_b, time_b = await loop.run_in_executor(None, _run_one, request.model_b, False)
    job["result_b"] = result_b
    job["time_b_ms"] = time_b

    # --- No-memory baseline runs (optional) ---
    if request.no_memory_baseline:
        job["progress"] = "running_baseline_a"
        baseline_a, btime_a = await loop.run_in_executor(None, _run_one, request.model_a, True)
        job["baseline_a"] = baseline_a
        job["baseline_time_a_ms"] = btime_a

        job["progress"] = "running_baseline_b"
        baseline_b, btime_b = await loop.run_in_executor(None, _run_one, request.model_b, True)
        job["baseline_b"] = baseline_b
        job["baseline_time_b_ms"] = btime_b

    # --- Auto-judge (optional) ---
    if request.judge:
        job["progress"] = "judging"
        _judge_model = request.judge_model or _OLLAMA_MODEL
        verdict = await loop.run_in_executor(
            None, _judge, request.task, result_a, result_b, _judge_model
        )
        job["verdict"] = verdict

    job["progress"] = "done"
    job["status"] = "finished"
    job["output"] = f"Duell fertig — A: {time_a}ms, B: {time_b}ms"


class BenchmarkTasksRequest(BaseModel):
    model_a: str
    model_b: str
    category: str | None = None
    repo_slug: str | None = None
    timeout: float = 180.0
    judge_model: str | None = None


@app.post("/benchmark/tasks")
async def benchmark_tasks(
    request: BenchmarkTasksRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Run task suite on two models, score each answer and return comparison report."""
    import time
    import yaml
    from pathlib import Path as _Path
    from src.agents.pi import run_task_with_memory
    from src import ollama_client as _oc
    import json as _json

    suite_path = _Path(__file__).parent.parent.parent / "benchmarks" / "task_suite.yaml"
    if not suite_path.exists():
        raise HTTPException(status_code=404, detail="task_suite.yaml not found")

    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    tasks = suite.get("tasks", [])
    if request.category:
        tasks = [t for t in tasks if t.get("category") == request.category]
    if not tasks:
        raise HTTPException(status_code=404, detail=f"No tasks for category={request.category!r}")

    _repo_slug = request.repo_slug or os.getenv("PI_REPO_SLUG", "")
    _judge_model = request.judge_model or _OLLAMA_MODEL
    results = []

    def _score_answer(task_text: str, answer: str, keywords: list[str]) -> dict:
        hit = sum(1 for kw in keywords if kw.lower() in answer.lower())
        keyword_score = round(hit / len(keywords), 2) if keywords else 1.0
        return {"keyword_hits": hit, "keyword_total": len(keywords), "keyword_score": keyword_score}

    def _judge(task_text: str, answer_a: str, answer_b: str) -> dict:
        _prompt = (
            f"Aufgabe: {task_text}\n\n"
            f"Antwort A:\n{answer_a[:1200]}\n\n"
            f"Antwort B:\n{answer_b[:1200]}\n\n"
            "Bewerte beide Antworten sachlich. Antworte NUR mit validem JSON:\n"
            '{"winner":"A","score_a":8,"score_b":6,"reasoning":"..."}'
        )
        try:
            raw = _oc.generate(_OLLAMA_URL, _judge_model, _prompt, num_predict=512, keep_alive="0")
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.strip("`").lstrip("json").strip()
            return _json.loads(raw)
        except Exception as exc:
            return {"winner": "tie", "score_a": 0, "score_b": 0, "reasoning": f"Judge-Fehler: {exc}"}

    loop = asyncio.get_event_loop()

    for task_def in tasks:
        tid = task_def.get("id", "?")
        task_text = task_def.get("task", "")
        keywords = task_def.get("expected_keywords", [])
        requires_memory = task_def.get("requires_memory", False)

        t0 = time.monotonic()
        try:
            ans_a = await loop.run_in_executor(
                None,
                lambda: run_task_with_memory(
                    task=task_text, ollama_url=_OLLAMA_URL, model=request.model_a,
                    repo_slug=_repo_slug, timeout=request.timeout,
                ),
            )
        except Exception as exc:
            ans_a = f"[Fehler] {exc}"
        time_a = round((time.monotonic() - t0) * 1000, 1)

        t0 = time.monotonic()
        try:
            ans_b = await loop.run_in_executor(
                None,
                lambda: run_task_with_memory(
                    task=task_text, ollama_url=_OLLAMA_URL, model=request.model_b,
                    repo_slug=_repo_slug, timeout=request.timeout,
                ),
            )
        except Exception as exc:
            ans_b = f"[Fehler] {exc}"
        time_b = round((time.monotonic() - t0) * 1000, 1)

        score_a = _score_answer(task_text, ans_a, keywords)
        score_b = _score_answer(task_text, ans_b, keywords)
        verdict = await loop.run_in_executor(None, _judge, task_text, ans_a, ans_b)

        results.append({
            "task_id": tid,
            "category": task_def.get("category"),
            "requires_memory": requires_memory,
            "model_a": request.model_a,
            "model_b": request.model_b,
            "answer_a": ans_a[:500],
            "answer_b": ans_b[:500],
            "score_a": score_a,
            "score_b": score_b,
            "time_a_ms": time_a,
            "time_b_ms": time_b,
            "verdict": verdict,
        })

    wins_a = sum(1 for r in results if r["verdict"].get("winner") == "A")
    wins_b = sum(1 for r in results if r["verdict"].get("winner") == "B")
    avg_score_a = round(sum(r["verdict"].get("score_a", 0) for r in results) / len(results), 2)
    avg_score_b = round(sum(r["verdict"].get("score_b", 0) for r in results) / len(results), 2)

    return {
        "workspace_id": workspace_id,
        "tasks_run": len(results),
        "model_a": request.model_a,
        "model_b": request.model_b,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "avg_score_a": avg_score_a,
        "avg_score_b": avg_score_b,
        "results": results,
    }


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


@app.get("/wiki/graph")
async def wiki_graph(slug: str = "", workspace_id: str = Depends(get_workspace)) -> dict:
    """Return wiki cluster graph + recent Pi-agent search activations for Brain visualization."""
    import json as _json
    import time as _time_g
    from src.config import CACHE_DIR
    from src.api.memory_service import _RECENT_ACTIVATIONS

    cluster_path = CACHE_DIR / f"{slug}_wiki_clusters_emb.json"
    index_path = CACHE_DIR / f"{slug}_wiki_index.json"

    clusters: list[dict] = []
    raw: list[dict] = []
    if cluster_path.exists():
        raw = _json.loads(cluster_path.read_text())
        clusters = [
            {
                "name": c["name"],
                "files": c.get("files", []),
                "labels": c.get("labels", []),
                "size": max(1, len(c.get("files", []))),
            }
            for c in raw
        ]
    elif index_path.exists():
        idx = _json.loads(index_path.read_text())
        clusters = [{"name": k, "files": [], "labels": [], "size": 1} for k in idx]

    edges: list[dict] = []
    for c in raw:
        for edge in c.get("edges", []):
            # edges stored as [target_name, weight, rules_list]
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                edges.append({
                    "source": c["name"],
                    "target": edge[0],
                    "weight": edge[1],
                    "rules": edge[2] if len(edge) > 2 else [],
                })

    now = _time_g.time()
    activations: list[dict] = []
    for ev in _RECENT_ACTIVATIONS:
        if ev["workspace_id"] != workspace_id or now - ev["ts"] > 60:
            continue
        hit: set[str] = set()
        for sid in ev["source_ids"]:
            # source_id: "repo:owner/name:path" or "paper:arxiv:id:title"
            path = sid.split(":")[-1]
            for c in clusters:
                if any(
                    path == f or path.endswith("/" + f) or f.endswith("/" + path) or path == f.split("/")[-1]
                    for f in c["files"]
                ):
                    hit.add(c["name"])
        activations.append({
            "query": ev["query"][:80],
            "clusters": list(hit),
            "age_s": round(now - ev["ts"], 1),
        })

    return {"clusters": clusters, "edges": edges, "activations": activations}


def main() -> None:
    import uvicorn
    import os
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("API_PORT", "8080")))


if __name__ == "__main__":
    main()
