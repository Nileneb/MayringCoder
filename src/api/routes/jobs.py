from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from src.api.auth import get_workspace
from src.api.job_queue import (
    get_job as _get_job,
    make_job as _make_job,
    python_exe as _python_exe,
    run_checker_job as _run_checker_job,
    _JOBS,
)
from src.api.routes.models import (
    AmbientSnapshotRequest,
    AnalyzeRequest,
    BenchmarkRequest,
    IssuesIngestRequest,
    PaperIngestRequest,
    PopulateRequest,
    PredictiveRebuildRequest,
    RepoRequest,
    TurbulenceRequest,
    WikiGenerateRequest,
)

router = APIRouter(tags=["jobs"])

_ROOT = Path(__file__).parent.parent.parent.parent
_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

from src import config as _config


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

    for flag, label in (
        ("--generate-ambient", "ambient"),
        ("--rebuild-transitions", "predictive"),
    ):
        vid = _make_job(workspace_id)
        v2_jobs[label] = vid
        asyncio.create_task(_run_checker_job(vid, ["--repo", repo, flag], workspace_id))

    img_id = _make_job(workspace_id)
    v2_jobs["images"] = img_id
    asyncio.create_task(_run_checker_job(
        img_id,
        ["--ingest-images", repo, "--no-limit"],
        workspace_id,
    ))

    overview_id = _make_job(workspace_id)
    wiki_id = _make_job(workspace_id)
    v2_jobs["overview"] = overview_id
    v2_jobs["wiki"] = wiki_id
    _JOBS[job_id]["v2_jobs"] = v2_jobs

    async def _overview_then_wiki() -> None:
        await _run_checker_job(overview_id, ["--repo", repo, "--mode", "overview", "--time-budget", str(_config.ANALYSIS_TIME_BUDGET)], workspace_id)
        if _JOBS.get(overview_id, {}).get("status") == "done":
            await _run_checker_job(wiki_id, ["--repo", repo, "--generate-wiki"], workspace_id)
        else:
            _JOBS[wiki_id]["status"] = "error"
            _JOBS[wiki_id]["output"] = "skipped — overview job failed"

    asyncio.create_task(_overview_then_wiki())


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Poll status of a background job."""
    job = _get_job(job_id)
    if not job or job["workspace_id"] != workspace_id:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/analyze")
async def trigger_analysis(
    request: AnalyzeRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Submit a code analysis job. Returns job_id; fires v2-chain (wiki/ambient/images) on success."""
    args = ["--repo", request.repo, "--rag-enrichment"]
    if request.full:
        args.append("--full")
    if request.adversarial:
        args.append("--adversarial")
    if request.no_pi:
        args.append("--no-pi")
    if request.budget is not None:
        args.extend(["--budget", str(request.budget)])
    if request.second_opinion:
        args.extend(["--second-opinion", request.second_opinion])

    job_id = _make_job(workspace_id)
    asyncio.create_task(_run_with_v2_postingest(job_id, args, workspace_id, request.repo))
    return {"job_id": job_id, "status": "started", "workspace_id": workspace_id, "repo": request.repo}


@router.post("/overview")
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


@router.post("/turbulence")
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


@router.post("/benchmark")
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


@router.post("/issues/ingest")
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


@router.post("/populate")
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


@router.post("/wiki/generate")
async def trigger_wiki_generate(
    request: WikiGenerateRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Rebuild wiki for a workspace_id (all repos + conversations) or a single repo."""
    job_id = _make_job(workspace_id)
    wid = request.workspace_id or workspace_id
    if wid and not request.repo:
        args = ["--generate-wiki", "--workspace-id", wid]
    else:
        args = ["--generate-wiki", "--wiki-type", request.wiki_type]
        if request.repo:
            args = ["--repo", request.repo] + args
    asyncio.create_task(_run_checker_job(job_id, args, workspace_id))
    return {"job_id": job_id, "status": "started", "workspace_id": wid, "repo": request.repo}


@router.post("/ambient/snapshot")
async def trigger_ambient_snapshot(
    request: AmbientSnapshotRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Generate a fresh ambient-context snapshot for the given repo."""
    job_id = _make_job(workspace_id)
    args = ["--repo", request.repo, "--generate-ambient"]
    asyncio.create_task(_run_checker_job(job_id, args, workspace_id))
    return {"job_id": job_id, "status": "started", "repo": request.repo}


@router.post("/predictive/rebuild-transitions")
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


@router.post("/papers/ingest")
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
