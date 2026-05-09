"""IGIO classifier coverage + admin-triggered backfill.

Tracks Issue #141: existing chunks have an empty ``igio_axis`` because the
classifier was never run retroactively. These endpoints expose the coverage
ratio and an admin-only trigger that spawns ``python -m src.cli
--classify-igio`` as a SUBPROCESS — keeping the FastAPI worker free during
long Ollama loops.

Mounted under ``/stats/`` so the production nginx whitelist (``/stats/*``)
already covers it — no nginx config change required.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.api.auth import get_token_info
from src.api.jwt_auth import TokenInfo
from src.memory.store import init_memory_db

router = APIRouter()
_log = logging.getLogger(__name__)

_IGIO_JOBS: dict[str, dict[str, Any]] = {}
_ROOT = Path(__file__).parent.parent.parent.parent


def _python_exe() -> str:
    venv = _ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else "python"


def _conn():
    from src.config import CACHE_DIR
    return init_memory_db(CACHE_DIR / "memory.db")


def _is_admin(info: TokenInfo) -> bool:
    return "*" in info.scopes or "admin" in info.scopes


@router.get("/stats/igio-coverage")
async def igio_coverage(
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """Ratio of active chunks with non-empty ``igio_axis``.

    Service token (workspace_id='system', scope='*') counts ALL workspaces.
    Regular JWTs count only chunks belonging to the caller's workspace.
    """
    conn = _conn()
    if _is_admin(info):
        total = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE is_active = 1"
        ).fetchone()[0]
        with_axis = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE is_active = 1 AND igio_axis != ''"
        ).fetchone()[0]
        scope = "all"
    else:
        total = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE is_active = 1 AND workspace_id = ?",
            (info.workspace_id,),
        ).fetchone()[0]
        with_axis = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE is_active = 1 AND igio_axis != '' "
            "AND workspace_id = ?",
            (info.workspace_id,),
        ).fetchone()[0]
        scope = "workspace"
    ratio = round(with_axis / total, 4) if total else 0.0
    return {
        "workspace_id": info.workspace_id,
        "scope": scope,
        "total_active": total,
        "with_axis": with_axis,
        "ratio": ratio,
    }


async def _run_backfill_subprocess(
    job_id: str, limit: int, threshold: float, model: str,
    workspace_id: str | None,
) -> None:
    """Spawn ``python -m src.cli --classify-igio ...`` so the LLM loop runs
    in its own process. Keeps the FastAPI worker pool free, survives
    container reload of THIS request (subprocess detaches), and shares
    code with the local CLI path so behaviour cannot diverge."""
    state = _IGIO_JOBS[job_id]
    state.update(status="running", started_at=time.time())
    args = [
        "-m", "src.cli", "--classify-igio",
        "--igio-limit", str(limit),
        "--igio-min-confidence", str(threshold),
        "--model", model,
    ]
    # IGIO-Classification operiert cross-tenant: das Modell läuft pro
    # Chunk und schreibt igio_axis zurück, unabhängig vom Workspace
    # des Aufrufers. Wenn der admin-Trigger keinen workspace_id mitgibt,
    # default auf 'system' (Service-Token Maintenance-Bucket). Vorher
    # fiel resolve_cli_workspace auf 'default' zurück, was nach dem
    # Workspace-Refactor (commit 577d2d8) als UnknownWorkspaceError
    # crashte → Backfill-Cron lief stündlich grün, persisted=0/0.
    args += ["--workspace-id", workspace_id or "system"]
    try:
        proc = await asyncio.create_subprocess_exec(
            _python_exe(), *args,
            cwd=str(_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        out = stdout.decode(errors="replace") if stdout else ""
        # CLI prints "[igio] persisted=<N>/<total>" on success.
        persisted = 0
        picked = 0
        for line in out.splitlines():
            if "persisted=" in line and "[igio]" in line:
                try:
                    persisted_part = line.split("persisted=", 1)[1].split(" ", 1)[0]
                    if "/" in persisted_part:
                        p, t = persisted_part.split("/", 1)
                        persisted = int(p)
                        picked = int(t)
                except (ValueError, IndexError):
                    pass
        state.update(
            status="done" if proc.returncode == 0 else "error",
            picked=picked,
            persisted=persisted,
            returncode=proc.returncode,
            tail=out[-2000:],
            ended_at=time.time(),
        )
    except Exception as e:
        state.update(status="error", error=str(e), ended_at=time.time())
        _log.exception("igio backfill job %s failed", job_id)


@router.post("/stats/igio-backfill")
async def trigger_igio_backfill(
    info: TokenInfo = Depends(get_token_info),
    limit: int = 200,
    min_confidence: float = 0.5,
    model: str | None = None,
    workspace_id: str | None = None,
) -> dict:
    """Trigger an IGIO-axis backfill for chunks where igio_axis = ''.

    Admin-only (service token or ``scope='admin'``). Runs the same code path
    as ``python -m src.cli --classify-igio`` in a background thread so the
    request returns immediately with a job_id.
    Status via GET ``/stats/igio-backfill/{job_id}``.
    """
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    if model is None:
        # Issue #88: ModelRouter is the only allowed path so admin
        # /stats/admin/model-routes overrides actually take effect.
        from src.model_router import ModelRouter
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        model = ModelRouter(ollama_url).resolve("text") or "qwen2.5-coder:7b"
    job_id = f"igio-{int(time.time() * 1000)}"
    _IGIO_JOBS[job_id] = {
        "status": "queued",
        "limit": limit,
        "min_confidence": min_confidence,
        "model": model,
        "workspace_id": workspace_id,
        "queued_at": time.time(),
    }
    asyncio.create_task(_run_backfill_subprocess(
        job_id, limit, min_confidence, model, workspace_id,
    ))
    return {"job_id": job_id, "status": "queued"}


@router.get("/stats/igio-backfill/{job_id}")
async def get_igio_backfill_status(
    job_id: str,
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    state = _IGIO_JOBS.get(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="job not found")
    return {"job_id": job_id, **state}
