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

from mayring_core.memory.tasks import list_tasks
from src.api.auth import get_token_info
from src.api.dependencies import get_conn as _conn
from src.api.jwt_auth import TokenInfo
from src.wiki_v2.igio_classifier import VALID_AXES

router = APIRouter()
_log = logging.getLogger(__name__)

_IGIO_JOBS: dict[str, dict[str, Any]] = {}
_ROOT = Path(__file__).parent.parent.parent.parent


def _python_exe() -> str:
    venv = _ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else "python"


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


@router.get("/stats/igio-lens")
async def igio_lens(
    limit: int = 10,
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """Per-IGIO-axis distribution + recent chunks for the IGIO-Lens view.

    Workspace-scoped exactly like /stats/igio-coverage: a service token / admin
    (scope '*') sees all workspaces; a regular JWT sees only its own. Uncached —
    a GROUP BY on idx_chunks_workspace_id is cheap, and caching a live feed caused
    write-then-read staleness elsewhere.
    """
    limit = max(1, min(limit, 50))
    conn = _conn()
    admin = _is_admin(info)
    ws = info.workspace_id
    ws_clause = "" if admin else "AND workspace_id = ?"
    ws_params: tuple = () if admin else (ws,)
    scope = "all" if admin else "workspace"

    counts = {
        (row[0] or ""): row[1]
        for row in conn.execute(
            f"SELECT igio_axis, COUNT(*) FROM chunks "
            f"WHERE is_active = 1 {ws_clause} GROUP BY igio_axis",
            ws_params,
        ).fetchall()
    }

    axes: dict[str, dict] = {}
    for axis in VALID_AXES:
        rows = conn.execute(
            f"SELECT chunk_id, summary, text, source_id, igio_confidence, created_at "
            f"FROM chunks WHERE is_active = 1 AND igio_axis = ? {ws_clause} "
            f"ORDER BY created_at DESC LIMIT ?",
            (axis, *ws_params, limit),
        ).fetchall()
        axes[axis] = {
            "count": counts.get(axis, 0),
            "chunks": [
                {
                    "chunk_id":  r[0],
                    "preview":   (r[1] or r[2] or "")[:160],
                    "source_id": r[3],
                    "confidence": round(r[4] or 0.0, 3),
                    "created_at": r[5],
                }
                for r in rows
            ],
        }

    # Attach the workspace's task-list to the intervention axis so the lens
    # intervention column can render them (open first, then recently-done).
    # Admin scope sees all workspaces for chunks but todos are always scoped
    # to the caller's own workspace — task ownership is per-workspace.
    todo_ws = ws if not admin else ws
    open_todos = list_tasks(_conn(), todo_ws, status="open")
    done_todos = list_tasks(_conn(), todo_ws, status="done")
    axes["intervention"]["todos"] = [
        {
            "task_id":     t["task_id"],
            "title":       t["title"],
            "status":      t["status"],
            "created_by":  t.get("created_by"),
            "created_at":  t.get("created_at"),
            "completed_at": t.get("completed_at"),
        }
        for t in [*open_todos, *done_todos][:limit]
    ]

    return {
        "workspace_id": ws,
        "scope": scope,
        "axes": axes,
        "unclassified": {"count": counts.get("", 0)},
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
        from mayring_core.model_router import ModelRouter
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
