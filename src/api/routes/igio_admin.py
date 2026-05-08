"""IGIO classifier coverage + admin-triggered backfill.

Tracks Issue #141: existing chunks have an empty ``igio_axis`` because the
classifier was never run retroactively. These endpoints expose the coverage
ratio and an admin-only trigger that runs the same code path as
``python -m src.cli --classify-igio`` in a FastAPI background thread.

Mounted under ``/stats/`` so the production nginx whitelist (``/stats/*``)
already covers it — no nginx config change required.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.api.auth import get_token_info
from src.api.jwt_auth import TokenInfo
from src.memory.store import init_memory_db

router = APIRouter()
_log = logging.getLogger(__name__)

_IGIO_JOBS: dict[str, dict[str, Any]] = {}


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


def _run_backfill_sync(
    job_id: str, limit: int, threshold: float, model: str,
    workspace_id: str | None,
) -> None:
    """Blocking backfill loop. Mirrors ``_cmd_classify_igio`` from src/cli.py
    so behaviour stays identical to the local CLI path."""
    from src.wiki_v2.igio_classifier import classify_chunk, now_iso
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    state = _IGIO_JOBS[job_id]
    state.update(status="running", started_at=time.time())
    conn = _conn()
    try:
        if workspace_id:
            rows = conn.execute(
                "SELECT chunk_id, text, category_labels FROM chunks "
                "WHERE igio_axis = '' AND is_active = 1 AND text != '' "
                "AND workspace_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (workspace_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT chunk_id, text, category_labels FROM chunks "
                "WHERE igio_axis = '' AND is_active = 1 AND text != '' "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        state.update(picked=len(rows))
        persisted = 0
        counts: dict[str, int] = {a: 0 for a in
                                  ("issue", "goal", "intervention", "outcome", "")}
        for r in rows:
            cats = [c for c in (r["category_labels"] or "").split(",") if c]
            try:
                verdict = classify_chunk(
                    r["text"], cats, ollama_url=ollama_url, model=model,
                )
            except Exception as e:
                _log.warning("igio classify failed for %s: %s", r["chunk_id"], e)
                continue
            counts[verdict.axis] = counts.get(verdict.axis, 0) + 1
            if verdict.axis and verdict.confidence >= threshold:
                conn.execute(
                    "UPDATE chunks SET igio_axis = ?, igio_confidence = ?, "
                    "igio_classified_at = ? WHERE chunk_id = ?",
                    (verdict.axis, verdict.confidence, now_iso(), r["chunk_id"]),
                )
                persisted += 1
        conn.commit()
        state.update(status="done", persisted=persisted,
                     counts=counts, ended_at=time.time())
    except Exception as e:
        state.update(status="error", error=str(e), ended_at=time.time())
        _log.exception("igio backfill job %s failed", job_id)
    finally:
        conn.close()


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
        model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
    job_id = f"igio-{int(time.time() * 1000)}"
    _IGIO_JOBS[job_id] = {
        "status": "queued",
        "limit": limit,
        "min_confidence": min_confidence,
        "model": model,
        "workspace_id": workspace_id,
        "queued_at": time.time(),
    }
    asyncio.create_task(asyncio.to_thread(
        _run_backfill_sync, job_id, limit, min_confidence, model, workspace_id,
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
