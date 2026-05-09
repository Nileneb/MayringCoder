"""Read-only dashboard endpoints.

Pure aggregations over data that already lives in ``memory.db`` (or in-process
ring buffers). No new tables — see the per-endpoint docstrings for source.
The Laravel ``MemoryDashboard`` Livewire component fans out to these.

Authentication is shared with the rest of the API (workspace-scoped via
``get_workspace`` dependency). Admins see cross-workspace data when an
explicit ``workspace_id`` query param is given.
"""
from __future__ import annotations

import json as _json
from typing import Any

from fastapi import APIRouter, Depends

from src.api.auth import get_workspace
from src.memory.store import init_memory_db

router = APIRouter()


# Single shared connection — same pattern memory.py uses, avoids per-request
# init cost. Lazy because pytest fixtures may swap CACHE_DIR.
def _conn():
    from src.config import CACHE_DIR
    return init_memory_db(CACHE_DIR / "memory.db")


# ---------------------------------------------------------------------------
# 1. recent ingestion events  →  ingestion_log
# ---------------------------------------------------------------------------

@router.get("/stats/recent-ops")
async def recent_ops(
    source_id: str | None = None,
    since_minutes: int | None = None,
    limit: int = 50,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Live ingestion timeline. Same shape as ``/stats/summary.recent_ops``
    but filterable + standalone for view-side polling without the full summary."""
    sql = ["SELECT event_type, source_id, payload, created_at FROM ingestion_log WHERE 1=1"]
    params: list = []
    if source_id:
        sql.append("AND source_id = ?")
        params.append(source_id)
    if since_minutes:
        sql.append("AND created_at > datetime('now', ?)")
        params.append(f"-{since_minutes} minutes")
    sql.append("ORDER BY created_at DESC LIMIT ?")
    params.append(limit)
    rows = _conn().execute(" ".join(sql), params).fetchall()
    return {
        "workspace_id": workspace_id,
        "ops": [
            {
                "event_type": r[0],
                "source_id": r[1],
                "payload": _safe_json(r[2]),
                "created_at": r[3],
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# 2. job history  →  in-memory _JOBS dict (now JSON-persisted at shutdown)
# ---------------------------------------------------------------------------

# Path is /stats/jobs-history rather than /jobs/history because jobs.py owns
# /jobs/{job_id} as a catch-all path-param — registering /jobs/history would
# either be shadowed by it or shadow it depending on include order, both bad.
@router.get("/stats/jobs-history")
async def jobs_history(
    status: str | None = None,
    limit: int = 50,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """All checker/wiki/duel jobs from the in-process queue plus persisted state.

    Persistence: ``_JOBS`` is written to ``cache/jobs_state.json`` whenever a
    job's status changes; loaded on FastAPI startup. So a container restart
    no longer wipes the visible history.
    """
    from src.api.job_queue import _JOBS

    # Multi-Tenant: Nur Jobs des aufrufenden Workspaces zurückgeben.
    # 'system' (Service-Token) sieht alles — ist Maintenance-Bucket.
    # Ohne diesen Filter zeigte das memory-dashboard.blade jobs aus
    # smoke-runs (ws=system) im User-Account, das war confusing.
    def _ws_match(j: dict) -> bool:
        if workspace_id == "system":
            return True
        return j.get("workspace_id") == workspace_id

    items = sorted(
        (j for j in _JOBS.values()
         if _ws_match(j) and (not status or j.get("status") == status)),
        key=lambda x: x.get("started_at", ""),
        reverse=True,
    )[:limit]
    # Bei status='error' liefere die letzten 1200 chars des output-Logs
    # zurück (typischerweise Traceback). Ohne dieses Feld zeigt das
    # app.linn.games-Memory-Dashboard nur status='error' ohne Detail —
    # User sieht nicht warum der Job gefailed ist.
    def _error_tail(j: dict) -> str | None:
        if j.get("status") != "error":
            return None
        out = j.get("output") or ""
        if not out:
            return None
        # Letzte 1200 chars; wenn output länger ist, mit Ellipsis prefix.
        TAIL = 1200
        if len(out) <= TAIL:
            return out
        return "…\n" + out[-TAIL:]

    return {
        "workspace_id": workspace_id,
        "jobs": [
            {
                "job_id": j.get("job_id"),
                "status": j.get("status"),
                "started_at": j.get("started_at"),
                "stages": j.get("stages", {}),
                "progress": j.get("progress"),
                "workspace_id": j.get("workspace_id"),
                "error_tail": _error_tail(j),
            }
            for j in items
        ],
    }


# ---------------------------------------------------------------------------
# 3. context feedback log  →  context_feedback_log
# ---------------------------------------------------------------------------

@router.get("/stats/feedback-log")
async def feedback_log(
    limit: int = 50,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """How often does an injected chunk actually get referenced by Claude?

    Killer-metric for memory effectiveness. ``referenced_rate`` answers
    "are we wasting prompt tokens?" at a glance.
    """
    conn = _conn()
    total_24h = conn.execute(
        "SELECT COUNT(*) FROM context_feedback_log WHERE captured_at > datetime('now','-24 hours')"
    ).fetchone()[0]
    referenced_24h = conn.execute(
        "SELECT COUNT(*) FROM context_feedback_log "
        "WHERE captured_at > datetime('now','-24 hours') AND was_referenced = 1"
    ).fetchone()[0]
    rows = conn.execute(
        "SELECT trigger_ids, context_text, was_referenced, led_to_retrieval, "
        "       relevance_score, captured_at "
        "FROM context_feedback_log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return {
        "workspace_id": workspace_id,
        "injections_24h": total_24h,
        "referenced_24h": referenced_24h,
        "referenced_rate": round(referenced_24h / total_24h, 3) if total_24h else 0.0,
        "recent": [
            {
                "trigger_ids": _safe_json(r[0]),
                "context_preview": (r[1] or "")[:200],
                "was_referenced": bool(r[2]),
                "led_to_retrieval": bool(r[3]),
                "relevance_score": r[4],
                "captured_at": r[5],
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# 4. cross-source chunk refs  →  chunk_source_refs
# ---------------------------------------------------------------------------

@router.get("/stats/source-refs")
async def source_refs(
    limit: int = 50,
    min_sources: int = 2,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Chunks that surface in multiple sources — dedup awareness."""
    rows = _conn().execute(
        "SELECT canonical_chunk_id, COUNT(DISTINCT source_id) AS n, "
        "       GROUP_CONCAT(source_id, '||') AS sids "
        "FROM chunk_source_refs "
        "WHERE workspace_id = ? "
        "GROUP BY canonical_chunk_id HAVING n >= ? "
        "ORDER BY n DESC LIMIT ?",
        (workspace_id, min_sources, limit),
    ).fetchall()
    return {
        "workspace_id": workspace_id,
        "refs": [
            {
                "canonical_chunk_id": r[0],
                "source_count": r[1],
                "sources": (r[2] or "").split("||")[:20],
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# 5. wiki triggers  →  trigger_stats
# ---------------------------------------------------------------------------

@router.get("/stats/triggers")
async def triggers(
    only_active: bool = True,
    limit: int = 50,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Wiki trigger fire/reference rates. Low ratio → trigger fires but is
    never useful, prime candidate for retirement."""
    sql = ["SELECT trigger_id, fire_count, ref_count, is_active, last_fired FROM trigger_stats"]
    if only_active:
        sql.append("WHERE is_active = 1")
    sql.append("ORDER BY fire_count DESC LIMIT ?")
    rows = _conn().execute(" ".join(sql), (limit,)).fetchall()
    return {
        "workspace_id": workspace_id,
        "triggers": [
            {
                "trigger_id": r[0],
                "fire_count": r[1],
                "ref_count": r[2],
                "is_active": bool(r[3]),
                "last_fired": r[4],
                "ratio": round(r[2] / r[1], 3) if r[1] else 0.0,
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# 6. topic transitions  →  topic_transitions
# ---------------------------------------------------------------------------

@router.get("/stats/topic-flow")
async def topic_flow(
    from_topic: str | None = None,
    limit: int = 50,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Predictive memory paths — "Du arbeitest an X, als nächstes typischerweise Y/Z"."""
    sql = ["SELECT from_topic, to_topic, count, last_seen FROM topic_transitions"]
    params: list = []
    if from_topic:
        sql.append("WHERE from_topic = ?")
        params.append(from_topic)
    sql.append("ORDER BY count DESC LIMIT ?")
    params.append(limit)
    rows = _conn().execute(" ".join(sql), params).fetchall()
    return {
        "workspace_id": workspace_id,
        "flows": [
            {
                "from_topic": r[0],
                "to_topic": r[1],
                "count": r[2],
                "last_seen": r[3],
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# 7. pi-agent task queue  →  pi_jobs
# ---------------------------------------------------------------------------

# Path is /stats/pi-tasks (not /pi/tasks): the production nginx whitelist
# only matches `pi-task|pi_task` as top-level segments, so /pi/... falls
# through to the default location and gets routed to the MCP server
# (404). /stats/* is already whitelisted.
@router.get("/stats/pi-tasks")
async def pi_tasks(
    status: str | None = None,
    limit: int = 50,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """All pi-agent tasks for the caller's workspace, newest first."""
    sql = [
        "SELECT job_id, task_text, status, prefer, scope, model, error, "
        "       created_at, updated_at FROM pi_jobs "
        "WHERE workspace_id = ?"
    ]
    params: list = [workspace_id]
    if status:
        sql.append("AND status = ?")
        params.append(status)
    sql.append("ORDER BY created_at DESC LIMIT ?")
    params.append(limit)
    try:
        rows = _conn().execute(" ".join(sql), params).fetchall()
    except Exception:
        # pi_jobs table may not exist on older DBs; fail soft
        return {"workspace_id": workspace_id, "tasks": []}
    return {
        "workspace_id": workspace_id,
        "tasks": [
            {
                "job_id": r[0],
                "task_preview": (r[1] or "")[:120],
                "status": r[2],
                "prefer": r[3],
                "scope": r[4],
                "model": r[5],
                "error": r[6] or None,
                "created_at": r[7],
                "updated_at": r[8],
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# 8. recent search activations  →  memory_service._RECENT_ACTIVATIONS
# ---------------------------------------------------------------------------

@router.get("/stats/activations")
async def activations(
    limit: int = 50,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Last N memory searches across all workspaces (admins) or just the
    caller's workspace (tenants)."""
    from src.api.memory_service import _RECENT_ACTIVATIONS
    from src.api.mcp_auth import _TOKEN_CTX

    info = _TOKEN_CTX.get()
    is_admin = bool(info and "admin" in (info.scopes or ()))
    items = list(_RECENT_ACTIVATIONS)
    if not is_admin:
        items = [a for a in items if a.get("workspace_id") == workspace_id]
    items = sorted(items, key=lambda a: a.get("ts", 0), reverse=True)[:limit]
    return {
        "workspace_id": workspace_id,
        "activations": items,
    }


# ---------------------------------------------------------------------------
# 9. workspace breakdown  →  GROUP BY chunks/sources
# ---------------------------------------------------------------------------

@router.get("/stats/workspaces")
async def workspaces(workspace_id: str = Depends(get_workspace)) -> dict:
    """Per-workspace activity. Admins see all, tenants see just their own.

    Useful to spot drift (one user with 3 parallel workspaces) or stale
    workspaces that haven't been touched in months.
    """
    from src.api.mcp_auth import _TOKEN_CTX

    info = _TOKEN_CTX.get()
    is_admin = bool(info and "admin" in (info.scopes or ()))

    conn = _conn()
    if is_admin:
        rows = conn.execute(
            "SELECT c.workspace_id, "
            "       COUNT(DISTINCT c.chunk_id), "
            "       COUNT(DISTINCT s.source_id), "
            "       MAX(c.created_at) "
            "FROM chunks c LEFT JOIN sources s ON s.workspace_id = c.workspace_id "
            "GROUP BY c.workspace_id ORDER BY 4 DESC"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT ?, COUNT(DISTINCT c.chunk_id), "
            "       COUNT(DISTINCT s.source_id), MAX(c.created_at) "
            "FROM chunks c LEFT JOIN sources s ON s.workspace_id = c.workspace_id "
            "WHERE c.workspace_id = ?",
            (workspace_id, workspace_id),
        ).fetchall()

    return {
        "workspace_id": workspace_id,
        "workspaces": [
            {
                "workspace_id": r[0],
                "chunks": r[1],
                "sources": r[2],
                "last_activity": r[3],
            }
            for r in rows if r[0]
        ],
    }


# ---------------------------------------------------------------------------
# 10. vector search trends  →  llm_calls_log with call_type='vector_search'
# ---------------------------------------------------------------------------

@router.get("/stats/llm-call-types")
async def llm_call_types(
    days: int = 1,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Per-call_type aggregate counts from ``llm_calls_log`` over the
    last `days` days. Smoke probe for #101 (categorization logging)
    deepens the existing 'logged_24h > 0' check by asserting
    ``call_type='categorization'`` specifically appears — proves the
    Mayring categorisation pipeline is logging, not just other LLM
    paths happening to push the counter."""
    days = max(1, min(days, 30))
    conn = _conn()
    rows = conn.execute(
        "SELECT call_type, COUNT(*) AS n FROM llm_calls_log "
        "WHERE created_at > datetime('now', ?) "
        "GROUP BY call_type ORDER BY n DESC",
        (f"-{days} days",),
    ).fetchall()
    return {
        "workspace_id": workspace_id,
        "window_days": days,
        "counts": {row[0]: int(row[1]) for row in rows},
    }


@router.get("/stats/vector-trend")
async def vector_trend(
    limit: int = 50,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Vector-stage success-rate over time. Reuses ``llm_calls_log`` with a
    dedicated ``call_type='vector_search'`` so we don't introduce yet
    another write-heavy table. Logging runs on every search."""
    conn = _conn()
    last_24h = conn.execute(
        "SELECT COUNT(*) FROM llm_calls_log "
        "WHERE call_type = 'vector_search' AND created_at > datetime('now','-24 hours')"
    ).fetchone()[0]
    rows = conn.execute(
        "SELECT prompt, response, duration_ms, created_at FROM llm_calls_log "
        "WHERE call_type = 'vector_search' "
        "ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    successes = 0
    score_sum = 0.0
    for _q, resp_json, _ms, _ts in rows:
        diag = _safe_json(resp_json)
        if isinstance(diag, dict):
            stage = str(diag.get("vector_stage", ""))
            if stage.startswith("ok("):
                successes += 1
            try:
                # crude parse of "ok(max_score=0.5,matches=...,mean_dist=...)"
                if "max_score=" in stage:
                    score_sum += float(stage.split("max_score=")[1].split(",")[0])
            except (ValueError, IndexError):
                pass
    return {
        "workspace_id": workspace_id,
        "logged_24h": last_24h,
        "success_rate": round(successes / len(rows), 3) if rows else 0.0,
        "mean_max_score": round(score_sum / max(1, successes), 3) if successes else 0.0,
        "recent": [
            {
                "query": (r[0] or "")[:120],
                "diagnostics": _safe_json(r[1]),
                "duration_ms": r[2],
                "created_at": r[3],
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _safe_json(raw: Any) -> Any:
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str):
        return raw
    try:
        return _json.loads(raw)
    except (ValueError, TypeError):
        return raw
