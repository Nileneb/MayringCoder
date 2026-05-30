"""Read-only dashboard endpoints.

Pure aggregations over data that already lives in ``memory.db`` (or in-process
ring buffers). No new tables — see the per-endpoint docstrings for source.
The Laravel ``MemoryDashboard`` Livewire component fans out to these.

Authentication is shared with the rest of the API (workspace-scoped via
``get_workspace`` dependency). Admins see cross-workspace data when an
explicit ``workspace_id`` query param is given.
"""
from __future__ import annotations

import functools as _functools
import json as _json
import time as _time
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.auth import get_token_info, get_workspace
from src.api.dependencies import get_conn as _conn
from src.api import shared_state as _shared_state
from src.api.jwt_auth import TokenInfo

router = APIRouter()


# ---------------------------------------------------------------------------
# Short TTL cache for the read-only aggregation endpoints below.
# ---------------------------------------------------------------------------

_DASH_CACHE: dict[str, tuple[float, Any]] = {}
_DASH_CACHE_TTL = 15.0
_DASH_CACHE_MAX = 2000


def _dashboard_ttl_cache(fn):
    """Per-(endpoint, args, workspace) TTL cache for read-only dashboard reads.

    WHY(api-saturation 2026-05-24): the Livewire dashboards poll ~10 of these
    endpoints every 10s and each is a fresh SQLite COUNT/GROUP-BY scan over
    million-row tables. On the single-worker API, concurrent polls saturated
    the event loop (every request 14s+, even /health starved). A short TTL
    means most polls hit memory instead of the DB; staleness is fine for
    observability widgets.

    Keyed by the resolved kwargs — crucially ``workspace_id`` (a Depends
    kwarg) — so a cache entry can never bleed across tenants. ``functools.wraps``
    preserves the original signature so FastAPI still resolves Depends/query
    params. Apply BELOW ``@router.get`` so the router registers the wrapper.
    """
    @_functools.wraps(fn)
    async def _wrapper(**kwargs):
        key = fn.__name__ + ":" + repr(sorted(kwargs.items()))
        now = _time.monotonic()
        # Shared L2 (Redis) — consistent across uvicorn workers when reachable.
        shared = _shared_state.cache_get("dash:" + key)
        if shared is not None:
            return shared
        # Per-process L1 (and the sole cache if Redis is down).
        hit = _DASH_CACHE.get(key)
        if hit is not None and hit[0] > now:
            return hit[1]
        value = await fn(**kwargs)
        if len(_DASH_CACHE) >= _DASH_CACHE_MAX:
            _DASH_CACHE.clear()
        _DASH_CACHE[key] = (now + _DASH_CACHE_TTL, value)
        _shared_state.cache_set("dash:" + key, value, _DASH_CACHE_TTL)
        return value

    return _wrapper


# ---------------------------------------------------------------------------
# 1. recent ingestion events  →  ingestion_log
# ---------------------------------------------------------------------------

@router.get("/stats/recent-ops")
async def recent_ops(  # NOT cached: live ingest feed (WHY, smoke-fix 2026-05-24).
    # The 15s @_dashboard_ttl_cache served the pre-ingest list on a write-then-read
    # (watcher_hook_fires polled within 8s and never saw the just-PUT source until
    # the TTL expired). ORDER BY created_at DESC LIMIT is index-backed (v13), so
    # running it per-poll is cheap — same call as the /stats/feedback-log fix.
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
    include_smoke: bool = False,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """All checker/wiki/duel jobs from the in-process queue plus persisted state.

    Persistence: ``_JOBS`` is written to ``cache/jobs_state.json`` whenever a
    job's status changes; loaded on FastAPI startup. So a container restart
    no longer wipes the visible history.
    """
    from src.api.job_queue import _JOBS, _load_jobs

    # Multi-Tenant: Nur Jobs des aufrufenden Workspaces zurückgeben.
    # 'system' (Service-Token) sieht alles — ist Maintenance-Bucket.
    # Ohne diesen Filter zeigte das memory-dashboard.blade jobs aus
    # smoke-runs (ws=system) im User-Account, das war confusing.
    def _ws_match(j: dict) -> bool:
        if workspace_id == "system":
            return True
        return j.get("workspace_id") == workspace_id

    # WHY(pi-dashboard multi-worker 2026-05-25): _JOBS is per-PROCESS — under
    # uvicorn --workers 4 a request hits one worker and saw only ~1/4 of jobs
    # (often none) → the Pi-Agent dashboard job list looked empty/broken.
    # _save_jobs writes the WHOLE registry to the shared jobs_state.json on every
    # status change, so _load_jobs() is the cross-worker source of truth. Merge
    # local _JOBS on top for the freshest in-process status.
    # WHY(#253): smoke-triggered populate jobs (source="smoke") fail by design
    # against a known-bad repo and pile up in workspace:system as noise. Hide
    # them by default; include_smoke=true surfaces them for debugging.
    def _smoke_match(j: dict) -> bool:
        return include_smoke or j.get("source") != "smoke"

    merged: dict[str, dict] = {**_load_jobs(), **_JOBS}
    items = sorted(
        (j for j in merged.values()
         if _ws_match(j) and _smoke_match(j)
         and (not status or j.get("status") == status)),
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
                "source": j.get("source", ""),
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

    NOT cached (WHY, smoke-fix 2026-05-24): this is a LIVE-movement metric —
    ``injections_24h`` ticks up on every search. The 15s shared cache served the
    pre-search value on an immediate re-read (feedback_log_movement smoke red once
    the cartesian-join saturation that used to expire the cache between calls was
    fixed). The COUNT is a range-scan on idx_context_feedback_captured (v12), so
    running it per-poll is cheap.
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
# 3b. per-prompt retrieval trace  →  context_feedback_log (query + stage_scores)
# ---------------------------------------------------------------------------

@router.get("/stats/prompt-trace")
@_dashboard_ttl_cache
async def prompt_trace(
    limit: int = 20,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Per-prompt retrieval trace — "what actually happened for THIS prompt?".

    For each recent memory search: the query that ran, which chunks were
    injected, their per-stage reranker scores (vector/symbolic/recency/
    affinity/feedback/llm/predicted-topic/rationale/final), the reranker
    version, and whether Claude ended up referencing the context. Sourced from
    ``context_feedback_log`` (query + trigger_ids + stage_scores), enriched with
    each chunk's ``source_id`` (batch-resolved in one query — no N+1).

    Distinct from ``/stats/feedback-log``, which is the 24h aggregate metric and
    deliberately omits the per-prompt query + scores.
    """
    conn = _conn()
    rows = conn.execute(
        "SELECT id, query, trigger_ids, stage_scores, reranker_version, "
        "       was_referenced, relevance_score, captured_at "
        "FROM context_feedback_log "
        "WHERE workspace_id = ? AND query != '' "
        "ORDER BY id DESC LIMIT ?",
        (workspace_id, limit),
    ).fetchall()

    parsed: list[tuple] = []
    all_ids: set[str] = set()
    for r in rows:
        ids = _safe_json(r[2]) or []
        if not isinstance(ids, list):
            ids = []
        scores = _safe_json(r[3]) or {}
        if not isinstance(scores, dict):
            scores = {}
        all_ids.update(i for i in ids if isinstance(i, str))
        parsed.append((r, ids, scores))

    # Batch-resolve source_id for every chunk across all rows (no N+1).
    src_map: dict[str, str] = {}
    if all_ids:
        ids_list = list(all_ids)
        qp = ",".join("?" * len(ids_list))
        for cid, sid in conn.execute(
            f"SELECT chunk_id, source_id FROM chunks WHERE chunk_id IN ({qp})",
            tuple(ids_list),
        ).fetchall():
            src_map[cid] = sid

    prompts = []
    for r, ids, scores in parsed:
        chunks = []
        for cid in ids:
            sc = scores.get(cid, {}) if isinstance(scores.get(cid), dict) else {}
            chunks.append({
                "chunk_id": cid,
                "source_id": src_map.get(cid),
                "final": sc.get("f"),
                "scores": sc,
            })
        # Order by the reranker's final score — what it actually preferred.
        chunks.sort(key=lambda c: c["final"] if c["final"] is not None else -1.0,
                    reverse=True)
        prompts.append({
            "id": r[0],
            "query": r[1],
            "reranker_version": r[4],
            "was_referenced": bool(r[5]),
            "relevance_score": r[6],
            "captured_at": r[7],
            "chunk_count": len(ids),
            "chunks": chunks,
        })
    return {"workspace_id": workspace_id, "prompts": prompts}


# ---------------------------------------------------------------------------
# 4. cross-source chunk refs  →  chunk_source_refs
# ---------------------------------------------------------------------------

@router.get("/stats/source-refs")
@_dashboard_ttl_cache
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
@_dashboard_ttl_cache
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
@_dashboard_ttl_cache
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
@_dashboard_ttl_cache
async def pi_tasks(
    status: str | None = None,
    limit: int = 50,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """All pi-agent tasks for the caller's workspace, newest first."""
    sql = [
        # NOTE: pi_jobs has no `updated_at` column — selecting it errored and the
        # fail-soft except below returned an empty list, so this endpoint showed
        # nothing even when rows existed. `finished_at` is the real terminal-time
        # column; map it onto the response's updated_at to keep the shape.
        "SELECT job_id, task_text, status, prefer, scope, model, error, "
        "       created_at, finished_at FROM pi_jobs "
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


class _PiTaskRecord(BaseModel):
    job_id: str
    task_text: str = ""
    status: str = "completed"
    prefer: str = "auto"
    model: str = ""
    scope: str = "local"
    repo_slug: str = ""
    result: str | None = None
    error: str | None = None
    claimed_by: str = ""
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


# WHY(pi-observability): pi-tasks run locally in the plugin (Phase-1 always-local)
# and never reached the cloud, so this dashboard was structurally empty. The MCP
# server mirrors each task here so the Pi-Agent view reflects real activity.
# Path stays under /stats/ so the prod nginx whitelist already routes it (no
# allowlist change). Upsert by job_id, scoped to the caller's workspace.
@router.post("/stats/pi-tasks/record")
async def record_pi_task(
    rec: _PiTaskRecord,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Mirror a (locally-executed) pi-task into the cloud pi_jobs table."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    result_json = _json.dumps({"text": rec.result}) if rec.result is not None else ""
    try:
        conn = _conn()
        conn.execute(
            "INSERT INTO pi_jobs (job_id, task_text, repo_slug, workspace_id, "
            "status, prefer, ollama_url, model, result_json, error, timeout_s, "
            "scope, capability_required, claimed_by, claimed_at, created_at, "
            "started_at, finished_at) "
            "VALUES (?, ?, ?, ?, ?, ?, '', ?, ?, ?, 0, ?, '', ?, '', ?, ?, ?) "
            "ON CONFLICT(job_id) DO UPDATE SET "
            "  status=excluded.status, model=excluded.model, "
            "  result_json=excluded.result_json, error=excluded.error, "
            "  claimed_by=excluded.claimed_by, started_at=excluded.started_at, "
            "  finished_at=excluded.finished_at",
            (
                rec.job_id, rec.task_text, rec.repo_slug, workspace_id,
                rec.status, rec.prefer, rec.model, result_json, rec.error or "",
                rec.scope, rec.claimed_by, rec.created_at or now,
                rec.started_at or "", rec.finished_at or "",
            ),
        )
        conn.commit()
    except Exception as exc:  # fail-soft: observability must never 500 a caller
        import logging
        # Log the detail server-side; never echo the exception text to the caller
        # (py/stack-trace-exposure). The caller only needs the soft-fail signal.
        logging.getLogger(__name__).warning("record_pi_task failed: %s", exc)
        return {"ok": False, "error": "record failed", "workspace_id": workspace_id}
    return {"ok": True, "job_id": rec.job_id, "workspace_id": workspace_id}


# ---------------------------------------------------------------------------
# 8. recent search activations  →  memory_service._RECENT_ACTIVATIONS
# ---------------------------------------------------------------------------

@router.get("/stats/activations")
async def activations(
    limit: int = 50,
    workspace_id: str = Depends(get_workspace),
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """Last N memory searches across all workspaces (admins) or just the
    caller's workspace (tenants)."""
    from src.api.memory_service import _RECENT_ACTIVATIONS

    # WHY(tenancy-audit 2026-05-31): identity MUST come from the request
    # dependency, NOT mcp_auth._TOKEN_CTX — that ContextVar is only set by the
    # MCP ASGI sub-app's middleware, so on every REST route it is None → admins
    # silently lost their cross-workspace view. get_token_info also carries the
    # verified X-Act-As-* override.
    is_admin = info.is_admin
    # Shared (Redis) activations span all workers; None → Redis down, use local deque.
    items = _shared_state.activations_redis()
    if items is None:
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
async def workspaces(
    workspace_id: str = Depends(get_workspace),
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """Per-workspace activity.

    Admins see all workspaces in the DB. Tenants see every workspace they're
    a member of — derived from the JWT `memberships[]` claim (V2). For
    backward-compat with old JWTs without memberships, falls back to the
    single active workspace_id.
    """
    # WHY(tenancy-audit 2026-05-31): identity from get_token_info, NOT
    # mcp_auth._TOKEN_CTX (None on every REST route → admins lost the all-view,
    # multi-membership users saw only their active ws → smoke stats_workspaces
    # red). get_token_info also honours the X-Act-As-* override.
    is_admin = info.is_admin
    # WHY(#195, v2): pre-V2 the non-admin branch listed exactly one row
    # (the active workspace). After V2 a user in N workspaces sees all N.
    # Source-of-truth is the JWT `memberships[]` claim from Laravel; falls
    # back to {workspace_id} for legacy JWTs without the field.
    member_ws_ids: set[str] = {workspace_id}
    member_ws_types: dict[str, str] = {}
    if info.memberships:
        for m in info.memberships:
            if m.id:
                member_ws_ids.add(m.id)
                member_ws_types[m.id] = m.type

    conn = _conn()
    # WHY(smoke-fix 2026-05-24): the old query did
    #   chunks c LEFT JOIN sources s ON s.workspace_id = c.workspace_id
    # — a per-workspace CARTESIAN product (every chunk × every source in the same
    # ws) before COUNT(DISTINCT). On bene's workspace (~60k chunks × ~4k sources)
    # that's ~250M intermediate rows → /stats/workspaces timed out (smoke
    # workspace_scoping/dashboard_endpoints red, box saturated). Count chunks and
    # sources INDEPENDENTLY per workspace (index GROUP BY on workspace_id) and
    # merge — identical result, O(chunks+sources) instead of O(chunks×sources).
    if is_admin:
        chunk_rows = conn.execute(
            "SELECT workspace_id, COUNT(*), MAX(created_at) "
            "FROM chunks GROUP BY workspace_id"
        ).fetchall()
        src_rows = conn.execute(
            "SELECT workspace_id, COUNT(*) FROM sources GROUP BY workspace_id"
        ).fetchall()
    else:
        # SAFE: placeholders is "?,?,?" — no user-input concatenation.
        placeholders = ",".join("?" * len(member_ws_ids))
        chunk_rows = conn.execute(
            f"SELECT workspace_id, COUNT(*), MAX(created_at) FROM chunks "
            f"WHERE workspace_id IN ({placeholders}) GROUP BY workspace_id",
            tuple(member_ws_ids),
        ).fetchall()
        src_rows = conn.execute(
            f"SELECT workspace_id, COUNT(*) FROM sources "
            f"WHERE workspace_id IN ({placeholders}) GROUP BY workspace_id",
            tuple(member_ws_ids),
        ).fetchall()
    _src_counts = {r[0]: r[1] for r in src_rows}
    # chunks drive the row set (matches the old chunks-LEFT-JOIN-sources shape:
    # a ws with chunks but no sources still appears with sources=0).
    rows = [(r[0], r[1], _src_counts.get(r[0], 0), r[2]) for r in chunk_rows]
    rows.sort(key=lambda r: (r[3] or ""), reverse=True)  # ORDER BY last_activity DESC

    return {
        "workspace_id": workspace_id,
        "workspaces": [
            {
                "workspace_id": r[0],
                "type": member_ws_types.get(r[0], "personal" if r[0] == workspace_id else "unknown"),
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
@_dashboard_ttl_cache
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
