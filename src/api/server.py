"""MayringCoder Multi-Tenant API Server.

FastAPI HTTP layer. Auth: **RS256-JWT** (2026-04 umgestellt von Sanctum).

JWTs werden von app.linn.games' JwtIssuer ausgestellt und tragen
``workspace_id``, ``sub``, ``scope`` sowie BYO-Provider-Claims. Dieser
Server validiert sie **offline** gegen den Public-Key unter
JWT_PUBLIC_KEY_PATH — keine Laravel-DB-Roundtrip mehr nötig.

Start:
    .venv/bin/python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8090

Required env:
    JWT_PUBLIC_KEY_PATH   — path to RS256 public key (PEM)
    JWT_ISSUER            — expected `iss` claim (default: https://app.linn.games)
    JWT_AUDIENCE          — expected `aud` claim (default: mayringcoder)
    OLLAMA_URL            — Ollama endpoint (default three.linn.games)
    EMBED_BATCH_SIZE      — embedding batch size (default 32)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_ROOT / ".env")

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:
    raise ImportError("Missing dependency: pip install fastapi uvicorn")

from src.api.dependencies import get_conn as _get_conn
from src.api.training import router as _training_router
from fastapi import Depends, Header
from src.api.auth import get_workspace, get_token_info
from src.api.routes import memory, wiki, jobs, duel, reports
from src.api.routes.sync import router as _sync_router
from src.api.job_queue import _JOBS, run_checker_job as _run_checker_job
from src.api.routes import pi_stats as _pi_stats

# JSON-Log-File für log-ingest-Cron. Wird vor FastAPI() konfiguriert,
# damit die Startup-Logs (route-include, db-init) auch landen.
from mayring_core.logging_setup import configure_json_logging
configure_json_logging()

# Wire embed/generate/vision into mayring_core so a standalone
# `uvicorn src.api.server:app` registers them too, not only `python -m src.main` (#267).
from src.provider_setup import setup_providers
setup_providers()

app = FastAPI(title="MayringCoder API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(_training_router)
app.include_router(memory.router)
app.include_router(wiki.router)
app.include_router(jobs.router)
app.include_router(duel.router)
app.include_router(reports.router)
app.include_router(_sync_router)
from src.api.routes import dashboard as _dashboard
app.include_router(_dashboard.router)
from src.api.routes import stats_admin as _stats_admin
app.include_router(_stats_admin.router)
from src.api.routes import model_router_admin as _model_router_admin
app.include_router(_model_router_admin.router)
from src.api.routes import wiki_second_opinion as _wiki_second_opinion
app.include_router(_wiki_second_opinion.router)
from src.api.routes import retrieval_metrics as _retrieval_metrics
app.include_router(_retrieval_metrics.router)
from src.api.routes import reranker_admin as _reranker_admin
app.include_router(_reranker_admin.router)
app.include_router(_pi_stats.router)
from src.api.routes import reconcile_admin as _reconcile_admin
app.include_router(_reconcile_admin.router)
from src.api.routes import admin_logs as _admin_logs
app.include_router(_admin_logs.router)
from src.api.routes import tasks as _tasks
app.include_router(_tasks.router)
from src.api.routes import devices as _devices
app.include_router(_devices.router)
from src.api.routes import embed_pool as _embed_pool  # #365 Schicht 3: distributed embedding pool
app.include_router(_embed_pool.router)
from src.api.routes import codebooks as _codebooks  # #workspace-uuid-sot v2.0 P1.3
app.include_router(_codebooks.router)
from src.api.routes import projects as _projects  # Project Router Slice 1
app.include_router(_projects.router)
from src.api.routes import repo_events as _repo_events  # Repo-watching Task 2
app.include_router(_repo_events.router)
from src.api.routes import ambient_admin as _ambient_admin  # ambient snapshot refresh
app.include_router(_ambient_admin.router)
from src.api.routes import backfill_admin as _backfill_admin  # tenancy phase A: chroma visibility backfill
app.include_router(_backfill_admin.router)
from src.api.routes import role_permissions as _role_permissions  # tenancy phase B
app.include_router(_role_permissions.router)
from src.api.routes import sources_list as _sources_list  # tenancy phase C
app.include_router(_sources_list.router)
from src.api.routes import purge_admin as _purge_admin  # smoke self-clean (#253)
app.include_router(_purge_admin.router)
from src.api.routes import watch_repos as _watch_repos  # dashboard-managed repo-watch list
app.include_router(_watch_repos.router)
from src.api.routes import workspace_claim as _workspace_claim  # claim unclaimed:<device> buckets
app.include_router(_workspace_claim.router)
from src.api.routes import agent_keys as _agent_keys  # per-agent A2A API keys (X-API-Key)
app.include_router(_agent_keys.router)
from src.api.routes import text_models as _text_models  # runtime text-model override
app.include_router(_text_models.router)

# A2A research-relay (Langdock → cloud queue → laptop worker). Mounts the
# agent-card + JSON-RPC (/a2a) onto this app. db_path + workspace_id MUST match
# what the worker's claim reads (devices.py claim_cloud_next), or the job is
# never claimed — see the relay spec.
if os.getenv("MAYRING_A2A_RELAY_ENABLED", "1") == "1":
    from src.api.a2a_relay import register_a2a_relay
    from src.api.routes.devices import _job_db_path
    register_a2a_relay(
        app,
        base_url=os.getenv("MAYRING_A2A_BASE_URL", "https://mcp.linn.games"),
        model=os.getenv("MAYRING_A2A_MODEL", "qwen3.5:9b"),
        # Full UUID — MUST equal what get_workspace resolves the worker's JWT to
        # (claim_cloud_next filters by exact string). The short "019e14d6" would
        # never match → job enqueued but never claimed. Single-user MVP pin;
        # per-request workspace derivation = multi-tenant follow-up.
        workspace_id=os.getenv("MAYRING_A2A_WORKSPACE_ID", "019e14d6-0489-7348-bca8-e29c11293cb7"),
        db_path=_job_db_path(),
    )


@app.get("/auth/verify")
async def _auth_verify(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> dict:
    """Auth gate for nginx auth_request (/a2a + /searxng). Accepts EITHER a
    per-agent API key (X-API-Key, for external A2A clients like Langdock) OR a
    Bearer JWT. 200 on valid, 401 otherwise."""
    from fastapi import HTTPException
    if x_api_key:
        from src.api import agent_keys
        res = agent_keys.verify(x_api_key)
        if res:
            return {"ok": True, "auth": "api_key", "workspace_id": res["workspace_id"]}
        raise HTTPException(status_code=401, detail="invalid api key")
    if authorization and authorization.lower().startswith("bearer "):
        from src.api.jwt_auth import validate_jwt_token
        if validate_jwt_token(authorization.split(" ", 1)[1].strip()) is not None:
            return {"ok": True, "auth": "jwt"}
        raise HTTPException(status_code=401, detail="invalid token")
    raise HTTPException(status_code=401, detail="no credentials")


@app.on_event("startup")
def _run_pending_schema_migrations() -> None:
    """Force the idempotent schema migration on the cloud DB at boot.

    The lazy `get_conn()` path *should* trigger this via init_memory_db,
    but a series of production 500s ("no such column: visibility",
    "no such column: scope") proved that the live DB pre-dates several
    schema additions and the lazy path was not reaching them — likely
    because the connection was set up before the relevant migration was
    added to the codebase, and never re-initialised.
    Calling init_memory_db here on every container start is cheap (a
    no-op when fully migrated) and removes the foot-gun for good.

    Path resolution mirrors `get_conn()` exactly: when MAYRING_LOCAL_DB
    is set (production), migrate that file; otherwise fall back to the
    default MEMORY_DB_PATH. Otherwise the startup migration would touch
    a different file than the request handlers and miss the live DB.
    """
    import logging
    import os
    from pathlib import Path
    from mayring_core.memory.store import init_memory_db
    logger = logging.getLogger(__name__)
    db_path = os.environ.get("MAYRING_LOCAL_DB", "")
    target = Path(db_path) if db_path else None
    try:
        init_memory_db(target).close()
        logger.info(
            "server.startup: schema migrations applied at %s",
            target or "<default MEMORY_DB_PATH>",
        )
    except Exception:
        # Don't block server boot — get_conn() will retry on first request,
        # and the new defence-in-depth in sync.py / mcp_pi_tools handles the
        # remaining gap if this step somehow failed.
        logger.exception("server.startup: schema migration failed (non-fatal)")


def _plain_ollama_generate(
    prompt: str, model: str, ollama_url: str, timeout: float,
    num_predict: int = 1024,
    response_format: str | None = None,
    options: dict | None = None,
) -> dict:
    """Pure Ollama /api/generate — NO memory augmentation. Used for every
    non-'pi-task' kind (judge/categorize/summarize/…) so all those jobs go
    through the PiQueue (bounded/distributed, no direct GPU hammering) WITHOUT
    injecting unrelated memory. Returns {'content': text}.

    Routes via ollama_client.generate so it inherits the cloud-split
    (OLLAMA_CLOUD_PRIMARY_RATIO → % of generate jobs to ollama.com) + fallback —
    judge/categorize get distributed like every other generate job, not pinned
    to one host. Embeddings stay local (they never go through generate()).

    ``response_format`` ("json") forwards Ollama's top-level format field so the
    JSON-mode pi_* tools (judge-relevance) get structured output through
    the queue. num_predict default is 1024 (a cap, not a target) — conservative
    cap that covers the longest remaining queue jobs without truncation."""
    from mayring_core.ollama_client import generate as _ollama_generate
    text = _ollama_generate(
        url=ollama_url,
        model=model,
        prompt=prompt,
        stream=False,
        timeout=timeout,
        num_predict=num_predict,
        think=False,
        # WHY(hook-latency 2026-06-08): force LOCAL — these are latency-critical hook/search
        # feeders (categorize, judge, summarize). The 50% cloud-primary split routed half of
        # them to a slower cloud model (gemma3:4b), adding 4-8s variance per prompt-categorize
        # and blowing the memory-inject 9s budget. The local GPU answers in ~1s. Heavy
        # pi-tasks (run_task_with_memory) keep their own cloud-split for throughput.
        cloud_primary=False,
        # Caller-options (z.B. temperature=0 + fixer seed der Mayring-Reduktion) erhalten;
        # ohne explizite options bleibt der bisherige Default temperature=0.
        options=options if options is not None else {"temperature": 0.0},
        response_format=response_format,
        label="pi-queue-job",
    )
    return {"content": (text or "").strip()}


@app.on_event("startup")
async def _start_pi_queue() -> None:
    import asyncio
    from mayring_pi_agent.pi_queue import get_pi_queue
    from mayring_pi_agent.pi_jobs import PiJob

    queue = get_pi_queue()

    async def _handler(job: PiJob) -> dict:
        import time as _time
        from mayring_pi_agent.pi import run_task_with_memory
        # T4: ModelRouter mit job_class-hint. 'mini' kann auf phi3:3.8b
        # routen wenn yaml es definiert, 'standard' bleibt mistral:7b.
        # Plus per-class timeout — kleine Tasks dürfen nicht 240s warten.
        resolved_model = job.model or _model_for_job_class(job.job_class)
        resolved_timeout = (
            float(job.timeout_s) if job.timeout_s
            else _timeout_for_job_class(job.job_class, fallback=240.0)
        )
        _ollama = os.getenv("OLLAMA_URL", "http://localhost:11434")
        loop = asyncio.get_event_loop()
        start = _time.monotonic()

        def _run() -> dict:
            # Central routing (2026-05-28): ONLY 'pi-task' gets memory
            # augmentation. Every other kind (judge, categorize, summarize,
            # second-opinion, derivation, …) is a pure-prompt job routed
            # through the SAME bounded PiQueue — so NO caller hits Ollama
            # directly anymore; all llama jobs are distributed from here.
            if job.kind == "pi-task":
                return run_task_with_memory(
                    task=job.task_text,
                    ollama_url=_ollama,
                    model=resolved_model,
                    repo_slug=job.repo_slug,
                    system_prompt=job.system_prompt,
                    timeout=resolved_timeout,
                )
            return _plain_ollama_generate(
                job.task_text, resolved_model, _ollama, resolved_timeout,
                response_format=job.response_format or None,
                options=job.options,
            )

        try:
            result = await loop.run_in_executor(None, _run)
            job.latency_ms = int((_time.monotonic() - start) * 1000)
            job.model_used = resolved_model
            return result
        except Exception:
            job.latency_ms = int((_time.monotonic() - start) * 1000)
            job.model_used = resolved_model  # log-für-stats: was wäre genutzt worden
            raise

    queue.set_handler(_handler)
    await queue.start()


@app.on_event("startup")
async def _prewarm_token_cache() -> None:
    """Pre-warm this worker's symbolic-score token cache so its first searches don't
    pay the ~3.95s cold re-tokenisation of the whole workspace (stage-timing
    2026-06-21). Runs in a daemon thread → never delays worker readiness or /health;
    best-effort (lazy per-query warming still covers any miss). MAYRING_PREWARM_TOKENS=0
    disables it."""
    if os.getenv("MAYRING_PREWARM_TOKENS", "1").lower() in ("0", "false", "no"):
        return
    import threading

    def _warm() -> None:
        import logging
        import time as _time
        log = logging.getLogger(__name__)
        try:
            from src.api.memory_service import prewarm_token_cache
            t0 = _time.monotonic()
            n = prewarm_token_cache(_get_conn())
            log.info("token-cache prewarm: %d chunks in %.1fs (pid=%d)",
                     n, _time.monotonic() - t0, os.getpid())
        except Exception as exc:  # noqa: BLE001 — best-effort; lazy warming still works
            log.warning("token-cache prewarm skipped (%s)", exc)

    threading.Thread(target=_warm, name="token-prewarm", daemon=True).start()


@app.on_event("startup")
async def _reap_stale_jobs() -> None:
    """Reap orphaned 'started' jobs whose process died mid-run (deploy/restart/crash).
    WHY(zombie-debounce 2026-06-21): jobs_state.json froze a populate at 'started' for
    8 days, and enqueue_populate's debounce reused it → all re-ingest silently stopped.
    On boot we mark anything older than STALE_JOB_SECONDS as failed so debounce is free."""
    import logging
    try:
        from src.api.job_queue import reconcile_stale_jobs
        n = reconcile_stale_jobs()
        if n:
            logging.getLogger(__name__).warning(
                "startup: reaped %d stale/orphaned job(s) frozen at 'started'", n)
    except Exception as exc:  # noqa: BLE001 — best-effort; debounce staleness still guards
        logging.getLogger(__name__).warning("startup: stale-job reconcile skipped (%s)", exc)


@app.on_event("shutdown")
async def _stop_pi_queue() -> None:
    from mayring_pi_agent.pi_queue import get_pi_queue
    await get_pi_queue().shutdown()


def _model_for_job_class(job_class: str) -> str:
    """T4: ModelRouter.resolve('text', job_class) routet 'mini' → kleineres
    Modell wenn yaml einen classes-Block hat. 'standard'/unknown fallen
    auf den outer route — kein Breaking-Change weil yaml-Default keine
    classes hat."""
    from mayring_core.model_router import ModelRouter
    return ModelRouter(os.getenv("OLLAMA_URL", "http://localhost:11434")).resolve(
        "text", job_class=job_class,
    )


def _timeout_for_job_class(job_class: str, fallback: float = 240.0) -> float:
    """T4: per-class timeout aus model_routes.yaml — 'mini' typically 30s
    statt 240s damit kleine Tasks nicht endlos auf langsame Modelle warten."""
    from mayring_core.model_router import ModelRouter
    return float(
        ModelRouter(os.getenv("OLLAMA_URL", "http://localhost:11434")).timeout_for(
            "text", job_class=job_class,
        ) or fallback
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "1.0.0"}


# WHY(2026-05-11): /stats/summary macht 9 sqlite-queries inkl. mehrerer
# COUNT-WHERE-datetime-scans über millionen-rows tabellen — Dashboard auf
# app.linn.games timeoutet wiederholt (cURL error 28: 5002ms). Fix: in-
# process TTL-cache (30s) + stale-fallback wenn DB-query crasht. Damit
# zeigt das dashboard im worst-case last-known-good statt 504 — was die
# user-experience signifikant verbessert. Single-replica = process-local
# cache OK.
_STATS_CACHE: dict[str, Any] = {"fresh": None, "stale": None, "expires_at": 0.0}
_STATS_CACHE_TTL = 30.0


def bust_stats_cache() -> None:
    """Invalidate the /stats/summary fresh-slot so the next call recomputes.

    WHY(2026-05-11, smoke-fix): the 30s TTL-cache broke the post-deploy
    smoke (`feedback_count_delta`, `stop_hook_e2e`) — those checks post
    feedback then immediately re-read /stats/summary expecting the count
    to have grown, but the cache served the stale count. Callers that
    mutate counts (feedback insert, ingest) must call this. The 'stale'
    slot is kept (it's only the disaster-fallback if a live query crashes).
    """
    _STATS_CACHE["fresh"] = None
    _STATS_CACHE["expires_at"] = 0.0
    # Also drop the shared (cross-worker) copy so every worker recomputes.
    from src.api import shared_state
    shared_state.cache_del("stats:summary")


def _stats_summary_uncached() -> dict:
    """The actual sqlite-heavy work. Wrapped by stats_summary() with cache."""
    from src.api.job_queue import _JOBS
    conn = _get_conn()
    active = conn.execute("SELECT COUNT(*) FROM chunks WHERE is_active=1").fetchone()[0]
    total  = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    sources = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    # WHY(2026-05-10 user-mandate "no legacy"): keine positive/negative/
    # neutral-keys mehr im response. Konsumenten MÜSSEN auf stars[]
    # migrieren — fail-loud wenn sie das nicht tun (KeyError statt
    # silent fallback auf abgeleitete buckets).
    fb_rows = conn.execute("SELECT signal, COUNT(*) FROM chunk_feedback GROUP BY signal").fetchall()
    fb = {r[0]: r[1] for r in fb_rows}
    feedback_summary = {
        "stars":  {str(i): fb.get(str(i), 0) for i in range(1, 6)},
        "total":  sum(fb.get(str(i), 0) for i in range(1, 6)),
        "avg":    (
            sum(int(s) * fb.get(str(s), 0) for s in range(1, 6))
            / max(sum(fb.get(str(i), 0) for i in range(1, 6)), 1)
        ),
    }
    last_hour = conn.execute(
        "SELECT COUNT(*) FROM ingestion_log WHERE created_at > datetime('now', '-1 hour')"
    ).fetchone()[0]
    last_24h = conn.execute(
        "SELECT COUNT(*) FROM ingestion_log WHERE created_at > datetime('now', '-24 hours')"
    ).fetchone()[0]
    recent = [
        {"event_type": r[0], "source_id": r[1], "created_at": r[2]}
        for r in conn.execute(
            "SELECT event_type, source_id, created_at FROM ingestion_log "
            "ORDER BY created_at DESC LIMIT 20"
        ).fetchall()
    ]
    # WHY(pi-dashboard multi-worker 2026-05-25): _JOBS is per-process; under
    # --workers 4 this widget saw only the serving worker's jobs. Merge the
    # cross-worker shared file (_load_jobs) with local _JOBS for a complete view.
    from src.api.job_queue import _load_jobs as _load_jobs_shared
    _jobs_merged = {**_load_jobs_shared(), **_JOBS}
    recent_jobs = [
        {
            "job_id":     j["job_id"],
            "status":     j["status"],
            "started_at": j.get("started_at"),
            "stages":     j.get("stages", {}),
            "progress":   j.get("progress"),
            "v2_jobs":    {k: _jobs_merged.get(v, {}).get("status") for k, v in j.get("v2_jobs", {}).items()},
        }
        for j in sorted(_jobs_merged.values(), key=lambda x: x.get("started_at", ""), reverse=True)[:5]
    ]
    try:
        llm_recent = [
            {
                "call_type":   r[0],
                "model":       r[1],
                "prompt":      r[2],
                "response":    r[3],
                "tool_calls":  r[4],
                "duration_ms": r[5],
                "created_at":  r[6],
            }
            for r in conn.execute(
                "SELECT call_type, model, prompt, response, tool_calls, duration_ms, created_at"
                " FROM llm_calls_log ORDER BY created_at DESC LIMIT 20"
            ).fetchall()
        ]
        llm_24h = conn.execute(
            "SELECT COUNT(*) FROM llm_calls_log"
            " WHERE created_at > datetime('now','-24 hours')"
        ).fetchone()[0]
    except Exception:
        llm_recent, llm_24h = [], 0
    return {
        "chunks":      {"active": active, "total": total},
        "sources":     {"count": sources},
        "feedback":    feedback_summary,
        "ingestion":   {"last_hour": last_hour, "last_24h": last_24h},
        "recent_ops":  recent,
        "recent_jobs": recent_jobs,
        "llm_calls":   {"last_24h": llm_24h, "recent": llm_recent},
    }


@app.get("/stats/summary")
def stats_summary(workspace_id: str = Depends(get_workspace)) -> dict:
    """Cached + stale-fallback wrapper around _stats_summary_uncached.

    Flow:
      1. Cache hit & fresh (<30s) → return cached immediately. ~1ms.
      2. Cache miss or expired → run uncached query. On success: refresh
         both 'fresh' and 'stale' slots, return result.
      3. Uncached query crashes/timeouts → return stale cache (any age)
         with `_cache_status='stale'` marker; only fail with 503 if there
         is NO cache at all (first-ever call + DB down).
    """
    import time as _t
    from src.api import shared_state
    _ = workspace_id  # auth-only; response is global

    now = _t.time()

    # Shared L2 (Redis) — authoritative across uvicorn workers when reachable,
    # so the dashboard shows the same numbers regardless of which worker answers.
    shared = shared_state.cache_get("stats:summary")
    if shared is not None:
        return {**shared, "_cache_status": "hit"}

    # Per-process L1 fast path — ONLY when Redis is down. WHY(feedback_count_delta):
    # with Redis up, a cache_get miss means the key was busted (feedback/ingest
    # called bust_stats_cache → cache_del) or expired → recompute. Serving the
    # local L1 here would return a stale count that a bust on ANOTHER worker never
    # cleared (delta=0). Redis is the single source when reachable.
    if (not shared_state.enabled()
            and _STATS_CACHE["fresh"] is not None and now < _STATS_CACHE["expires_at"]):
        return {**_STATS_CACHE["fresh"], "_cache_status": "hit"}

    # Slow path — run the heavy query
    try:
        result = _stats_summary_uncached()
        _STATS_CACHE["fresh"] = result
        _STATS_CACHE["stale"] = result
        _STATS_CACHE["expires_at"] = now + _STATS_CACHE_TTL
        shared_state.cache_set("stats:summary", result, _STATS_CACHE_TTL)
        return {**result, "_cache_status": "fresh"}
    except Exception as exc:
        # DB-crash or timeout — fall back to stale cache if we have one.
        if _STATS_CACHE["stale"] is not None:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "stats/summary: serving stale cache after live-query fail: %s", exc,
            )
            # WHY(2026-05-11, codeql py/stack-trace-exposure): no raw exc
            # text in the body — only the exception *class* (safe, helps the
            # dashboard show "why stale" without leaking internals). Full
            # detail is in the warning log above.
            return {
                **_STATS_CACHE["stale"],
                "_cache_status": "stale",
                "_stale_reason": type(exc).__name__,
            }
        # No cache at all — first call ever AND DB broken.
        # WHY(2026-05-11, codeql py/stack-trace-exposure): don't leak the
        # exception text in the HTTP body — log it server-side, return a
        # generic message. The detailed error is in the warning log above.
        import logging as _logging
        _logging.getLogger(__name__).error("stats/summary unavailable (no cache, DB error): %s", exc)
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="stats/summary temporarily unavailable")


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("API_PORT", "8080")))


if __name__ == "__main__":
    main()
