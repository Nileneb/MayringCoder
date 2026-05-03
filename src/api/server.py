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
from src.api.routes import memory, wiki, jobs, duel, reports
from src.api.routes.sync import router as _sync_router
from src.api.job_queue import _JOBS, run_checker_job as _run_checker_job

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
    from src.memory.store import init_memory_db
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


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "1.0.0"}


@app.get("/stats/summary")
def stats_summary() -> dict:
    from src.api.job_queue import _JOBS
    conn = _get_conn()
    active = conn.execute("SELECT COUNT(*) FROM chunks WHERE is_active=1").fetchone()[0]
    total  = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    sources = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    fb_rows = conn.execute("SELECT signal, COUNT(*) FROM chunk_feedback GROUP BY signal").fetchall()
    fb = {r[0]: r[1] for r in fb_rows}
    star_pos   = sum(fb.get(str(s), 0) for s in (4, 5))
    star_neg   = sum(fb.get(str(s), 0) for s in (1, 2))
    star_neu   = fb.get("3", 0)
    feedback_summary = {
        "positive": fb.get("positive", 0) + star_pos,
        "negative": fb.get("negative", 0) + star_neg,
        "neutral":  fb.get("neutral",  0) + star_neu,
        "stars":    {str(i): fb.get(str(i), 0) for i in range(1, 6)},
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
    recent_jobs = [
        {
            "job_id":     j["job_id"],
            "status":     j["status"],
            "started_at": j.get("started_at"),
            "stages":     j.get("stages", {}),
            "progress":   j.get("progress"),
            "v2_jobs":    {k: _JOBS.get(v, {}).get("status") for k, v in j.get("v2_jobs", {}).items()},
        }
        for j in sorted(_JOBS.values(), key=lambda x: x.get("started_at", ""), reverse=True)[:5]
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


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("API_PORT", "8080")))


if __name__ == "__main__":
    main()
