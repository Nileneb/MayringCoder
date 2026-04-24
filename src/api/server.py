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
    OLLAMA_MODEL          — default model (e.g. gemma4:e4b)
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


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "1.0.0"}


@app.get("/stats/summary")
def stats_summary() -> dict:
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
    return {
        "chunks":    {"active": active, "total": total},
        "sources":   {"count": sources},
        "feedback":  feedback_summary,
        "ingestion": {"last_hour": last_hour, "last_24h": last_24h},
        "recent_ops": recent,
    }


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("API_PORT", "8080")))


if __name__ == "__main__":
    main()
