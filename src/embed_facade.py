"""Single entry point for producing an embedding (#365 Schicht 3).

If at least EMBED_REPLICATION devices are eligible, the embedding goes through the
verified pool (dual-send + cosine agreement). Otherwise it falls back to the direct
Spec-B path (local GPU host, deadline->cloud) — the pool is relief, not a single point."""
from __future__ import annotations

import json
import time
from typing import Any

from mayring_core import config as cfg
from mayring_core.memory import devices as device_store
from mayring_core.memory import embed_pool as ep


def _direct_embed(text: str, *, model: str = "bge-m3") -> list[float]:
    """Fallback: today's direct path. URL resolution follows the existing Spec-B
    worker routing (local GPU host; deadline escalates to cloud elsewhere)."""
    import os
    from mayring_core.ollama_client import embed_single
    url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    return embed_single(url, model, text)


def _poll_verified(conn: Any, embed_id: str, *, timeout_s: float) -> list[float] | None:
    """Poll the embed job until verified (agreed vector) or timeout. Returns the
    agreed vector, or None on timeout/divergence (caller falls back).

    ep.get returns result_a as a JSON string (TEXT column, not deserialized) — must
    json.loads here before returning."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = ep.get(conn, embed_id)
        if job and job["status"] == "verified" and job["result_a"]:
            return json.loads(job["result_a"])
        if job and job["status"] in ("diverged", "failed"):
            return None
        time.sleep(1.0)
    return None


def verified_embedding(conn: Any, *, text: str, workspace_id: str, projekt_id: str,
                       chunk_ref: str, model: str = "bge-m3",
                       now: str | None = None) -> list[float]:
    if now is None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    eligible = device_store.eligible_embed_devices(
        conn, workspace_id, now=now, fresh_seconds=cfg.EMBED_HEARTBEAT_FRESH_SECONDS)
    if len(eligible) < cfg.EMBED_REPLICATION:
        return _direct_embed(text, model=model)
    eid = ep.enqueue(conn, workspace_id=workspace_id, projekt_id=projekt_id,
                     text=text, chunk_ref=chunk_ref, model=model)
    vec = _poll_verified(conn, eid, timeout_s=cfg.EMBED_DUAL_CLAIM_TIMEOUT_SECONDS)
    if vec is None:
        return _direct_embed(text, model=model)  # pool didn't agree in time -> fallback
    return vec


__all__ = ("verified_embedding",)
