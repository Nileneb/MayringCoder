"""Offline LLM relevance judge for the reranker training pipeline.

Used ONLY at training-export time (``tools/export_retrieval_dataset.py
--span-judge``), never in the retrieval hot path. WHY: a heavyweight
Ollama relevance judge in the inject path would blow the memory-inject
budget (Inject-Advisor incident 2026-05-24, 12.5s > 9s → 0 chunks).
Here it runs offline as a TEACHER signal — the expensive span judge
refines noisy ``was_referenced`` labels so the cheap linear v2 reranker
trains on better data (distillation). The judge score is deliberately
NOT a runtime model feature: ``score_v2`` reads ``stage_scores`` logged
at retrieval time and can never recompute this offline-only signal.

Mirrors the prompt + ModelRouter 'text' route of
``src.api.mcp_agent_tools.pi_judge_relevance`` but stays self-contained
so the host-side export script needs no FastAPI-server import. Scores are
cached in ``memory.db`` (``span_judge_cache``) so retrains do not re-call
Ollama for the same (query, chunk) pairs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone

_log = logging.getLogger(__name__)

MAX_CHUNKS_PER_CALL = int(os.getenv("SPAN_JUDGE_MAX_CHUNKS", "20"))
CHUNK_TEXT_LIMIT = 600

# Identical rubric for the Ollama judge AND the Claude pre-warm path
# (tools/span_judge_prewarm.py) so both produce consistent teacher labels.
RELEVANCE_RUBRIC = (
    "You are a relevance judge. For each chunk, decide how relevant it is "
    "to the query on a 0..1 scale:\n"
    "  0.0 = not relevant at all\n"
    "  0.3 = tangential\n"
    "  0.6 = relevant context\n"
    "  0.9 = primary source\n"
    "  1.0 = directly answers the query\n\n"
    'Output STRICT JSON only: {"scores":{"<chunk_id>":<0..1>,...}}\n'
    "No prose, no explanation. ALL chunks MUST appear in the output."
)

# WHY(reranker-gpu-relief): the span-judge export pegged the GPU ~10min straight.
# Cool the LOCAL model down by sleeping after every Nth real model call (cache
# hits don't count). 0 on either knob disables.
_COOLDOWN_EVERY = int(os.getenv("SPAN_JUDGE_COOLDOWN_EVERY", "15"))
_COOLDOWN_SECONDS = float(os.getenv("SPAN_JUDGE_COOLDOWN_SECONDS", "2.5"))

# WHY(2026-06-08 live-run): a 50/50 split still sent half the judge calls
# LOCAL-first, and the saturated single-slot GPU host (mistral:7b + bge-m3, no
# OLLAMA_NUM_PARALLEL) hit the 60s ReadTimeout on each before the cloud fallback
# — the export crawled AND starved the live inject-advisor (hot-path ReadTimeouts).
# Hard call-budget: a retrain can do at most N fresh judge calls, then span_judge
# returns {} (keeps the raw was_referenced label) so it can NEVER hammer
# unboundedly regardless of the uncached-pair count. Claude pre-warm + cloud-first
# routing shrink the calls; this is the backstop. 0 = unlimited.
_MAX_CALLS = int(os.getenv("SPAN_JUDGE_MAX_CALLS", "0"))
# Per-call wall-clock cap. Lower than the old 60s so a slow cloud/local call
# fails fast (→ {} → was_referenced) instead of stalling the whole export.
_TIMEOUT = float(os.getenv("SPAN_JUDGE_TIMEOUT", "45"))
_model_calls = 0
_budget_logged = False


def _budget_exhausted() -> bool:
    """True once the fresh-call budget is spent — gate BEFORE a model call."""
    global _budget_logged
    if _MAX_CALLS and _model_calls >= _MAX_CALLS:
        if not _budget_logged:
            _log.warning(
                "span_judge call budget %d reached — keeping raw labels for the rest",
                _MAX_CALLS,
            )
            _budget_logged = True
        return True
    return False


def _note_and_cooldown() -> None:
    """Count a real model call and sleep periodically for GPU headroom."""
    global _model_calls
    _model_calls += 1
    if _COOLDOWN_EVERY <= 0 or _COOLDOWN_SECONDS <= 0:
        return
    if _model_calls % _COOLDOWN_EVERY == 0:
        _log.info(
            "span_judge cooldown: %.1fs after %d model calls",
            _COOLDOWN_SECONDS, _model_calls,
        )
        time.sleep(_COOLDOWN_SECONDS)


def _loads_lenient(raw: str):
    """json.loads tolerant of markdown fences / leading prose that some Ollama
    backends emit even under format:json. Genuinely broken JSON still raises →
    judge_relevance's caller falls back to was_referenced (no silent swallow)."""
    s = (raw or "").strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        i, j = s.find("{"), s.rfind("}")
        if i >= 0 and j > i:
            return json.loads(s[i:j + 1])
        raise


def _ollama_url() -> str:
    return os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")


def _judge_model(ollama_url: str | None = None) -> str:
    from mayring_core.model_router import ModelRouter
    return ModelRouter(ollama_url or _ollama_url()).resolve("text") or "qwen2.5-coder:7b"


def query_hash(query: str) -> str:
    return hashlib.sha256((query or "").encode("utf-8")).hexdigest()


def ensure_cache_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS span_judge_cache ("
        "  query_hash  TEXT NOT NULL,"
        "  chunk_id    TEXT NOT NULL,"
        "  score       REAL NOT NULL,"
        "  model       TEXT,"
        "  computed_at TEXT,"
        "  PRIMARY KEY (query_hash, chunk_id)"
        ")"
    )
    conn.commit()


def _read_cache(
    conn: sqlite3.Connection, qhash: str, chunk_ids: list[str]
) -> dict[str, float]:
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT chunk_id, score FROM span_judge_cache "
        f"WHERE query_hash = ? AND chunk_id IN ({placeholders})",
        (qhash, *chunk_ids),
    ).fetchall()
    return {r[0]: float(r[1]) for r in rows}


def _write_cache(
    conn: sqlite3.Connection, qhash: str, scores: dict[str, float], model: str
) -> None:
    if not scores:
        return
    ts = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        "INSERT OR REPLACE INTO span_judge_cache "
        "(query_hash, chunk_id, score, model, computed_at) VALUES (?,?,?,?,?)",
        [(qhash, cid, float(sc), model, ts) for cid, sc in scores.items()],
    )
    conn.commit()


def _chunk_texts(
    conn: sqlite3.Connection, chunk_ids: list[str]
) -> dict[str, str]:
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({placeholders})",
        tuple(chunk_ids),
    ).fetchall()
    return {r[0]: (r[1] or "") for r in rows}


def judge_relevance(
    query: str,
    items: list[tuple[str, str]],
    *,
    ollama_url: str | None = None,
    model: str | None = None,
    timeout: float = _TIMEOUT,
) -> dict[str, float]:
    """Batched Ollama relevance judge → {chunk_id: 0..1}.

    ``items`` is ``[(chunk_id, text), ...]``. On ANY failure (Ollama
    unreachable, malformed JSON) returns ``{}`` and logs a WARNING — the
    export then falls back to the raw ``was_referenced`` label rather
    than crashing or silently swallowing a real bug.
    """
    if not query or not items:
        return {}
    url = (ollama_url or _ollama_url()).rstrip("/")
    mdl = model or _judge_model(url)
    norm: list[tuple[str, str]] = []
    for cid, text in items[:MAX_CHUNKS_PER_CALL]:
        t = (text or "")[:CHUNK_TEXT_LIMIT]
        if cid and t:
            norm.append((cid, t))
    if not norm:
        return {}
    chunks_text = "\n\n".join(f"[{cid}]\n{t}" for cid, t in norm)
    prompt = RELEVANCE_RUBRIC + f"\n\nQuery: {query}\n\nChunks:\n{chunks_text}"
    # WHY(reranker-gpu-relief): route through ollama_client.generate instead of a
    # direct httpx.post so the export inherits its cloud-primary split
    # (OLLAMA_CLOUD_PRIMARY_RATIO, set to 0.5 for the export subprocess →
    # ~half the judge calls offload to Ollama-Cloud, qwen2.5-coder:7b →
    # qwen3-coder-next), model-map and local fallback. num_predict=2048: 600
    # truncated the format:json scores object mid-structure for ≤20-chunk batches
    # → invalid JSON → whole refinement silently skipped (2026-05-28).
    from mayring_core.ollama_client import generate
    try:
        raw = generate(
            url,
            mdl,
            prompt,
            stream=False,
            response_format="json",
            num_predict=2048,
            options={"temperature": 0.1},
            timeout=timeout,
            label="span_judge",
            # fast-fail like the old direct httpx.post: one local attempt, then
            # cloud-fallback (if key) — no multi-second retry storm stalling the
            # export when local Ollama is down. Cloud-primary split is unaffected.
            max_retries=1,
        ).strip()
        data = _loads_lenient(raw)
        return {
            str(k): max(0.0, min(1.0, float(v)))
            for k, v in (data.get("scores") or {}).items()
        }
    except Exception as exc:
        _log.warning(
            "span_judge unavailable (%s) — skipping refinement for this query", exc
        )
        return {}


def scores_for_query(
    conn: sqlite3.Connection,
    query: str,
    chunk_ids: list[str],
    *,
    ollama_url: str | None = None,
    timeout: float | None = None,
) -> dict[str, float]:
    """Cache-first relevance scores for one query's candidate chunks.

    Reads ``span_judge_cache``, judges only the misses (one batched
    Ollama call over their chunk texts), persists, returns the merged
    ``{chunk_id: score}``. Cache miss + Ollama down → those chunks have
    no score and the caller keeps the raw ``was_referenced`` label.
    """
    if not query or not chunk_ids:
        return {}
    ensure_cache_table(conn)
    qhash = query_hash(query)
    cached = _read_cache(conn, qhash, chunk_ids)
    missing = [cid for cid in chunk_ids if cid not in cached]
    if missing and not _budget_exhausted():
        texts = _chunk_texts(conn, missing)
        items = [(cid, texts[cid]) for cid in missing if texts.get(cid)]
        if items:
            model = _judge_model(ollama_url)
            fresh = judge_relevance(
                query, items, ollama_url=ollama_url, model=model,
                timeout=timeout if timeout is not None else _TIMEOUT,
            )
            _note_and_cooldown()  # count the call + periodic GPU breather
            if fresh:
                # WHY(2026-05-28): the export runs in-container next to the live
                # API, which holds write locks on the shared memory.db. An
                # uncaught OperationalError("database is locked") here crashed
                # the WHOLE export (returncode≠0 → train never ran → no model
                # written — that's why the in-prod span_judge run produced
                # nothing). The cache is an optimization, not load-bearing —
                # skip the write on contention, keep the fresh scores. Fail-soft
                # like the rest of span_judge.
                try:
                    _write_cache(conn, qhash, fresh, model)
                except sqlite3.OperationalError as exc:
                    _log.warning("span_judge cache write skipped (db busy): %s", exc)
                cached.update(fresh)
    return cached
