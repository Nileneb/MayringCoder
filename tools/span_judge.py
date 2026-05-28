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
from datetime import datetime, timezone

_log = logging.getLogger(__name__)

MAX_CHUNKS_PER_CALL = 20
CHUNK_TEXT_LIMIT = 600


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
    timeout: float = 60.0,
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
    sys_prompt = (
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
    chunks_text = "\n\n".join(f"[{cid}]\n{t}" for cid, t in norm)
    prompt = sys_prompt + f"\n\nQuery: {query}\n\nChunks:\n{chunks_text}"
    import httpx
    try:
        resp = httpx.post(
            f"{url}/api/generate",
            json={
                "model": mdl,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                # WHY(2026-05-28): 600 truncated the format:json scores object
                # mid-structure for ≤20-chunk batches → incomplete (invalid)
                # JSON → parse fail → span_judge silently skipped the retrain's
                # whole refinement. 2048 gives ample headroom; format:json keeps
                # it valid once it completes.
                "options": {"temperature": 0.1, "num_predict": 2048},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
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
    timeout: float = 60.0,
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
    if missing:
        texts = _chunk_texts(conn, missing)
        items = [(cid, texts[cid]) for cid in missing if texts.get(cid)]
        if items:
            model = _judge_model(ollama_url)
            fresh = judge_relevance(
                query, items, ollama_url=ollama_url, model=model, timeout=timeout
            )
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
