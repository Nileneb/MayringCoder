"""Memory-Injection v2.0 — retrieval-quality metrics.

Joins ``context_feedback_log`` (one row per /memory/search call, with the
top-K chunk_ids the search returned) with ``chunk_feedback`` (positive /
negative ratings the Stop hook posts after the assistant's reply) to
compute the metrics that actually answer "is the ranker getting better":

  precision@K  — share of top-K chunks the user marked positive
  ndcg@K       — discounted-cumulative gain, position-weighted (log2 decay)
  recall@K     — share of all positively-rated chunks that appeared in top-K

The current dashboard only counted feedback events. With these we can
compare reranker versions, see degradations early, and (with Pipeline 2
from Issue #87) train a learned reranker on the same feature vectors.

Mounted under ``/stats/`` so the production nginx whitelist already
covers the new paths.
"""
from __future__ import annotations

import json
import math
from typing import Any

from fastapi import APIRouter, Depends

from src.api.auth import get_token_info
from src.api.jwt_auth import TokenInfo
from src.memory.store import init_memory_db

router = APIRouter()


def _conn():
    from src.config import CACHE_DIR
    return init_memory_db(CACHE_DIR / "memory.db")


def _is_admin(info: TokenInfo) -> bool:
    return "*" in info.scopes or "admin" in info.scopes


def _label_map(rows: list[Any]) -> dict[str, int]:
    """Aggregate {chunk_id: 1 if any positive feedback, 0 otherwise}.

    A chunk with mixed signals counts as positive when the positive count
    strictly exceeds the negative — a single negative on an otherwise
    well-rated chunk shouldn't drop it out of the dataset.
    """
    counts: dict[str, dict[str, int]] = {}
    for r in rows:
        cid = r["chunk_id"]
        sig = r["signal"]
        bucket = counts.setdefault(cid, {"pos": 0, "neg": 0})
        if sig in ("positive", "1", "2", "3", "4", "5"):
            bucket["pos"] += 1
        elif sig == "negative":
            bucket["neg"] += 1
    return {cid: 1 if v["pos"] > v["neg"] else 0 for cid, v in counts.items()}


def _ndcg(labels: list[int], k: int) -> float:
    """Standard NDCG@K with binary relevance + log2 position discount."""
    if not labels:
        return 0.0
    gains = labels[:k]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(labels, reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


@router.get("/stats/retrieval-metrics")
async def retrieval_metrics(
    info: TokenInfo = Depends(get_token_info),
    days: int = 7,
    k: int = 5,
) -> dict:
    """Compute precision@K + NDCG@K + recall@K over the last `days` days.

    Service token ('*'/'admin' scope): cross-workspace.
    Regular JWT: scoped to the caller's workspace.
    """
    if k < 1 or k > 20:
        k = 5
    if days < 1 or days > 90:
        days = 7
    conn = _conn()
    is_admin = _is_admin(info)
    where_ws = "" if is_admin else " AND (workspace_id = ? OR workspace_id = '')"
    params: list[Any] = [f"-{days} days"]
    if not is_admin:
        params.append(info.workspace_id)
    log_rows = conn.execute(
        f"SELECT id, query, trigger_ids, captured_at, workspace_id "
        f"FROM context_feedback_log "
        f"WHERE captured_at > datetime('now', ?){where_ws} "
        f"ORDER BY captured_at DESC LIMIT 2000",
        params,
    ).fetchall()

    fb_rows = conn.execute(
        "SELECT chunk_id, signal FROM chunk_feedback "
        "WHERE created_at > datetime('now', ?)",
        (f"-{days} days",),
    ).fetchall()
    labels = _label_map(fb_rows)

    queries_with_feedback = 0
    p_at_k_sum = 0.0
    ndcg_sum = 0.0
    relevant_total = 0
    relevant_in_topk = 0
    for row in log_rows:
        try:
            chunk_ids = json.loads(row["trigger_ids"])
        except (TypeError, ValueError):
            continue
        topk = chunk_ids[:k]
        topk_labels = [labels.get(cid, 0) for cid in topk]
        if not any(labels.get(cid, 0) for cid in chunk_ids):
            continue
        queries_with_feedback += 1
        p_at_k_sum += sum(topk_labels) / max(len(topk), 1)
        ndcg_sum += _ndcg(topk_labels, k)
        all_relevant = sum(1 for cid in chunk_ids if labels.get(cid, 0))
        if all_relevant:
            relevant_total += all_relevant
            relevant_in_topk += sum(topk_labels)

    n = max(queries_with_feedback, 1)
    return {
        "scope": "all" if is_admin else "workspace",
        "workspace_id": info.workspace_id,
        "window_days": days,
        "k": k,
        "queries_logged": len(log_rows),
        "queries_with_feedback": queries_with_feedback,
        "feedback_events": len(fb_rows),
        "precision_at_k": round(p_at_k_sum / n, 4),
        "ndcg_at_k": round(ndcg_sum / n, 4),
        "recall_at_k": (
            round(relevant_in_topk / relevant_total, 4)
            if relevant_total else 0.0
        ),
    }


@router.get("/stats/retrieval-stage-attribution")
async def retrieval_stage_attribution(
    info: TokenInfo = Depends(get_token_info),
    days: int = 7,
) -> dict:
    """For chunks the user later marked positive: which stage carried them?

    Reads ``stage_scores`` JSON from context_feedback_log. For each
    positive chunk in the top-K of a query, find the stage with the
    highest contribution among {vector, symbolic, recency, source_affinity}.
    Counts "wins" per stage to surface which signal is doing real work.

    If 80%+ of wins come from one stage, the others are dead weight in
    the current weighting and a learned reranker should fix that.
    """
    if days < 1 or days > 90:
        days = 7
    conn = _conn()
    is_admin = _is_admin(info)
    where_ws = "" if is_admin else " AND (workspace_id = ? OR workspace_id = '')"
    params: list[Any] = [f"-{days} days"]
    if not is_admin:
        params.append(info.workspace_id)
    rows = conn.execute(
        f"SELECT trigger_ids, stage_scores FROM context_feedback_log "
        f"WHERE captured_at > datetime('now', ?){where_ws} "
        f"AND stage_scores != '{{}}' AND stage_scores != '' "
        f"ORDER BY captured_at DESC LIMIT 2000",
        params,
    ).fetchall()
    fb_rows = conn.execute(
        "SELECT chunk_id, signal FROM chunk_feedback "
        "WHERE created_at > datetime('now', ?)",
        (f"-{days} days",),
    ).fetchall()
    labels = _label_map(fb_rows)

    wins = {"v": 0, "s": 0, "r": 0, "a": 0}
    counted = 0
    for row in rows:
        try:
            stage = json.loads(row["stage_scores"])
        except (TypeError, ValueError):
            continue
        for cid, scores in stage.items():
            if labels.get(cid, 0) != 1 or not isinstance(scores, dict):
                continue
            best_stage = max(("v", "s", "r", "a"),
                             key=lambda s: scores.get(s, 0.0))
            wins[best_stage] += 1
            counted += 1
    return {
        "scope": "all" if is_admin else "workspace",
        "workspace_id": info.workspace_id,
        "window_days": days,
        "positive_chunks_attributed": counted,
        "stage_wins": {
            "vector":          wins["v"],
            "symbolic":        wins["s"],
            "recency":         wins["r"],
            "source_affinity": wins["a"],
        },
        "stage_share": {
            "vector":          round(wins["v"] / max(counted, 1), 4),
            "symbolic":        round(wins["s"] / max(counted, 1), 4),
            "recency":         round(wins["r"] / max(counted, 1), 4),
            "source_affinity": round(wins["a"] / max(counted, 1), 4),
        },
    }
