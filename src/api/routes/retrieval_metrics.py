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
from src.api.dependencies import get_conn as _conn
from src.api.jwt_auth import TokenInfo

router = APIRouter()  # redeploy-marker 2026-06-05 (counterfactual)


def _is_admin(info: TokenInfo) -> bool:
    return "*" in info.scopes or "admin" in info.scopes


def _label_map(rows: list[Any]) -> dict[str, int]:
    """Aggregate {chunk_id: 1 if avg rating >= 3.5, 0 otherwise}.

    WHY(2026-05-10 rating-migration): NDCG braucht binary label. Mapping:
    avg rating >= 3.5 → 1 (relevant), sonst 0. Replaces alte positive>negative-
    logik — eine rating 5 + rating 2 ergibt jetzt avg 3.5 → 1, statt
    "1pos 1neg → 0" wie vorher.
    """
    accum: dict[str, list[int]] = {}
    for r in rows:
        cid = r["chunk_id"]
        sig = r["signal"]
        try:
            rating = int(sig)
            if 1 <= rating <= 5:
                accum.setdefault(cid, []).append(rating)
        except (TypeError, ValueError):
            continue
    return {
        cid: 1 if (sum(rs) / len(rs)) >= 3.5 else 0
        for cid, rs in accum.items()
    }


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
def retrieval_metrics(
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


@router.get("/stats/retrieval-ab")
def retrieval_ab(
    info: TokenInfo = Depends(get_token_info),
    days: int = 7,
    k: int = 5,
) -> dict:
    """A/B compare reranker versions on the same metrics.

    Splits ``context_feedback_log`` rows by ``reranker_version`` and runs
    the same precision@K + NDCG@K logic per group. Lets us tell whether
    v2 actually beats v1 on real traffic, not just on hold-out test
    AUC. Numbers are workspace-scoped unless caller has admin scope.
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
    rows = conn.execute(
        f"SELECT trigger_ids, reranker_version "
        f"FROM context_feedback_log "
        f"WHERE captured_at > datetime('now', ?){where_ws} "
        f"AND query != '' "
        f"ORDER BY captured_at DESC LIMIT 4000",
        params,
    ).fetchall()
    fb_rows = conn.execute(
        "SELECT chunk_id, signal FROM chunk_feedback "
        "WHERE created_at > datetime('now', ?)",
        (f"-{days} days",),
    ).fetchall()
    labels = _label_map(fb_rows)

    buckets: dict[str, dict[str, float]] = {}
    for row in rows:
        version = (row["reranker_version"] or "v1") or "v1"
        try:
            chunk_ids = json.loads(row["trigger_ids"])
        except (TypeError, ValueError):
            continue
        topk = chunk_ids[:k]
        if not any(labels.get(cid, 0) for cid in chunk_ids):
            continue
        bucket = buckets.setdefault(version, {
            "queries": 0, "p_sum": 0.0, "ndcg_sum": 0.0,
        })
        bucket["queries"] += 1
        topk_labels = [labels.get(cid, 0) for cid in topk]
        bucket["p_sum"] += sum(topk_labels) / max(len(topk), 1)
        bucket["ndcg_sum"] += _ndcg(topk_labels, k)

    summary: dict[str, dict[str, float]] = {}
    for v, b in buckets.items():
        n = max(b["queries"], 1)
        summary[v] = {
            "queries": int(b["queries"]),
            "precision_at_k": round(b["p_sum"] / n, 4),
            "ndcg_at_k": round(b["ndcg_sum"] / n, 4),
        }
    p_v1 = (summary.get("v1") or {}).get("precision_at_k", 0.0)
    p_v2 = (summary.get("v2") or {}).get("precision_at_k", 0.0)
    n_v1 = (summary.get("v1") or {}).get("ndcg_at_k", 0.0)
    n_v2 = (summary.get("v2") or {}).get("ndcg_at_k", 0.0)
    return {
        "scope": "all" if is_admin else "workspace",
        "workspace_id": info.workspace_id,
        "window_days": days,
        "k": k,
        "by_version": summary,
        "uplift": {
            "precision_at_k": round(p_v2 - p_v1, 4),
            "ndcg_at_k":      round(n_v2 - n_v1, 4),
        },
    }


@router.get("/stats/retrieval-stage-attribution")
def retrieval_stage_attribution(
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


@router.get("/stats/admin/reranker-counterfactual")
def reranker_counterfactual(
    info: TokenInfo = Depends(get_token_info),
    baseline: str | None = None,
    candidate: str | None = None,
    days: int = 7,
    k: int = 5,
) -> dict:
    """Fairer Head-to-Head OHNE Live-Traffic für ein neues Modell: nimm die Queries,
    die ``baseline`` real serviert hat (deren gespeicherte Reihenfolge = baseline's
    echtes Ranking), und RE-RANKE dieselben Kandidaten mit ``candidate`` (score_v2 über
    die geloggten stage_scores). precision@K + nDCG@K je Query, gemittelt — identisches
    Query-Set, echte Labels. So sieht man v1 vs ein nie-serviertes v<N> mit Zahlen.

    Werden baseline/candidate nicht übergeben, leitet der Endpunkt das Paar aus
    den aktiven Reranker-Versionen ab (read_active_versions). Genau 2 aktive nötig."""
    if not _is_admin(info):
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="admin scope required")
    if k < 1 or k > 20:
        k = 5
    if days < 1 or days > 90:
        days = 7
    from mayring_core.memory.reranker_v2 import _model_path, read_active_versions, score_v2

    # WHY: Query-Params sind Optional — fehlen sie, wird das Paar aus den aktiv
    # gesetzten Versionen abgeleitet (read_active_versions). So vergleicht der
    # Endpunkt immer die tatsächlich laufenden Versionen statt hartcodierter
    # v1/v4-Konstanten, die nach einem Retraining sofort veraltet wären.
    if baseline is None or candidate is None:
        from fastapi import HTTPException
        active = read_active_versions()
        if len(active) != 2:
            raise HTTPException(
                status_code=400,
                detail="A/B braucht genau 2 aktive Reranker-Versionen",
            )
        baseline, candidate = active[0], active[1]

    from fastapi import HTTPException
    cand_path = _model_path(candidate)
    if not cand_path.exists():
        raise HTTPException(status_code=404, detail=f"no model file for {candidate}")
    cand_model = json.loads(cand_path.read_text(encoding="utf-8"))

    conn = _conn()
    rows = conn.execute(
        "SELECT trigger_ids, stage_scores FROM context_feedback_log "
        "WHERE captured_at > datetime('now', ?) AND reranker_version = ? "
        "AND stage_scores != '{}' AND stage_scores != '' "
        "ORDER BY captured_at DESC LIMIT 4000",
        (f"-{days} days", baseline),
    ).fetchall()
    fb = conn.execute("SELECT chunk_id, signal FROM chunk_feedback "
                      "WHERE created_at > datetime('now', ?)", (f"-{days} days",)).fetchall()
    labels = _label_map(fb)

    def _scores(topk: list[str]) -> tuple[float, float]:
        labs = [labels.get(c, 0) for c in topk[:k]]
        return sum(labs) / max(len(labs), 1), _ndcg(labs, k)

    base_p = base_n = cand_p = cand_n = 0.0
    counted = 0
    for row in rows:
        try:
            cands = json.loads(row["trigger_ids"])
            stage = json.loads(row["stage_scores"])
        except (TypeError, ValueError):
            continue
        if not isinstance(cands, list) or not any(labels.get(c, 0) for c in cands):
            continue  # nur Queries mit mind. einem positiv-gerateten Kandidaten
        reranked = sorted(cands, key=lambda c: score_v2(stage.get(c, {}) or {}, cand_model),
                          reverse=True)
        bp, bn = _scores(cands)        # baseline = servierte Reihenfolge
        cp, cn = _scores(reranked)     # candidate = re-rankt
        base_p += bp; base_n += bn; cand_p += cp; cand_n += cn
        counted += 1

    if not counted:
        return {"queries": 0, "baseline": baseline, "candidate": candidate,
                "note": f"keine {baseline}-servierten Queries mit Positiv-Label in {days}d"}
    return {
        "window_days": days, "k": k, "queries": counted,
        "baseline": {"version": baseline, "precision_at_k": round(base_p / counted, 4),
                     "ndcg_at_k": round(base_n / counted, 4)},
        "candidate": {"version": candidate, "precision_at_k": round(cand_p / counted, 4),
                      "ndcg_at_k": round(cand_n / counted, 4)},
        "delta": {"precision_at_k": round((cand_p - base_p) / counted, 4),
                  "ndcg_at_k": round((cand_n - base_n) / counted, 4)},
    }
