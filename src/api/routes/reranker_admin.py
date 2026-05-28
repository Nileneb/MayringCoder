"""Admin endpoints to inspect retrieval data + run reranker training.

Pipeline 2 of Issue #87: the production memory.db sits on the server,
not on a developer laptop, so the export+train scripts in tools/ need
a server-side trigger. This module wraps:

  GET  /stats/admin/training-data-counts
       Returns row counts so we know if there's enough data to bother
       running a full training pass.

  POST /stats/admin/train-reranker
       Runs the export + train pipeline in a background subprocess.
       Writes cache/finetuning/retrieval_dataset.jsonl and
       cache/rerank_v2.json. Returns a job_id; status via
       GET /stats/admin/train-reranker/{job_id}.

The actual training job uses ``tools/export_retrieval_dataset.py`` and
``tools/train_reranker.py`` so the CLI path and the API path can never
drift. Same code, same defaults.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.api.auth import get_token_info
from src.api.dependencies import get_conn as _conn
from src.api.jwt_auth import TokenInfo

router = APIRouter()
_log = logging.getLogger(__name__)
_ROOT = Path(__file__).parent.parent.parent.parent
_TRAIN_JOBS: dict[str, dict[str, Any]] = {}


def _python_exe() -> str:
    venv = _ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else "python"


def _is_admin(info: TokenInfo) -> bool:
    return "*" in info.scopes or "admin" in info.scopes


@router.get("/stats/admin/training-data-counts")
async def training_data_counts(
    info: TokenInfo = Depends(get_token_info),
    days: int = 30,
) -> dict:
    """Row-count snapshot to decide whether a training run makes sense.

    Includes ``since_last_training`` deltas read from the model JSON
    (cache/rerank_v2.json: trained_at, n_train). The auto-retrain logic
    in .github/workflows/train-reranker.yml reads ``ready_to_train``
    plus a ``ready_to_retrain`` (delta-based) gate so we don't burn
    cycles on repeats with no new data.
    """
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    conn = _conn()
    log_count = conn.execute(
        "SELECT COUNT(*) FROM context_feedback_log "
        "WHERE captured_at > datetime('now', ?) "
        "AND query != '' AND stage_scores != '{}'",
        (f"-{days} days",),
    ).fetchone()[0]
    fb_total = conn.execute(
        "SELECT COUNT(*) FROM chunk_feedback "
        "WHERE created_at > datetime('now', ?)",
        (f"-{days} days",),
    ).fetchone()[0]
    # WHY(2026-05-10 rating-migration): nur noch rating 1..5. fb_pos =
    # rating >= 4, fb_neg = rating <= 2. Binary positive/negative entfernt.
    fb_pos = conn.execute(
        "SELECT COUNT(*) FROM chunk_feedback "
        "WHERE created_at > datetime('now', ?) "
        "AND signal IN ('4','5')",
        (f"-{days} days",),
    ).fetchone()[0]
    fb_neg = conn.execute(
        "SELECT COUNT(*) FROM chunk_feedback "
        "WHERE created_at > datetime('now', ?) "
        "AND signal IN ('1','2')",
        (f"-{days} days",),
    ).fetchone()[0]

    last_trained_at: str | None = None
    n_rows_last_train: int = 0
    last_metrics: dict | None = None
    try:
        from mayring_core.memory.reranker_v2 import _load_model
        m = _load_model()
        if isinstance(m, dict):
            last_trained_at = m.get("trained_at")
            n_rows_last_train = int(m.get("n_train", 0)) + int(m.get("n_test", 0))
            last_metrics = m.get("metrics")
    except Exception:
        pass

    new_rows_since_train = max(0, log_count - n_rows_last_train) if last_trained_at else log_count
    cold_start_ready = log_count >= 50 and fb_pos >= 10
    retrain_ready = (
        last_trained_at is not None
        and new_rows_since_train >= 50
        and fb_pos >= 10
    )
    return {
        "window_days": days,
        "retrieval_log_with_features": log_count,
        "feedback_total": fb_total,
        "feedback_positive": fb_pos,
        "feedback_negative": fb_neg,
        "last_trained_at": last_trained_at,
        "n_rows_at_last_train": n_rows_last_train,
        "new_rows_since_train": new_rows_since_train,
        "last_metrics": last_metrics,
        "ready_to_train": cold_start_ready,
        "ready_to_retrain": retrain_ready,
        "min_required": {
            "cold_start": {"retrieval_log_rows": 50, "positives": 10},
            "retrain":    {"new_rows_since_train": 50, "positives": 10},
        },
    }


async def _run_train_subprocess(
    job_id: str, days: int, span_judge: bool = False
) -> None:
    """Spawn export → train as a subprocess so the API stays responsive."""
    state = _TRAIN_JOBS[job_id]
    state.update(status="running", started_at=time.time())
    from mayring_core.config import CACHE_DIR
    out_jsonl = CACHE_DIR / "finetuning" / "retrieval_dataset.jsonl"
    out_model = CACHE_DIR / "rerank_v2.json"
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    try:
        export_cmd = [
            _python_exe(), "tools/export_retrieval_dataset.py",
            "--days", str(days),
            "--out", str(out_jsonl),
        ]
        if span_judge:
            # Offline-LLM-Judge verfeinert Labels (tools/span_judge.py).
            # Läuft IM Container; Ollama via three.linn.games.
            export_cmd.append("--span-judge")
        proc = await asyncio.create_subprocess_exec(
            *export_cmd, cwd=str(_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        export_out, _ = await proc.communicate()
        export_log = (export_out or b"").decode(errors="replace")
        state.update(export_returncode=proc.returncode,
                     export_log=export_log[-1500:])
        if proc.returncode != 0:
            state.update(status="error",
                         error="export failed",
                         ended_at=time.time())
            return
        rows_written = 0
        if out_jsonl.exists():
            try:
                with out_jsonl.open(encoding="utf-8") as f:
                    rows_written = sum(1 for _ in f)
            except OSError:
                pass
        state.update(rows_exported=rows_written)
        if rows_written < 50:
            state.update(
                status="error",
                error=f"only {rows_written} rows exported — need ≥50 for training",
                ended_at=time.time(),
            )
            return
        train_cmd = [
            _python_exe(), "tools/train_reranker.py",
            "--in", str(out_jsonl),
            "--out", str(out_model),
        ]
        proc2 = await asyncio.create_subprocess_exec(
            *train_cmd, cwd=str(_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        train_out, _ = await proc2.communicate()
        train_log = (train_out or b"").decode(errors="replace")
        state.update(train_returncode=proc2.returncode,
                     train_log=train_log[-1500:])
        model_data: dict | None = None
        if out_model.exists():
            try:
                model_data = json.loads(out_model.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        state.update(
            status="done" if proc2.returncode == 0 else "error",
            model=model_data,
            model_path=str(out_model.relative_to(_ROOT)) if out_model.exists() else None,
            ended_at=time.time(),
        )
    except Exception as e:
        state.update(status="error", error=str(e), ended_at=time.time())
        _log.exception("train-reranker job %s failed", job_id)


@router.post("/stats/admin/train-reranker")
async def trigger_train_reranker(
    info: TokenInfo = Depends(get_token_info),
    days: int = 30,
    span_judge: bool = False,
) -> dict:
    """Kick off the export + train pipeline. Admin scope only.

    Workflow inside the spawned subprocess:
      1. tools/export_retrieval_dataset.py → cache/finetuning/retrieval_dataset.jsonl
      2. tools/train_reranker.py            → cache/rerank_v2.json

    Each step uses the SAME script the CLI uses, so behaviour cannot
    drift between local-dev runs and production runs.

    span_judge: when true, the export refines noisy labels with the
    offline Ollama relevance judge (tools/span_judge.py). Adds Ollama
    latency to the export step only — never the retrieval hot path.
    """
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    job_id = f"train-{int(time.time() * 1000)}"
    _TRAIN_JOBS[job_id] = {
        "status": "queued",
        "days": days,
        "span_judge": span_judge,
        "queued_at": time.time(),
    }
    asyncio.create_task(_run_train_subprocess(job_id, days, span_judge))
    return {"job_id": job_id, "status": "queued", "days": days,
            "span_judge": span_judge}


@router.get("/stats/admin/train-reranker/{job_id}")
async def get_train_reranker_status(
    job_id: str,
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    state = _TRAIN_JOBS.get(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="job not found")
    return {"job_id": job_id, **state}


@router.get("/stats/admin/reranker-default")
async def get_reranker_default(
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """Current persisted default reranker version.

    Reads ``cache/rerank_default.txt``. Default 'auto' means 50/50 A/B.
    """
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    from mayring_core.memory.reranker_v2 import _read_runtime_default
    return {"default_version": _read_runtime_default()}


@router.post("/stats/admin/reranker-default")
async def set_reranker_default(
    info: TokenInfo = Depends(get_token_info),
    version: str = "auto",
) -> dict:
    """Persist a new default reranker version. Auto-rollout cron writes
    here when one version's NDCG@5 beats the other by ≥25%."""
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    if version not in ("v1", "v2", "auto"):
        raise HTTPException(status_code=400, detail="version must be v1/v2/auto")
    from mayring_core.memory.reranker_v2 import write_runtime_default
    written = write_runtime_default(version)
    _log.info("reranker default set to %s by workspace=%s", written, info.workspace_id)
    return {"default_version": written}


@router.post("/stats/admin/reranker-rollout-decision")
async def reranker_rollout_decision(
    info: TokenInfo = Depends(get_token_info),
    days: int = 7,
    k: int = 5,
    threshold_pct: float = 2.0,
    apply: bool = False,
) -> dict:
    """Inspect the A/B uplift and (optionally) flip the runtime default.

    WHY(#180, 2026-05-12): the decision metric is now **precision@K**, not
    NDCG@K. NDCG@K saturates near the ceiling (~0.85+) here, so a relative
    threshold on it is almost unreachable (a 25% bump would need NDCG > 1.0)
    — that's why a clearly-better v2 (+16% precision, +9.6% NDCG vs v1) kept
    sitting in 'auto' instead of being rolled out. Precision@K is also the
    metric that actually matters for "did we inject the right chunks" and it
    has real headroom. If one version beats the other by ≥ ``threshold_pct``
    % on precision@K AND has ≥30 queries of evidence, it becomes the new
    default IF apply=True.

    apply=False → returns the recommendation only, doesn't mutate.
    Designed for the auto-rollout workflow to call once per day.

    Decision rules:
      * Insufficient data (queries < 30 in either bucket) → keep 'auto'
      * v2.precision ≥ v1.precision * (1 + threshold/100) → switch to 'v2'
      * v1.precision ≥ v2.precision * (1 + threshold/100) → switch to 'v1'
      * else → keep 'auto' (uncertain, let A/B keep running)
    """
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    from src.api.routes.retrieval_metrics import retrieval_ab as _ab
    from mayring_core.memory.reranker_v2 import _read_runtime_default, write_runtime_default
    ab = await _ab(info=info, days=days, k=k)
    by_version = ab.get("by_version") or {}
    v1 = by_version.get("v1") or {}
    v2 = by_version.get("v2") or {}
    n_v1 = int(v1.get("queries") or 0)
    n_v2 = int(v2.get("queries") or 0)
    p_v1 = float(v1.get("precision_at_k") or 0.0)
    p_v2 = float(v2.get("precision_at_k") or 0.0)
    ndcg_v1 = float(v1.get("ndcg_at_k") or 0.0)  # kept in response for context, not the decision
    ndcg_v2 = float(v2.get("ndcg_at_k") or 0.0)
    current = _read_runtime_default()
    decision = "keep"
    target = current
    reason = ""
    min_queries = 30
    factor = 1.0 + (threshold_pct / 100.0)
    if n_v1 < min_queries or n_v2 < min_queries:
        decision = "keep"
        target = "auto"
        reason = (
            f"insufficient data (v1.queries={n_v1}, v2.queries={n_v2}, "
            f"min={min_queries}); staying 'auto' until both sides have evidence"
        )
    elif p_v1 == 0 and p_v2 == 0:
        decision = "keep"
        target = "auto"
        reason = "both precision=0 — no labelled queries yet, staying 'auto'"
    elif p_v2 >= p_v1 * factor and p_v2 > 0:
        decision = "switch"
        target = "v2"
        reason = (
            f"v2 precision {p_v2:.3f} ≥ v1 precision {p_v1:.3f} × "
            f"{factor:.2f} → v2 wins by ≥{threshold_pct}%"
        )
    elif p_v1 >= p_v2 * factor and p_v1 > 0:
        decision = "switch"
        target = "v1"
        reason = (
            f"v1 precision {p_v1:.3f} ≥ v2 precision {p_v2:.3f} × "
            f"{factor:.2f} → v1 wins by ≥{threshold_pct}%"
        )
    else:
        decision = "keep"
        target = "auto"
        reason = (
            f"neither beats the other by ≥{threshold_pct}% on precision "
            f"(v1={p_v1:.3f}, v2={p_v2:.3f}); 'auto' continues A/B"
        )
    applied = False
    if apply and target != current:
        write_runtime_default(target)
        applied = True
        _log.info(
            "auto-rollout flipped default %s → %s (%s)", current, target, reason,
        )
    return {
        "current": current,
        "target": target,
        "decision": decision,
        "applied": applied,
        "reason": reason,
        "metrics": {
            "v1": {"queries": n_v1, "precision_at_k": p_v1, "ndcg_at_k": ndcg_v1},
            "v2": {"queries": n_v2, "precision_at_k": p_v2, "ndcg_at_k": ndcg_v2},
        },
        "decision_metric": "precision_at_k",
        "threshold_pct": threshold_pct,
        "min_queries": min_queries,
        "window_days": days,
    }
