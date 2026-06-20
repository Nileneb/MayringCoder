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
import re
import threading
import time
from pathlib import Path
from typing import Any

# Export prints `PROGRESS <done>/<total>` per batch of events; the subprocess
# runner parses these into job-state.progress for the live frontend bar.
_PROGRESS_RE = re.compile(r"^PROGRESS\s+(\d+)/(\d+)")

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from src.api.auth import get_token_info, get_workspace
from src.api.dependencies import get_conn as _conn
from src.api.jwt_auth import TokenInfo

router = APIRouter()
_log = logging.getLogger(__name__)
_ROOT = Path(__file__).parent.parent.parent.parent

# WHY(multi-worker, 2026-05-28): under uvicorn --workers the train job is
# created in ONE worker's in-memory _TRAIN_JOBS; a status GET routed to another
# worker found nothing ("status: None") → the dashboard "Status prüfen" button
# + the train_reranker MCP tool polled blind. Mirror job_queue's shared-file
# pattern (atomic tmp+rename) so every worker sees the same job state. Schema
# differs from populate jobs, so it gets its own file.
_TRAIN_JOBS_FILE = Path(
    os.environ.get("MAYRING_TRAIN_JOBS_STATE", str(_ROOT / "cache" / "train_jobs_state.json"))
)
_TRAIN_JOBS_LOCK = threading.Lock()


def _load_train_jobs() -> dict[str, dict[str, Any]]:
    try:
        if _TRAIN_JOBS_FILE.exists():
            return json.loads(_TRAIN_JOBS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        pass  # corrupt state must not take the API down — next save overwrites
    return {}


def _save_train_job(job_id: str) -> None:
    """Merge one job into the shared file atomically (read-modify-write so a
    concurrent worker's jobs are never clobbered). Best-effort; never raises."""
    try:
        _TRAIN_JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _TRAIN_JOBS_LOCK:
            shared = _load_train_jobs()
            shared[job_id] = _TRAIN_JOBS.get(job_id, {})
            tmp = _TRAIN_JOBS_FILE.with_suffix(_TRAIN_JOBS_FILE.suffix + ".tmp")
            tmp.write_text(json.dumps(shared, default=str), encoding="utf-8")
            tmp.replace(_TRAIN_JOBS_FILE)
    except OSError:
        pass


_TRAIN_JOBS: dict[str, dict[str, Any]] = _load_train_jobs()


def _python_exe() -> str:
    venv = _ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else "python"


def _is_admin(info: TokenInfo) -> bool:
    return "*" in info.scopes or "admin" in info.scopes


@router.get("/stats/admin/training-data-counts")
def training_data_counts(
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
        from mayring_core.memory.reranker_v2 import _load_model, list_reranker_versions
        m = _load_model()
        if not isinstance(m, dict):
            # WHY(versioning 2026-06-13): Trainings akkumulieren als rerank_v3..vN —
            # das Legacy-Single-Slot rerank_v2.json existiert nicht mehr, _load_model()
            # (default v2) lieferte None → Dashboard zeigte "nie trainiert" trotz v7.
            # Fallback: das NEUESTE trainierte Modell aus der Versionsliste.
            trained = [v for v in list_reranker_versions() if v.get("trained_at")]
            if trained:
                latest = max(trained, key=lambda v: str(v["trained_at"]))
                m = {
                    "trained_at": latest["trained_at"],
                    "n_train": latest.get("n_train") or 0,
                    "n_test": latest.get("n_test") or 0,
                    "metrics": latest.get("metrics"),
                }
        if isinstance(m, dict):
            last_trained_at = m.get("trained_at")
            n_rows_last_train = int(m.get("n_train", 0)) + int(m.get("n_test", 0))
            last_metrics = m.get("metrics")
    except Exception as e:
        _log.warning("training-data-counts: model state unreadable: %s", e)

    if last_trained_at:
        # WHY(2026-05-28): count rows actually logged SINCE the model trained —
        # NOT (windowed_count − all_time_trainset_size). The old
        # `max(0, log_count − n_rows_last_train)` subtracted the model's all-time
        # train size (n_train+n_test, here 18336) from a `days`-window count
        # (6718); once the trainset exceeds the window the delta goes negative →
        # clamped 0 FOREVER → ready_to_retrain never fired → the reranker never
        # retrained despite ~749 new injections/day (the stalled-loop bug).
        # datetime() normalizes the ISO model-ts (…+00:00) vs the no-tz
        # captured_at (utcnow().isoformat()) so the same-day compare is correct
        # (a raw string compare breaks on the 'T' vs ' ' separator).
        new_rows_since_train = conn.execute(
            "SELECT COUNT(*) FROM context_feedback_log "
            "WHERE datetime(captured_at) > datetime(?) "
            "AND query != '' AND stage_scores != '{}'",
            (last_trained_at,),
        ).fetchone()[0]
    else:
        new_rows_since_train = log_count
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
    job_id: str, days: int, span_judge: bool = False,
    span_judge_max_calls: int | None = None,
) -> None:
    """Spawn export → train as a subprocess so the API stays responsive."""
    state = _TRAIN_JOBS[job_id]

    def _upd(**kw: Any) -> None:
        state.update(**kw)
        _save_train_job(job_id)  # persist to shared file so any worker's GET sees it

    _upd(status="running", started_at=time.time())
    from mayring_core.config import CACHE_DIR
    out_jsonl = CACHE_DIR / "finetuning" / "retrieval_dataset.jsonl"
    # WHY(reranker-version-table 2026-05-30): write the NEXT accumulating version
    # (v3, v4 …) instead of overwriting rerank_v2.json, so the dashboard table
    # gains a new row each retrain and you click-to-activate. Mirrors the CLI path.
    try:
        from tools.train_reranker import _next_version_path
        out_model = _next_version_path()
    except Exception:  # fallback: keep the legacy single-slot behaviour
        out_model = CACHE_DIR / "rerank_v2.json"
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    if span_judge:
        # WHY(reranker-gpu-relief, 2026-06-08 live-run): scope these to the EXPORT
        # subprocess only — its fresh ollama_client import reads
        # OLLAMA_CLOUD_PRIMARY_RATIO at load, so the live API process keeps its own
        # (low) hot-path ratio. Default ratio 1.0 = the heavy LLM judge runs
        # CLOUD-FIRST (local big model only as fallback): the live run showed a
        # 50/50 split sent half the calls local-first into the saturated single-slot
        # GPU host's 60s ReadTimeout, crawling AND starving the inject-advisor. The
        # user's standing rule: LLM calls → cloud, embeddings → local GPU.
        # SPAN_JUDGE_MAX_CALLS caps fresh judge calls so a retrain can never hammer
        # unboundedly (backstop next to Claude pre-warm + cooldown). All via GitHub env.
        env["OLLAMA_CLOUD_PRIMARY_RATIO"] = os.getenv("SPAN_JUDGE_CLOUD_RATIO", "1.0")
        # WHY(Pfad-A claude-teacher): span_judge_max_calls=0 → CACHE-ONLY: der
        # Export refined NUR mit vorgewärmten Claude-Labels (span_judge_cache),
        # KEIN frischer Ollama-Call → der schwache ministral-3:3b kann v nicht mehr
        # vergiften (v7/v8: v invertiert). (max_calls=0 heißt im budget-check
        # 'unbegrenzt', daher der separate SPAN_JUDGE_CACHE_ONLY-Schalter.)
        # max_calls>0 = harter Fresh-Call-Backstop; None = 400-Default.
        if span_judge_max_calls == 0:
            env["SPAN_JUDGE_CACHE_ONLY"] = "1"
        else:
            _mc = span_judge_max_calls if span_judge_max_calls is not None \
                else int(os.getenv("SPAN_JUDGE_MAX_CALLS", "400"))
            env["SPAN_JUDGE_MAX_CALLS"] = str(_mc)
        env["SPAN_JUDGE_TIMEOUT"] = os.getenv("SPAN_JUDGE_TIMEOUT", "45")
        env["SPAN_JUDGE_COOLDOWN_EVERY"] = os.getenv("SPAN_JUDGE_COOLDOWN_EVERY", "15")
        env["SPAN_JUDGE_COOLDOWN_SECONDS"] = os.getenv("SPAN_JUDGE_COOLDOWN_SECONDS", "2.5")
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
        # Stream stdout line-by-line so the frontend sees LIVE progress
        # (PROGRESS x/y markers from the export) instead of only the final log
        # via communicate(). PROGRESS lines update job-state.progress (persisted
        # to the shared file, returned by the status GET) and are filtered out
        # of the stored export_log to keep it readable.
        export_lines: list[str] = []
        _upd(progress={"phase": "export", "current": 0, "total": 0, "pct": 0})
        assert proc.stdout is not None
        async for _raw in proc.stdout:
            line = _raw.decode(errors="replace")
            m = _PROGRESS_RE.match(line.strip())
            if m:
                cur, tot = int(m.group(1)), int(m.group(2))
                _upd(progress={"phase": "export", "current": cur, "total": tot,
                               "pct": (round(100 * cur / tot) if tot else 0)})
            else:
                export_lines.append(line)
        await proc.wait()
        export_log = "".join(export_lines)
        _upd(export_returncode=proc.returncode,
             export_log=export_log[-1500:])
        if proc.returncode != 0:
            _upd(status="error",
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
        _upd(rows_exported=rows_written)
        if rows_written < 50:
            _upd(
                status="error",
                error=f"only {rows_written} rows exported — need ≥50 for training",
                ended_at=time.time(),
            )
            return
        _upd(progress={"phase": "train", "current": 0, "total": 0, "pct": 0})
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
        _upd(train_returncode=proc2.returncode,
             train_log=train_log[-1500:])
        model_data: dict | None = None
        if out_model.exists():
            try:
                model_data = json.loads(out_model.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        _upd(
            status="done" if proc2.returncode == 0 else "error",
            model=model_data,
            model_path=str(out_model.relative_to(_ROOT)) if out_model.exists() else None,
            progress={"phase": "done", "current": 0, "total": 0, "pct": 100},
            ended_at=time.time(),
        )
    except Exception as e:
        _upd(status="error", error=str(e), ended_at=time.time())
        _log.exception("train-reranker job %s failed", job_id)


@router.post("/stats/admin/train-reranker")
async def trigger_train_reranker(
    info: TokenInfo = Depends(get_token_info),
    days: int = 30,
    span_judge: bool = False,
    span_judge_max_calls: int | None = None,
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
        "span_judge_max_calls": span_judge_max_calls,
        "queued_at": time.time(),
    }
    _save_train_job(job_id)  # persist before the task starts so a GET on any worker finds it
    asyncio.create_task(
        _run_train_subprocess(job_id, days, span_judge, span_judge_max_calls))
    return {"job_id": job_id, "status": "queued", "days": days,
            "span_judge": span_judge, "span_judge_max_calls": span_judge_max_calls}


@router.get("/stats/admin/train-reranker/{job_id}")
def get_train_reranker_status(
    job_id: str,
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    # Read the shared file FIRST (the job may run in a different worker); fall
    # back to this worker's in-memory copy.
    state = _load_train_jobs().get(job_id) or _TRAIN_JOBS.get(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="job not found")
    return {"job_id": job_id, **state}


@router.get("/stats/admin/reranker-default")
def get_reranker_default(
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """Current persisted default reranker version.

    Reads ``cache/rerank_default.txt``. Default 'auto' means 50/50 A/B.
    """
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    from mayring_core.memory.reranker_v2 import _read_runtime_default
    return {"default_version": _read_runtime_default()}


@router.get("/stats/admin/reranker-versions")
def list_reranker_versions_endpoint(
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """All selectable reranker versions for the dashboard table (v1 baseline +
    every cache/rerank_v<N>.json) with metadata + active flag.

    ``active`` is now a list of 1–2 versions (A/B pair or single active).
    """
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    from mayring_core.memory.reranker_v2 import read_active_versions, list_reranker_versions
    active = read_active_versions()
    versions = list_reranker_versions()
    # WHY(2026-06-20): list_reranker_versions derives `active` from the legacy
    # rerank_default.txt, which diverges from the real serving SoT (rerank_active.json
    # via read_active_versions) — that mismatch is why the dashboard showed v4 active
    # while v3 was actually serving. Re-stamp the flag from the SoT so there is ONE truth.
    active_set = set(active)
    for v in versions:
        v["active"] = v["version"] in active_set
    return {"active": active, "versions": versions}


def _assert_active_quality(versions: list[str], force: bool) -> None:
    """Quality invariant: a trained model may only become active if it scores
    ≥ the v1 vector baseline on the leakage-free clean-eval. This is the single
    gate that stops degenerate/below-baseline models (v4–v7) from creeping back
    into the serving A/B pool — no matter which write path put them there
    (manual, deprecated alias, future auto-rollout, migration fallback).

    Fail-soft: when there is no claude-labelled evidence yet, clean-eval is empty
    → we cannot judge → we do NOT block (the existence/format check still runs).
    `force=True` is an explicit, logged human override for a deliberate below-
    baseline activation. v1 (the baseline itself) is always allowed."""
    if force:
        return
    scores = _clean_eval_scores()
    if not scores:
        return
    baseline = scores.get("v1")
    if baseline is None:
        return
    for v in (str(x).strip().lower() for x in versions):
        if v == "v1":
            continue
        s = scores.get(v)
        if s is not None and s < baseline:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{v} clean-eval nDCG {s:.4f} < v1 baseline {baseline:.4f} — "
                    f"refusing to activate a below-baseline model. Pass force=true "
                    f"to override deliberately."
                ),
            )


class RerankerActiveReq(BaseModel):
    versions: list[str]


@router.post("/stats/admin/reranker-active")
def set_reranker_active(
    body: RerankerActiveReq,
    info: TokenInfo = Depends(get_token_info),
    force: bool = False,
) -> dict:
    """Set 1–2 active reranker versions (A/B pair or single).

    Replaces the old single-default + autorollout model.  Each version must
    be 'v1' or a 'v<N>' whose model file exists; passing 3+ versions returns
    HTTP 422.  A non-v1 version below the v1 clean-eval baseline is rejected
    (422) unless force=true. Admin scope required.
    """
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    _assert_active_quality(body.versions, force)
    from mayring_core.memory.reranker_v2 import write_active_versions
    try:
        written = write_active_versions(body.versions)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _log.info(
        "reranker active versions set to %s (force=%s) by workspace=%s",
        written, force, info.workspace_id,
    )
    return {"active": written}


@router.post("/stats/admin/reranker-default")
def set_reranker_default(
    info: TokenInfo = Depends(get_token_info),
    version: str = "auto",
) -> dict:
    """DEPRECATED-Alias: setzt EINE aktive Reranker-Version. Nutze /reranker-active.

    WHY(2026-06-06): das Serving liest jetzt rerank_active.json (read_active_versions),
    NICHT mehr rerank_default.txt. Dieser Endpoint schrieb sonst still eine Datei, die
    niemand liest (Footgun). Er leitet jetzt auf write_active_versions um. 'auto' ist
    bedeutungslos (zwei aktive Versionen SIND das A/B) → 400."""
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    if (version or "").strip().lower() == "auto":
        raise HTTPException(status_code=400,
                            detail="'auto' entfällt — setze 2 aktive Versionen via /reranker-active")
    _assert_active_quality([version], force=False)
    from mayring_core.memory.reranker_v2 import write_active_versions
    try:
        written = write_active_versions([version])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _log.info("reranker active set to %s (via default-alias) by workspace=%s", written, info.workspace_id)
    return {"default_version": written[0], "active": written}


@router.delete("/stats/admin/reranker-versions/{version}")
def delete_reranker_version_endpoint(
    version: str,
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """Lösche ein trainiertes Reranker-Modell (rerank_v<N>.json) aus der Liste.
    Guard: v1/auto + das aktive Modell sind geschützt."""
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    from mayring_core.memory.reranker_v2 import delete_reranker_version
    try:
        deleted = delete_reranker_version(version)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _log.info("reranker version %s deleted=%s by workspace=%s", version, deleted, info.workspace_id)
    return {"version": version, "deleted": deleted}



@router.post("/stats/admin/reembed-categories")
def reembed_categories(info: TokenInfo = Depends(get_token_info)) -> dict:
    """Re-embed active codebook_categories into the Chroma 'codebook_categories'
    collection. Admin scope only.

    Repairs the silent reranker-v3 cat_match death after a Chroma cutover: the
    categories persist in SQLite (with embedding_id), but their vectors vanish
    from the Chroma collection when the chroma service restarts → query→category
    derivation returns empty → cat_match goes inert and category-themed searches
    stop surfacing matches. There was NO repair path (the codebook import is a
    host-side tool, not auto-run on deploy), so this had to be fixed by hand each
    time. Idempotent upsert; safe to run any time. Sync def → FastAPI threadpool
    so the blocking Ollama embed calls don't stall the event loop.
    """
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    import os
    from mayring_core.ollama_client import embed_batch
    from mayring_core.memory.store import get_chroma_collection
    conn = _conn()
    rows = conn.execute(
        "SELECT embedding_id, name, COALESCE(description, name) "
        "FROM categories WHERE status='active' AND embedding_id != ''"
    ).fetchall()
    if not rows:
        return {"categories": 0, "embedded": 0,
                "detail": "no active categories with embedding_id"}
    url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    # WHY(bge-m3-migration): MUSS dasselbe Modell wie der Core nutzen (config.EMBEDDING_MODEL,
    # env-driven → prod=bge-m3). Vorher hartes MAYRING_EMBED_MODEL → embeddete Kategorien
    # weiter mit nomic@768, während memory_chunks auf bge-m3@1024 lief → Dim-Mismatch,
    # query→category-Derivation (cat_match) still inert. MAYRING_EMBED_MODEL bleibt als
    # expliziter Override.
    from mayring_core.config import EMBEDDING_MODEL as _CORE_EMBED_MODEL
    model = os.getenv("MAYRING_EMBED_MODEL") or _CORE_EMBED_MODEL
    # WHY(#343): fail-fast embed-Timeout statt hardcoded 120 — ein hängendes/
    # überlastetes Ollama soll den (im Threadpool laufenden) Reembed-Batch nicht
    # minutenlang blockieren. OLLAMA_EMBED_TIMEOUT=30 default.
    from mayring_core.config import OLLAMA_EMBED_TIMEOUT
    col = get_chroma_collection("codebook_categories")
    ids = [r[0] for r in rows]
    texts = [f"{r[1]}: {r[2]}" for r in rows]
    embedded = 0
    _reset_done = False
    from mayring_core.memory.store import reset_chroma_collection
    for i in range(0, len(ids), 64):
        embs = embed_batch(url, model, texts[i:i + 64], timeout=OLLAMA_EMBED_TIMEOUT)
        if not embs:
            continue
        try:
            col.upsert(ids=ids[i:i + 64], embeddings=embs,
                       documents=texts[i:i + 64])
        except Exception as exc:
            # WHY(bge-m3 dim-mismatch 2026-06-08): the codebook_categories collection was
            # created at the old nomic dim (768); re-embedding at bge-m3 (1024) raised
            # "Collection expecting embedding with dimension of 768, got 1024" → cat_match
            # stayed inert. Drop+recreate at the new dim once, then retry. (Same self-heal
            # as memory_sync; now a shared core helper.)
            if "dimension" not in str(exc).lower() or _reset_done:
                raise
            _log.warning("reembed-categories: dim mismatch (%s) — recreating collection", exc)
            col = reset_chroma_collection("codebook_categories")
            _reset_done = True
            col.upsert(ids=ids[i:i + 64], embeddings=embs,
                       documents=texts[i:i + 64])
        embedded += len(embs)
    _log.info("reembed-categories: %d/%d embedded by workspace=%s",
              embedded, len(rows), info.workspace_id)
    return {"categories": len(rows), "embedded": embedded,
            "collection_count": col.count()}


@router.post("/stats/admin/dedup-categories")
def dedup_categories(
    threshold: float = 0.93,
    dry_run: bool = True,
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """#340 Ebene b — Kategorie-Dedup (Mayring S7 für System #2: die `categories`-
    Tabelle + chunk_categories-FK, NICHT die chunks.category_labels-CSV).

    Problem: query→category (derive_query_category_ids, cosine-nearest gegen die
    Chroma-`codebook_categories`) und die chunk_categories-FK der Treffer landen auf
    VERSCHIEDENEN, aber quasi-identischen Kategorie-IDs (z.B. 262 `user_authentication
    _and_login_processes` vs 261) → leere Schnittmenge → reranker-v3 cat_match=0.

    Fix: Near-Dup-Kategorien (cosine >= threshold auf ihren Codebook-Embeddings)
    in EINE kanonische ID kollabieren. Kanonisch = die meist-verlinkte (chunk_
    categories COUNT, Tie → kleinere id). Pro Dup: chunk_categories + codebook_
    proposals-FKs auf die kanonische ID repointen (OR IGNORE gegen den UNIQUE-
    (chunk_id,category_id)-Index, dann Reste löschen), Dup-Chroma-Embedding aus
    `codebook_categories` entfernen, Dup-Row löschen.

    Idempotent. `dry_run=True` (default) mutiert NICHTS — liefert nur den Plan
    (read-only). Sync def → Threadpool (#343), blockiert den Event-Loop nicht.
    """
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    import numpy as np
    from mayring_core.memory.store import get_chroma_collection
    conn = _conn()
    rows = conn.execute(
        "SELECT id, name, embedding_id FROM categories "
        "WHERE status='active' AND embedding_id != ''"
    ).fetchall()
    if len(rows) < 2:
        return {"dry_run": dry_run, "clusters": 0, "merges": [], "detail": "< 2 active categories"}

    col = get_chroma_collection("codebook_categories")
    emb_ids = [r[2] for r in rows]
    got = col.get(ids=emb_ids, include=["embeddings"])
    # Chroma kann IDs fehlen lassen (verwaiste embedding_ids) → nur vorhandene nutzen.
    emb_by_id = {i: e for i, e in zip(got.get("ids", []), got.get("embeddings", []))}
    cats = [
        {"id": r[0], "name": r[1], "emb_id": r[2], "vec": np.asarray(emb_by_id[r[2]], dtype=float)}
        for r in rows if r[2] in emb_by_id and emb_by_id[r[2]] is not None
    ]
    if len(cats) < 2:
        return {"dry_run": dry_run, "clusters": 0, "merges": [],
                "detail": f"{len(cats)} categories with a Chroma embedding (need >=2)"}

    mat = np.vstack([c["vec"] for c in cats])
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sim = (mat / norms) @ (mat / norms).T

    # Union-Find über alle Paare mit cosine >= threshold → Near-Dup-Cluster.
    parent = list(range(len(cats)))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(cats)):
        for b in range(a + 1, len(cats)):
            if sim[a][b] >= threshold:
                ra, rb = _find(a), _find(b)
                if ra != rb:
                    parent[ra] = rb

    clusters: dict[int, list[int]] = {}
    for i in range(len(cats)):
        clusters.setdefault(_find(i), []).append(i)

    def _link_count(cat_id: int) -> int:
        return conn.execute(
            "SELECT COUNT(*) FROM chunk_categories WHERE category_id=?", (cat_id,)
        ).fetchone()[0]

    merges: list[dict] = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        # Index in `cats` durchreichen (KEIN cats.index(d) — die Dicts tragen ein
        # numpy 'vec', dict-Gleichheit darauf wirft "ambiguous truth value").
        ranked = sorted(
            members,
            key=lambda i: (-_link_count(cats[i]["id"]), cats[i]["id"]),
        )
        ci = ranked[0]
        canon = cats[ci]
        dup_idxs = ranked[1:]
        merges.append({
            "canonical": {"id": canon["id"], "name": canon["name"],
                          "links": _link_count(canon["id"])},
            "dups": [{"id": cats[i]["id"], "name": cats[i]["name"],
                      "links": _link_count(cats[i]["id"]),
                      "cosine": round(float(sim[ci][i]), 3)} for i in dup_idxs],
            "chunk_links_repointed": sum(_link_count(cats[i]["id"]) for i in dup_idxs),
        })

    if dry_run or not merges:
        return {"dry_run": dry_run, "threshold": threshold,
                "active_with_emb": len(cats), "clusters_with_dups": len(merges),
                "merges": merges}

    # --- APPLY (mutiert Prod) -------------------------------------------------
    applied = 0
    for m in merges:
        canon_id = m["canonical"]["id"]
        for d in m["dups"]:
            dup_id = d["id"]
            # chunk_categories: auf canon repointen; OR IGNORE gegen UNIQUE(chunk_id,
            # category_id), dann verbliebene Dup-Links löschen.
            conn.execute("UPDATE OR IGNORE chunk_categories SET category_id=? WHERE category_id=?",
                         (canon_id, dup_id))
            conn.execute("DELETE FROM chunk_categories WHERE category_id=?", (dup_id,))
            # codebook_proposals-FKs (falls Tabelle existiert) ebenfalls repointen.
            for fk in ("category_id", "parent_hint_id"):
                try:
                    conn.execute(
                        f"UPDATE codebook_proposals SET {fk}=? WHERE {fk}=?",
                        (canon_id, dup_id))
                except Exception:  # noqa: BLE001 — Tabelle/Spalte optional
                    pass
            # Dup-Chroma-Embedding entfernen, damit query→category nicht mehr darauf zeigt.
            dup_emb = conn.execute(
                "SELECT embedding_id FROM categories WHERE id=?", (dup_id,)).fetchone()
            if dup_emb and dup_emb[0]:
                try:
                    col.delete(ids=[dup_emb[0]])
                except Exception as exc:  # noqa: BLE001
                    _log.warning("dedup: chroma embedding %s nicht entfernt: %s", dup_emb[0], exc)
            conn.execute("DELETE FROM categories WHERE id=?", (dup_id,))
            applied += 1
    conn.commit()
    _log.info("dedup-categories: %d dup-Kategorien in %d Cluster gemergt (threshold=%s) by ws=%s",
              applied, len(merges), threshold, info.workspace_id)
    return {"dry_run": False, "threshold": threshold,
            "clusters_with_dups": len(merges), "dups_merged": applied,
            "merges": merges, "codebook_count": col.count()}


@router.post("/stats/admin/relink-chunks")
def relink_chunks(
    dry_run: bool = True,
    top_n: int = 2,
    info: TokenInfo = Depends(get_token_info),
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """#340 Ebene b (echter Fix) — bestehende Chunks deduktiv gegen das AKTUELLE
    Codebook neu verlinken.

    Root-Cause cat_match=0: die chunk_categories-Tags wurden zu einem FRÜHEREN
    Codebook-Stand erzeugt; `derive_query_category_ids` (Query-Seite) nutzt das
    HEUTIGE Codebook → die vektor-nah abgerufenen Chunks tragen andere category_ids
    (245/261/631) als die Query ableitet (262/300) → leere Schnittmenge. KEIN
    Duplikat-Problem (Dedup-Dry-Run bewies: 262/261 clustern nicht mal bei 0.93).

    Fix: `link_chunks_deductive` (dieselbe `_best_match`+`_HYBRID_MIN`-Logik wie die
    Query-Ableitung) für alle aktiven Chunks des Workspace neu ausführen → die
    abgerufenen Chunks bekommen die Kategorien, die die Query heute ableitet →
    cat_match feuert. ADDITIV (INSERT OR REPLACE pro (chunk,cat); bestehende Links
    bleiben, kein Datenverlust). top_n=2: jeder Chunk wird mit seinen 2 besten
    aktuellen Kategorien verlinkt (erhöht die Überlapp-Chance mit der top_n=3-Query).
    Workspace-scoped. dry_run=True (default) zeigt nur Zahlen. Sync def → Threadpool.
    """
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    import numpy as np
    from mayring_core.memory.store import get_chroma_collection
    from mayring_core.memory.ingestion.mayring_process import (
        _category_embeddings, _link_chunk, _HYBRID_MIN,
    )
    conn = _conn()
    rows = conn.execute(
        "SELECT chunk_id FROM chunks WHERE workspace_id=? AND is_active=1", (workspace_id,)
    ).fetchall()
    chunk_ids = [r[0] for r in rows]
    if not chunk_ids:
        return {"dry_run": dry_run, "workspace_id": workspace_id, "active_chunks": 0,
                "detail": "keine aktiven Chunks in diesem Workspace"}

    chunks_col = get_chroma_collection("memory_chunks")
    pairs: list[tuple[str, list]] = []
    _B = 500
    for i in range(0, len(chunk_ids), _B):
        batch = chunk_ids[i:i + _B]
        got = chunks_col.get(ids=batch, include=["embeddings"])
        for cid, emb in zip(got.get("ids", []), got.get("embeddings", [])):
            if emb is not None and len(emb):
                pairs.append((cid, emb))

    if dry_run:
        return {"dry_run": True, "workspace_id": workspace_id,
                "active_chunks": len(chunk_ids), "chunks_with_embedding": len(pairs),
                "top_n": top_n,
                "detail": "apply mit dry_run=false verlinkt diese Chunks gegen das aktuelle Codebook"}

    # Aktive Codebook-Kategorien + Embeddings (kanonischer Fetch).
    codebook_col = get_chroma_collection("codebook_categories")
    crows = conn.execute(
        "SELECT id, name, parent_id, embedding_id FROM categories "
        "WHERE status='active' AND embedding_id != '' AND project_id IS NULL ORDER BY id"
    ).fetchall()
    cats = [{"id": r[0], "name": r[1], "parent_id": r[2],
             "embedding_id": r[3]} for r in crows]
    cat_pairs = _category_embeddings(codebook_col, cats)
    if not cat_pairs:
        return {"dry_run": False, "workspace_id": workspace_id, "links_written": 0,
                "detail": "keine aktiven Kategorie-Embeddings"}
    cat_objs = [c for c, _ in cat_pairs]
    C = np.asarray([e for _, e in cat_pairs], dtype=float)
    cn = np.linalg.norm(C, axis=1, keepdims=True); cn[cn == 0] = 1.0
    C = C / cn

    # WHY(#340 perf): EIN numpy-matmul pro Batch (X @ Cᵀ) statt per-Chunk-Python-
    # _cosine über alle Kategorien (4346×170 Python-Loops = Minuten + Riesen-Txn,
    # blockierte /memory/search). Batch-Commit alle 1000 Chunks → Write-Lock wird
    # periodisch freigegeben (kein "database is locked" für parallele Writer).
    min_score = _HYBRID_MIN
    written = 0
    CHUNK_BATCH = 1000
    for start in range(0, len(pairs), CHUNK_BATCH):
        cb = pairs[start:start + CHUNK_BATCH]
        X = np.asarray([e for _, e in cb], dtype=float)
        xn = np.linalg.norm(X, axis=1, keepdims=True); xn[xn == 0] = 1.0
        sims = (X / xn) @ C.T
        for ri, (cid, _) in enumerate(cb):
            row = sims[ri]
            for j in np.argsort(row)[::-1][:top_n]:
                s = float(row[j])
                if s >= min_score:
                    _link_chunk(conn, cid, cat_objs[j]["id"], version=1,
                                confidence=s, source="deductive")
                    written += 1
        conn.commit()
    _log.info("relink-chunks: %d Links für %d Chunks (top_n=%d, %d Kat.) ws=%s",
              written, len(pairs), top_n, len(cat_objs), workspace_id)
    return {"dry_run": False, "workspace_id": workspace_id,
            "chunks_relinked": len(pairs), "links_written": written,
            "categories": len(cat_objs), "top_n": top_n}


@router.get("/stats/admin/cat-match-debug")
def cat_match_debug(
    query: str = "user authentication login session token oauth jwt password",
    info: TokenInfo = Depends(get_token_info),
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Pinpoint a red reranker_cat_match_fires: report BOTH sides — does the
    query derive category_ids (query side), and does the corpus have any
    chunk_categories FK rows (chunk side) — instead of guessing. Read-only."""
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    import os
    conn = _conn()
    out: dict[str, Any] = {"query": query}
    for label, sql in (
        ("codebook_active", "SELECT COUNT(*) FROM categories WHERE status='active' AND embedding_id != ''"),
        ("chunk_categories_total", "SELECT COUNT(*) FROM chunk_categories"),
        ("chunks_with_categories", "SELECT COUNT(DISTINCT chunk_id) FROM chunk_categories"),
    ):
        try:
            out[label] = conn.execute(sql).fetchone()[0]
        except Exception as e:  # noqa: BLE001 — diagnostic, surface the error
            out[label] = f"ERR {type(e).__name__}: {e}"
    try:
        from mayring_core.ollama_client import embed_batch
        from mayring_core.memory.store import get_chroma_collection
        from mayring_core.memory.ingestion.mayring_process import derive_query_category_ids
        url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        model = os.getenv("MAYRING_EMBED_MODEL", "nomic-embed-text")
        col = get_chroma_collection("codebook_categories")
        out["chroma_codebook_count"] = col.count()
        qemb = (embed_batch(url, model, [query], timeout=60) or [None])[0]
        out["query_embedded"] = qemb is not None
        if qemb is not None:
            ids = derive_query_category_ids(conn, col, qemb)
            out["query_category_ids"] = sorted(ids)
            # #340 Diagnose (read-only): die VOLLE Rangliste der Query gegen alle
            # Kategorie-Embeddings (mit Scores) — zeigt, wo die chunk-seitigen
            # Kategorien (z.B. 261/687/637) für die Query ranken → datengetriebene
            # Wahl von min_score/top_n bzw. Synonym-Expansion statt Raten.
            try:
                from mayring_core.memory.ingestion.mayring_process import (
                    _active_category_pairs, _cosine,
                )
                pairs = _active_category_pairs(conn, col, None)
                ranked = sorted(
                    ((float(_cosine(qemb, emb)), cat["id"], cat["name"]) for cat, emb in pairs),
                    key=lambda t: t[0], reverse=True,
                )
                out["query_top_categories"] = [
                    {"id": cid, "name": nm, "cos": round(s, 3)} for s, cid, nm in ranked[:15]
                ]
                # Nachbarn der top-1-Query-Kategorie (Synonym-Distanz): wie nah ist
                # 262 zu 261 & Co.? → entscheidet, ob Synonym-Expansion greift.
                if ranked:
                    top_id = ranked[0][1]
                    emb_by_id = {cat["id"]: emb for cat, emb in pairs}
                    temb = emb_by_id.get(top_id)
                    if temb is not None:
                        nbrs = sorted(
                            ((float(_cosine(temb, emb)), cat["id"], cat["name"])
                             for cat, emb in pairs if cat["id"] != top_id),
                            key=lambda t: t[0], reverse=True,
                        )
                        out["top_query_cat_neighbors"] = [
                            {"id": cid, "name": nm, "cos": round(s, 3)} for s, cid, nm in nbrs[:10]
                        ]
            except Exception as e:  # noqa: BLE001 — diagnostic
                out["ranking_err"] = f"{type(e).__name__}: {e}"
            if ids:
                ph = ",".join("?" for _ in ids)
                out["query_category_names"] = [
                    r[0] for r in conn.execute(
                        f"SELECT name FROM categories WHERE id IN ({ph})",
                        tuple(ids)).fetchall()
                ]
            # Chunk side: which category_ids do chunks actually link to? If the
            # query-derived ids are duplicate "auth" ids that chunks never link
            # to, the set-intersection is empty even though both sides look
            # populated (id fragmentation across the codebook).
            top = conn.execute(
                "SELECT cc.category_id, cat.name, COUNT(*) c "
                "FROM chunk_categories cc JOIN categories cat ON cat.id = cc.category_id "
                "GROUP BY cc.category_id ORDER BY c DESC LIMIT 15"
            ).fetchall()
            out["chunk_side_top_category_ids"] = [
                {"id": r[0], "name": r[1], "chunks": r[2]} for r in top
            ]
            if ids:
                chunk_ids_with_derived = conn.execute(
                    f"SELECT COUNT(DISTINCT chunk_id) FROM chunk_categories "
                    f"WHERE category_id IN ({','.join('?' for _ in ids)})",
                    tuple(ids)).fetchone()[0]
                out["chunks_linked_to_query_ids"] = chunk_ids_with_derived
    except Exception as e:  # noqa: BLE001 — diagnostic, surface the error
        out["derive_err"] = f"{type(e).__name__}: {e}"
    # The smoke's exact path: run the workspace-scoped search and inspect whether
    # the RETRIEVED candidates are linked to the query-derived ids. If not, the
    # query/chunk overlap exists globally but the searched workspace's retrieved
    # chunks aren't linked → cat_match 0 (workspace coverage, not a global gap).
    try:
        from src.api.memory_service import run_search
        from src.api.dependencies import get_chroma as _get_chroma
        sres = run_search(query, conn, _get_chroma(),
                          os.getenv("OLLAMA_URL", "http://localhost:11434"),
                          {"top_k": 10, "workspace_id": workspace_id, "llm_prefilter": False})
        results = sres.get("results", []) or []
        cand_ids = [r.get("chunk_id") for r in results if r.get("chunk_id")]
        out["search_workspace"] = workspace_id
        out["search_n"] = len(results)
        out["search_cat_match_hits"] = sum(
            1 for r in results if (r.get("score_cat_match") or 0) > 0)
        if cand_ids:
            ph = ",".join("?" for _ in cand_ids)
            rows = conn.execute(
                f"SELECT chunk_id, category_id FROM chunk_categories WHERE chunk_id IN ({ph})",
                tuple(cand_ids)).fetchall()
            out["candidates_with_any_category"] = len({r[0] for r in rows})
            out["candidate_category_ids"] = sorted({r[1] for r in rows})
    except Exception as e:  # noqa: BLE001 — diagnostic, surface the error
        out["search_err"] = f"{type(e).__name__}: {e}"
    return out


def _backfill_chunk_categories(after_rowid: int, limit: int) -> dict[str, Any]:
    """One cursor-paginated window of the deductive chunk→category backfill.

    Phase 3.2 links chunks to codebook categories per-ingest, but the existing
    corpus (and chunks ingested while the codebook_categories Chroma collection
    was empty post-cutover) were never linked → reranker-v3 cat_match had almost
    no coverage on the retrieved set (verified: 8/10 candidates uncategorised).
    LLM-free (cosine vs the 108 cached category embeddings); idempotent
    (chunk_categories PK upsert), so re-running over the whole table is safe."""
    import os as _os
    from mayring_core.memory.store import get_chroma_collection
    from mayring_core.memory.ingestion.mayring_process import link_chunks_deductive
    conn = _conn()
    # Only UNLINKED chunks (rowid-cursor advances past already-linked + the
    # unmatchable so it always terminates). Keeps the recurring post-deploy run
    # cheap once the one-time full backfill is done — re-linking the whole corpus
    # every deploy would load the box + flake the smoke.
    rows = conn.execute(
        "SELECT rowid, chunk_id FROM chunks "
        "WHERE rowid > ? AND chunk_id NOT IN (SELECT chunk_id FROM chunk_categories) "
        "ORDER BY rowid LIMIT ?",
        (after_rowid, limit),
    ).fetchall()
    if not rows:
        return {"processed": 0, "linked": 0, "next_after": after_rowid, "has_more": False}
    chunk_ids = [r[1] for r in rows]
    max_rowid = rows[-1][0]
    chunks_col = get_chroma_collection("memory_chunks")
    got = chunks_col.get(ids=chunk_ids, include=["embeddings"])
    got_ids = got.get("ids") or []
    got_embs = got.get("embeddings")
    got_embs = list(got_embs) if got_embs is not None else []
    pairs = [
        (cid, emb) for cid, emb in zip(got_ids, got_embs)
        if emb is not None and len(emb)
    ]
    linked = 0
    if pairs:
        linked = link_chunks_deductive(
            conn, get_chroma_collection("codebook_categories"), pairs)
        conn.commit()
    return {
        "processed": len(rows),
        "embeddings_found": len(pairs),
        "linked": linked,
        "next_after": max_rowid,
        "has_more": len(rows) == limit,
    }


@router.post("/stats/admin/chunk-categories-backfill")
async def chunk_categories_backfill(
    after: int = 0,
    limit: int = 500,
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """Deductive chunk→category backfill (one window per call; loop with
    ``next_after`` until ``has_more`` is false). Restores reranker-v3 cat_match
    coverage on chunks ingested before Phase 3.2 or while the category Chroma
    was cold. LLM-free + idempotent. Admin-only; runs in the threadpool so the
    cosine loop doesn't stall the event loop."""
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    return await run_in_threadpool(_backfill_chunk_categories, after, max(1, min(limit, 2000)))


@router.post("/stats/admin/label-advisor")
async def label_advisor(
    after: int = 0,
    limit: int = 15,
    confidence_threshold: float = 0.75,
    info: TokenInfo = Depends(get_token_info),
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """LLM label refinement for LOW-cosine-confidence chunks (SubQ/SSA-style).

    The category-consolidation derives category_labels from the cosine
    chunk_categories FK; where the cosine match was weak (best link confidence <
    threshold) the labels can be imprecise. This re-labels ONLY those chunks via
    an LLM CONSTRAINED to the active codebook (picks names from the fixed list —
    no free-derive, so labels stay aligned with the codebook/FK SoT). One batched
    LLM call routed through the central PiQueue (cloud-split aware). Cursor-
    paginated (loop next_after until has_more=false). Admin-only; NOT on the hot
    ingest path — meant for post-deploy-ingest / manual runs."""
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    lim = max(1, min(limit, 40))
    conn = _conn()
    rows = conn.execute(
        "SELECT c.rowid, c.chunk_id, COALESCE(c.summary, c.text, '') "
        "FROM chunks c WHERE c.is_active = 1 AND c.category_source = 'deductive-link' "
        "AND c.rowid > ? "
        "AND (SELECT MAX(cc.confidence) FROM chunk_categories cc WHERE cc.chunk_id = c.chunk_id) < ? "
        "ORDER BY c.rowid LIMIT ?",
        (after, confidence_threshold, lim),
    ).fetchall()
    if not rows:
        return {"processed": 0, "advised": 0, "next_after": after, "has_more": False}
    max_rowid = rows[-1][0]
    cats = [r[0] for r in conn.execute(
        "SELECT name FROM categories WHERE status='active' AND name != '' ORDER BY name"
    ).fetchall()]
    if not cats:
        return {"processed": len(rows), "advised": 0, "next_after": max_rowid,
                "has_more": len(rows) == lim, "detail": "no active codebook categories"}
    cat_set = {c.lower() for c in cats}
    items = [(r[1], (r[2] or "")[:500]) for r in rows]
    prompt = (
        "You are a Mayring category labeler. For each chunk, assign the 1-3 "
        "categories from THIS fixed list that its content most supports (judge by "
        "the actual content, not surface keywords). Use ONLY names from the list, "
        "verbatim.\n"
        f"Categories: {', '.join(cats)}\n\n"
        'Output STRICT JSON only: {"<chunk_id>": ["<name>", ...], ...}. No prose. '
        "Every chunk_id MUST appear.\n\nChunks:\n"
        + "\n\n".join(f"[{cid}]\n{txt}" for cid, txt in items)
    )
    import os as _os
    import uuid as _uuid
    from datetime import datetime, timezone
    from mayring_pi_agent.pi_queue import get_pi_queue
    from mayring_pi_agent.pi_jobs import PiJob
    from mayring_core.model_router import ModelRouter
    from mayring_core.memory.store import kv_get, kv_put
    from src.api.mcp_agent_tools import _loads_json_lenient
    _ollama = _os.getenv("OLLAMA_URL", "http://localhost:11434")
    model = ModelRouter(_ollama).resolve("text") or "mistral:7b-instruct"
    job = PiJob(
        job_id=_uuid.uuid4().hex[:16], task_text=prompt, workspace_id=workspace_id,
        kind="label-advise", job_class="standard", model=model,
        response_format="json", timeout_s=90.0,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    try:
        result = await get_pi_queue().enqueue(job)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"label-advise queue failed: {exc}")
    content = (result.get("content") if isinstance(result, dict) else str(result)) or ""
    try:
        data = _loads_json_lenient(content)
    except Exception:
        data = {}
    advised = 0
    for cid, _txt in items:
        labels = data.get(cid) if isinstance(data, dict) else None
        if not isinstance(labels, list):
            continue
        # CONSTRAINED: keep only labels that exist verbatim in the codebook.
        valid = [s for s in (str(x).strip() for x in labels) if s.lower() in cat_set][:3]
        if not valid:
            continue
        conn.execute(
            "UPDATE chunks SET category_labels = ?, category_source = 'llm-advised' "
            "WHERE chunk_id = ? AND is_active = 1",
            (",".join(valid), cid),
        )
        cached = kv_get(cid)
        if cached is not None:
            cached["category_labels"] = valid
            cached["category_source"] = "llm-advised"
            kv_put(cid, cached)
        advised += 1
    conn.commit()
    return {"processed": len(rows), "advised": advised,
            "next_after": max_rowid, "has_more": len(rows) == lim}


def _memory_db_conn():
    """Fresh sqlite3 connection to the prod memory.db with Row factory + a
    busy timeout — the span-judge prewarm helpers (and the export) expect
    name-indexed rows and tolerate the live API holding WAL write locks."""
    import sqlite3
    from mayring_core.config import CACHE_DIR
    conn = sqlite3.connect(str(CACHE_DIR / "memory.db"), timeout=15.0)
    conn.row_factory = sqlite3.Row
    return conn


@router.get("/stats/admin/span-judge/uncached-pairs")
def span_judge_uncached_pairs(
    days: int = 7,
    limit: int = 300,
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """Emit the uncached (query, chunk) pairs Claude should judge as the strong
    teacher for the reranker (Pfad A). Mirrors ``tools/span_judge_prewarm.py
    --dump`` but server-side so it runs against the prod memory.db without a
    host exec. The caller judges each batch per span_judge.RELEVANCE_RUBRIC and
    POSTs the scores back to /span-judge/ingest; the next --span-judge retrain
    then reads those Claude labels cache-first instead of the weak Ollama judge
    (ministral-3:3b poisoned v7: v=-3.26). Admin-only; read-only."""
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    from tools.span_judge_prewarm import dump
    lim = max(0, min(limit, 600))
    conn = _memory_db_conn()
    try:
        batches = dump(conn, days, lim)
    finally:
        conn.close()
    n_pairs = sum(len(b["chunks"]) for b in batches)
    return {"window_days": days, "limit": lim, "batches": len(batches),
            "pairs": n_pairs, "data": batches}


class SpanJudgeIngestReq(BaseModel):
    scores: list[dict]


@router.post("/stats/admin/span-judge/ingest")
def span_judge_ingest(
    body: SpanJudgeIngestReq,
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """Write Claude's relevance scores into span_judge_cache tagged
    'claude-prewarm' (Pfad A). Body: {"scores": [{"query": str,
    "scores": {chunk_id: 0..1}}, ...]}. Idempotent (INSERT OR REPLACE per
    (query_hash, chunk_id)). Admin-only."""
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    from tools.span_judge_prewarm import ingest
    conn = _memory_db_conn()
    try:
        written = ingest(conn, body.scores)
    finally:
        conn.close()
    _log.info("span-judge prewarm ingest: %d scores by workspace=%s",
              written, info.workspace_id)
    return {"ingested": written}


def _dcg(gains: list[float]) -> float:
    import math
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _ndcg_at_k(ranked_rels: list[float], k: int) -> float:
    """Graded nDCG@k: ranked_rels = Claude relevance in MODEL-ranked order."""
    dcg = _dcg(ranked_rels[:k])
    idcg = _dcg(sorted(ranked_rels, reverse=True)[:k])
    return (dcg / idcg) if idcg > 0 else 0.0


def _clean_eval_compute(k: int = 5) -> dict:
    """Leakage-free reranker eval, factored out so BOTH the endpoint and the
    active-versions quality gate use the identical measure. Ranks each query's
    chunks by every model and scores nDCG@k against CLAUDE's relevance labels
    (span_judge_cache claude-prewarm), NOT the recency-leaked was_referenced
    label. Returns {} (no by_version) when there is no claude-labelled evidence
    yet — callers treat that as 'cannot judge', never as 'all zero'."""
    import json as _json
    try:
        from tools import span_judge as _sj
    except ImportError:
        import span_judge as _sj
    from mayring_core.memory.reranker_v2 import _load_model, score_v2, list_reranker_versions

    import sqlite3 as _sqlite3
    try:
        conn = _memory_db_conn()
    except Exception as exc:  # no DB (fresh install / test env) → no evidence
        _log.warning("clean-eval: memory.db unavailable (%s) — no quality evidence", exc)
        return {}
    try:
        # WHY(2026-06-20): a fresh/empty DB (CI, new install) lacks span_judge_cache /
        # context_feedback_log → OperationalError. That legitimately means "no claude
        # labels yet → cannot judge quality" → return {} (the gate then fail-soft does
        # NOT block). Catching the missing-table here, not at connect, was the bug that
        # failed 4 activation tests against a fresh CI DB.
        h2q: dict[str, str] = {}
        for (q,) in conn.execute(
                "SELECT DISTINCT query FROM context_feedback_log WHERE query != ''"):
            h2q.setdefault(_sj.query_hash(q), q)
        # claude relevance per query_hash → {chunk_id: rel}
        claude: dict[str, dict[str, float]] = {}
        for qh, cid, score in conn.execute(
                "SELECT query_hash, chunk_id, score FROM span_judge_cache "
                "WHERE model = 'claude-prewarm'"):
            claude.setdefault(qh, {})[cid] = float(score)
        # latest stage_scores per query (the feature dict the reranker sees)
        eval_queries: list[dict] = []
        for qh, rels in claude.items():
            query = h2q.get(qh)
            if not query or len(rels) < 2:  # need ≥2 chunks to rank
                continue
            row = conn.execute(
                "SELECT stage_scores FROM context_feedback_log "
                "WHERE query = ? AND stage_scores != '{}' "
                "ORDER BY captured_at DESC LIMIT 1", (query,)).fetchone()
            if not row:
                continue
            try:
                stage = _json.loads(row[0])
            except (TypeError, ValueError):
                continue
            pairs = [(cid, rels[cid], stage.get(cid))
                     for cid in rels if isinstance(stage.get(cid), dict)]
            if len(pairs) >= 2:
                eval_queries.append({"query": query, "pairs": pairs})
    except _sqlite3.OperationalError as exc:  # missing table = no labels yet
        _log.warning("clean-eval: %s — no quality evidence, gate fail-soft", exc)
        return {}
    finally:
        conn.close()

    if not eval_queries:
        return {}

    versions = [v["version"] for v in list_reranker_versions()]
    out: dict[str, float] = {}
    n_pairs = sum(len(q["pairs"]) for q in eval_queries)
    for ver in versions:
        model = _load_model(ver) if ver != "v1" else None
        ndcgs: list[float] = []
        for q in eval_queries:
            if model is not None:
                scored = [(score_v2(st, model), rel) for _cid, rel, st in q["pairs"]]
            else:  # v1 baseline: rank by raw vector stage-score
                scored = [(float(st.get("v", 0.0)), rel) for _cid, rel, st in q["pairs"]]
            scored.sort(key=lambda t: t[0], reverse=True)
            ndcgs.append(_ndcg_at_k([rel for _s, rel in scored], k))
        out[ver] = round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0.0
    return {"by_version": out, "eval_queries": len(eval_queries), "eval_pairs": n_pairs}


def _clean_eval_scores(k: int = 5) -> dict[str, float]:
    """ver → clean nDCG@k. Empty dict = no claude-labelled evidence → no gating."""
    return _clean_eval_compute(k).get("by_version", {})


def _sufficiency_agreement(eval_queries: list[dict], judge_fn,
                           rel_threshold: float = 0.6) -> dict:
    """Does gemma's sufficiency verdict agree with the Claude labels?

    Ground-truth proxy: a query is OBJECTIVELY sufficient if at least one of its
    chunks has Claude relevance ≥ rel_threshold (there is a real primary source);
    otherwise objectively insufficient (only tangential chunks). gemma judges the
    same chunk set. We report agreement + the two error directions:
      * false_pass = gemma says sufficient but objectively NOT → dangerous (the
        gate would let weak context through without a re-retrieval loop).
      * false_loop = gemma says insufficient but objectively yes → wasted loop.
    eval_queries: [{"query": str, "chunks": [{"chunk_id","text"}], "rels": [float]}].
    judge_fn(query, chunks) → {"sufficient": bool, ...} (injected for tests)."""
    n = len(eval_queries)
    if not n:
        return {}
    agree = false_pass = false_loop = obj_suff = gemma_suff = 0
    for q in eval_queries:
        objective = max(q["rels"], default=0.0) >= rel_threshold
        verdict = judge_fn(q["query"], q["chunks"])
        gemma = bool(verdict.get("sufficient"))
        obj_suff += int(objective)
        gemma_suff += int(gemma)
        if gemma == objective:
            agree += 1
        elif gemma and not objective:
            false_pass += 1
        else:
            false_loop += 1
    return {
        "eval_queries": n,
        "agreement": round(agree / n, 4),
        "false_pass_rate": round(false_pass / n, 4),
        "false_loop_rate": round(false_loop / n, 4),
        "objective_sufficient": obj_suff,
        "gemma_sufficient": gemma_suff,
        "rel_threshold": rel_threshold,
    }


@router.get("/stats/admin/sufficiency-eval")
def sufficiency_eval(
    rel_threshold: float = 0.6,
    limit: int = 40,
    think: bool = False,
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """Outcome proof for the Mythos sufficiency gate: how well does gemma's
    sufficient/insufficient verdict track the Claude relevance labels? Makes one
    gemma call per claude-labelled query (bounded by `limit`; read-only, ACT path
    only — never the hot path). Empty → no claude labels yet."""
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    import sqlite3 as _sqlite3
    try:
        from tools import span_judge as _sj
        from tools.sufficiency_gate import judge_sufficiency
    except ImportError:
        import span_judge as _sj
        from sufficiency_gate import judge_sufficiency

    try:
        conn = _memory_db_conn()
    except Exception as exc:
        _log.warning("sufficiency-eval: memory.db unavailable (%s)", exc)
        return {"error": "memory.db unavailable"}
    try:
        h2q: dict[str, str] = {}
        for (q,) in conn.execute(
                "SELECT DISTINCT query FROM context_feedback_log WHERE query != ''"):
            h2q.setdefault(_sj.query_hash(q), q)
        rows = conn.execute(
            "SELECT s.query_hash, s.chunk_id, s.score, c.text "
            "FROM span_judge_cache s JOIN chunks c ON c.chunk_id = s.chunk_id "
            "WHERE s.model = 'claude-prewarm'").fetchall()
    except _sqlite3.OperationalError as exc:
        _log.warning("sufficiency-eval: %s — no claude labels", exc)
        return {"error": "no claude labels yet"}
    finally:
        conn.close()

    by_q: dict[str, dict] = {}
    for qh, cid, score, text in rows:
        query = h2q.get(qh)
        if not query or not text:
            continue
        e = by_q.setdefault(qh, {"query": query, "chunks": [], "rels": []})
        e["chunks"].append({"chunk_id": cid, "text": text})
        e["rels"].append(float(score))
    eval_queries = [e for e in by_q.values() if len(e["chunks"]) >= 1][:max(1, limit)]
    if not eval_queries:
        return {"error": "no claude-labelled queries with chunk text"}

    url = _ollama_url_for_eval()
    res = _sufficiency_agreement(
        eval_queries,
        lambda q, ch: judge_sufficiency(q, ch, url, think=think),
        rel_threshold=rel_threshold,
    )
    res["metric"] = "gemma_sufficiency_vs_claude_labels"
    return res


def _ollama_url_for_eval() -> str:
    import os as _os
    return _os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")


@router.get("/stats/admin/reranker-clean-eval")
def reranker_clean_eval(
    k: int = 5,
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """Leakage-free reranker eval: rank each query's chunks by every model and
    score nDCG@k against CLAUDE's relevance labels (span_judge_cache claude-prewarm),
    NOT the recency-leaked was_referenced label. This is the honest "is model X
    better than v5" measure — the offline auc/p@1 from training reward the very
    recency leakage that crippled v5's served vector weight. Read-only; admin."""
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    res = _clean_eval_compute(k)
    if not res:
        return {"error": "no claude-labelled queries with stage_scores + ≥2 chunks",
                "hint": "run span-judge prewarm ingest first"}
    out = res["by_version"]
    ranked = sorted(out.items(), key=lambda t: t[1], reverse=True)
    return {
        "metric": f"clean_ndcg_at_{k}_vs_claude",
        "eval_queries": res["eval_queries"],
        "eval_pairs": res["eval_pairs"],
        "by_version": out,
        "best": ranked[0][0] if ranked else None,
        "ranking": ranked,
    }


@router.post("/stats/admin/reranker-rollout-decision")
def reranker_rollout_decision(
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
    from mayring_core.memory.reranker_v2 import (
        _read_runtime_default, write_runtime_default,
    )
    from src.api.routes.retrieval_metrics import retrieval_ab as _ab
    ab = _ab(info=info, days=days, k=k)
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


@router.get("/stats/admin/db-export")
def db_export(info: TokenInfo = Depends(get_token_info)):
    """#backup — streamt einen tar.gz der MayringCoder-DB (memory.db + Chroma) für
    systemd-unabhängiges Backup + Dev-Sync. Symmetrisch zum app.linn.games
    pg_dump-Endpoint, nutzt aber die bestehende Admin-Auth (service-token/JWT) +
    nginx /stats/admin-Allowlist. Exportiert NUR die DB-Artefakte:
    memory.db(+wal/shm) + memory_chroma/ — KEINE Secrets (agent_keys.json) oder
    Wiki-/Job-State. memory.db-wal mitnehmen → SQLite repliziert beim Öffnen →
    konsistenter Snapshot. Sync def → Threadpool (#343), tar streamt ~190 MB ohne
    Event-Loop-Block.
    """
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    import subprocess
    from fastapi.responses import StreamingResponse
    from mayring_core.config import CACHE_DIR

    members = [f for f in ("memory.db", "memory.db-wal", "memory.db-shm", "memory_chroma")
               if (CACHE_DIR / f).exists()]
    if not members:
        raise HTTPException(status_code=500, detail="no DB artefacts in cache dir")

    cmd = ["tar", "-czf", "-", "-C", str(CACHE_DIR), *members]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _gen():
        try:
            assert proc.stdout is not None
            for chunk in iter(lambda: proc.stdout.read(1 << 16), b""):
                yield chunk
        finally:
            proc.stdout and proc.stdout.close()
            rc = proc.wait()
            if rc != 0:
                err = (proc.stderr.read() if proc.stderr else b"")[:500]
                _log.error("db-export: tar exit %s: %s", rc, err)

    stamp = time.strftime("%Y-%m-%d_%H%M%S", time.gmtime())
    return StreamingResponse(_gen(), media_type="application/gzip", headers={
        "Content-Disposition": f'attachment; filename="mayring_db_{stamp}.tar.gz"',
        "Cache-Control": "no-store",
        "X-Accel-Buffering": "no",
    })
