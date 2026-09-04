"""Shared memory search and ingest logic used by server.py and mcp.py."""
from __future__ import annotations

import logging as _logging
import os as _os
import re as _re
import threading as _threading
import time as _time
from collections import deque
from typing import Any

_log = _logging.getLogger(__name__)

from mayring_core.memory.ingest import ingest
from mayring_core.memory.retrieval import compress_for_prompt, search
from mayring_core.memory.schema import Source

# Brain visualization: recent search activations (ring buffer, shared in-process)
_RECENT_ACTIVATIONS: deque[dict] = deque(maxlen=200)

# Standalone trivial prompts (greeting/ack/test) — whole-string match only, so
# "ok lass uns X bauen" still counts as real work.
_TRIVIAL_RE = _re.compile(
    r"^(ping|ok|okay|k|ja|nein|test|hi|hallo|hey|danke|thx|thanks|yes|no|"
    r"stop|weiter|los|go|fertig|done)[\s!?.]*$", _re.I)
_MIN_QUERY_LEN = 8


# Task-loop HTTP-fanout cap (FALLE 1, 2026-06-20): the act-path loop POSTs each
# sub-question to our OWN /memory/search so uvicorn's worker PROCESSES run the
# CPU-bound symbolic search truly in parallel (separate GILs) — in-process threads
# only parallelised the I/O, the symbolic rerank stayed GIL-serialised. This
# BoundedSemaphore caps TOTAL in-flight internal searches per process so the loop
# never starves the 4 api workers it shares with the hot-path inject. Start at 2
# (conservative); raise via env once measured under load.
_FANOUT_CAP = max(1, int(_os.getenv("TASK_SEARCH_FANOUT_CAP", "2")))
_FANOUT_SEM = _threading.BoundedSemaphore(_FANOUT_CAP)


def _http_retrieve_fn(retrieve_url: str, bearer: str, opts: dict[str, Any],
                      char_budget: int):
    """Build a retrieve_fn that fans sub-questions out as HTTP POSTs to our own
    /memory/search (true cross-process parallelism), instead of an in-process call.

    Preserves the caller's user/workspace scope by forwarding the raw bearer (FALLE
    2) and the same opts the in-process path uses. llm_prefilter=False: the loop
    already judges sufficiency with gemma, so each search skips the redundant
    PI-advisor stage and stays fast."""
    import httpx

    url = retrieve_url.rstrip("/") + "/memory/search"
    body_base: dict[str, Any] = {
        "top_k": opts.get("top_k", 8),
        "char_budget": char_budget,
        "llm_prefilter": False,
    }
    for src_key, dst_key in (("repo", "repo"), ("scope_key", "scope"),
                             ("project_id", "project"), ("session_id", "session_id"),
                             ("category_hint", "category_hint")):
        val = opts.get(src_key)
        if val:
            body_base[dst_key] = val
    auth = bearer if bearer.lower().startswith("bearer ") else f"Bearer {bearer}"
    headers = {"Authorization": auth}
    timeout = httpx.Timeout(connect=5.0, read=25.0, write=5.0, pool=5.0)

    def retrieve(q: str) -> list[dict]:
        try:
            with _FANOUT_SEM:  # cap total in-flight internal searches per process
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(url, json={"query": q, **body_base}, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
        except httpx.HTTPError as exc:
            # One sub-question's search failed (network/auth/timeout). Degrade THIS
            # question to no-chunks so the loop continues with the others instead of
            # 500-ing the whole act-path — logged loudly, never silently swallowed.
            _log.warning("task-loop HTTP fanout failed for sub-question %r: %s", q, exc)
            return []
        return [{"chunk_id": r.get("chunk_id", ""), "text": r.get("text", "") or ""}
                for r in data.get("results", []) if r.get("chunk_id")]

    return retrieve


def _is_corpus_worthy(raw_query: str | None, task: str | None) -> bool:
    """Should this (raw_query → task) pair enter the finetune corpus?

    Gates LOGGING only (never the search). Rejects trivial/test prompts, too-short
    queries, empty tasks, and rows where distillation extracted nothing (task ==
    raw → derive_task fell back to the raw prompt). Keeps the corpus clean so the
    judge finetune learns from real tasks, not 'ping'."""
    rq = (raw_query or "").strip()
    t = (task or "").strip()
    if len(rq) < _MIN_QUERY_LEN:
        return False
    if not t:
        return False
    if _TRIVIAL_RE.match(rq):
        return False
    if t.lower() == rq.lower():  # no real distillation happened
        return False
    return True


def run_search(
    query: str,
    conn: Any,
    chroma: Any,
    ollama_url: str,
    opts: dict[str, Any],
    char_budget: int = 6000,
    session_compacted: bool = False,
) -> dict[str, Any]:
    """Run hybrid search and compress results. Returns {results, prompt_context}."""
    results = search(
        query=query,
        conn=conn,
        chroma_collection=chroma,
        ollama_url=ollama_url,
        opts=opts,
        session_compacted=session_compacted,
    )
    workspace_id = opts.get("workspace_id", "default")
    _activation = {
        "workspace_id": workspace_id,
        "query": query,
        "source_ids": [r.source_id for r in results],
        "ts": _time.time(),
    }
    _RECENT_ACTIVATIONS.append(_activation)  # per-process L1 fallback
    # Write-through to the shared ring so every uvicorn worker (and the
    # dashboard, whichever worker serves it) sees this search (§5.3).
    from src.api import shared_state
    shared_state.activation_push(_activation)
    response = {
        "results": [r.to_dict() for r in results],
        "prompt_context": compress_for_prompt(results, char_budget),
        "diagnostics": {
            "vector_stage": opts.get("_vector_diag", "unknown"),
            "candidates": len(results),
            "reranker_version": opts.get("_reranker_used", "v1"),
        },
    }

    # Inject-effizienz tracking: every search that produces hits also
    # produces a row in context_feedback_log so the "Memory-Effizienz (24h)"
    # card on the dashboard counts hook-injections too. Without this only
    # the legacy MCP-tool path (mcp_memory_tools.py) wrote rows here, so
    # the counter froze the moment everything moved to the hook-path.
    if results:
        try:
            import json as _json
            from datetime import datetime, timezone
            _ids = _json.dumps([r.chunk_id for r in results])
            # Per-stage scores keyed by chunk_id — the dataset that lets us
            # train a learned reranker (Issue #87 Pipeline 2). Logged for
            # every search; missing fields default to 0 so old rows still
            # work in metric queries.
            # CRITICAL: log the SAME value the runtime score_v2 sees, or the
            # trained weights are off. _rerank() uses sv_eff (stretched
            # per-query so the best Chroma hit reaches 1.0) for ranking,
            # but RetrievalRecord.score_vector stores sv_raw (typically
            # [0, 0.5]) for the API response. If we log sv_raw and infer on
            # sv_eff the model learns on the wrong value range — that's the
            # real reason 'v' came out negative in the first two training
            # runs. We now log sv_eff under the 'v' key (model feature),
            # and keep sv_raw under 'v_raw' for diagnostics only.
            #
            # Per-query sv_eff = sv_raw / max(sv_raw across results); the
            # max-normalisation is the same one _rerank() applies.
            _vmax = max(
                (getattr(r, "score_vector", 0.0) or 0.0) for r in results
            )
            def _sv_eff(r):
                raw = getattr(r, "score_vector", 0.0) or 0.0
                return raw / _vmax if _vmax > 0 else 0.0
            _stage = _json.dumps({
                r.chunk_id: {
                    "v":  round(_sv_eff(r), 4),
                    "v_raw": round(getattr(r, "score_vector", 0.0) or 0.0, 4),
                    "s":  round(getattr(r, "score_symbolic", 0.0) or 0.0, 4),
                    "r":  round(getattr(r, "score_recency", 0.0) or 0.0, 4),
                    "a":  round(getattr(r, "score_source_affinity", 0.0) or 0.0, 4),
                    # v2 features — aggregate signals, not user verdicts.
                    # 0.5 = "no signal yet" (chunk had zero feedback events
                    # at retrieval time, or LLM advisor was disabled).
                    # User feedback itself stays binary in chunk_feedback;
                    # what we log here is the per-chunk RATIO computed
                    # from those binary events.
                    "sf": round(getattr(r, "score_feedback", 0.5) or 0.5, 4),
                    "sl": round(getattr(r, "score_llm", 0.5) or 0.5, 4),
                    # WHY(#187): score_predicted_topic + rationale-presence als
                    # Trainings-Features. Ohne diese loggt der context_feedback_log
                    # die Werte nicht und der Trainer kann sie nicht lernen —
                    # `pt` und `re` wären phantom-features im API-Response.
                    "pt": round(getattr(r, "score_predicted_topic", 0.0) or 0.0, 4),
                    "re": 1.0 if (getattr(r, "rationale_edges", None) or []) else 0.0,
                    # WHY(#270 reranker-v3): cat_match (query↔chunk codebook-category
                    # overlap) must be logged so the daily trainer can learn its
                    # weight (Phase B) — otherwise only the deterministic 0.08 boost
                    # (Phase A) ever uses it and it's a phantom feature in the model.
                    "cat_match": round(getattr(r, "score_cat_match", 0.0) or 0.0, 4),
                    "f":  round(getattr(r, "score_final", 0.0) or 0.0, 4),
                }
                for r in results
            })
            from mayring_core.memory.store import log_context_injection
            log_context_injection(
                conn, trigger_ids=_ids, context_text=response["prompt_context"],
                query=query, stage_scores=_stage, workspace_id=workspace_id,
                reranker_version=opts.get("_reranker_used", "v1"))
        except Exception:
            pass  # non-critical; never block the search result

    return response


def ensure_task_search_log(conn) -> None:
    """The clean finetune corpus: every task-anchored search logs
    (raw_query → task → sub-questions → halt → chunks). MayringCoder-local table
    (not a core-memory concern) so no mayring-core migration is needed. Shared by
    the REST endpoint and the MCP tool so there is ONE corpus writer."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS task_search_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id  TEXT NOT NULL,
            raw_query     TEXT NOT NULL,
            task          TEXT NOT NULL,
            questions     TEXT NOT NULL DEFAULT '[]',
            halted_by     TEXT NOT NULL DEFAULT '',
            loops         INTEGER NOT NULL DEFAULT 0,
            n_chunks      INTEGER NOT NULL DEFAULT 0,
            chunk_ids     TEXT NOT NULL DEFAULT '[]',
            created_at    TEXT NOT NULL
        )"""
    )
    conn.commit()


def run_task_search(
    query: str,
    conn: Any,
    chroma: Any,
    ollama_url: str,
    opts: dict[str, Any],
    *,
    char_budget: int = 6000,
    max_loops: int = 2,
    budget_s: float = 25.0,
    max_q: int = 4,
    think: bool = False,
    already_task: bool = False,
    anchor_only: bool = False,
    conn_factory: Any = None,
    parallelism: int = 4,
    retrieve_url: str | None = None,
    bearer: str | None = None,
) -> dict[str, Any]:
    """Task-anchored Mythos retrieval, shared by REST + MCP.

    Distills the prompt to a task anchor, then either:
      - anchor_only=True  → ONE search with the task (hot-path safe, ~7s): the
        +13pp task-anchor win without the decomposition loop. This is what the
        UserPromptSubmit inject uses (replaces the 3 redundant same-query lenses;
        the full loop measured 12-15s, too slow for the 9s budget).
      - anchor_only=False → fan the task into sub-questions and loop until
        answered/no-progress (act-path depth, via the MCP tool).
    Both log the clean corpus row. Returns {task, questions, halted_by, loops, chunks}."""
    import json as _json
    from tools.sufficiency_gate import derive_task, derive_and_decompose, run_task_loop

    workspace_id = opts.get("workspace_id", "default")
    # Task derivation, by mode:
    #  - already_task: the query IS the task (recap/clean caller) → no LLM call.
    #  - anchor_only: only the task is needed (no loop) → derive_task alone.
    #  - full loop: fuse derive+decompose into ONE gemma call (~1.5s saved vs the
    #    separate derive_task + in-loop decompose). pre_questions feeds the loop so
    #    it skips its own decompose.
    pre_questions: list[str] | None = None
    if already_task:
        task = query.strip()
    elif anchor_only:
        task = derive_task(query, ollama_url)
    else:
        task, pre_questions = derive_and_decompose(query, ollama_url, max_q=max_q)

    # Sub-question retrieval, two modes (act-path only; anchor_only never loops):
    #  - HTTP fanout (retrieve_url + bearer given): POST to our own /memory/search
    #    so the CPU-bound search runs in OTHER uvicorn worker PROCESSES → true
    #    parallelism past the GIL. Cap = _FANOUT_CAP (conservative 2).
    #  - In-process: a fresh thread-local SQLite conn per thread (conn_factory);
    #    Chroma is a shared thread-safe server. Without conn_factory the shared
    #    conn would race, so we fall back to sequential (parallelism=1). This stays
    #    GIL-serialised for the symbolic stage — the fallback, not the fast path.
    # Kill-switch: TASK_SEARCH_FANOUT=0 forces the in-process path in prod without a
    # redeploy (instant rollback if the fanout misbehaves under load).
    _fanout_on = _os.getenv("TASK_SEARCH_FANOUT", "1").lower() not in ("0", "false", "no")
    use_http = bool(_fanout_on and retrieve_url and bearer and not anchor_only)
    if use_http:
        retrieve_fn = _http_retrieve_fn(retrieve_url, bearer, opts, char_budget)
        _par = min(parallelism, _FANOUT_CAP)
    else:
        _par = parallelism if conn_factory else 1

        def retrieve_fn(q: str) -> list[dict]:
            c = conn_factory() if conn_factory else conn
            sr = run_search(q, c, chroma, ollama_url, opts, char_budget)
            return [{"chunk_id": r.get("chunk_id", ""), "text": r.get("text", "") or ""}
                    for r in sr.get("results", []) if r.get("chunk_id")]

    extra: dict[str, Any] = {}
    if anchor_only:
        # one search; pass the FULL run_search response through (results +
        # prompt_context) so the inject hook can render + persist exactly like a
        # normal /memory/search, with source_id intact for the Stop-hook feedback.
        sr = run_search(task, conn, chroma, ollama_url, opts, char_budget)
        results = sr.get("results", [])
        chunks = [{"chunk_id": r.get("chunk_id", ""), "text": r.get("text", "") or ""}
                  for r in results if r.get("chunk_id")]
        questions, halted_by, loops = [task], "anchor_only", 0
        extra = {"results": results, "prompt_context": sr.get("prompt_context", ""),
                 "diagnostics": sr.get("diagnostics", {})}
    else:
        loop = run_task_loop(task, retrieve_fn, ollama_url, think=think,
                             max_loops=max_loops, budget_s=budget_s, max_q=max_q,
                             parallelism=_par, questions=pre_questions)
        chunks = loop["final_chunks"]
        questions, halted_by, loops = loop["questions"], loop["halted_by"], loop["loops"]

    # Corpus hygiene: only log real (prompt → task) pairs. Trivial/test prompts
    # and no-op distillations would poison the finetune corpus — the exact failure
    # mode this whole effort fixed. The search itself already ran; we just don't
    # record junk.
    if _is_corpus_worthy(query, task):
        try:
            ensure_task_search_log(conn)
            conn.execute(
                "INSERT INTO task_search_log (workspace_id, raw_query, task, questions, "
                "halted_by, loops, n_chunks, chunk_ids, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (workspace_id, query, task, _json.dumps(questions),
                 halted_by, loops, len(chunks),
                 _json.dumps([c["chunk_id"] for c in chunks]),
                 _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())),
            )
            conn.commit()
        except Exception:
            pass  # corpus logging is best-effort; never fail the search

    return {"task": task, "questions": questions, "halted_by": halted_by,
            "loops": loops, "chunks": chunks, **extra}


def run_ingest(
    source_dict: dict[str, Any],
    content: str,
    conn: Any,
    chroma: Any,
    ollama_url: str,
    model: str,
    opts: dict[str, Any],
    workspace_id: str = "default",
) -> dict[str, Any]:
    """Create Source from dict and ingest into memory. Returns ingest result dict."""
    # Compute content_hash from the actual content when the caller didn't
    # set one. /memory/put doesn't pass content_hash in the request, and
    # the state-detection logic (core.py: NEW/CHANGED/UNCHANGED) needs it
    # to know whether anything changed. Without this fix every PUT returns
    # state="new" even when re-ingesting identical content — the exact
    # acceptance-criterion failure for closed Issue #137.
    import hashlib as _hashlib
    given_hash = source_dict.get("content_hash", "")
    if not given_hash and content is not None:
        given_hash = "sha256:" + _hashlib.sha256(
            (content or "").encode("utf-8")
        ).hexdigest()

    src = Source(
        source_id=source_dict.get("source_id") or Source.make_id(
            source_dict.get("repo", ""), source_dict.get("path", "")
        ),
        source_type=source_dict.get("source_type", "repo_file"),
        repo=source_dict.get("repo", ""),
        path=source_dict.get("path", ""),
        branch=source_dict.get("branch", "main"),
        commit=source_dict.get("commit", ""),
        content_hash=given_hash,
        visibility=source_dict.get("visibility") or "private",
        org_id=source_dict.get("org_id"),
        user_id=source_dict.get("user_id"),
        scope_key=source_dict.get("scope_key"),  # #252
        source_class=source_dict.get("source_class") or "code",  # reference-doc-layer
    )
    return ingest(
        source=src,
        content=content,
        conn=conn,
        chroma_collection=chroma,
        ollama_url=ollama_url,
        model=model,
        opts=opts,
        workspace_id=workspace_id,
    )
