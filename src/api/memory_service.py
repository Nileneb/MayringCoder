"""Shared memory search and ingest logic used by server.py and mcp.py."""
from __future__ import annotations

import time as _time
from collections import deque
from typing import Any

from mayring_core.memory.ingest import ingest
from mayring_core.memory.retrieval import compress_for_prompt, search
from mayring_core.memory.schema import Source

# Brain visualization: recent search activations (ring buffer, shared in-process)
_RECENT_ACTIVATIONS: deque[dict] = deque(maxlen=200)


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
    _RECENT_ACTIVATIONS.append({
        "workspace_id": workspace_id,
        "query": query,
        "source_ids": [r.source_id for r in results],
        "ts": _time.time(),
    })
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
                    "f":  round(getattr(r, "score_final", 0.0) or 0.0, 4),
                }
                for r in results
            })
            conn.execute(
                "INSERT INTO context_feedback_log"
                " (trigger_ids,context_text,was_referenced,led_to_retrieval,"
                "  relevance_score,captured_at,query,stage_scores,workspace_id,"
                "  reranker_version)"
                " VALUES (?,?,0,0,0.0,?,?,?,?,?)",
                (_ids, response["prompt_context"][:2000],
                 datetime.now(timezone.utc).isoformat(),
                 query[:1000], _stage, workspace_id,
                 opts.get("_reranker_used", "v1")),
            )
            conn.commit()
        except Exception:
            pass  # non-critical; never block the search result

    return response


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
