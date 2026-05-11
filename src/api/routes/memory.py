from __future__ import annotations

import hashlib
import logging
import os
import threading as _threading
from typing import Any

_log = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, HTTPException

from src.api.auth import get_token_info, get_workspace
from src.api.dependencies import get_chroma as _get_chroma, get_conn as _get_conn
from src.api.jwt_auth import TokenInfo
from src.api.memory_service import run_ingest as _run_ingest, run_search as _run_search
from src.api.routes.models import (
    ConversationMicroBatchRequest,
    LogEvent,
    LogEventBatch,
    MemoryFeedbackRequest,
    MemoryInvalidateRequest,
    MemoryPutRequest,
    MemoryReindexRequest,
    MemorySearchRequest,
    PatchVisibilityRequest,
    PiTaskRequest,
)

router = APIRouter(tags=["memory"])

_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def _bg_wiki_rebuild(workspace_id: str) -> None:
    try:
        from src.wiki_v2.graph import WikiGraph
        from src.wiki_v2.edge_detector import EdgeDetector
        from src.wiki_v2.clustering import ClusterEngine
        conn = _get_conn()
        wiki_db = WikiGraph(workspace_id=workspace_id, repo_slug="")
        edges = EdgeDetector(ollama_url=_OLLAMA_URL).detect_from_overview({}, conn, workspace_id, "")
        for e in edges:
            wiki_db.add_edge(e)
        ClusterEngine().cluster(wiki_db, strategy="louvain")
    except Exception:
        _log.exception("Background wiki rebuild failed for workspace_id=%s", workspace_id)

def _model(task: str = "text") -> str:
    from src.model_router import ModelRouter
    return ModelRouter(_OLLAMA_URL).resolve(task)


@router.post("/pi-task")
async def pi_task(
    request: PiTaskRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Run a task via the Pi-agent (memory-augmented reasoning).

    Issue #183 T3: jobs go through the in-process PiQueue with bounded
    concurrency (PI_CONCURRENCY=2 default). API contract stays
    backward-compatible — callers still await directly on the result.
    """
    import uuid as _uuid
    from datetime import datetime, timezone
    from src.agents.pi_queue import get_pi_queue
    from src.agents.pi_jobs import PiJob, classify_pi_job

    _repo_slug = request.repo_slug or os.getenv("PI_REPO_SLUG", "")
    job = PiJob(
        job_id=_uuid.uuid4().hex[:16],
        task_text=request.task,
        system_prompt=request.system_prompt or "",
        repo_slug=_repo_slug,
        workspace_id=workspace_id,
        kind="pi-task",
        job_class=classify_pi_job(request.task, request.system_prompt or ""),
        timeout_s=request.timeout,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    queue = get_pi_queue()
    fut = queue.enqueue(job)
    try:
        result = await fut
        return {"workspace_id": workspace_id, "content": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/memory/search")
async def memory_search(
    request: MemorySearchRequest,
    workspace_id: str = Depends(get_workspace),
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """Search workspace memory."""
    try:
        # WHY(L7, v2-workspaces): without user_id + org_ids the retrieval
        # scope_filter sees only public+private — chunks shared via
        # visibility='user' or 'org' silently never surface for REST callers
        # (Laravel, hooks). MCP path already does this; REST was the gap.
        opts: dict[str, Any] = {
            "top_k": request.top_k,
            "workspace_id": workspace_id,
            "user_id": info.sub,
            "org_ids": info.org_ids,
        }
        if request.repo:
            opts["repo"] = request.repo
        if request.source_type:
            opts["source_type"] = request.source_type
        if request.task_context:
            opts["task_context"] = request.task_context
        if request.llm_prefilter is not None:
            opts["llm_prefilter"] = request.llm_prefilter
        if request.reranker_version:
            # Forward per-request override to _rerank() — used by /stats/
            # retrieval-ab to compare v1/v2 head-to-head on the same query.
            opts["reranker_version"] = request.reranker_version
        if request.category_hint:
            # WHY(2026-05-10): chunks die mit prompt-categories überlappen
            # bekommen einen score-boost im _rerank. Normalize zu lowercase.
            opts["category_hint"] = [
                c.lower().strip() for c in request.category_hint if c and c.strip()
            ]

        # WHY(2026-05-11, task-categorization): Mayring-konformer
        # task-derive aus dem query. Cascade: erst embedding-similarity
        # zu existierenden tasks, sonst mistral-call. Resultat wird zum
        # boost im _rerank für chunks die bei semantisch ähnlichen tasks
        # schon positiv bewertet wurden. Ist die saubere Ablösung der
        # alten tech-label-cluster ([neu]configuration etc).
        try:
            from src.memory.task_derivation import derive_task
            task = derive_task(
                request.query, _get_conn(), _OLLAMA_URL, workspace_id,
            )
            if task:
                opts["task_id"] = task["task_id"]
        except Exception as exc:
            # task-derive ist optional — fail soft (Logger im modul warnt).
            import logging
            logging.getLogger(__name__).warning("task-derive skipped: %s", exc)

        result = _run_search(request.query, _get_conn(), _get_chroma(), _OLLAMA_URL,
                             opts, request.char_budget)
        # Task-id im response damit feedback-calls sie zurück-senden können
        # (sonst geht der task-chunk-link verloren).
        if "task_id" in opts:
            result["task_id"] = opts["task_id"]
        return {"workspace_id": workspace_id, **result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _signature_for_event(ev: LogEvent) -> str:
    """Stable hash über logger + msg-skeleton (timestamps/IDs/UUIDs raus).
    Identische Logik-Errors aus verschiedenen Requests bekommen die
    gleiche signature → Dedup beim claude-error-triage trigger.
    """
    import re as _re
    skeleton = ev.msg or ""
    skeleton = _re.sub(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b",
                       "<UUID>", skeleton, flags=_re.I)
    skeleton = _re.sub(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\S*\b",
                       "<TS>", skeleton)
    skeleton = _re.sub(r"\b\d+\b", "<N>", skeleton)
    skeleton = _re.sub(r"\s+", " ", skeleton).strip()
    base = f"{ev.logger}|{ev.level}|{skeleton}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


@router.post("/memory/log-event")
async def memory_log_event(
    batch: LogEventBatch,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Batch-Ingest von App-Logger-Events.

    Pro Event:
      1. error_signature berechnen (falls nicht mitgesendet)
      2. ALS chunk in workspace ingesten (typisch 'bene:logs' via
         X-Workspace-Id-Header bei service-token caller)
      3. Wenn level ≥ ERROR und KEIN bestehender chunk in
         '<workspace>:analyses' mit dieser signature → repository_dispatch
         an claude-error-triage. Verhindert Doppel-Agent-Spawn für
         denselben Bug.
    """
    triggered = []
    skipped_known = []
    ingested = 0
    severe_levels = {"ERROR", "CRITICAL", "EXCEPTION", "FATAL"}
    conn = _get_conn()
    chroma = _get_chroma()
    for ev in batch.events:
        sig = ev.error_signature or _signature_for_event(ev)
        # 1. Ingest as memory chunk
        source_id = f"log:{batch.service}:{sig}:{ev.ts}"
        content = (
            f"[{ev.ts}] {ev.level} [{ev.logger}] {ev.msg}"
            + (f"\n{ev.exc}" if ev.exc else "")
        )
        try:
            _run_ingest(
                {"source_id": source_id, "source_type": "log_event",
                 "repo": "", "path": batch.service},
                content, conn, chroma, _OLLAMA_URL, _model("text"),
                {"categorize": False, "error_signature": sig},
                workspace_id,
            )
            ingested += 1
        except Exception as exc:
            _log.warning("log-event ingest failed: %s", exc)
            continue

        # 2. Trigger-decision für severe levels
        if ev.level.upper() not in severe_levels:
            continue

        # Memory-Injection-check: existiert eine Analyse für diese
        # signature? Wenn ja → kein neuer Agent.
        analyses_ws = f"{workspace_id}:analyses"
        try:
            sr = _run_search(
                f"error_signature:{sig}", analyses_ws, conn, chroma,
                _OLLAMA_URL, top_k=1,
            )
            if sr.get("results"):
                skipped_known.append(sig)
                continue
        except Exception:
            pass  # bei search-fail trotzdem triggern (besser doppelt
            # als gar nicht)
        triggered.append({"signature": sig, "level": ev.level,
                          "logger": ev.logger})

    # KEIN repository_dispatch mehr — Triage läuft via existierender
    # Claude-Code-Session (claude.ai/code) ODER Cloud-environments mit
    # Repo-Connect. Ingest-only bleibt so simple wie möglich. Die
    # `triggered`-Liste im Response zeigt dem Caller nur, welche
    # signatures ungesehen waren (Memory-Dashboard kann das nutzen
    # um neue ERROR-cards hervorzuheben).
    return {
        "workspace_id": workspace_id,
        "ingested": ingested,
        "triggered": len(triggered),
        "skipped_known_signatures": len(skipped_known),
        "new_signatures": [t["signature"] for t in triggered],
    }


@router.post("/memory/put")
async def memory_put(
    request: MemoryPutRequest,
    workspace_id: str = Depends(get_workspace),
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """Ingest content into workspace memory.

    WHY(L3, v2-workspaces): when caller asks for visibility='org' the source
    needs an org_id stamp — otherwise the chunk is invisible to org members
    (scope_filter requires s.org_id = caller_org). We resolve org_id from
    the JWT memberships: if the caller passed one, verify they're a member
    (else 403); if they didn't, default to their first org-membership.
    """
    try:
        source_dict: dict[str, Any] = {
            "source_id": request.source_id,
            "source_type": request.source_type,
            "repo": request.repo,
            "path": request.path,
        }
        if request.visibility:
            source_dict["visibility"] = request.visibility
        if request.visibility == "org":
            org_member_ids = set(info.org_ids)
            if request.org_id:
                if request.org_id not in org_member_ids:
                    raise HTTPException(
                        status_code=403,
                        detail=f"caller is not a member of org_id={request.org_id!r}",
                    )
                source_dict["org_id"] = request.org_id
            else:
                if not org_member_ids:
                    raise HTTPException(
                        status_code=400,
                        detail="visibility='org' requires org membership or explicit org_id",
                    )
                # First org-membership as default — log so silent picks
                # are visible in production grepping.
                first_org = next(iter(info.org_ids))
                _log.warning(
                    "memory.put visibility=org without org_id — "
                    "defaulting to first membership %r (caller=%s)",
                    first_org, info.sub,
                )
                source_dict["org_id"] = first_org
        elif request.org_id:
            # Honor explicit org_id even outside visibility=org (e.g. when
            # caller wants to stamp the source for later promotion).
            source_dict["org_id"] = request.org_id

        result = _run_ingest(source_dict, request.content, _get_conn(), _get_chroma(),
                             _OLLAMA_URL, _model("text"), {"categorize": request.categorize},
                             workspace_id)
        if source_dict.get("source_type") == "paper":
            _threading.Thread(
                target=_bg_wiki_rebuild,
                args=(workspace_id,),
                daemon=True,
            ).start()
        return {"workspace_id": workspace_id, **result}
    except HTTPException:
        # Re-raise as-is so the 403 (membership/auth) and 404 codes
        # survive — without this guard the broader Exception handler
        # below would mask them as 500. Not a no-op.
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/conversation/micro-batch")
async def conversation_micro_batch(
    request: ConversationMicroBatchRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Accept a batch of raw Claude turns from a remote conversation watcher,
    produce a summary on the server side (so the user doesn't need their own
    Ollama), and ingest as a ``conversation_summary`` source.

    This is the endpoint the per-user `tools/conversation_watcher.py` in
    RemoteHttpSink-Modus calls. Dedup: the source_id is deterministic
    (``conversation:<workspace_slug>:<session_id>``); when the same content
    is re-posted, ingest() detects it via content_hash and skips.
    """
    try:
        from tools.ingest_conversations import _summarize as _summarize_turns

        turns_dicts = [t.model_dump() for t in request.turns]
        if not turns_dicts:
            raise HTTPException(status_code=400, detail="turns must not be empty")

        # WHY(security, multi-tenant): workspace_slug aus dem Body ist
        # user-controlled. Code-review fand: ein User in workspace 'bene'
        # könnte body.workspace_slug='alice' setzen → source_id, repo,
        # path UND topic_transitions würden unter alice's Slug indexiert.
        # topic_transitions hat keine workspace_id-Spalte → cross-tenant
        # Markov-poisoning. Server bindet den Slug jetzt strikt an
        # workspace_id (JWT). Default-slug 'default' wird auch overwritten —
        # Watcher braucht keinen body-slug mehr.
        body_slug = (request.workspace_slug or "").strip().lower()
        # 'default' war der pydantic-default — den droppen wir silent.
        if body_slug and body_slug != "default" and body_slug != workspace_id.lower():
            raise HTTPException(
                status_code=403,
                detail=f"workspace_slug={body_slug!r} mismatch with "
                       f"authenticated workspace={workspace_id!r}",
            )
        # Ab hier: server-derived slug, nicht user-controlled
        slug = workspace_id

        first_ts = turns_dicts[0].get("timestamp", "")[:10]
        batch_key = f"{request.session_id}:{len(turns_dicts)}:{turns_dicts[-1].get('timestamp', '')}"
        content_hash = "sha256:" + hashlib.sha256(batch_key.encode()).hexdigest()[:16]
        source_id = f"conversation:{slug}:{request.session_id[:16]}"

        summary = (
            request.presumarized
            or _summarize_turns(turns_dicts, "", _OLLAMA_URL, _model("text"))
        )
        content = (
            f"# Session {first_ts or 'unbekannt'} | {slug}\n\n"
            f"{summary}\n"
        )
        source_dict = {
            "source_id": source_id,
            "source_type": "conversation_summary",
            "repo": slug,
            "path": f"{slug}/incremental",
            "branch": "local",
            "commit": "",
            "content_hash": content_hash,
        }
        result = _run_ingest(
            source_dict, content, _get_conn(), _get_chroma(),
            _OLLAMA_URL, _model("text"),
            {"categorize": True, "codebook": "social", "mode": "hybrid"},
            workspace_id,
        )

        # Predictive Memory v2 (Issue #55): bei jeder neuen
        # conversation_summary inkrementell die Markov-Transitions
        # bumpen, statt auf Cron + 100-chunk-rebuild zu warten.
        # Best-effort: ein Fehler hier darf den Ingest nicht failen.
        transitions_updated = 0
        try:
            from src.memory.predictive import update_transitions_incremental
            # Slug ist server-derived (siehe oben) — Markov-Transitions
            # landen im richtigen Bucket, kein cross-tenant-poisoning mehr.
            transitions_updated = update_transitions_incremental(
                content, _get_conn(), slug,
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "predictive incremental update failed: %s", exc,
            )
        return {
            "workspace_id": workspace_id,
            "source_id": source_id,
            "transitions_updated": transitions_updated,
            **result,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/memory/chunk/{chunk_id}")
async def memory_get_chunk(
    chunk_id: str,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    from src.memory.store import kv_get, get_chunk
    cached = kv_get(chunk_id)
    if cached is not None:
        return {"workspace_id": workspace_id, "chunk": cached}
    chunk = get_chunk(_get_conn(), chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail="chunk not found")
    return {"workspace_id": workspace_id, "chunk": chunk.to_dict()}


@router.post("/memory/invalidate")
async def memory_invalidate(
    request: MemoryInvalidateRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    from src.memory.store import deactivate_chunks_by_source, log_ingestion_event
    from src.memory.retrieval import invalidate_query_cache
    conn = _get_conn()
    count = deactivate_chunks_by_source(conn, request.source_id)
    log_ingestion_event(conn, request.source_id, "invalidated", {"count": count})
    invalidate_query_cache()
    return {"workspace_id": workspace_id, "source_id": request.source_id, "deactivated_count": count}


@router.get("/memory/chunks/{source_id}")
async def memory_list_by_source(
    source_id: str,
    active_only: bool = True,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    from src.memory.store import get_chunks_by_source
    chunks = get_chunks_by_source(_get_conn(), source_id, active_only=active_only)
    return {
        "workspace_id": workspace_id,
        "source_id": source_id,
        "count": len(chunks),
        "chunks": [c.to_dict() for c in chunks],
    }


@router.get("/memory/explain/{chunk_id}")
async def memory_explain(
    chunk_id: str,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    from src.memory.store import get_chunk, get_source
    from src.memory.schema import make_memory_key, source_fingerprint
    chunk = get_chunk(_get_conn(), chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail="chunk not found")
    cats = chunk.category_labels[0] if chunk.category_labels else "uncategorized"
    fp = source_fingerprint(chunk.source_id)
    hash_prefix = chunk.text_hash.replace("sha256:", "")[:8]
    memory_key = make_memory_key("repo", cats, fp, hash_prefix)
    source = get_source(_get_conn(), chunk.source_id)
    return {
        "workspace_id": workspace_id,
        "chunk_id": chunk_id,
        "memory_key": memory_key,
        "source_id": chunk.source_id,
        "category_labels": chunk.category_labels,
        "chunk_level": chunk.chunk_level,
        "ordinal": chunk.ordinal,
        "created_at": chunk.created_at,
        "is_active": chunk.is_active,
        "superseded_by": chunk.superseded_by,
        "quality_score": chunk.quality_score,
        "source": source.to_dict() if source else {},
    }


@router.post("/memory/reindex")
async def memory_reindex(
    request: MemoryReindexRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    try:
        from src.analysis.context import _embed_texts
        from src.memory.store import get_chunks_by_source, get_chunk
        from src.memory.retrieval import invalidate_query_cache

        chroma = _get_chroma()
        conn = _get_conn()

        if request.source_id:
            chunks = get_chunks_by_source(conn, request.source_id, active_only=True)
        else:
            rows = conn.execute(
                "SELECT chunk_id FROM chunks WHERE is_active = 1"
            ).fetchall()
            chunk_ids = [r[0] for r in rows]
            chunks = [c for cid in chunk_ids if (c := get_chunk(conn, cid)) is not None]

        reindexed = 0
        errors = 0

        for chunk in chunks:
            try:
                emb = _embed_texts([chunk.text[:500]], _OLLAMA_URL)[0]
                if chroma is not None:
                    _ws_row = conn.execute(
                        "SELECT workspace_id FROM chunks WHERE chunk_id = ?", (chunk.chunk_id,)
                    ).fetchone()
                    _ws_id = _ws_row[0] if _ws_row else "default"
                    chroma.upsert(
                        ids=[chunk.chunk_id],
                        documents=[chunk.text[:500]],
                        embeddings=[emb],
                        metadatas=[{
                            "workspace_id": _ws_id,
                            "source_id": chunk.source_id,
                            "chunk_level": chunk.chunk_level,
                            "category_labels": ",".join(chunk.category_labels),
                            "is_active": 1,
                        }],
                    )
                reindexed += 1
            except Exception:
                errors += 1

        invalidate_query_cache()
        return {"workspace_id": workspace_id, "reindexed_count": reindexed, "errors": errors}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/memory/feedback")
async def memory_feedback(
    request: MemoryFeedbackRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    # WHY(2026-05-10 rating-migration): only rating 1..5 accepted. Binary
    # positive/negative dominated reranker-v2-weights (issue #180) and
    # the auto-rater's path-match heuristik produced gift-signale für
    # generic files. Rating ist skalar — reranker bekommt echten gradient,
    # judge kann "wichtig aber nicht primär" ausdrücken statt "ja/nein".
    if request.signal not in ("1", "2", "3", "4", "5"):
        raise HTTPException(
            status_code=400,
            detail="signal must be a rating '1'..'5' — binary positive/negative no longer accepted"
        )
    from src.memory.store import add_feedback, get_chunks_by_source

    def _mark_referenced(chunk_ids: list[str]) -> None:
        """Bridge to context_feedback_log: when feedback comes in, the
        injection events that surfaced these chunks are flagged
        was_referenced=1. Without this hook the dashboard's Memory-
        Effizienz quote (referenced/injections) stays near zero — every
        positive/negative the user gives via REST was invisible to the
        injection-effectiveness counter, only the legacy MCP-tool path
        wired this up.
        """
        if not chunk_ids:
            return
        try:
            conn = _get_conn()
            for cid in chunk_ids:
                # Match on substring of the JSON-array trigger_ids — the
                # log stores `["chk_a", "chk_b", ...]`. LIKE %"<id>"%
                # is good enough; collisions are functionally harmless.
                conn.execute(
                    "UPDATE context_feedback_log SET was_referenced = 1 "
                    "WHERE was_referenced = 0 AND trigger_ids LIKE ?",
                    (f'%"{cid}"%',),
                )
            conn.commit()
        except Exception:
            pass  # never block feedback on log-bookkeeping

    # Slug-tolerance (Issue #138 — solution B): the SessionStart inject
    # surfaces source_id strings like "github-issue:mayringcoder:jwt-..." but
    # used to demand the opaque chunk_id for /feedback. Now we accept either:
    # if `chunk_id` looks like a source_id (no chk_ prefix, contains ':'),
    # we fan the signal out to every active chunk of that source.
    cid = request.chunk_id
    looks_like_source = ":" in cid and not cid.startswith("chk_")
    if looks_like_source:
        chunks = get_chunks_by_source(_get_conn(), cid, active_only=True)
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail=f"no active chunks found for source_id={cid!r}",
            )
        for ch in chunks:
            add_feedback(_get_conn(), ch.chunk_id, request.signal, request.metadata or {})
        # rating >= 4 flips referenced-flag (chunks die der user als
        # wichtig oder primärquelle bewertet hat — pendant zum alten
        # positive-signal für die Memory-Effizienz-quote).
        if int(request.signal) >= 4:
            _mark_referenced([ch.chunk_id for ch in chunks])
        _bust_stats()
        return {
            "workspace_id": workspace_id,
            "source_id": cid,
            "chunk_ids": [ch.chunk_id for ch in chunks],
            "applied_to": len(chunks),
            "recorded": True,
        }
    add_feedback(_get_conn(), cid, request.signal, request.metadata or {})
    if int(request.signal) >= 4:
        _mark_referenced([cid])
    _bust_stats()
    return {"workspace_id": workspace_id, "chunk_id": cid, "recorded": True}


def _bust_stats() -> None:
    """Lazy-import bust_stats_cache to avoid an import cycle (server↔routes)."""
    try:
        from src.api.server import bust_stats_cache
        bust_stats_cache()
    except Exception:
        pass  # cache-invalidation is best-effort; a stale stats page isn't fatal


@router.post("/search")
async def search_alias(
    request: MemorySearchRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Alias for /memory/search — used by Laravel MayringMcpClient."""
    return await memory_search(request, workspace_id)


@router.post("/ingest")
async def ingest_alias(
    request: MemoryPutRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Alias for /memory/put — used by Laravel MayringMcpClient."""
    return await memory_put(request, workspace_id)


@router.patch("/sources/{source_id}/visibility")
async def patch_source_visibility(
    source_id: str,
    request: PatchVisibilityRequest,
    workspace_id: str = Depends(get_workspace),
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """Update visibility (and optionally org_id) for a source.

    WHY(L8, v2-workspaces): without ownership-check this endpoint allowed
    cross-tenant vandalism — any authed caller could flip visibility on
    another user's sources. V2 enforces: owner (same workspace OR same sub)
    or admin (scope * / admin). 'user' added to the visibility whitelist.
    """
    if request.visibility not in ("private", "org", "public", "user"):
        raise HTTPException(
            status_code=400,
            detail="visibility must be private|org|public|user",
        )
    conn = _get_conn()
    row = conn.execute(
        "SELECT workspace_id, user_id FROM sources WHERE source_id = ?",
        (source_id,),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="source not found")
    src_ws = row["workspace_id"] if hasattr(row, "keys") else row[0]
    src_user = row["user_id"] if hasattr(row, "keys") else row[1]

    is_admin = "*" in info.scopes or "admin" in info.scopes
    is_owner = (
        src_ws == workspace_id
        or (src_user is not None and info.sub is not None and src_user == info.sub)
    )
    if not (is_admin or is_owner):
        raise HTTPException(
            status_code=403,
            detail="not authorized to change visibility of this source",
        )

    conn.execute(
        "UPDATE sources SET visibility = ?, org_id = ? WHERE source_id = ?",
        (request.visibility, request.org_id, source_id),
    )
    conn.commit()
    return {"source_id": source_id, "visibility": request.visibility, "org_id": request.org_id}
