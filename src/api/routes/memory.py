from __future__ import annotations

import hashlib
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.api.auth import get_workspace
from src.api.dependencies import get_chroma as _get_chroma, get_conn as _get_conn
from src.api.memory_service import run_ingest as _run_ingest, run_search as _run_search
from src.api.routes.models import (
    ConversationMicroBatchRequest,
    MemoryFeedbackRequest,
    MemoryInvalidateRequest,
    MemoryPutRequest,
    MemoryReindexRequest,
    MemorySearchRequest,
    PiTaskRequest,
)

router = APIRouter(tags=["memory"])

_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "")


@router.post("/pi-task")
async def pi_task(
    request: PiTaskRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Run a task via the Pi-agent (memory-augmented reasoning)."""
    import asyncio
    from src.agents.pi import run_task_with_memory
    _repo_slug = request.repo_slug or os.getenv("PI_REPO_SLUG", "")
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_task_with_memory(
                task=request.task,
                ollama_url=_OLLAMA_URL,
                model=_OLLAMA_MODEL,
                repo_slug=_repo_slug,
                system_prompt=request.system_prompt,
                timeout=request.timeout,
            ),
        )
        return {"workspace_id": workspace_id, "content": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/memory/search")
async def memory_search(
    request: MemorySearchRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Search workspace memory."""
    try:
        opts: dict[str, Any] = {"top_k": request.top_k, "workspace_id": workspace_id}
        if request.repo:
            opts["repo"] = request.repo
        if request.source_type:
            opts["source_type"] = request.source_type
        result = _run_search(request.query, _get_conn(), _get_chroma(), _OLLAMA_URL,
                             opts, request.char_budget)
        return {"workspace_id": workspace_id, **result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/memory/put")
async def memory_put(
    request: MemoryPutRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Ingest content into workspace memory."""
    try:
        source_dict = {"source_id": request.source_id, "source_type": request.source_type,
                       "repo": request.repo, "path": request.path}
        result = _run_ingest(source_dict, request.content, _get_conn(), _get_chroma(),
                             _OLLAMA_URL, _OLLAMA_MODEL, {"categorize": request.categorize},
                             workspace_id)
        return {"workspace_id": workspace_id, **result}
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

        first_ts = turns_dicts[0].get("timestamp", "")[:10]
        batch_key = f"{request.session_id}:{len(turns_dicts)}:{turns_dicts[-1].get('timestamp', '')}"
        content_hash = "sha256:" + hashlib.sha256(batch_key.encode()).hexdigest()[:16]
        source_id = f"conversation:{request.workspace_slug}:{request.session_id[:16]}"

        summary = (
            request.presumarized
            or _summarize_turns(turns_dicts, "", _OLLAMA_URL, _OLLAMA_MODEL)
        )
        content = (
            f"# Session {first_ts or 'unbekannt'} | {request.workspace_slug}\n\n"
            f"{summary}\n"
        )
        source_dict = {
            "source_id": source_id,
            "source_type": "conversation_summary",
            "repo": request.workspace_slug,
            "path": f"{request.workspace_slug}/incremental",
            "branch": "local",
            "commit": "",
            "content_hash": content_hash,
        }
        result = _run_ingest(
            source_dict, content, _get_conn(), _get_chroma(),
            _OLLAMA_URL, _OLLAMA_MODEL,
            {"categorize": True, "codebook": "social", "mode": "hybrid"},
            workspace_id,
        )
        return {"workspace_id": workspace_id, "source_id": source_id, **result}
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
    if request.signal not in ("positive", "negative", "neutral"):
        raise HTTPException(status_code=400, detail="signal must be positive|negative|neutral")
    from src.memory.store import add_feedback
    add_feedback(_get_conn(), request.chunk_id, request.signal, request.metadata or {})
    return {"workspace_id": workspace_id, "chunk_id": request.chunk_id, "recorded": True}


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
