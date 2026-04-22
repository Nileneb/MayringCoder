"""Core ingestion orchestrator: chunking → dedup → embed → store → log."""
from __future__ import annotations

from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.model_router import ModelRouter

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **_kw):  # type: ignore[misc]
        return it

try:
    import chromadb as _chromadb  # noqa: F401
    _HAS_CHROMADB = True
except ImportError:
    _HAS_CHROMADB = False

from src.config import CACHE_DIR
from src.memory.chunker import structural_chunk
from src.memory.ingestion.categorization import (
    _INGEST_DEFAULTS,
    _INGEST_DEFAULT_FALLBACK,
    mayring_categorize,
)
from src.memory.ingestion.image import (
    _IMAGE_EXTENSIONS,
    _is_image_file,
    ingest_image,
)
from src.memory.ingestion.multiview import generate_multiview_chunks
from src.memory.ingestion.utils import log_memory_event, now_iso
from src.memory.schema import Chunk, Source
from src.memory.store import (
    add_source_ref,
    find_by_text_hash,
    get_source,
    insert_chunk,
    kv_put,
    log_ingestion_event,
    upsert_source,
)


MEMORY_CHROMA_DIR: Path = CACHE_DIR / "memory_chroma"


def resolve_dedup(
    conn: Any,
    chunk: Chunk,
    workspace_id: str = "default",
) -> tuple[Chunk, bool]:
    """Exact dedup via text_hash (workspace-scoped).

    Returns (existing_chunk, True) if duplicate found in same workspace.
    Returns (chunk, False) if no duplicate — caller should insert.
    """
    existing = find_by_text_hash(conn, chunk.text_hash, workspace_id=workspace_id)
    if existing is not None:
        return existing, True
    return chunk, False


def get_or_create_chroma_collection(chroma_dir: Path | None = None):
    """Get or create the 'memory_chunks' ChromaDB collection (process singleton)."""
    if not _HAS_CHROMADB:
        return None
    from src.memory.store import get_chroma_collection as get_collection
    return get_collection("memory_chunks", path=chroma_dir)


def ingest(
    source: Source,
    content: str,
    conn: Any,
    chroma_collection: Any,
    ollama_url: str,
    model: str,
    opts: dict | None = None,
    router: "ModelRouter | None" = None,
    workspace_id: str = "default",
) -> dict:
    """Orchestrate the full ingestion pipeline for one source.

    opts:
        categorize (bool, default True): run mayring_categorize()
        log        (bool, default False): write JSONL event
        codebook   (str):  codebook name (default "auto")
        mode       (str):  categorisation mode (default "hybrid")
        multiview  (bool): use view-chunking for github_issue

    Returns:
        {source_id, chunk_ids, indexed, deduped, superseded}
    """
    opts = opts or {}

    # Image fast-path — route known image extensions to ingest_image() when
    # a vision model is registered in the router. Keeps a single pipeline
    # with consistent chunk_level="image_caption".
    if _is_image_file(source.path):
        if source.source_type != "image":
            source = _dc_replace(source, source_type="image")
        if router is not None and router.is_available("vision"):
            try:
                return ingest_image(
                    source=source,
                    image_path=Path(source.path),
                    conn=conn,
                    chroma_collection=chroma_collection,
                    ollama_url=ollama_url,
                    model=model,
                    vision_model=router.resolve("vision"),
                    workspace_id=workspace_id,
                )
            except Exception:
                pass  # fall through to generic pipeline with source_type=image

    if router is not None and not model:
        _task = "mayring_code" if source.source_type == "repo_file" else "mayring_social"
        if router.is_available(_task):
            model = router.resolve(_task)

    defaults = _INGEST_DEFAULTS.get(source.source_type, _INGEST_DEFAULT_FALLBACK)
    effective = {**defaults, **opts}

    do_categorize: bool = bool(effective.get("categorize", True)) and bool(model)
    do_log:        bool = bool(effective.get("log", False))
    do_multiview:  bool = bool(effective.get("multiview", False))
    do_force:      bool = bool(effective.get("force", False))
    mode:          str  = effective.get("mode", "hybrid")
    codebook_choice: str = effective.get("codebook", "auto")

    from src.analysis.context import _embed_texts

    # Skip re-ingestion unless caller passes opts={"force": True} — that lifts
    # the cache completely (used by /populate?force_reingest=true so
    # re-runs actually re-chunk, re-categorize, re-embed).
    if source.content_hash and not do_force:
        existing_src = get_source(conn, source.source_id)
        if existing_src and existing_src.content_hash == source.content_hash:
            return {
                "source_id": source.source_id,
                "chunk_ids": [], "indexed": False,
                "deduped": 0, "superseded": 0, "skipped": True,
            }

    upsert_source(conn, source, workspace_id=workspace_id)
    log_ingestion_event(conn, source.source_id, "ingest_start", {"path": source.path})

    if do_multiview and source.source_type == "github_issue" and model:
        chunks = generate_multiview_chunks(source.source_id, content, ollama_url, model)
    else:
        chunks = structural_chunk(content, source.source_id, source.path)

    if do_categorize and model:
        chunks = mayring_categorize(
            chunks, ollama_url, model,
            mode=mode, codebook=codebook_choice,
            source_type=source.source_type,
            conn=conn,
            router=router,
        )

    new_chunk_ids: list[str] = []
    deduped_count = 0
    indexed = False

    for chunk in _tqdm(chunks, desc="Chunks embedden", unit="chunk", leave=False):
        canonical, is_dup = resolve_dedup(conn, chunk, workspace_id=workspace_id)
        if is_dup:
            deduped_count += 1
            add_source_ref(conn, canonical.chunk_id, source.source_id, workspace_id)
            continue

        insert_chunk(conn, chunk, workspace_id=workspace_id)
        add_source_ref(conn, chunk.chunk_id, source.source_id, workspace_id)

        try:
            emb = _embed_texts([chunk.text[:500]], ollama_url)[0]
        except Exception:
            emb = None

        if chroma_collection is not None and emb is not None:
            try:
                chroma_collection.upsert(
                    ids=[chunk.chunk_id],
                    documents=[chunk.text[:500]],
                    embeddings=[emb],
                    metadatas=[{
                        "workspace_id": workspace_id,
                        "source_id": chunk.source_id,
                        "chunk_level": chunk.chunk_level,
                        "category_labels": ",".join(chunk.category_labels),
                        "category_source": chunk.category_source,
                        "category_confidence": chunk.category_confidence,
                        "is_active": 1,
                    }],
                )
                indexed = True
            except Exception:
                pass

        kv_put(chunk.chunk_id, chunk.to_dict())
        new_chunk_ids.append(chunk.chunk_id)

    log_ingestion_event(
        conn,
        source.source_id,
        "ingest_done",
        {"chunks": len(new_chunk_ids), "deduped": deduped_count},
    )

    result = {
        "source_id": source.source_id,
        "chunk_ids": new_chunk_ids,
        "indexed": indexed,
        "deduped": deduped_count,
        "superseded": 0,
    }

    if do_log:
        log_memory_event({"event": "ingest", "ts": now_iso(), **result})

    return result


__all__ = [
    "MEMORY_CHROMA_DIR",
    "_HAS_CHROMADB",
    "_IMAGE_EXTENSIONS",
    "_is_image_file",
    "get_or_create_chroma_collection",
    "ingest",
    "resolve_dedup",
]
