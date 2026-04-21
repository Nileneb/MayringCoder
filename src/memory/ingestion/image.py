"""Image ingestion — vision captioning for single image files."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.memory.schema import Chunk, Source
from src.memory.store import (
    add_source_ref,
    insert_chunk,
    kv_put,
    log_ingestion_event,
    upsert_source,
)
from src.memory.ingestion.utils import now_iso


_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp"})


def _is_image_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in _IMAGE_EXTENSIONS


def ingest_image(
    source: Source,
    image_path: Path,
    conn: Any,
    chroma_collection: Any,
    ollama_url: str,
    model: str,
    vision_model: str = "qwen2.5vl:3b",
    workspace_id: str = "default",
) -> dict:
    """Ingest a single image file via vision captioning.

    SVGs are ingested as raw text. Raster images are captioned via the Ollama
    multimodal model (vision_model) and stored as text chunks with embeddings.

    Returns:
        {source_id, chunk_ids, indexed, deduped, superseded}
    """
    from src.agents.vision import caption_image, get_image_metadata
    from src.analysis.context import _embed_texts
    # Late import avoids circular dep core -> image -> core
    from src.memory.ingestion.core import resolve_dedup

    upsert_source(conn, source, workspace_id=workspace_id)
    log_ingestion_event(conn, source.source_id, "ingest_start", {"path": source.path})

    metadata = get_image_metadata(image_path) or {}

    caption = caption_image(image_path, ollama_url, vision_model)
    if not caption.strip():
        fmt = metadata.get("format", "")
        w = metadata.get("width", 0)
        h = metadata.get("height", 0)
        size = metadata.get("file_size", 0)
        caption = (
            f"Image file: {image_path.name}"
            + (f" ({fmt}, {w}x{h} px, {size} bytes)" if fmt else f" ({size} bytes)")
        )

    is_svg = Path(source.path).suffix.lower() == ".svg"
    category_labels = ["diagram"] if is_svg else ["image"]

    text_hash = Chunk.compute_text_hash(caption)
    chunk = Chunk(
        chunk_id=Chunk.make_id(source.source_id, 0, "image_caption"),
        source_id=source.source_id,
        chunk_level="image_caption",
        ordinal=0,
        start_offset=0,
        end_offset=len(caption),
        text=caption,
        text_hash=text_hash,
        dedup_key=text_hash,
        category_labels=category_labels,
        created_at=now_iso(),
    )

    canonical, is_dup = resolve_dedup(conn, chunk, workspace_id=workspace_id)
    new_chunk_ids: list[str] = []
    deduped_count = 0
    indexed = False

    if is_dup:
        deduped_count += 1
        add_source_ref(conn, canonical.chunk_id, source.source_id, workspace_id)
    else:
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

    return {
        "source_id": source.source_id,
        "chunk_ids": new_chunk_ids,
        "indexed": indexed,
        "deduped": deduped_count,
        "superseded": 0,
    }
