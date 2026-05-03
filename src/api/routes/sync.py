from __future__ import annotations

import logging
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.api.auth import get_workspace
from src.api.dependencies import get_chroma as _get_chroma, get_conn as _get_conn

router = APIRouter(prefix="/memory", tags=["sync"])
logger = logging.getLogger(__name__)


class ChunkSyncItem(BaseModel):
    chunk_id: str
    source_id: str
    text: str
    workspace_id: str
    created_at: str
    is_active: bool
    text_hash: str | None
    dedup_key: str | None
    embedding: list[float] | None


class MemorySyncResponse(BaseModel):
    cursor: str
    chunks: list[ChunkSyncItem]


def _ensure_visibility_column(db) -> None:
    """If the cloud DB was created before the visibility migration landed,
    `s.visibility` is NaN and the SELECT below fails with `no such column`.
    The migration is in src.memory.store but only fires when init_memory_db
    is called. Production was deployed before that migration was added, so
    the column may be missing on the live DB. Add it lazily here so a
    legacy DB still answers /memory/changes — idempotent: ALTER TABLE
    re-applied raises OperationalError which we swallow."""
    try:
        cols = {r[1] for r in db.execute("PRAGMA table_info(sources)").fetchall()}
    except sqlite3.Error:
        return
    if "visibility" not in cols:
        try:
            db.execute(
                "ALTER TABLE sources ADD COLUMN visibility "
                "TEXT NOT NULL DEFAULT 'private'"
            )
            logger.warning("sync: applied missing visibility column on sources")
        except sqlite3.Error as e:
            logger.error("sync: failed to add visibility column: %s", e)


@router.get("/changes", response_model=MemorySyncResponse)
def get_changes(
    since: str = Query(..., description="ISO 8601 cursor — only chunks created after this"),
    limit: int = Query(500, le=2000),
    workspace_id: str = Depends(get_workspace),
) -> MemorySyncResponse:
    db = _get_conn()
    _ensure_visibility_column(db)

    try:
        rows = db.execute(
            """
            SELECT c.chunk_id, c.source_id, c.text, c.workspace_id,
                   c.created_at, c.is_active, c.text_hash, c.dedup_key
            FROM chunks c
            JOIN sources s ON c.source_id = s.source_id
            WHERE c.created_at > ?
              AND (s.visibility = 'public'
                   OR (s.visibility = 'private' AND c.workspace_id = ?))
            ORDER BY c.created_at ASC
            LIMIT ?
            """,
            (since, workspace_id, limit),
        ).fetchall()
    except sqlite3.Error as e:
        # Don't 500 — that hits the client's UserPromptSubmit hook on every
        # keystroke. Surface the real DB error to the response so the client
        # log shows what's actually broken.
        logger.exception("sync: SQL query failed")
        raise HTTPException(status_code=503, detail=f"DB query failed: {e}") from e

    if not rows:
        return MemorySyncResponse(cursor=since, chunks=[])

    chunk_ids = [r[0] for r in rows]
    embedding_map: dict[str, list[float] | None] = {cid: None for cid in chunk_ids}
    try:
        col = _get_chroma()
        result = col.get(ids=chunk_ids, include=["embeddings"])
        ids = result.get("ids") or []
        embs = result.get("embeddings") or []
        for cid, emb in zip(ids, embs):
            if emb is not None:
                # Coerce numpy.ndarray (chromadb >=0.5) to plain list.
                embedding_map[cid] = list(emb)
    except Exception:
        # Non-fatal: chunks still flow back without embeddings, the client
        # can re-embed locally. Logged with stack so the cause is visible.
        logger.exception("sync: chroma fetch failed — continuing without embeddings")

    items = [
        ChunkSyncItem(
            chunk_id=r[0],
            source_id=r[1],
            text=r[2],
            workspace_id=r[3],
            created_at=r[4],
            is_active=bool(r[5]),
            text_hash=r[6],
            dedup_key=r[7],
            embedding=embedding_map.get(r[0]),
        )
        for r in rows
    ]
    new_cursor = items[-1].created_at if items else since
    return MemorySyncResponse(cursor=new_cursor, chunks=items)
