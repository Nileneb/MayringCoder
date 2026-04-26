from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from src.api.auth import get_workspace
from src.api.dependencies import get_chroma as _get_chroma, get_conn as _get_conn

router = APIRouter(prefix="/memory", tags=["sync"])


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


@router.get("/changes", response_model=MemorySyncResponse)
def get_changes(
    since: str = Query(..., description="ISO 8601 cursor — only chunks created after this"),
    workspace_id: str = Query(...),
    limit: int = Query(500, le=2000),
    _ws: str = Depends(get_workspace),
) -> MemorySyncResponse:
    db = _get_conn()
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

    if not rows:
        return MemorySyncResponse(cursor=since, chunks=[])

    chunk_ids = [r[0] for r in rows]
    embedding_map: dict[str, list[float] | None] = {cid: None for cid in chunk_ids}
    try:
        col = _get_chroma()
        result = col.get(ids=chunk_ids, include=["embeddings"])
        for cid, emb in zip(result["ids"], result["embeddings"]):
            if emb is not None:
                embedding_map[cid] = list(emb)
    except Exception:
        pass

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
