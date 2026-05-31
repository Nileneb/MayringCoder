"""One-time, idempotent backfill: stamp visibility/org_id/user_id into the Chroma
metadata of every existing chunk (Phase A). Reads the authoritative values from
SQLite `sources`. Safe to re-run."""
from __future__ import annotations

from typing import Any


def backfill_visibility_metadata(conn: Any, collection: Any, batch: int = 500) -> int:
    rows = conn.execute(
        "SELECT c.chunk_id, s.visibility, s.org_id, s.user_id "
        "FROM chunks c JOIN sources s ON c.source_id = s.source_id"
    ).fetchall()
    by_id = {}
    for r in rows:
        cid = r["chunk_id"] if hasattr(r, "keys") else r[0]
        by_id[cid] = {
            "visibility": (r["visibility"] if hasattr(r, "keys") else r[1]) or "private",
            "org_id": (r["org_id"] if hasattr(r, "keys") else r[2]) or "",
            "user_id": (r["user_id"] if hasattr(r, "keys") else r[3]) or "",
        }
    if not by_id:
        return 0
    ids = list(by_id.keys())
    updated = 0
    for i in range(0, len(ids), batch):
        chunk_ids = ids[i:i + batch]
        collection.update(ids=chunk_ids, metadatas=[by_id[c] for c in chunk_ids])
        updated += len(chunk_ids)
    return updated
