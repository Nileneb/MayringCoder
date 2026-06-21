"""One-time, idempotent backfill: stamp repo + source_class into the Chroma
metadata of every existing chunk. Reads the authoritative values from SQLite
`sources`. Safe to re-run.

WHY(repo-scoping-hardfilter + reference-doc-layer, 2026-06-21): build_chroma_where
now hard-filters vector candidates by `repo` and default-excludes
source_class='reference'. Chunks ingested before this layer lack those metadata
keys, so the Chroma `$where` would drop them from the vector top-K until this
backfill stamps them from SQLite `sources`. Correctness never depended on it
(the SQL _scope_filter is the boundary) — this restores the vector ranking.
"""
from __future__ import annotations

from typing import Any


def backfill_repo_source_class_metadata(conn: Any, collection: Any, batch: int = 500) -> int:
    rows = conn.execute(
        "SELECT c.chunk_id, s.repo, s.source_class "
        "FROM chunks c JOIN sources s ON c.source_id = s.source_id"
    ).fetchall()
    by_id = {}
    for r in rows:
        cid = r["chunk_id"] if hasattr(r, "keys") else r[0]
        by_id[cid] = {
            "repo": (r["repo"] if hasattr(r, "keys") else r[1]) or "",
            "source_class": (r["source_class"] if hasattr(r, "keys") else r[2]) or "code",
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
