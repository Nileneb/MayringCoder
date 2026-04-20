"""One-shot migration: relabel all existing workspace_ids → 'bene-workspace'.

Touches:
    SQLite  (cache/memory.db):  chunks, sources, chunk_source_refs
    ChromaDB (cache/memory_chroma): metadata.workspace_id on all docs

Idempotent: running twice is a no-op.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CACHE_DIR
from src.memory.store import get_chroma_collection

TARGET_WS = "bene-workspace"
DB_PATH = CACHE_DIR / "memory.db"


def migrate_sqlite() -> dict[str, int]:
    touched = {}
    conn = sqlite3.connect(DB_PATH)
    try:
        for table in ("chunks", "sources", "chunk_source_refs"):
            cur = conn.execute(
                f"UPDATE {table} SET workspace_id = ? WHERE workspace_id != ?",
                (TARGET_WS, TARGET_WS),
            )
            touched[table] = cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return touched


def migrate_chroma() -> dict[str, int]:
    collection = get_chroma_collection("memory_chunks")
    if collection is None:
        return {"skipped": 1, "reason": "chromadb not installed"}

    got = collection.get(include=["metadatas"])
    ids = got.get("ids", [])
    metas = got.get("metadatas", []) or []

    if not ids:
        return {"relabeled": 0, "total": 0}

    BATCH = 500
    relabeled = 0
    for i in range(0, len(ids), BATCH):
        batch_ids = ids[i : i + BATCH]
        batch_metas = metas[i : i + BATCH]
        new_metas = []
        dirty_ids = []
        for _id, _m in zip(batch_ids, batch_metas):
            m = dict(_m or {})
            if m.get("workspace_id") != TARGET_WS:
                m["workspace_id"] = TARGET_WS
                new_metas.append(m)
                dirty_ids.append(_id)
        if dirty_ids:
            collection.update(ids=dirty_ids, metadatas=new_metas)
            relabeled += len(dirty_ids)
    return {"relabeled": relabeled, "total": len(ids)}


def main() -> None:
    print(f"Target workspace_id = {TARGET_WS!r}")
    print(f"SQLite  → {DB_PATH}")
    sqlite_result = migrate_sqlite()
    print(f"  {sqlite_result}")
    print("ChromaDB → memory_chunks")
    chroma_result = migrate_chroma()
    print(f"  {chroma_result}")
    print("done.")


if __name__ == "__main__":
    main()
