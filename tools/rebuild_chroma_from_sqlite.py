"""Re-embed the Chroma `memory_chunks` collection from SQLite (source of truth).

Two modes:

1. **In-place recovery** (default `--collection memory_chunks`, no `--since`):
   drops + recreates the collection via an embedded PersistentClient. Use to
   repair a corrupted HNSW index. WHY(#workspace-uuid-sot): the bulk
   workspace-migration corrupted the index — `collection.count()` SEGFAULTed,
   every vector query returned `chroma_query_empty`. Run with the chroma-using
   containers STOPPED to avoid concurrent-write corruption.

2. **Blue-green migration** (`--collection memory_chunks_bge`, optionally
   `--since TIMESTAMP`): builds a NEW collection alongside the live one via the
   running Chroma *server* (server-aware `get_chroma_collection`, HttpClient when
   MAYRING_CHROMA_HOST is set) — NO drop, idempotent upsert, so reads keep
   serving the old collection during the build. Used for the bge-m3 cutover
   (768→1024-dim). `--since` does the catch-up pass for chunks created after the
   build started. Embedding model from MAYRING_EMBED_MODEL / EMBEDDING_MODEL.

--dry-run default; --apply writes.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EMBED_MODEL = os.environ.get(
    "MAYRING_EMBED_MODEL", os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
)
BATCH = 64


def rebuild(
    db_path: Path,
    chroma_path: Path,
    ollama_url: str,
    apply: bool,
    collection: str = "memory_chunks",
    since: str | None = None,
) -> dict:
    from mayring_core.ollama_client import embed_batch

    # Blue-green when targeting a non-default collection OR doing a catch-up pass.
    # In-place recovery only for the default name without a --since filter.
    blue_green = collection != "memory_chunks" or since is not None

    where = "is_active=1 AND text != ''"
    params: list = []
    if since is not None:
        where += " AND created_at >= ?"
        params.append(since)

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        f"SELECT chunk_id, text, workspace_id, source_id, chunk_level, "
        f"category_labels, category_source, category_confidence "
        f"FROM chunks WHERE {where}",
        params,
    ).fetchall()
    conn.close()
    print(f"active chunks to embed: {len(rows)}"
          + (f" (since {since})" if since else ""))
    if not apply:
        from collections import Counter
        dist = Counter(r[2] for r in rows)
        for ws, n in dist.most_common(12):
            print(f"   {n:6} {ws}")
        return {"would_embed": len(rows), "collection": collection,
                "mode": "blue-green" if blue_green else "in-place"}

    if blue_green:
        # Server-aware: writes into the LIVE Chroma server (no containers stopped,
        # no drop). The remap of "memory_chunks" only fires for that exact name,
        # so an explicit target like "memory_chunks_bge" is used verbatim.
        from mayring_core.memory.store import get_chroma_collection
        col = get_chroma_collection(collection, path=chroma_path)
        if col is None:
            return {"status": "chromadb_unavailable"}
        print(f"blue-green upsert into '{collection}' (no drop, server-aware)")
    else:
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_path))
        try:
            client.delete_collection("memory_chunks")
            print("deleted old collection")
        except Exception as e:  # noqa: BLE001
            print(f"delete_collection failed ({e}) — recreating anyway")
        col = client.get_or_create_collection("memory_chunks")

    done = 0
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        texts = [r[1][:2000] for r in batch]
        t = time.time()
        embs = embed_batch(ollama_url, EMBED_MODEL, texts, timeout=120)
        if not embs or len(embs) != len(batch):
            print(f"  embed batch {i} FAILED — abort (got {embs and len(embs)})")
            return {"embedded": done, "status": "embed_failed_at", "at": i}
        col.upsert(
            ids=[r[0] for r in batch],
            embeddings=embs,
            documents=texts,
            metadatas=[{
                "workspace_id": r[2] or "",
                "source_id": r[3] or "",
                "chunk_level": r[4] or "",
                "category_labels": r[5] or "",
                "category_source": r[6] or "",
                "category_confidence": float(r[7] or 0.0),
                "is_active": 1,
            } for r in batch],
        )
        done += len(batch)
        if i % (BATCH * 10) == 0:
            print(f"  {done}/{len(rows)} ({time.time()-t:.1f}s/batch)")
    print(f"re-embedded {done} chunks into '{collection}'")
    return {"embedded": done, "count": col.count(), "collection": collection}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(ROOT / "cache" / "memory.db"))
    ap.add_argument("--chroma", default=str(ROOT / "cache" / "memory_chroma"))
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--collection", default="memory_chunks",
                    help="target collection (use memory_chunks_bge for blue-green)")
    ap.add_argument("--since", default=None,
                    help="catch-up: only chunks with created_at >= TIMESTAMP (ISO)")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== re-embed chroma '{args.collection}' from {args.db}  [{mode}] ===")
    print(f"chroma={args.chroma}  ollama={args.ollama_url}  model={EMBED_MODEL}")
    rep = rebuild(Path(args.db), Path(args.chroma), args.ollama_url, args.apply,
                  collection=args.collection, since=args.since)
    print(rep)
    blue_green = args.collection != "memory_chunks" or args.since is not None
    if args.apply:
        print("\nAPPLIED — verify count()/search."
              + (" Flip MEMORY_CHUNKS_COLLECTION + EMBEDDING_MODEL to switch live."
                 if blue_green else " Restart api."))
    else:
        print("\nDRY-RUN — re-run with --apply."
              + ("" if blue_green else " (containers stopped for in-place)"))


if __name__ == "__main__":
    main()
