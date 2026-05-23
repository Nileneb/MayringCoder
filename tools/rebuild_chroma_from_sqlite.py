"""Rebuild the Chroma `memory_chunks` collection from SQLite (source of truth).

WHY(#workspace-uuid-sot): the bulk workspace-migration (metadata updates + ~2564
dedup deletes) corrupted the Chroma HNSW index — `collection.count()` SEGFAULTs
(exit 139), every vector query returns `chroma_query_empty`, searches take 11s+,
hooks time out. SQLite (chunks.text + metadata) is intact, so we drop the broken
collection and re-embed every active chunk fresh.

Run with the chroma-using containers STOPPED (api/mcp/pi) to avoid concurrent-write
corruption. --dry-run default; --apply rebuilds. Idempotent (recreates from scratch).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EMBED_MODEL = os.environ.get("MAYRING_EMBED_MODEL", "nomic-embed-text")
BATCH = 64


def rebuild(db_path: Path, chroma_path: Path, ollama_url: str, apply: bool) -> dict:
    from mayring_core.ollama_client import embed_batch
    import chromadb

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT chunk_id, text, workspace_id, source_id, chunk_level, "
        "category_labels, category_source, category_confidence "
        "FROM chunks WHERE is_active=1 AND text != ''"
    ).fetchall()
    conn.close()
    print(f"active chunks to embed: {len(rows)}")
    if not apply:
        from collections import Counter
        dist = Counter(r[2] for r in rows)
        for ws, n in dist.most_common(12):
            print(f"   {n:6} {ws}")
        return {"would_embed": len(rows)}

    client = chromadb.PersistentClient(path=str(chroma_path))
    # Drop the corrupted collection (best-effort), then recreate fresh.
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
    print(f"re-embedded {done} chunks")
    return {"embedded": done, "count": col.count()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(ROOT / "cache" / "memory.db"))
    ap.add_argument("--chroma", default=str(ROOT / "cache" / "memory_chroma"))
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "https://three.linn.games"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== rebuild chroma memory_chunks from {args.db}  [{mode}] ===")
    print(f"chroma={args.chroma}  ollama={args.ollama_url}  model={EMBED_MODEL}")
    rep = rebuild(Path(args.db), Path(args.chroma), args.ollama_url, args.apply)
    print(rep)
    print("\n" + ("APPLIED — verify count()/search; restart api." if args.apply
                  else "DRY-RUN — re-run with --apply (containers stopped)."))


if __name__ == "__main__":
    main()
