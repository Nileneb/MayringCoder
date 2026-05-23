"""Embed all projects into the Chroma collection 'projects' (embedding_id=proj:<id>).

Idempotent upsert. Run with the chroma containers reachable. --dry-run default.
Mirrors tools/import_codebooks_to_db.py.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run(db_path: Path, apply: bool) -> dict:
    from mayring_core.ollama_client import embed_batch
    from mayring_core.memory.store import get_chroma_collection
    from src.api.routes.projects import project_embed_text

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, name, source_ref, source_type FROM projects").fetchall()
    rep = {"projects": len(rows), "embedded": 0}
    if not apply or not rows:
        return rep
    url = os.environ.get("OLLAMA_URL", "https://three.linn.games")
    model = os.environ.get("MAYRING_EMBED_MODEL", "nomic-embed-text")
    col = get_chroma_collection("projects")
    ids = [f"proj:{r[0]}" for r in rows]
    texts = [project_embed_text(r[1] or "", r[2] or "", r[3] or "") for r in rows]
    metas = [{"project_id": r[0]} for r in rows]
    for i in range(0, len(ids), 64):
        embs = embed_batch(url, model, texts[i:i + 64], timeout=120)
        if embs:
            col.upsert(ids=ids[i:i + 64], embeddings=embs,
                       documents=texts[i:i + 64], metadatas=metas[i:i + 64])
            rep["embedded"] += len(embs)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(ROOT / "cache" / "memory.db"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    rep = run(Path(args.db), args.apply)
    print(f"projects={rep['projects']} embedded={rep['embedded']} "
          f"({'APPLIED' if args.apply else 'DRY-RUN'})")


if __name__ == "__main__":
    main()
