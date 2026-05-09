#!/usr/bin/env python3
"""Pre-launch-Cleanup: Lösche alle Daten in user-N + default Buckets.

Hintergrund: Workspace-IDs sind seit 2026-05-09 email-Slugs (z.B. 'bene'
für bene@linn.games), nicht mehr 'user-2' oder 'default'. Pre-launch-
Beta = keine echten User-Daten unter den Legacy-Buckets — daher hard
DELETE statt Migration.

Affected tables (memory.db):
    chunks, sources, workspaces, workspace_aliases,
    context_feedback_log, chunk_source_refs, llm_calls_log,
    ingestion_log (wenn workspace-tagged)

Affected ChromaDB collections: memory_chunks (where metadata.workspace_id
matches a deleted chunk).

Usage:
    python tools/purge_legacy_workspaces.py            # dry-run
    python tools/purge_legacy_workspaces.py --execute  # echte Löschung
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project-root is on sys.path so 'src' imports work when run directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CACHE_DIR
from src.memory.store import init_memory_db


LEGACY_PREDICATE_WORKSPACE_ID = (
    "workspace_id LIKE 'user-%' OR workspace_id = 'default'"
)
LEGACY_PREDICATE_ID = "id LIKE 'user-%' OR id = 'default'"

# Tabellen mit workspace_id-Spalte
WS_TABLES = [
    "chunks",
    "sources",
    "context_feedback_log",
    "chunk_source_refs",
    "llm_calls_log",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--execute", action="store_true",
                        help="Führe DELETEs aus (default: dry-run)")
    args = parser.parse_args()

    conn = init_memory_db(CACHE_DIR / "memory.db")
    print(f"# DB: {CACHE_DIR / 'memory.db'}")
    print(f"# Mode: {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print()

    counts: dict[str, int] = {}

    # 1. Tabellen mit workspace_id-Spalte
    for table in WS_TABLES:
        try:
            row = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {LEGACY_PREDICATE_WORKSPACE_ID}"
            ).fetchone()
            counts[table] = int(row[0]) if row else 0
        except Exception as e:
            counts[table] = -1
            print(f"  {table}: query-error ({e})")

    # 2. workspaces table mit id-Spalte
    row = conn.execute(
        f"SELECT COUNT(*) FROM workspaces WHERE {LEGACY_PREDICATE_ID}"
    ).fetchone()
    counts["workspaces"] = int(row[0]) if row else 0

    # 3. workspace_aliases die auf user-N zeigen
    row = conn.execute(
        "SELECT COUNT(*) FROM workspace_aliases "
        "WHERE alias LIKE 'user-%' OR alias = 'default' "
        "OR workspace_id LIKE 'user-%' OR workspace_id = 'default'"
    ).fetchone()
    counts["workspace_aliases"] = int(row[0]) if row else 0

    print("# Legacy-Rows (user-* + default):")
    for table, cnt in counts.items():
        marker = "✓" if cnt == 0 else "→"
        print(f"  {marker} {table:25s} {cnt:>6d}")
    total = sum(c for c in counts.values() if c > 0)
    print(f"  {'─' * 35}")
    print(f"    TOTAL                    {total:>6d}")

    if not args.execute:
        print("\n# Dry-run only. Re-run with --execute to delete.")
        return 0

    if total == 0:
        print("\n# Nichts zu löschen.")
        return 0

    print("\n# Executing DELETEs…")
    for table in WS_TABLES:
        if counts.get(table, 0) > 0:
            n = conn.execute(
                f"DELETE FROM {table} WHERE {LEGACY_PREDICATE_WORKSPACE_ID}"
            ).rowcount
            print(f"  {table}: {n} rows deleted")

    if counts["workspaces"] > 0:
        n = conn.execute(
            f"DELETE FROM workspaces WHERE {LEGACY_PREDICATE_ID}"
        ).rowcount
        print(f"  workspaces: {n} rows deleted")

    if counts["workspace_aliases"] > 0:
        n = conn.execute(
            "DELETE FROM workspace_aliases "
            "WHERE alias LIKE 'user-%' OR alias = 'default' "
            "OR workspace_id LIKE 'user-%' OR workspace_id = 'default'"
        ).rowcount
        print(f"  workspace_aliases: {n} rows deleted")

    conn.commit()

    # ChromaDB cleanup — separate Collection
    try:
        from src.memory.store import get_chroma_collection
        col = get_chroma_collection("memory_chunks")
        if col is not None:
            for prefix in ("user-",):
                # Where-Query auf metadata.workspace_id mit prefix nicht
                # nativ unterstützt → list + filter
                pass
            # Pragmatisch: alle chunks deren metadata.workspace_id matched
            # via Python-Loop (kleine DBs ok).
            all_meta = col.get(include=["metadatas"]).get("metadatas") or []
            all_ids = col.get().get("ids") or []
            to_drop = [
                cid for cid, meta in zip(all_ids, all_meta)
                if isinstance(meta, dict) and (
                    str(meta.get("workspace_id", "")).startswith("user-")
                    or meta.get("workspace_id") == "default"
                )
            ]
            if to_drop:
                col.delete(ids=to_drop)
                print(f"  chroma:memory_chunks: {len(to_drop)} chunks deleted")
            else:
                print("  chroma:memory_chunks: keine legacy-chunks")
    except Exception as e:
        print(f"  chroma cleanup failed: {e}")

    print("\n# Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
