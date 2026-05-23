"""Remove redundant chunks/sources within ONE workspace (#workspace-uuid-sot).

After collapsing fake-user/system buckets into the canonical workspace, the same
content exists multiple times (same file ingested under different old buckets).
The user wants redundancy GONE ("entferne lieber redundante Chunks"). Two passes,
in this order (user choice "source_id, dann Content"):

1. **Source dedup** — sources sharing the same non-empty `content_hash` within the
   workspace: keep the newest (`captured_at`), delete the rest. Their chunks +
   chunk_source_refs go too (explicit delete + Chroma cleanup).
2. **Chunk dedup** — remaining chunks sharing the same non-empty `dedup_key`
   (= text_hash) within the workspace: keep the newest (`created_at`), delete rest.

Chroma docs are keyed by `chunk_id` (ingestion/core.py) — deleted chunks are
removed from the `memory_chunks` collection too. --dry-run default; --apply writes.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _chroma_delete(chunk_ids: list[str]) -> int:
    if not chunk_ids:
        return 0
    try:
        from mayring_core.memory.store import get_chroma_collection
        col = get_chroma_collection("memory_chunks")
        if col is None:
            return 0
        for i in range(0, len(chunk_ids), 500):
            col.delete(ids=chunk_ids[i:i + 500])
        return len(chunk_ids)
    except Exception as e:  # noqa: BLE001
        print(f"  chroma delete skipped: {e}", file=sys.stderr)
        return 0


def dedup(db_path: Path, workspace: str, apply: bool) -> dict:
    if not db_path.exists():
        return {"status": "MISSING"}
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout=20000")
    conn.execute("PRAGMA foreign_keys=ON")
    rep: dict = {}
    chroma_victims: list[str] = []
    try:
        # --- 1. source dedup by content_hash ---------------------------------
        src_groups = conn.execute(
            "SELECT content_hash, GROUP_CONCAT(source_id, '\x1f') "
            "FROM sources WHERE workspace_id = ? AND content_hash != '' "
            "GROUP BY content_hash HAVING count(*) > 1", (workspace,)
        ).fetchall()
        src_victims: list[str] = []
        for _h, ids_str in src_groups:
            ids = ids_str.split("\x1f")
            # keep newest captured_at
            order = conn.execute(
                f"SELECT source_id FROM sources WHERE source_id IN "
                f"({','.join('?' * len(ids))}) ORDER BY captured_at DESC", ids
            ).fetchall()
            src_victims.extend(r[0] for r in order[1:])  # all but newest
        rep["sources_removed"] = len(src_victims)
        if src_victims:
            ph = ",".join("?" * len(src_victims))
            chroma_victims += [r[0] for r in conn.execute(
                f"SELECT chunk_id FROM chunks WHERE source_id IN ({ph})", src_victims)]
            if apply:
                conn.execute(
                    f"DELETE FROM chunk_source_refs WHERE source_id IN ({ph})", src_victims)
                conn.execute(f"DELETE FROM chunks WHERE source_id IN ({ph})", src_victims)
                conn.execute(f"DELETE FROM sources WHERE source_id IN ({ph})", src_victims)

        # --- 2. chunk dedup by dedup_key (text_hash) -------------------------
        ch_groups = conn.execute(
            "SELECT dedup_key, GROUP_CONCAT(chunk_id, '\x1f') "
            "FROM chunks WHERE workspace_id = ? AND dedup_key != '' "
            "GROUP BY dedup_key HAVING count(*) > 1", (workspace,)
        ).fetchall()
        ch_victims: list[str] = []
        for _k, ids_str in ch_groups:
            ids = ids_str.split("\x1f")
            order = conn.execute(
                f"SELECT chunk_id FROM chunks WHERE chunk_id IN "
                f"({','.join('?' * len(ids))}) ORDER BY created_at DESC", ids
            ).fetchall()
            ch_victims.extend(r[0] for r in order[1:])
        rep["chunks_removed"] = len(ch_victims)
        if ch_victims:
            chroma_victims += ch_victims
            if apply:
                ph = ",".join("?" * len(ch_victims))
                # chunk_source_refs references the chunk via canonical_chunk_id
                # and has NO FK cascade → delete explicitly.
                conn.execute(
                    f"DELETE FROM chunk_source_refs WHERE canonical_chunk_id IN ({ph})",
                    ch_victims)
                conn.execute(f"DELETE FROM chunks WHERE chunk_id IN ({ph})", ch_victims)

        if apply:
            conn.commit()
        rep["chroma_to_delete"] = len(chroma_victims)
        rep["_chroma_victims"] = chroma_victims if apply else []
        return rep
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--db", default=str(ROOT / "cache" / "memory.db"))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--no-chroma", action="store_true")
    args = ap.parse_args()
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== dedup workspace {args.workspace}  [{mode}] ===")
    rep = dedup(Path(args.db), args.workspace, args.apply)
    if rep.get("status") == "MISSING":
        print("db MISSING"); return
    print(f"  redundant sources (by content_hash) → remove: {rep['sources_removed']}")
    print(f"  redundant chunks  (by dedup_key)     → remove: {rep['chunks_removed']}")
    print(f"  chroma docs to delete: {rep['chroma_to_delete']}")
    if args.apply and not args.no_chroma:
        n = _chroma_delete(rep.get("_chroma_victims", []))
        print(f"  chroma deleted: {n}")
    print("\n" + ("APPLIED." if args.apply else "DRY-RUN — re-run with --apply."))


if __name__ == "__main__":
    main()
