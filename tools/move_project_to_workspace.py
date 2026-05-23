"""Move a project's chunks/sources to another workspace (#workspace-uuid-sot).

Use case: the `logs` "project" is actually mayring-api INFRA logs (the old
`bene:logs` log-ingest target) — genuine system logs, not user content. They
must live in `system` (Filament-Admin-Sicht), not pollute the personal workspace
+ memory search. This moves chunks/sources WHERE project_id=<from-project> (in
<from-workspace>) → workspace_id=<to-workspace>, project_id=NULL, incl.
chunk_source_refs + Chroma metadata. --dry-run default, idempotent.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def move(db_path: Path, from_ws: str, from_project: str, to_ws: str,
         apply: bool) -> dict:
    if not db_path.exists():
        return {"status": "MISSING"}
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout=20000")
    rep: dict = {}
    try:
        moved_chunk_ids = [r[0] for r in conn.execute(
            "SELECT chunk_id FROM chunks WHERE workspace_id=? AND project_id=?",
            (from_ws, from_project))]
        moved_source_ids = [r[0] for r in conn.execute(
            "SELECT source_id FROM sources WHERE workspace_id=? AND project_id=?",
            (from_ws, from_project))]
        rep = {"chunks": len(moved_chunk_ids), "sources": len(moved_source_ids)}
        if apply and (moved_chunk_ids or moved_source_ids):
            conn.execute(
                "UPDATE chunks SET workspace_id=?, project_id=NULL "
                "WHERE workspace_id=? AND project_id=?", (to_ws, from_ws, from_project))
            conn.execute(
                "UPDATE sources SET workspace_id=?, project_id=NULL "
                "WHERE workspace_id=? AND project_id=?", (to_ws, from_ws, from_project))
            # chunk_source_refs has no project_id — move by the moved sources.
            if moved_source_ids:
                ph = ",".join("?" * len(moved_source_ids))
                conn.execute(
                    f"UPDATE chunk_source_refs SET workspace_id=? "
                    f"WHERE workspace_id=? AND source_id IN ({ph})",
                    (to_ws, from_ws, *moved_source_ids))
            conn.commit()
        rep["_chunk_ids"] = moved_chunk_ids if apply else []
        return rep
    finally:
        conn.close()


def move_chroma(chunk_ids: list[str], to_ws: str, apply: bool) -> dict:
    if not chunk_ids:
        return {"relabeled": 0}
    try:
        from mayring_core.memory.store import get_chroma_collection
        col = get_chroma_collection("memory_chunks")
        if col is None:
            return {"status": "skipped"}
    except Exception as e:  # noqa: BLE001
        return {"status": f"skipped: {e}"}
    n = 0
    for i in range(0, len(chunk_ids), 500):
        batch = chunk_ids[i:i + 500]
        got = col.get(ids=batch, include=["metadatas"])
        ids = got.get("ids", []) or []
        metas = got.get("metadatas", []) or []
        new_metas = []
        for m in metas:
            mm = dict(m or {})
            mm["workspace_id"] = to_ws
            mm.pop("project_id", None)
            new_metas.append(mm)
        if ids and apply:
            col.update(ids=ids, metadatas=new_metas)
        n += len(ids)
    return {"relabeled": n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-workspace", required=True)
    ap.add_argument("--from-project", required=True)
    ap.add_argument("--to-workspace", required=True)
    ap.add_argument("--db", default=str(ROOT / "cache" / "memory.db"))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--no-chroma", action="store_true")
    args = ap.parse_args()
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== move project '{args.from_project}' ({args.from_workspace}) "
          f"→ workspace '{args.to_workspace}'  [{mode}] ===")
    rep = move(Path(args.db), args.from_workspace, args.from_project,
               args.to_workspace, args.apply)
    if rep.get("status") == "MISSING":
        print("db MISSING"); return
    print(f"  chunks: {rep['chunks']}  sources: {rep['sources']}")
    if not args.no_chroma:
        print(f"  chroma: {move_chroma(rep.get('_chunk_ids', []), args.to_workspace, args.apply)}")
    print("\n" + ("APPLIED." if args.apply else "DRY-RUN — re-run with --apply."))


if __name__ == "__main__":
    main()
