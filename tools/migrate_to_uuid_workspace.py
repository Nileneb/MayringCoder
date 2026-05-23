"""One-shot, idempotent re-key: collapse all user workspace-slugs → ONE canonical
app.linn.games workspace UUID (V2 source-of-truth identity switch).

WHY(#workspace-uuid-sot): app.linn.games signs the authoritative workspace UUID
into the JWT (`workspace_id` claim). MayringCoder used to derive an email-slug
("bene") instead and stored data under it. After jwt_auth.py trusts the UUID, all
historical data keyed by slugs ("bene", "bene-workspace", "default", "bene:*", …)
must move to that UUID or it becomes invisible to search. The user confirmed ALL
existing memory belongs to ONE UUID, so this is a flat re-key. `system` (and
`public`) are protected — service/maintenance + public-share buckets keep their id.

Touches every `workspace_id`-bearing table in memory.db + wiki_v2.db (discovered
dynamically via PRAGMA), the Chroma `memory_chunks` collection metadata, and
registers old slugs as `workspace_aliases → uuid`.

Safety: --dry-run is the DEFAULT (prints the exact plan + per-table counts, writes
nothing). Pass --apply to write. PK/UNIQUE tables (devices, wiki_nodes,
wiki_clusters) use UPDATE OR IGNORE + leftover-delete to avoid composite-key
collisions. Idempotent: re-running after success is a no-op (everything already ==
target). Reuses the Chroma batch pattern from tools/migrate_workspace_to_bene.py.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# WHY(#workspace-uuid-sot): the prod data is NOT one bucket. Past buggy identity
# created fake-USER buckets (email-slug "bene", "user-2", an extra UUID) instead
# of doing PROJECT separation on the workspace axis. Project/repo buckets
# (dronedetect, battlefield, bene:logs, …) and "system" are LEGITIMATE and MUST be
# preserved. So this migration folds ONLY an explicit --merge set (the fake-user /
# personal slugs) into the canonical personal UUID; everything else is untouched.


def _tables_with_workspace_col(conn: sqlite3.Connection) -> dict[str, bool]:
    """{table_name: workspace_id_is_part_of_pk} for every table carrying the column."""
    out: dict[str, bool] = {}
    tabs = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    for t in tabs:
        info = list(conn.execute(f"PRAGMA table_info({t})"))
        cols = {r[1] for r in info}
        if "workspace_id" not in cols:
            continue
        ws_in_pk = any(r[1] == "workspace_id" and r[5] > 0 for r in info)
        # Also detect workspace_id inside a UNIQUE index (not just the PK) —
        # e.g. wiki_edges UNIQUE(source,target,type,workspace_id). Re-keying such
        # a table with a plain UPDATE blows up on collisions; these need the
        # OR IGNORE + leftover-delete path (which also dedups, as the user wants).
        ws_collidable = ws_in_pk
        if not ws_collidable:
            for idx in conn.execute(f"PRAGMA index_list({t})"):
                if idx[2]:  # unique index
                    idx_cols = [r[2] for r in conn.execute(
                        f"PRAGMA index_info({idx[1]})")]
                    if "workspace_id" in idx_cols:
                        ws_collidable = True
                        break
        out[t] = ws_collidable
    return out


def _distribution(conn: sqlite3.Connection, table: str) -> list[tuple[str, int]]:
    return list(conn.execute(
        f"SELECT workspace_id, count(*) FROM {table} GROUP BY 1 ORDER BY 2 DESC"))


def migrate_sqlite(db_path: Path, target: str, merge_set: tuple[str, ...],
                   apply: bool) -> dict:
    if not db_path.exists():
        return {"db": str(db_path), "status": "MISSING"}
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout=20000")
    report: dict = {"db": str(db_path), "tables": {}}
    try:
        tables = _tables_with_workspace_col(conn)
        # Move ONLY the explicit merge_set (fake-user/personal slugs) → target.
        # Everything else (projects, system, the target itself) is untouched.
        placeholders = ",".join("?" for _ in merge_set)
        merge_clause = f"workspace_id IN ({placeholders}) AND workspace_id != ?"
        for table, ws_in_pk in tables.items():
            before = _distribution(conn, table)
            to_move = [(ws, n) for ws, n in before
                       if ws in merge_set and ws != target]
            moved = sum(n for _, n in to_move)
            report["tables"][table] = {
                "ws_in_pk": ws_in_pk,
                "move_rows": moved,
                "from": [ws for ws, _ in to_move],
            }
            if apply and moved:
                params = (target, *merge_set, target)  # SET + WHERE
                where_params = (*merge_set, target)    # WHERE only
                if ws_in_pk:
                    # Composite PK/UNIQUE → OR IGNORE avoids collisions with an
                    # already-migrated (key, target) row; sweep skipped leftovers.
                    conn.execute(
                        f"UPDATE OR IGNORE {table} SET workspace_id=? "
                        f"WHERE {merge_clause}", params)
                    conn.execute(
                        f"DELETE FROM {table} WHERE {merge_clause}", where_params)
                else:
                    conn.execute(
                        f"UPDATE {table} SET workspace_id=? "
                        f"WHERE {merge_clause}", params)

        # --- workspaces registry + aliases -----------------------------------
        if "workspaces" in {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}:
            slugs = [r[0] for r in conn.execute(
                f"SELECT id FROM workspaces WHERE id IN ({placeholders}) "
                "AND id != ?", (*merge_set, target))]
            report["workspaces_slugs_to_alias"] = slugs
            if apply:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "INSERT OR IGNORE INTO workspaces(id, kind, display_name, "
                    "created_at, updated_at) VALUES (?, 'user', ?, ?, ?)",
                    (target, "Personal (app.linn.games)", now, now))
                for slug in slugs:
                    conn.execute(
                        "INSERT OR IGNORE INTO workspace_aliases(alias, "
                        "workspace_id, created_at) VALUES (?, ?, ?)",
                        (slug, target, now))
                    # Old slug workspace row no longer the identity → drop it
                    # (its alias now points at the UUID). Done AFTER alias insert.
                    conn.execute("DELETE FROM workspaces WHERE id=?", (slug,))
        if apply:
            conn.commit()
        return report
    finally:
        conn.close()


def migrate_chroma(target: str, merge_set: tuple[str, ...], apply: bool) -> dict:
    try:
        from mayring_core.memory.store import get_chroma_collection
    except Exception as e:  # noqa: BLE001
        return {"status": "skipped", "reason": f"import failed: {e}"}
    collection = get_chroma_collection("memory_chunks")
    if collection is None:
        return {"status": "skipped", "reason": "chromadb not installed"}
    got = collection.get(include=["metadatas"])
    ids = got.get("ids", []) or []
    metas = got.get("metadatas", []) or []
    if not ids:
        return {"relabeled": 0, "total": 0}
    BATCH = 500
    relabeled = 0
    for i in range(0, len(ids), BATCH):
        b_ids = ids[i:i + BATCH]
        b_metas = metas[i:i + BATCH]
        dirty_ids, new_metas = [], []
        for _id, _m in zip(b_ids, b_metas):
            m = dict(_m or {})
            ws = m.get("workspace_id")
            if ws in merge_set and ws != target:
                m["workspace_id"] = target
                dirty_ids.append(_id)
                new_metas.append(m)
        if dirty_ids and apply:
            collection.update(ids=dirty_ids, metadatas=new_metas)
        relabeled += len(dirty_ids)
    return {"relabeled": relabeled, "total": len(ids)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--uuid", required=True,
                    help="canonical target workspace UUID (app.linn.games SoT)")
    ap.add_argument("--db", default=str(ROOT / "cache" / "memory.db"))
    ap.add_argument("--wiki-db", default=str(ROOT / "cache" / "wiki_v2.db"))
    ap.add_argument("--merge", required=True,
                    help="comma-separated workspace_ids to FOLD into --uuid "
                         "(the fake-user/personal slugs). Everything else "
                         "— projects, system — is left untouched.")
    ap.add_argument("--apply", action="store_true",
                    help="write changes (default: dry-run, writes nothing)")
    ap.add_argument("--no-chroma", action="store_true")
    args = ap.parse_args()

    target = args.uuid.strip()
    merge_set = tuple(p.strip() for p in args.merge.split(",") if p.strip())
    if not merge_set:
        ap.error("--merge must list at least one source workspace_id")
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== workspace re-key  merge {list(merge_set)} → {target}  [{mode}] ===")
    print("everything NOT in --merge (projects, system, …) stays untouched.\n")

    for label, path in (("memory.db", Path(args.db)), ("wiki_v2.db", Path(args.wiki_db))):
        rep = migrate_sqlite(path, target, merge_set, args.apply)
        print(f"[{label}] {rep.get('db')}")
        if rep.get("status") == "MISSING":
            print("  MISSING — skipped"); continue
        for t, info in rep["tables"].items():
            if info["move_rows"]:
                pk = " (PK-rebuild)" if info["ws_in_pk"] else ""
                print(f"  {t:26} move {info['move_rows']:5} rows from "
                      f"{info['from']}{pk}")
        if rep.get("workspaces_slugs_to_alias"):
            print(f"  workspaces → alias: {rep['workspaces_slugs_to_alias']}")
        print()

    if not args.no_chroma:
        cr = migrate_chroma(target, merge_set, args.apply)
        print(f"[chroma memory_chunks] {cr}")

    if not args.apply:
        print("\nDRY-RUN — nothing written. Re-run with --apply to commit.")
    else:
        print("\nAPPLIED. Re-run (idempotent) to sweep stragglers after the flip.")


if __name__ == "__main__":
    main()
