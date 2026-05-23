"""Move legacy per-project workspace_ids INTO the canonical workspace as a
`project_id` dimension (#workspace-uuid-sot v7, User-Diagramm).

WHY: past identity bugs created fake-separate workspace_ids per project
(dronedetect, battlefield, …). The target model is ONE workspace (the user's
UUID) that CONTAINS projects via `chunks/sources.project_id` + a `projects`
registry. This tool re-keys each project slug → (workspace_id=<uuid>,
project_id=<canonical>), normalizes duplicate slugs, registers `projects` rows,
and routes the shared `global` bucket to a `public` workspace
(visibility='public', readable by all) — NOT `system`.

Safety: --dry-run default, --apply to write, PRAGMA busy_timeout, idempotent.
PK/UNIQUE-collidable tables use UPDATE OR IGNORE + leftover-delete (dedups).
Reuses the table-discovery + collidability detection helpers from
migrate_to_uuid_workspace.py.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.migrate_to_uuid_workspace import _tables_with_workspace_col  # noqa: E402

# slug → canonical project_id (dup-normalisation baked in).
PROJECT_MAP = {
    "dronedetect": "dronedetect",
    "battlefield": "battlefield",
    "mayringcoder": "mayringcoder",
    "mayring-coder": "mayringcoder",        # dup → canonical
    "linn-games-research": "linn-games-research",
    "DiakonieWhisper": "DiakonieWhisper",
    "app-linn-games": "app-linn-games",
    "applinngames": "app-linn-games",        # dup → canonical
    "bene:logs": "logs",
}
# Shared bucket → public workspace (NOT a project, NOT system).
PUBLIC_SLUGS = {"global"}
PUBLIC_WS = "public"


def _project_name(pid: str) -> str:
    return pid.replace("-", " ").replace(":", " ").title()


def migrate(db_path: Path, target: str, apply: bool) -> dict:
    if not db_path.exists():
        return {"db": str(db_path), "status": "MISSING"}
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout=20000")
    report: dict = {"db": str(db_path), "projects": {}, "public": {}}
    try:
        tables = _tables_with_workspace_col(conn)
        has_projects = "projects" in {
            r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        # Which tables carry project_id (only some) — set it there too.
        proj_col_tables = {
            t for t in tables
            if "project_id" in {r[1] for r in conn.execute(f"PRAGMA table_info({t})")}
        }
        now = datetime.now(timezone.utc).isoformat()

        # --- project slugs → (target workspace + project_id) -----------------
        for slug, pid in PROJECT_MAP.items():
            moved = 0
            for table, collidable in tables.items():
                n = conn.execute(
                    f"SELECT count(*) FROM {table} WHERE workspace_id = ?", (slug,)
                ).fetchone()[0]
                if not n:
                    continue
                moved += n
                if not apply:
                    continue
                set_proj = ", project_id = ?" if table in proj_col_tables else ""
                params = ([target, pid, slug] if set_proj else [target, slug])
                stmt = (f"UPDATE {'OR IGNORE ' if collidable else ''}{table} "
                        f"SET workspace_id = ?{set_proj} WHERE workspace_id = ?")
                conn.execute(stmt, params)
                if collidable:
                    conn.execute(
                        f"DELETE FROM {table} WHERE workspace_id = ?", (slug,))
            if moved:
                report["projects"].setdefault(pid, 0)
                report["projects"][pid] += moved
            if apply and moved and has_projects:
                conn.execute(
                    "INSERT OR IGNORE INTO projects(id, workspace_id, name, "
                    "created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (pid, target, _project_name(pid), now, now))

        # --- global → public workspace (visibility='public') ----------------
        for slug in PUBLIC_SLUGS:
            for table, collidable in tables.items():
                n = conn.execute(
                    f"SELECT count(*) FROM {table} WHERE workspace_id = ?", (slug,)
                ).fetchone()[0]
                if not n:
                    continue
                report["public"][table] = n
                if not apply:
                    continue
                stmt = (f"UPDATE {'OR IGNORE ' if collidable else ''}{table} "
                        f"SET workspace_id = ? WHERE workspace_id = ?")
                conn.execute(stmt, (PUBLIC_WS, slug))
                if collidable:
                    conn.execute(
                        f"DELETE FROM {table} WHERE workspace_id = ?", (slug,))
            # sources of the public bucket become readable by all.
            if apply and "sources" in tables:
                conn.execute(
                    "UPDATE sources SET visibility = 'public' WHERE workspace_id = ?",
                    (PUBLIC_WS,))
        if apply:
            conn.commit()
        return report
    finally:
        conn.close()


def migrate_chroma(target: str, apply: bool) -> dict:
    try:
        from mayring_core.memory.store import get_chroma_collection
    except Exception as e:  # noqa: BLE001
        return {"status": "skipped", "reason": f"import failed: {e}"}
    col = get_chroma_collection("memory_chunks")
    if col is None:
        return {"status": "skipped"}
    got = col.get(include=["metadatas"])
    ids, metas = got.get("ids", []) or [], got.get("metadatas", []) or []
    BATCH, touched = 500, 0
    for i in range(0, len(ids), BATCH):
        b_ids, b_metas = ids[i:i + BATCH], metas[i:i + BATCH]
        d_ids, d_metas = [], []
        for _id, _m in zip(b_ids, b_metas):
            m = dict(_m or {})
            ws = m.get("workspace_id")
            if ws in PROJECT_MAP:
                m["workspace_id"] = target
                m["project_id"] = PROJECT_MAP[ws]
                d_ids.append(_id); d_metas.append(m)
            elif ws in PUBLIC_SLUGS:
                m["workspace_id"] = PUBLIC_WS
                d_ids.append(_id); d_metas.append(m)
        if d_ids and apply:
            col.update(ids=d_ids, metadatas=d_metas)
        touched += len(d_ids)
    return {"relabeled": touched, "total": len(ids)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--uuid", required=True, help="canonical workspace UUID")
    ap.add_argument("--db", default=str(ROOT / "cache" / "memory.db"))
    ap.add_argument("--wiki-db", default=str(ROOT / "cache" / "wiki_v2.db"))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--no-chroma", action="store_true")
    args = ap.parse_args()
    target = args.uuid.strip()
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== projects → dimension under {target}  [{mode}] ===")
    print(f"project map: {PROJECT_MAP}\npublic: {PUBLIC_SLUGS} → workspace '{PUBLIC_WS}'\n")
    for label, path in (("memory.db", Path(args.db)), ("wiki_v2.db", Path(args.wiki_db))):
        rep = migrate(path, target, args.apply)
        print(f"[{label}] {rep.get('db')}")
        if rep.get("status") == "MISSING":
            print("  MISSING"); continue
        for pid, n in rep["projects"].items():
            print(f"  project_id={pid:22} {n} rows")
        for t, n in rep["public"].items():
            print(f"  public  {t:22} {n} rows")
        print()
    if not args.no_chroma:
        print(f"[chroma] {migrate_chroma(target, args.apply)}")
    print("\n" + ("APPLIED." if args.apply else "DRY-RUN — re-run with --apply."))


if __name__ == "__main__":
    main()
