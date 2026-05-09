#!/usr/bin/env python3
"""Pre-launch-Migration: bewahre legacy-chunks, mappe sie auf email-slugs.

Hintergrund: Workspace-IDs sind seit 2026-05-09 email-Slugs (z.B. 'bene'
für bene@linn.games), nicht mehr 'user-N'. Echte User-Chunks die unter
'user-2' oder 'default' gesammelt wurden, sollen NICHT verloren gehen —
sie werden auf den email-slug-Workspace umgehängt.

Workflow:
    1. Migrate (--map): UPDATE workspace_id von user-N → email-slug.
       Alle 7 betroffenen Tabellen + ChromaDB.metadatas.
    2. Purge (--purge-unmapped): DELETE der Restposten, deren
       legacy-id nicht im Map-Argument vorkam.

Ohne Argumente: dry-run, zeigt nur Counts.

Affected tables (cache/memory.db):
    chunks, sources, workspaces, workspace_aliases,
    context_feedback_log, chunk_source_refs, llm_calls_log

Affected ChromaDB collections: memory_chunks (metadata.workspace_id).

Usage:
    # Dry-run: zeige was da ist
    python tools/purge_legacy_workspaces.py

    # Migrate: alle user-2-Chunks auf 'bene' umhängen, default-Chunks
    # ebenfalls auf 'bene' (Pre-launch single-user)
    python tools/purge_legacy_workspaces.py \\
        --map user-2:bene@linn.games,default:bene@linn.games \\
        --execute

    # Plus: lösche alles was nicht im Map war
    python tools/purge_legacy_workspaces.py \\
        --map user-2:bene@linn.games,default:bene@linn.games \\
        --purge-unmapped --execute
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CACHE_DIR
from src.identity.workspace_resolver import (
    email_to_slug,
    ensure_user_workspace,
)
from src.memory.store import init_memory_db


# Tabellen mit workspace_id-Spalte
WS_TABLES = [
    "chunks",
    "sources",
    "context_feedback_log",
    "chunk_source_refs",
    "llm_calls_log",
]


def parse_map(arg: str) -> dict[str, tuple[str, str | None]]:
    """`'user-2:bene@x.de,default:bene@x.de'` → {'user-2': ('bene','bene@x.de'), ...}"""
    out: dict[str, tuple[str, str | None]] = {}
    for pair in (arg or "").split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise SystemExit(f"--map: bad pair {pair!r}, expected 'legacy:email'")
        legacy, email = pair.split(":", 1)
        legacy = legacy.strip()
        email = email.strip() or None
        slug = email_to_slug(email or "") if email else legacy
        if not slug:
            raise SystemExit(f"--map: pair {pair!r} produces empty slug")
        out[legacy] = (slug, email)
    return out


def count_legacy(conn, predicate: str, params: tuple) -> int:
    return int(conn.execute(predicate, params).fetchone()[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--map", default="",
        help="Komma-separierte legacy:email-Pairs, z.B. 'user-2:bene@linn.games,default:bene@linn.games'",
    )
    parser.add_argument("--execute", action="store_true",
                        help="Führe Migration aus (default: dry-run)")
    parser.add_argument("--purge-unmapped", action="store_true",
                        help="Lösche alle legacy-rows die nicht im --map-Arg sind (nach Migration)")
    args = parser.parse_args()

    mapping = parse_map(args.map)
    conn = init_memory_db(CACHE_DIR / "memory.db")
    print(f"# DB: {CACHE_DIR / 'memory.db'}")
    print(f"# Mode: {'EXECUTE' if args.execute else 'DRY-RUN'}")
    if mapping:
        print("# Map:")
        for k, (slug, email) in mapping.items():
            print(f"    {k:20s} → {slug:20s}  ({email or '—'})")
    print()

    # ─── 1. Inventur ─────────────────────────────────────────────────
    legacy_workspace_ids = set()
    for table in WS_TABLES:
        rows = conn.execute(
            f"SELECT DISTINCT workspace_id FROM {table} "
            f"WHERE workspace_id LIKE 'user-%' OR workspace_id = 'default'"
        ).fetchall()
        legacy_workspace_ids.update(r[0] for r in rows)

    rows = conn.execute(
        "SELECT id FROM workspaces WHERE id LIKE 'user-%' OR id = 'default'"
    ).fetchall()
    legacy_workspace_ids.update(r[0] for r in rows)

    print(f"# Gefundene Legacy-Workspaces: {sorted(legacy_workspace_ids) or '—'}")

    counts_before: dict[str, dict[str, int]] = {}
    for table in WS_TABLES:
        counts_before[table] = {}
        for legacy in legacy_workspace_ids:
            n = count_legacy(
                conn,
                f"SELECT COUNT(*) FROM {table} WHERE workspace_id = ?",
                (legacy,),
            )
            counts_before[table][legacy] = n

    print("\n# Rows pro Legacy-Workspace × Tabelle:")
    print(f"  {'workspace':25s} " + "  ".join(f"{t:>20s}" for t in WS_TABLES))
    for legacy in sorted(legacy_workspace_ids):
        line = f"  {legacy:25s} "
        line += "  ".join(f"{counts_before[t].get(legacy, 0):>20d}" for t in WS_TABLES)
        print(line)

    if not args.execute:
        print("\n# Dry-run only. Re-run with --execute (+ --map / --purge-unmapped).")
        return 0

    # ─── 2. Migration: UPDATE workspace_id ──────────────────────────
    if mapping:
        print("\n# Executing migration…")
        for legacy, (new_slug, email) in mapping.items():
            # Upsert target-workspace-row (so Foreign-Key-style-Lookups halten)
            try:
                ensure_user_workspace(
                    conn, user_id=0, slug=new_slug, email=email,
                    display_name=email or new_slug,
                )
            except Exception:
                # User-Owner ist hier 0 (unknown) — beim nächsten echten
                # JWT-call wird owner_user_id korrekt gesetzt.
                pass
            for table in WS_TABLES:
                n = conn.execute(
                    f"UPDATE {table} SET workspace_id = ? WHERE workspace_id = ?",
                    (new_slug, legacy),
                ).rowcount
                if n:
                    print(f"  {table}: {n} {legacy} → {new_slug}")
            # workspaces-row selbst löschen wenn legacy-id existiert
            n = conn.execute(
                "DELETE FROM workspaces WHERE id = ?", (legacy,)
            ).rowcount
            if n:
                print(f"  workspaces: dropped legacy id {legacy}")

        conn.commit()

    # ─── 3. Purge unmapped legacy ───────────────────────────────────
    if args.purge_unmapped:
        unmapped = legacy_workspace_ids - set(mapping.keys())
        if not unmapped:
            print("\n# Keine unmapped legacy-rows.")
        else:
            print(f"\n# Purging unmapped: {sorted(unmapped)}")
            for legacy in unmapped:
                for table in WS_TABLES:
                    n = conn.execute(
                        f"DELETE FROM {table} WHERE workspace_id = ?",
                        (legacy,),
                    ).rowcount
                    if n:
                        print(f"  {table}: {n} rows of {legacy} deleted")
                n = conn.execute(
                    "DELETE FROM workspaces WHERE id = ?", (legacy,)
                ).rowcount
                if n:
                    print(f"  workspaces: dropped {legacy}")
        conn.commit()

    # ─── 4. ChromaDB ────────────────────────────────────────────────
    try:
        from src.memory.store import get_chroma_collection
        col = get_chroma_collection("memory_chunks")
        if col is not None:
            all_ids = col.get().get("ids") or []
            all_meta = col.get(include=["metadatas"]).get("metadatas") or []
            migrated_chroma = 0
            dropped_chroma = 0
            for cid, meta in zip(all_ids, all_meta):
                if not isinstance(meta, dict):
                    continue
                ws = meta.get("workspace_id")
                if ws in mapping:
                    new_slug = mapping[ws][0]
                    new_meta = {**meta, "workspace_id": new_slug}
                    col.update(ids=[cid], metadatas=[new_meta])
                    migrated_chroma += 1
                elif args.purge_unmapped and (
                    str(ws or "").startswith("user-") or ws == "default"
                ):
                    col.delete(ids=[cid])
                    dropped_chroma += 1
            if migrated_chroma:
                print(f"  chroma:memory_chunks: {migrated_chroma} chunks remapped")
            if dropped_chroma:
                print(f"  chroma:memory_chunks: {dropped_chroma} unmapped chunks deleted")
    except Exception as e:
        print(f"  chroma cleanup failed: {e}")

    print("\n# Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
