#!/usr/bin/env python3
"""Merge case-only Source duplicates created before URL canonicalization.

Background:
    Before commit eef1619, source_ids embedded the raw repo URL. GitHub treats
    "Nileneb/X" and "nileneb/X" as the same project, but our DB held them as
    two parallel sources. Each had its own chunks; Chroma usually only indexed
    one variant, so vector search silently returned 0 hits for the other half.

This script:
    1. Finds groups of source_ids that canonicalize to the same value.
    2. Picks one canonical source_id per group (lowercased URL).
    3. Re-points chunks + chunk_source_refs onto the canonical source.
    4. Drops the legacy source rows.

Idempotent: re-running on a clean DB is a no-op. Dry-run by default.

Usage:
    python tools/cleanup_canonical_repo_urls.py --db /path/to/memory.db
    python tools/cleanup_canonical_repo_urls.py --db ... --apply

Cloud-side: this script is invoked from src.api.dependencies on first init
(cleanup runs once, leaves a marker row in kv_store so it doesn't re-run).
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.memory.schema import canonicalize_repo_url


def _canonical_source_id(source_id: str) -> str:
    """Apply URL canonicalization to the repo segment of a source_id.

    Format expected: "repo:<URL>:<path>" (URL may contain colons after //).
    Other source_ids (paper:, note:, conversation:, ...) pass through unchanged.
    """
    if not source_id.startswith("repo:"):
        return source_id
    rest = source_id[len("repo:"):]
    # find the path separator: the LAST ":" that isn't part of "://"
    # Strategy: split URL prefix off via "://", then take everything before the
    # next ":" as the URL/host/path, and the remainder as the file path.
    if "://" in rest:
        scheme_split = rest.split("://", 1)
        scheme = scheme_split[0]
        after_scheme = scheme_split[1]
        # after_scheme = "github.com/owner/repo:src/file.py"
        if ":" in after_scheme:
            host_path, file_path = after_scheme.split(":", 1)
            url = f"{scheme}://{host_path}"
            return f"repo:{canonicalize_repo_url(url)}:{file_path}"
        return source_id
    # no scheme — "owner/name:path" form
    if ":" in rest:
        slug, file_path = rest.split(":", 1)
        return f"repo:{canonicalize_repo_url(slug)}:{file_path}"
    return source_id


def find_duplicate_groups(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Return {canonical_source_id: [legacy_source_id, ...]} for groups with >1 member."""
    rows = conn.execute("SELECT source_id FROM sources").fetchall()
    groups: dict[str, list[str]] = defaultdict(list)
    for (sid,) in rows:
        groups[_canonical_source_id(sid)].append(sid)
    return {canon: ids for canon, ids in groups.items() if len(ids) > 1 or ids[0] != canon}


def merge_group(
    conn: sqlite3.Connection,
    canonical_id: str,
    legacy_ids: list[str],
    apply: bool,
) -> dict[str, int]:
    """Move chunks + refs to canonical_id, then drop legacy sources.

    Stats: {chunks_moved, refs_moved, sources_dropped}.
    """
    stats = {"chunks_moved": 0, "refs_moved": 0, "sources_dropped": 0}
    if not apply:
        # Dry-run: just count what WOULD change
        for old in legacy_ids:
            if old == canonical_id:
                continue
            stats["chunks_moved"] += conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE source_id = ?", (old,)
            ).fetchone()[0]
            stats["refs_moved"] += conn.execute(
                "SELECT COUNT(*) FROM chunk_source_refs WHERE source_id = ?", (old,)
            ).fetchone()[0]
            stats["sources_dropped"] += 1
        return stats

    # Ensure canonical source row exists. If only legacy IDs are present (none
    # equals canonical), promote one of them by renaming. Crucially, re-point
    # chunks + refs at the new source_id too — otherwise they'd be left
    # pointing at a row that no longer exists.
    has_canonical = conn.execute(
        "SELECT 1 FROM sources WHERE source_id = ?", (canonical_id,)
    ).fetchone()
    if not has_canonical and legacy_ids:
        promoted = legacy_ids[0]
        conn.execute(
            "UPDATE sources SET source_id = ? WHERE source_id = ?",
            (canonical_id, promoted),
        )
        chunks_repointed = conn.execute(
            "UPDATE chunks SET source_id = ? WHERE source_id = ?",
            (canonical_id, promoted),
        ).rowcount or 0
        refs_repointed = conn.execute(
            "UPDATE OR IGNORE chunk_source_refs SET source_id = ? WHERE source_id = ?",
            (canonical_id, promoted),
        ).rowcount or 0
        stats["chunks_moved"] += chunks_repointed
        stats["refs_moved"] += refs_repointed
        legacy_ids = [s for s in legacy_ids if s != promoted]

    for old in legacy_ids:
        if old == canonical_id:
            continue
        stats["chunks_moved"] += conn.execute(
            "UPDATE chunks SET source_id = ? WHERE source_id = ?",
            (canonical_id, old),
        ).rowcount or 0
        stats["refs_moved"] += conn.execute(
            "UPDATE OR IGNORE chunk_source_refs SET source_id = ? WHERE source_id = ?",
            (canonical_id, old),
        ).rowcount or 0
        # Any refs that collided with existing canonical entries (UNIQUE) are
        # still pointing at the old source; drop them.
        conn.execute("DELETE FROM chunk_source_refs WHERE source_id = ?", (old,))
        conn.execute("DELETE FROM sources WHERE source_id = ?", (old,))
        stats["sources_dropped"] += 1
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", required=True, help="Path to memory.db")
    p.add_argument("--apply", action="store_true", help="Actually mutate (default: dry-run)")
    args = p.parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON")

    groups = find_duplicate_groups(conn)
    if not groups:
        print("No duplicate source_ids found — DB already canonical.")
        return

    print(f"Found {len(groups)} canonical groups with non-canonical members:")
    total = {"chunks_moved": 0, "refs_moved": 0, "sources_dropped": 0}
    for canon, legacy in groups.items():
        stats = merge_group(conn, canon, legacy, args.apply)
        for k, v in stats.items():
            total[k] += v
        print(f"  {canon}")
        for old in legacy:
            mark = "✓" if old == canon else "→ merge"
            print(f"    {mark} {old}")
        print(f"    {stats}")

    if args.apply:
        conn.commit()
        print(f"\nApplied. Totals: {total}")
    else:
        print(f"\nDry-run. Totals (would change): {total}")
        print("Pass --apply to commit.")


if __name__ == "__main__":
    main()
