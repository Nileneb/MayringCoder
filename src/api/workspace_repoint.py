"""Reusable workspace re-point: move ALL of a workspace's rows (and Chroma vectors)
onto another workspace and register an alias so old tokens keep resolving.

Generalised from tools/migrate_workspace_repoint.py so the same logic backs both the
one-off consolidation tool AND the dashboard claim endpoint. The caller owns backups
and (for SQLite) the surrounding transaction/commit decision.
"""
from __future__ import annotations

from datetime import datetime, timezone

# Tables whose workspace_id column must NOT be bulk-rewritten by the repoint loop:
# `workspaces` keys on `id` (no workspace_id col anyway); `workspace_aliases.workspace_id`
# is the alias TARGET — rewriting it would clobber unrelated aliases.
_EXCLUDE = {"workspace_aliases", "workspaces"}


def _tables_with_workspace_id(conn) -> list[str]:
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    out = []
    for t in sorted(tables):
        if t in _EXCLUDE:
            continue
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info('{t}')")]
        if "workspace_id" in cols:
            out.append(t)
    return out


def repoint_workspace(conn, old: str, new: str, *, chroma=None,
                      now: str | None = None, commit: bool = True) -> dict:
    """Move every row tagged `old` to `new`, register alias old→new, optionally
    re-tag Chroma `memory_chunks` metadata. Returns per-table counts moved.
    Idempotent-ish: re-running with an empty `old` is a no-op."""
    now = now or datetime.now(timezone.utc).isoformat()
    counts: dict[str, int] = {}
    for t in _tables_with_workspace_id(conn):
        n = conn.execute(f"SELECT COUNT(*) FROM {t} WHERE workspace_id=?", (old,)).fetchone()[0]
        if n:
            conn.execute(f"UPDATE {t} SET workspace_id=? WHERE workspace_id=?", (new, old))
            counts[t] = n
    conn.execute(
        "INSERT OR IGNORE INTO workspace_aliases(alias, workspace_id, created_at) "
        "VALUES (?,?,?)", (old, new, now))
    if commit:
        conn.commit()

    if chroma is not None:
        got = chroma.get(where={"workspace_id": old}, include=["metadatas"])
        ids = got.get("ids") or []
        metas = got.get("metadatas") or []
        B = 256
        for i in range(0, len(ids), B):
            chroma.update(
                ids=ids[i:i + B],
                metadatas=[{**(m or {}), "workspace_id": new} for m in metas[i:i + B]],
            )
        counts["chroma"] = len(ids)
    return counts
