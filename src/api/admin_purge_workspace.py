"""Purge ALL data for a single workspace (sources + chunks + every workspace_id
table + Chroma vectors). Used by the smoke harness to self-clean its ephemeral
`<prefix>-<ts>` workspaces (#253) and as an ops tool — no ssh/SQL needed.

A hard PROTECTED_WORKSPACES set refuses the real workspaces so a fat-fingered
call can never wipe live memory.
"""
from __future__ import annotations

from mayring_core.memory.db_adapter import DBAdapter

# WHY(#253 safety): never purge a real workspace, even on an explicit call.
PROTECTED_WORKSPACES = frozenset({
    "system", "public", "default", "bene:logs", "mayringcoder",
    "019d6933-002e-7153-a7df-f14e4c7d52b4",
    "019e14d6-0489-7348-bca8-e29c11293cb7",
})


def _workspace_tables(conn: DBAdapter) -> list[str]:
    tabs = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    return [t for t in tabs if "workspace_id" in conn.get_columns(t)]


def purge_workspace(conn: DBAdapter, chroma, workspace_id: str) -> dict:
    """Delete every row + vector for ``workspace_id``. Returns per-table counts."""
    if workspace_id in PROTECTED_WORKSPACES:
        raise ValueError(f"refusing to purge protected workspace {workspace_id!r}")

    chroma_removed = 0
    if chroma is not None:
        before = chroma.count()
        chroma.delete(where={"workspace_id": {"$in": [workspace_id]}})
        chroma_removed = before - chroma.count()

    rows: dict[str, int] = {}
    # FK off: delete across tables without ordering by parent/child relationships.
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        for t in _workspace_tables(conn):
            cur = conn.execute(f"DELETE FROM {t} WHERE workspace_id = ?", (workspace_id,))
            if cur.rowcount > 0:
                rows[t] = cur.rowcount
        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys = ON")
    return {"workspace_id": workspace_id, "chroma_removed": chroma_removed, "rows": rows}
