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


def purge_smoke_projects(conn: DBAdapter) -> dict:
    """Delete smoke-suite throwaway projects (+ links + leftover smoke groups)
    across ALL workspaces.

    WHY(2026-06-10): the C3 check creates one ``smoke/repo-c3-<ts>`` project per
    run and nothing deleted them; the broken NOT_SMOKE guard ('%/smoke/repo-%'
    vs the canonical slug 'smoke/repo-...' WITHOUT leading slash) had also
    claimed 87 of them into the user workspace. Pattern-gated -- owner 'smoke'
    is the suite's fake org, never a real repo.
    """
    ids = [r[0] for r in conn.execute(
        "SELECT id FROM projects WHERE lower(source_ref) LIKE '%smoke/repo-%'"
    ).fetchall()]
    links = 0
    if ids:
        ph = ",".join("?" * len(ids))
        links = conn.execute(
            f"DELETE FROM chunk_project_links WHERE project_id IN ({ph})", ids
        ).rowcount
        conn.execute(f"DELETE FROM projects WHERE id IN ({ph})", ids)
    groups = conn.execute(
        "DELETE FROM project_groups WHERE name LIKE 'smoke-%'"
    ).rowcount

    # smoke:%-Quellen (z.B. smoke:state:<ts>) — Checks ohne Self-Clean haben
    # ~200 Source-Zeilen an 3 Canonical-Chunks im realen Workspace angesammelt
    # (#253). Chunks via Standard-Invalidate-Pfad deaktivieren, dann Refs+Rows weg.
    from mayring_core.memory.store import deactivate_chunks_by_source
    smoke_sources = [r[0] for r in conn.execute(
        "SELECT source_id FROM sources WHERE source_id LIKE 'smoke:%'"
    ).fetchall()]
    deactivated = 0
    for sid in smoke_sources:
        deactivated += deactivate_chunks_by_source(conn, sid)
    if smoke_sources:
        ph = ",".join("?" * len(smoke_sources))
        conn.execute(f"DELETE FROM chunk_source_refs WHERE source_id IN ({ph})", smoke_sources)
        conn.execute(f"DELETE FROM sources WHERE source_id IN ({ph})", smoke_sources)
    conn.commit()
    return {"projects": len(ids), "chunk_project_links": max(links, 0),
            "project_groups": max(groups, 0), "smoke_sources": len(smoke_sources),
            "chunks_deactivated": deactivated}


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
