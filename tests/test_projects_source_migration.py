import sqlite3
from pathlib import Path

from mayring_core.memory.store import init_memory_db


def test_existing_pre_v8_projects_without_source_type_upgrades_cleanly(tmp_path: Path):
    """Regression (memory-agents -32000, 2026-05-27): an EXISTING DB whose
    projects table predates v8 (no source_type/source_ref) must upgrade without
    raising. idx_projects_source used to be created inside the main DDL block,
    which ran before _migrate_schema added the columns → 'no such column:
    source_type' → init_memory_db raised → the MCP server crashed on boot
    (JSON-RPC -32000). Fresh-DB tests missed it because CREATE TABLE adds the
    columns. Here we simulate the pre-v8 on-disk shape explicitly."""
    p = tmp_path / "legacy.db"
    raw = sqlite3.connect(p)
    raw.executescript(
        """
        CREATE TABLE projects (
            id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL,
            name TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );  -- NO owner_id / source_type / source_ref
        CREATE INDEX idx_projects_workspace ON projects(workspace_id);
        INSERT INTO projects VALUES ('p1', 'ws1', 'old project', 't', 't');
        """
    )
    raw.execute("PRAGMA user_version = 7")
    raw.commit()
    raw.close()

    conn = init_memory_db(p)  # must NOT raise
    cols = [r[1] for r in conn.execute("PRAGMA table_info(projects)").fetchall()]
    assert "source_type" in cols and "source_ref" in cols, (
        "v8 migration must back-fill projects.source_type/source_ref"
    )
    idx = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_projects_source'"
    ).fetchone()
    assert idx is not None, "idx_projects_source must exist after migration"
    assert conn.execute("PRAGMA user_version").fetchone()[0] >= 14
    # legacy data preserved through the upgrade
    assert conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 1


def test_fresh_db_creates_projects_source_index(tmp_path: Path):
    conn = init_memory_db(tmp_path / "fresh.db")
    idx = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_projects_source'"
    ).fetchone()
    assert idx is not None
    assert conn.execute("PRAGMA user_version").fetchone()[0] >= 14
