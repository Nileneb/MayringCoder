import sqlite3
from pathlib import Path

from mayring_core.memory.store import init_memory_db
from mayring_core.memory.tasks import create_task, list_tasks


def test_existing_v13_db_without_external_id_upgrades_cleanly(tmp_path: Path):
    """Regression (incident 2026-05-25): an EXISTING DB whose tasks table
    predates external_id must upgrade without raising. The v14 index creation
    used to run before the column was added → 'no such column: external_id' →
    the whole migration aborted → init_memory_db raised → every get_conn 500'd
    (prod search down). Fresh-DB tests missed it because CREATE TABLE adds the
    column. Here we simulate the v13 on-disk shape explicitly."""
    p = tmp_path / "legacy.db"
    raw = sqlite3.connect(p)
    raw.execute(
        """CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY, workspace_id TEXT, title TEXT,
            description TEXT, status TEXT, priority TEXT, due_date TEXT,
            tags TEXT, created_by TEXT, linked_chunk_id TEXT, scope_key TEXT,
            created_at TEXT, updated_at TEXT, completed_at TEXT)"""  # NO external_id
    )
    raw.execute("PRAGMA user_version = 13")
    raw.commit()
    raw.close()

    conn = init_memory_db(p)  # must NOT raise
    cols = [r[1] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()]
    assert "external_id" in cols, "v14 migration must add external_id to existing tasks"
    assert conn.execute("PRAGMA user_version").fetchone()[0] >= 14
    # and the upsert path works on the upgraded DB
    create_task(conn, workspace_id="ws", title="x", created_by="agent", external_id="e1")
    assert len(list_tasks(conn, "ws")) == 1


def test_external_id_upserts_not_duplicates(tmp_path: Path):
    conn = init_memory_db(tmp_path / "m.db")
    ws = "ws-1"
    t1 = create_task(conn, workspace_id=ws, title="do X", created_by="agent",
                     external_id="harness-42")
    t2 = create_task(conn, workspace_id=ws, title="do X (renamed)", created_by="agent",
                     external_id="harness-42")
    rows = list_tasks(conn, ws)
    assert len(rows) == 1, "same external_id must update, not duplicate"
    assert t1["task_id"] == t2["task_id"]
    assert t2["title"] == "do X (renamed)"


def test_no_external_id_always_inserts(tmp_path: Path):
    conn = init_memory_db(tmp_path / "m.db")
    create_task(conn, workspace_id="ws", title="a", created_by="agent")
    create_task(conn, workspace_id="ws", title="a", created_by="agent")
    assert len(list_tasks(conn, "ws")) == 2
