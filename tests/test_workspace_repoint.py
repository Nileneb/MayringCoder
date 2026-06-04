import sqlite3


def _db():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE chunks (chunk_id TEXT, workspace_id TEXT)")
    conn.execute("CREATE TABLE sources (source_id TEXT, workspace_id TEXT)")
    conn.execute("CREATE TABLE workspace_aliases (alias TEXT PRIMARY KEY, workspace_id TEXT, created_at TEXT NOT NULL)")
    conn.executemany("INSERT INTO chunks VALUES (?,?)", [("a", "old"), ("b", "old"), ("c", "keep")])
    conn.executemany("INSERT INTO sources VALUES (?,?)", [("s1", "old")])
    # a PRE-EXISTING alias that must NOT be clobbered by the repoint loop
    conn.execute("INSERT INTO workspace_aliases VALUES ('x', 'old', '2026-01-01')")
    conn.commit()
    return conn


def test_repoint_moves_rows_and_registers_alias():
    from src.api.workspace_repoint import repoint_workspace
    conn = _db()
    res = repoint_workspace(conn, "old", "new", now="2026-06-04T00:00:00Z")
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE workspace_id='new'").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE workspace_id='keep'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM sources WHERE workspace_id='new'").fetchone()[0] == 1
    # alias old->new registered
    assert conn.execute("SELECT workspace_id FROM workspace_aliases WHERE alias='old'").fetchone()[0] == "new"
    # pre-existing alias 'x' still points to its original target (loop excluded the table)
    assert conn.execute("SELECT workspace_id FROM workspace_aliases WHERE alias='x'").fetchone()[0] == "old"
    assert res["chunks"] == 2 and res["sources"] == 1


def test_repoint_noop_when_old_empty():
    from src.api.workspace_repoint import repoint_workspace
    conn = _db()
    res = repoint_workspace(conn, "ghost", "new", now="2026-06-04T00:00:00Z")
    assert sum(v for k, v in res.items() if isinstance(v, int)) == 0
