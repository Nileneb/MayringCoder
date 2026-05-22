from __future__ import annotations
import pytest
from unittest.mock import patch
from mayring_core.memory.db_adapter import DBAdapter
from mayring_core.memory.store import _init_schema


@pytest.fixture
def conn():
    db = DBAdapter.memory()
    _init_schema(db)
    return db


def test_derive_todo_creates_task_when_actionable(conn):
    import mayring_core.memory.todo_derivation as td
    with patch.object(td, "_embed_text", return_value=[0.1]*768), \
         patch.object(td, "_llm_todo", return_value={"actionable": True, "title": "Fix the auth bug"}):
        r = td.derive_todo("please fix the auth bug in jwt_auth", conn, "http://x", "ws1")
    assert r and r["created"] is True
    rows = conn.execute("SELECT title,status,created_by,tags FROM tasks WHERE workspace_id='ws1'").fetchall()
    assert [tuple(r) for r in rows] == [("Fix the auth bug", "open", "derived", "derived")]


def test_derive_todo_skips_when_not_actionable(conn):
    import mayring_core.memory.todo_derivation as td
    with patch.object(td, "_embed_text", return_value=[0.1]*768), \
         patch.object(td, "_llm_todo", return_value={"actionable": False, "title": ""}):
        r = td.derive_todo("what does this function do?", conn, "http://x", "ws1")
    assert r is None
    assert conn.execute("SELECT COUNT(*) FROM tasks WHERE workspace_id='ws1'").fetchone()[0] == 0


def test_derive_todo_dedups_near_identical_open_todo(conn):
    import mayring_core.memory.todo_derivation as td
    emb = [0.2]*768
    with patch.object(td, "_embed_text", return_value=emb), \
         patch.object(td, "_llm_todo", return_value={"actionable": True, "title": "Fix auth"}):
        td.derive_todo("fix the auth bug", conn, "http://x", "ws1")
        r2 = td.derive_todo("please fix that auth bug", conn, "http://x", "ws1")
    assert r2 is None
    assert conn.execute("SELECT COUNT(*) FROM tasks WHERE workspace_id='ws1'").fetchone()[0] == 1
