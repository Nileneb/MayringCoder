from __future__ import annotations
import pytest
from src.memory.db_adapter import DBAdapter
from src.memory.store import _init_schema
from src.memory import tasks as t


def _db():
    db = DBAdapter.memory(); _init_schema(db); return db


def test_create_task_persists_and_returns_row():
    db = _db()
    row = t.create_task(db, workspace_id="ws1", title="Fix auth", description="JWT bug",
                        priority="high", tags="auth,bug", created_by="42")
    assert row["task_id"].startswith("tsk_")
    assert row["status"] == "open"
    assert row["priority"] == "high"
    assert row["workspace_id"] == "ws1"
    assert row["completed_at"] is None
    stored = db.execute("SELECT title, tags FROM tasks WHERE task_id=?", (row["task_id"],)).fetchone()
    assert stored["title"] == "Fix auth"
    assert stored["tags"] == "auth,bug"


def test_list_tasks_filters_by_status_and_workspace():
    db = _db()
    t.create_task(db, workspace_id="ws1", title="a")
    done = t.create_task(db, workspace_id="ws1", title="b")
    t.complete_task(db, "ws1", done["task_id"])
    t.create_task(db, workspace_id="ws2", title="other-ws")
    open_ws1 = t.list_tasks(db, "ws1", status="open")
    assert [r["title"] for r in open_ws1] == ["a"]
    assert all(r["workspace_id"] == "ws1" for r in t.list_tasks(db, "ws1"))
    assert t.list_tasks(db, "ws2")[0]["title"] == "other-ws"


def test_complete_sets_completed_at_and_reopen_clears_it():
    db = _db()
    task = t.create_task(db, workspace_id="ws1", title="x")
    done = t.complete_task(db, "ws1", task["task_id"])
    assert done["status"] == "done" and done["completed_at"] is not None
    reopened = t.update_task(db, "ws1", task["task_id"], status="open")
    assert reopened["completed_at"] is None


def test_update_and_delete_are_workspace_scoped():
    db = _db()
    task = t.create_task(db, workspace_id="ws1", title="x")
    assert t.update_task(db, "ws2", task["task_id"], title="hijack") is None
    assert t.delete_task(db, "ws2", task["task_id"]) is False
    assert t.get_task(db, "ws1", task["task_id"])["title"] == "x"
    assert t.delete_task(db, "ws1", task["task_id"]) is True
    assert t.get_task(db, "ws1", task["task_id"]) is None
