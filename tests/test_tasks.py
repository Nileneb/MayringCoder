from __future__ import annotations
import pytest
from src.memory.db_adapter import DBAdapter
from src.memory.store import _init_schema
from src.memory import tasks as t


def _db():
    db = DBAdapter.memory(); _init_schema(db); return db


def _client_with(ws="ws1", sub="42"):
    from fastapi.testclient import TestClient
    from src.api.server import app
    from src.api.auth import get_workspace, get_token_info
    from src.api.jwt_auth import TokenInfo
    import src.api.dependencies as _deps
    db = _db()
    app.dependency_overrides[get_workspace] = lambda: ws
    app.dependency_overrides[get_token_info] = lambda: TokenInfo(workspace_id=ws, sub=sub, scopes=("mcp:memory",))
    _deps._conn = db
    return TestClient(app), db


def test_post_and_get_tasks_endpoint():
    client, db = _client_with()
    try:
        r = client.post("/tasks", json={"title": "API task", "priority": "high"}, headers={"Authorization": "Bearer t"})
        assert r.status_code == 200, r.text
        tid = r.json()["task_id"]
        assert r.json()["created_by"] == "42"
        lst = client.get("/tasks?status=open", headers={"Authorization": "Bearer t"})
        assert lst.status_code == 200
        assert any(x["task_id"] == tid for x in lst.json()["tasks"])
    finally:
        from src.api.server import app
        import src.api.dependencies as _deps
        app.dependency_overrides.clear(); _deps._conn = None


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


def test_patch_complete_delete_endpoints_and_cross_ws_404():
    client, db = _client_with(ws="ws1")
    try:
        tid = client.post("/tasks", json={"title": "x"}, headers={"Authorization": "Bearer t"}).json()["task_id"]
        pc = client.post(f"/tasks/{tid}/complete", headers={"Authorization": "Bearer t"})
        assert pc.status_code == 200 and pc.json()["status"] == "done"
        pa = client.patch(f"/tasks/{tid}", json={"priority": "low"}, headers={"Authorization": "Bearer t"})
        assert pa.status_code == 200 and pa.json()["priority"] == "low"
        miss = client.patch("/tasks/tsk_nope", json={"title": "y"}, headers={"Authorization": "Bearer t"})
        assert miss.status_code == 404
        assert client.delete(f"/tasks/{tid}", headers={"Authorization": "Bearer t"}).status_code == 200
    finally:
        from src.api.server import app
        import src.api.dependencies as _deps
        app.dependency_overrides.clear(); _deps._conn = None


def test_create_task_rejects_invalid_scope_key():
    db = _db()
    with pytest.raises(ValueError):
        t.create_task(db, workspace_id="ws1", title="x", scope_key="not-typed")


def test_post_task_invalid_scope_key_returns_422():
    client, db = _client_with()
    try:
        r = client.post("/tasks", json={"title": "x", "scope_key": "not-typed"},
                        headers={"Authorization": "Bearer t"})
        assert r.status_code == 422, r.text
    finally:
        from src.api.server import app
        import src.api.dependencies as _deps
        app.dependency_overrides.clear(); _deps._conn = None


def test_task_create_tool_invalid_priority_returns_error_dict(monkeypatch):
    import src.api.mcp_task_tools as mt
    import src.api.dependencies as _deps
    _deps._conn = _db()
    monkeypatch.setattr(mt, "_effective_workspace_id", lambda: "ws1")
    monkeypatch.setattr(mt, "_enforce_tenant", lambda w: w or "ws1")
    monkeypatch.setattr(mt, "_effective_user_id", lambda: None)
    captured = {}
    class FakeMCP:
        def tool(self):
            def deco(fn): captured[fn.__name__] = fn; return fn
            return deco
    mt.register_task_tools(FakeMCP())
    res = captured["task_create"](title="x", priority="bogus")
    assert "error" in res
    _deps._conn = None


def test_register_task_tools_create_and_list(monkeypatch):
    import src.api.mcp_task_tools as mt
    import src.api.dependencies as _deps
    db = _db()
    _deps._conn = db
    monkeypatch.setattr(mt, "_effective_workspace_id", lambda: "ws1")
    monkeypatch.setattr(mt, "_enforce_tenant", lambda w: w or "ws1")
    monkeypatch.setattr(mt, "_effective_user_id", lambda: None)  # agent path
    captured = {}
    class FakeMCP:
        def tool(self):
            def deco(fn): captured[fn.__name__] = fn; return fn
            return deco
    mt.register_task_tools(FakeMCP())
    created = captured["task_create"](title="agent task")
    assert created["created_by"] == "agent"
    listed = captured["task_list"]()
    assert any(x["task_id"] == created["task_id"] for x in listed["tasks"])
    _deps._conn = None


# ---------------------------------------------------------------------------
# Surface existing work (IGIO goals + research_questions) as read-only tasks
# ---------------------------------------------------------------------------

def _seed_existing(db, ws):
    now = "2026-05-20T00:00:00+00:00"
    db.execute("INSERT INTO research_questions (research_question_id, title, "
               "first_seen_at, last_used_at, occurrence_count, workspace_id) "
               "VALUES (?,'JWT-Validierung',?,?,3,?)", (f"rq1-{ws}", now, now, ws))
    db.execute("INSERT OR IGNORE INTO sources (source_id, source_type, captured_at, workspace_id) "
               "VALUES (?,'note',?,?)", (f"s1-{ws}", now, ws))
    db.execute("INSERT INTO chunks (chunk_id, source_id, text, summary, igio_axis, "
               "is_active, workspace_id, created_at) "
               "VALUES (?,?,'long goal text','Ship the task tracker','goal',1,?,?)",
               (f"gc1-{ws}", f"s1-{ws}", ws, now))
    db.commit()


def test_list_tracked_work_returns_goals_and_research_questions():
    db = _db()
    _seed_existing(db, "ws1")
    _seed_existing(db, "ws2")  # isolation
    rows = t.list_tracked_work(db, "ws1")
    by_src = {r["source"]: r for r in rows}
    assert by_src["research_question"]["title"] == "JWT-Validierung"
    assert by_src["research_question"]["read_only"] is True
    assert by_src["research_question"]["task_id"].startswith("rq:")
    assert by_src["goal"]["title"] == "Ship the task tracker"
    assert by_src["goal"]["status"] == "done"
    assert by_src["goal"]["task_id"].startswith("goal:")
    # workspace-scoped: only ws1's two rows
    assert all(r["workspace_id"] == "ws1" for r in rows)
    assert len(rows) == 2


def test_get_tasks_endpoint_includes_existing_by_default():
    client, db = _client_with(ws="ws1")
    _seed_existing(db, "ws1")
    try:
        r = client.get("/tasks", headers={"Authorization": "Bearer t"})
        assert r.status_code == 200, r.text
        sources = {x.get("source") for x in r.json()["tasks"]}
        assert "goal" in sources and "research_question" in sources
        # opting out returns only real tasks (none here)
        r2 = client.get("/tasks?include_existing=false", headers={"Authorization": "Bearer t"})
        assert r2.json()["tasks"] == []
    finally:
        from src.api.server import app
        import src.api.dependencies as _deps
        app.dependency_overrides.clear(); _deps._conn = None
