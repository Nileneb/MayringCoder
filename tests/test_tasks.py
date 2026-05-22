from __future__ import annotations
import pytest
from mayring_core.memory.db_adapter import DBAdapter
from mayring_core.memory.store import _init_schema
from mayring_core.memory import tasks as t


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


def test_tasks_has_derive_embedding_column():
    db = _db()
    assert "derive_embedding" in db.get_columns("tasks")


# --- Task 4: list_workspace_goals + GET /tasks/goals ---

import uuid as _uuid
from datetime import datetime, timezone as _tz


def _seed_goal_chunk(db, workspace_id: str, source_type: str, summary: str = "A goal") -> str:
    """Insert a source+chunk with igio_axis='goal' for testing. Returns chunk_id."""
    now = datetime.now(_tz.utc).isoformat()
    source_id = "src_" + _uuid.uuid4().hex[:12]
    chunk_id = "chk_" + _uuid.uuid4().hex[:12]
    db.execute(
        "INSERT INTO sources (source_id, source_type, repo, path, branch, \"commit\","
        " content_hash, captured_at, workspace_id, visibility)"
        " VALUES (?,?,?,?,?,?,?,?,?,'private')",
        (source_id, source_type, "repo", "path", "main", "", "hash_" + chunk_id, now, workspace_id),
    )
    db.execute(
        "INSERT INTO chunks (chunk_id, source_id, chunk_level, ordinal, start_offset, end_offset,"
        " text, text_hash, summary, category_labels, category_version, embedding_model,"
        " embedding_id, quality_score, dedup_key, category_source, category_confidence,"
        " igio_axis, igio_confidence, igio_classified_at, created_at, is_active, workspace_id)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (chunk_id, source_id, "file", 0, 0, 0, summary, "h_" + chunk_id, summary,
         "", "v1", "nomic-embed-text", "", 0.9, chunk_id, "test", 0.9,
         "goal", 0.9, now, now, 1, workspace_id),
    )
    db.commit()
    return chunk_id


def test_list_workspace_goals_filters_source_types():
    db = _db()
    # Seed: conversation goal (should appear), paper goal (should NOT), ambient goal (should NOT)
    _seed_goal_chunk(db, "ws1", "conversation_summary", "My dev goal")
    _seed_goal_chunk(db, "ws1", "paper", "Paper goal unrelated")
    _seed_goal_chunk(db, "ws1", "ambient_snapshot", "Ambient goal unrelated")

    goals = t.list_workspace_goals(db, "ws1")
    assert len(goals) == 1
    assert goals[0]["source"] == "goal"
    assert goals[0]["read_only"] is True
    assert "My dev goal" in goals[0]["title"]


def test_list_workspace_goals_is_workspace_scoped():
    db = _db()
    _seed_goal_chunk(db, "ws1", "conversation_summary", "ws1 goal")
    _seed_goal_chunk(db, "ws2", "conversation_summary", "ws2 goal")

    ws1_goals = t.list_workspace_goals(db, "ws1")
    ws2_goals = t.list_workspace_goals(db, "ws2")
    assert len(ws1_goals) == 1
    assert len(ws2_goals) == 1
    assert "ws1 goal" in ws1_goals[0]["title"]
    assert "ws2 goal" in ws2_goals[0]["title"]


def test_get_tasks_goals_endpoint():
    client, db = _client_with(ws="ws1")
    try:
        _seed_goal_chunk(db, "ws1", "conversation_summary", "Endpoint goal")
        r = client.get("/tasks/goals", headers={"Authorization": "Bearer t"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["workspace_id"] == "ws1"
        assert isinstance(data["goals"], list)
        assert len(data["goals"]) == 1
        assert data["goals"][0]["read_only"] is True
    finally:
        from src.api.server import app
        import src.api.dependencies as _deps
        app.dependency_overrides.clear(); _deps._conn = None


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

