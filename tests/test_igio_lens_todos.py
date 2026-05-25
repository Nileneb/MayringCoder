"""GET /stats/igio-lens must expose workspace tasks under intervention.todos."""
from __future__ import annotations

import asyncio
import datetime

from mayring_core.memory.store import init_memory_db
from mayring_core.memory.tasks import create_task
import src.api.routes.igio_admin as igio_admin
from src.api.jwt_auth import TokenInfo


def _seed_db(tmp_path, *, workspace_id: str = "ws-todos"):
    db = init_memory_db(tmp_path / "m.db")
    now = datetime.datetime.utcnow().isoformat()
    # minimal chunk row so the axes query doesn't crash on an empty table
    db.execute(
        "INSERT INTO sources (source_id, source_type, captured_at, workspace_id) "
        "VALUES (?,?,?,?)", ("src:t1", "note", now, workspace_id))
    db.execute(
        "INSERT INTO chunks (chunk_id, source_id, text, text_hash, dedup_key, "
        "created_at, workspace_id, igio_axis, igio_confidence, summary) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("t1", "src:t1", "text t1", "ht1", "dt1", now, workspace_id, "issue", 0.9, "s t1"))
    db.commit()
    return db


def _run(coro):
    return asyncio.run(coro)


def test_igio_lens_includes_intervention_todos(tmp_path, monkeypatch):
    """intervention.todos must contain any open task seeded for the workspace."""
    db = _seed_db(tmp_path, workspace_id="ws-todos")
    task = create_task(
        db, workspace_id="ws-todos", title="capture me",
        created_by="agent", external_id="ext-smoke-1",
    )
    monkeypatch.setattr(igio_admin, "_conn", lambda: db)

    info = TokenInfo(workspace_id="ws-todos", scopes=())
    res = _run(igio_admin.igio_lens(info=info, workspace="ws-todos"))

    assert res["workspace_id"] == "ws-todos"
    todos = res["axes"]["intervention"]["todos"]
    assert isinstance(todos, list), "intervention.todos must be a list"
    assert any(
        t["task_id"] == task["task_id"] and t["title"] == "capture me"
        for t in todos
    ), f"seeded task not found in todos: {todos}"


def test_igio_lens_todos_shape(tmp_path, monkeypatch):
    """Each todo entry must carry the six canonical fields."""
    db = _seed_db(tmp_path, workspace_id="ws-shape")
    create_task(db, workspace_id="ws-shape", title="shape check", created_by="agent")
    monkeypatch.setattr(igio_admin, "_conn", lambda: db)

    info = TokenInfo(workspace_id="ws-shape", scopes=())
    res = _run(igio_admin.igio_lens(info=info, workspace="ws-shape"))
    todos = res["axes"]["intervention"]["todos"]
    assert todos, "expected at least one todo"
    t = todos[0]
    for field in ("task_id", "title", "status", "created_by", "created_at", "completed_at"):
        assert field in t, f"missing field {field!r} in todo shape"


def test_igio_lens_todos_open_task_has_no_completed_at(tmp_path, monkeypatch):
    """An open (not-done) task must return completed_at=None, not crash."""
    db = _seed_db(tmp_path, workspace_id="ws-open")
    create_task(db, workspace_id="ws-open", title="open task", created_by="agent")
    monkeypatch.setattr(igio_admin, "_conn", lambda: db)

    info = TokenInfo(workspace_id="ws-open", scopes=())
    res = _run(igio_admin.igio_lens(info=info, workspace="ws-open"))
    todos = res["axes"]["intervention"]["todos"]
    assert todos
    assert todos[0]["completed_at"] is None


def test_igio_lens_intervention_count_still_present(tmp_path, monkeypatch):
    """Back-compat: intervention.count must remain present alongside todos."""
    db = _seed_db(tmp_path, workspace_id="ws-compat")
    monkeypatch.setattr(igio_admin, "_conn", lambda: db)

    info = TokenInfo(workspace_id="ws-compat", scopes=())
    res = _run(igio_admin.igio_lens(info=info, workspace="ws-compat"))
    assert "count" in res["axes"]["intervention"], "intervention.count must not be removed (back-compat)"
    assert "todos" in res["axes"]["intervention"], "intervention.todos must be added"
