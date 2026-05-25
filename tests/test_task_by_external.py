"""TDD: PATCH /tasks/by-external/{external_id} — hook-driven status sync."""
from __future__ import annotations

from mayring_core.memory.db_adapter import DBAdapter
from mayring_core.memory.store import _init_schema
from mayring_core.memory import tasks as t


def _db():
    db = DBAdapter.memory()
    _init_schema(db)
    return db


def _client_with(ws="ws1", sub="agent"):
    from fastapi.testclient import TestClient
    from src.api.server import app
    from src.api.auth import get_workspace, get_token_info
    from src.api.jwt_auth import TokenInfo
    import src.api.dependencies as _deps

    db = _db()
    app.dependency_overrides[get_workspace] = lambda: ws
    app.dependency_overrides[get_token_info] = lambda: TokenInfo(
        workspace_id=ws, sub=sub, scopes=("mcp:memory",)
    )
    _deps._conn = db
    return TestClient(app), db


def test_patch_by_external_updates_status_to_done():
    client, db = _client_with()
    try:
        cr = client.post(
            "/tasks",
            json={"title": "hook task", "external_id": "x"},
            headers={"Authorization": "Bearer t"},
        )
        assert cr.status_code == 200, cr.text

        pr = client.patch(
            "/tasks/by-external/x",
            json={"status": "done"},
            headers={"Authorization": "Bearer t"},
        )
        assert pr.status_code == 200, pr.text
        assert pr.json()["status"] == "done"
    finally:
        from src.api.server import app
        import src.api.dependencies as _deps
        app.dependency_overrides.clear()
        _deps._conn = None


def test_patch_by_external_unknown_returns_404():
    client, db = _client_with()
    try:
        pr = client.patch(
            "/tasks/by-external/unknown-ext",
            json={"status": "done"},
            headers={"Authorization": "Bearer t"},
        )
        assert pr.status_code == 404
    finally:
        from src.api.server import app
        import src.api.dependencies as _deps
        app.dependency_overrides.clear()
        _deps._conn = None
