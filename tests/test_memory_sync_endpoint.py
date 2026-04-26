"""Tests for GET /memory/changes — chunk export with embeddings."""
from __future__ import annotations

import datetime
import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from src.api.auth import get_workspace
import src.api.dependencies as _deps


@pytest.fixture(autouse=True)
def _override_workspace():
    prev = app.dependency_overrides.get(get_workspace)
    app.dependency_overrides[get_workspace] = lambda: "ws-test"
    yield
    if prev is None:
        app.dependency_overrides.pop(get_workspace, None)
    else:
        app.dependency_overrides[get_workspace] = prev


@pytest.fixture(autouse=True)
def _reset_conn(tmp_path, monkeypatch):
    from src.memory.db_adapter import DBAdapter
    from src.memory.store import _init_schema
    adapter = DBAdapter.create(tmp_path / "test.db", check_same_thread=False)
    _init_schema(adapter)
    monkeypatch.setattr(_deps, "_conn", adapter)
    yield adapter
    monkeypatch.setattr(_deps, "_conn", None)


@pytest.fixture
def client():
    return TestClient(app)


def _seed_chunk(conn, chunk_id: str, source_id: str, workspace_id: str,
                visibility: str = "private", created_at: str | None = None,
                is_active: int = 1) -> None:
    from src.memory.store import upsert_source, insert_chunk
    from src.memory.schema import Source, Chunk
    upsert_source(conn, Source(source_id=source_id, source_type="note", repo="r", path="p"),
                  workspace_id=workspace_id, visibility=visibility)
    conn.execute(
        """INSERT OR IGNORE INTO chunks
           (chunk_id, source_id, text, workspace_id, created_at, is_active, text_hash, dedup_key)
           VALUES (?,?,?,?,?,?,?,?)""",
        (chunk_id, source_id, "test text", workspace_id,
         created_at or datetime.datetime.utcnow().isoformat(), is_active, "h", "d"),
    )
    conn.commit()


def test_changes_returns_chunks_since_cursor(client, _reset_conn):
    _seed_chunk(_reset_conn, "chk1", "src1", "ws-test", created_at="2026-01-01T00:00:00")
    resp = client.get("/memory/changes",
                      params={"since": "2000-01-01T00:00:00", "workspace_id": "ws-test"})
    assert resp.status_code == 200
    data = resp.json()
    assert "cursor" in data
    assert len(data["chunks"]) == 1
    assert data["chunks"][0]["chunk_id"] == "chk1"


def test_changes_embedding_key_present(client, _reset_conn):
    _seed_chunk(_reset_conn, "chk2", "src2", "ws-test", created_at="2026-01-01T00:00:01")
    resp = client.get("/memory/changes",
                      params={"since": "2000-01-01T00:00:00", "workspace_id": "ws-test"})
    for chunk in resp.json()["chunks"]:
        assert "embedding" in chunk
        assert chunk["embedding"] is None or isinstance(chunk["embedding"], list)


def test_changes_respects_cursor(client, _reset_conn):
    _seed_chunk(_reset_conn, "chk3", "src3", "ws-test", created_at="2020-01-01T00:00:00")
    resp = client.get("/memory/changes",
                      params={"since": "2099-01-01T00:00:00", "workspace_id": "ws-test"})
    assert resp.json()["chunks"] == []


def test_changes_includes_inactive_chunks(client, _reset_conn):
    _seed_chunk(_reset_conn, "chk4", "src4", "ws-test",
                created_at="2026-01-01T00:00:02", is_active=0)
    resp = client.get("/memory/changes",
                      params={"since": "2000-01-01T00:00:00", "workspace_id": "ws-test"})
    chunks = resp.json()["chunks"]
    assert any(not c["is_active"] for c in chunks)
