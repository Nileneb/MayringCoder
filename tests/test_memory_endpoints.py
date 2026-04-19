"""Tests for the 6 new /memory/* HTTP endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from src.api.auth import get_workspace
import src.api.dependencies as _deps


def _fake_ws():
    return "test-ws"


app.dependency_overrides[get_workspace] = _fake_ws


@pytest.fixture(autouse=True)
def _reset_conn(tmp_path, monkeypatch):
    """Each test gets a fresh SQLite DB with check_same_thread=False for async TestClient."""
    import sqlite3 as _sqlite3
    from src.memory.store import _init_schema
    db_path = tmp_path / "test.db"
    conn = _sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = _sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    _init_schema(conn)
    monkeypatch.setattr(_deps, "_conn", conn)
    yield conn
    monkeypatch.setattr(_deps, "_conn", None)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def seeded_chunk(_reset_conn):
    """Insert a source + chunk into the test DB and return the chunk_id."""
    from src.memory.store import upsert_source, insert_chunk
    from src.memory.schema import Source, Chunk
    conn = _reset_conn
    upsert_source(
        conn,
        Source(
            source_id="src_test",
            source_type="repo_file",
            repo="test/repo",
            path="test.py",
            branch="main",
            commit="",
            content_hash="",
            captured_at="2026-04-19T00:00:00+00:00",
        ),
    )
    chunk = Chunk(
        chunk_id="chk_test123",
        source_id="src_test",
        text="hello world",
        text_hash=Chunk.compute_text_hash("hello world"),
        created_at="2026-04-19T00:00:00+00:00",
    )
    insert_chunk(conn, chunk)
    return "chk_test123"


# ---------------------------------------------------------------------------
# GET /memory/chunk/{chunk_id}
# ---------------------------------------------------------------------------

def test_memory_chunk_404(client):
    r = client.get("/memory/chunk/does-not-exist")
    assert r.status_code == 404


def test_memory_chunk_found(client, seeded_chunk):
    r = client.get(f"/memory/chunk/{seeded_chunk}")
    assert r.status_code == 200
    data = r.json()
    assert data["chunk"]["chunk_id"] == seeded_chunk
    assert data["workspace_id"] == "test-ws"


# ---------------------------------------------------------------------------
# GET /memory/explain/{chunk_id}
# ---------------------------------------------------------------------------

def test_memory_explain_404(client):
    r = client.get("/memory/explain/does-not-exist")
    assert r.status_code == 404


def test_memory_explain_found(client, seeded_chunk):
    r = client.get(f"/memory/explain/{seeded_chunk}")
    assert r.status_code == 200
    data = r.json()
    assert data["chunk_id"] == seeded_chunk
    assert "memory_key" in data
    assert data["source_id"] == "src_test"


# ---------------------------------------------------------------------------
# GET /memory/chunks/{source_id}
# ---------------------------------------------------------------------------

def test_memory_chunks_by_source_empty(client):
    r = client.get("/memory/chunks/nonexistent-source")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 0
    assert data["chunks"] == []


def test_memory_chunks_by_source_found(client, seeded_chunk):
    r = client.get("/memory/chunks/src_test")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 1
    assert data["chunks"][0]["chunk_id"] == seeded_chunk


# ---------------------------------------------------------------------------
# POST /memory/invalidate
# ---------------------------------------------------------------------------

def test_memory_invalidate_returns_count(client):
    r = client.post("/memory/invalidate", json={"source_id": "nope"})
    assert r.status_code == 200
    data = r.json()
    assert data["deactivated_count"] == 0
    assert data["source_id"] == "nope"


def test_memory_invalidate_deactivates_chunks(client, seeded_chunk):
    r = client.post("/memory/invalidate", json={"source_id": "src_test"})
    assert r.status_code == 200
    data = r.json()
    assert data["deactivated_count"] == 1


# ---------------------------------------------------------------------------
# POST /memory/feedback
# ---------------------------------------------------------------------------

def test_memory_feedback_records(client, seeded_chunk):
    r = client.post("/memory/feedback", json={
        "chunk_id": seeded_chunk, "signal": "positive", "metadata": {"label": "relevant"}
    })
    assert r.status_code == 200
    assert r.json()["recorded"] is True


def test_memory_feedback_rejects_invalid_signal(client):
    r = client.post("/memory/feedback", json={"chunk_id": "x", "signal": "bogus"})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# POST /memory/reindex
# ---------------------------------------------------------------------------

def test_memory_reindex_empty_db(client):
    r = client.post("/memory/reindex", json={})
    assert r.status_code == 200
    data = r.json()
    assert data["reindexed_count"] == 0
    assert data["errors"] == 0
