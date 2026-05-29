"""Tests for the 6 new /memory/* HTTP endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from src.api.auth import get_workspace
import src.api.dependencies as _deps


def _fake_ws():
    return "test-ws"


@pytest.fixture(autouse=True)
def _override_workspace():
    """Re-apply per test instead of once at import: another module's
    app.dependency_overrides.clear() (run before this file) would otherwise
    wipe the override, the real get_workspace would demand a JWT, and these
    token-free tests would 401. Order-independent now."""
    app.dependency_overrides[get_workspace] = _fake_ws
    yield
    app.dependency_overrides.pop(get_workspace, None)


@pytest.fixture(autouse=True)
def _reset_conn(tmp_path, monkeypatch):
    """Each test gets a fresh SQLite DB with check_same_thread=False for async TestClient."""
    from mayring_core.memory.db_adapter import DBAdapter
    from mayring_core.memory.store import _init_schema
    db_path = tmp_path / "test.db"
    adapter = DBAdapter.create(db_path, check_same_thread=False)
    _init_schema(adapter)
    monkeypatch.setattr(_deps, "_conn", adapter)
    yield adapter
    monkeypatch.setattr(_deps, "_conn", None)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def seeded_chunk(_reset_conn):
    """Insert a source + chunk into the test DB and return the chunk_id."""
    from mayring_core.memory.store import upsert_source, insert_chunk
    from mayring_core.memory.schema import Source, Chunk
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
        "chunk_id": seeded_chunk, "signal": "5", "metadata": {"label": "relevant"}
    })
    assert r.status_code == 200
    assert r.json()["recorded"] is True


def test_memory_feedback_rejects_invalid_signal(client):
    """Bogus + 'neutral' both rejected — 422 from pydantic Literal validator
    (preferred) or 400 from the in-route check, depending on which fires
    first. Both are 'rejected', the contract is binary-only."""
    for bad in ("bogus", "neutral"):
        r = client.post("/memory/feedback", json={"chunk_id": "x", "signal": bad})
        assert r.status_code in (400, 422), f"signal={bad!r} accepted (got {r.status_code})"


# ---------------------------------------------------------------------------
# POST /memory/reindex
# ---------------------------------------------------------------------------

def test_memory_reindex_empty_db(client):
    r = client.post("/memory/reindex", json={})
    assert r.status_code == 200
    data = r.json()
    assert data["reindexed_count"] == 0
    assert data["errors"] == 0


# ---------------------------------------------------------------------------
# POST /memory/put — igio_hint tagging (session→IGIO #2)
# ---------------------------------------------------------------------------

def test_memory_put_applies_igio_hint(client, _reset_conn, monkeypatch):
    """/memory/put with igio_hint='goal' MUST set igio_axis='goal' on the new chunk.

    Regression: the igio_hint tagging block existed only in /conversation/micro-batch,
    so the stop_hook's native-/goal capture POST to /memory/put was silently ignored
    (chunk got no axis → never appeared in /memory/goals or the IGIO-lens)."""
    import src.api.routes.memory as mem
    from src.api.auth import TokenInfo, get_token_info
    from mayring_core.memory.schema import Chunk, Source
    from mayring_core.memory.store import insert_chunk, upsert_source
    conn = _reset_conn

    app.dependency_overrides[get_token_info] = lambda: TokenInfo(workspace_id="test-ws", scopes=("*",))

    def _fake_ingest(source_dict, content, *a, **k):
        sid = source_dict["source_id"]
        upsert_source(conn, Source(source_id=sid, source_type=source_dict.get("source_type", "note"),
                                   repo="", path="", content_hash="h:" + sid))
        insert_chunk(conn, Chunk(chunk_id="chk_goalcap", source_id=sid, text=content,
                                 text_hash=Chunk.compute_text_hash(content)))
        return {"source_id": sid, "state": "new", "chunk_ids": ["chk_goalcap"]}

    monkeypatch.setattr(mem, "_run_ingest", _fake_ingest)
    try:
        r = client.post("/memory/put", json={
            "content": "make IGIO reflect the real session",
            "source_id": "session_goal:t1",
            "source_type": "session_goal",
            "igio_hint": "goal",
        })
        assert r.status_code == 200, r.text
        axis = conn.execute("SELECT igio_axis FROM chunks WHERE chunk_id='chk_goalcap'").fetchone()[0]
        assert axis == "goal"
    finally:
        app.dependency_overrides.pop(get_token_info, None)
