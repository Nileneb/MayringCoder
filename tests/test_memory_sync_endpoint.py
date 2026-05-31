"""Tests for GET /memory/changes — chunk export with embeddings."""
from __future__ import annotations

import datetime
import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from src.api.auth import get_token_info, get_workspace
from src.api.jwt_auth import TokenInfo
import src.api.dependencies as _deps


@pytest.fixture(autouse=True)
def _override_workspace():
    prev_ws = app.dependency_overrides.get(get_workspace)
    prev_ti = app.dependency_overrides.get(get_token_info)
    # V2: /memory/changes now reads memberships+sub from TokenInfo to scope
    # the visibility filter. Provide a minimal stub for legacy tests.
    test_info = TokenInfo(workspace_id="ws-test", sub="0", scopes=("mcp:memory",))
    app.dependency_overrides[get_workspace] = lambda: "ws-test"
    app.dependency_overrides[get_token_info] = lambda: test_info
    yield
    if prev_ws is None:
        app.dependency_overrides.pop(get_workspace, None)
    else:
        app.dependency_overrides[get_workspace] = prev_ws
    if prev_ti is None:
        app.dependency_overrides.pop(get_token_info, None)
    else:
        app.dependency_overrides[get_token_info] = prev_ti


@pytest.fixture(autouse=True)
def _reset_conn(tmp_path, monkeypatch):
    from mayring_core.memory.db_adapter import DBAdapter
    from mayring_core.memory.store import _init_schema
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
    from mayring_core.memory.store import upsert_source, insert_chunk
    from mayring_core.memory.schema import Source, Chunk
    # WHY(tenancy phase A): visibility='private' is now user_id-scoped (not
    # workspace-scoped). The /memory/changes caller stub has sub="0", so seed the
    # source with user_id="0" — otherwise the private arm (s.user_id = caller_sub)
    # never matches and the cross-device sync test sees nothing.
    upsert_source(conn, Source(source_id=source_id, source_type="note", repo="r", path="p"),
                  workspace_id=workspace_id, visibility=visibility, user_id="0")
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


def test_changes_handles_chroma_numpy_embeddings(client, _reset_conn, monkeypatch):
    """Regression: chroma >=0.5 liefert numpy.ndarray für embeddings.
    `result.get("embeddings") or []` löste dann
    `ValueError: truth value of an array is ambiguous` aus und der gesamte
    chroma-fetch lief in den except-Block → Embeddings nie an Clients geliefert.
    Nach dem Fix: numpy-arrays werden korrekt verarbeitet, embedding-Field
    enthält die Werte als plain list."""
    import numpy as np

    _seed_chunk(_reset_conn, "chk-np", "src-np", "ws-test",
                created_at="2026-01-01T00:00:03")

    class _FakeChromaCollection:
        def get(self, ids, include):
            # Genau das produziert chroma >=0.5 in production
            return {
                "ids": ids,
                "embeddings": np.array([[0.1, 0.2, 0.3]]),
            }

    import src.api.routes.sync as _sync_mod
    monkeypatch.setattr(_sync_mod, "_get_chroma", lambda: _FakeChromaCollection())

    resp = client.get("/memory/changes",
                      params={"since": "2000-01-01T00:00:00", "workspace_id": "ws-test"})
    assert resp.status_code == 200
    chunks = resp.json()["chunks"]
    np_chunk = next((c for c in chunks if c["chunk_id"] == "chk-np"), None)
    assert np_chunk is not None
    assert np_chunk["embedding"] == [0.1, 0.2, 0.3], \
        "numpy-array embedding muss als plain list zum Client gelangen"


def test_changes_handles_empty_chroma_response(client, _reset_conn, monkeypatch):
    """Edge: chroma liefert None für embeddings (z.B. bei leerer Collection).
    `or []` würde hier zwar funktionieren — aber unsere Defensive (`is not None`)
    muss auch diesen Fall sauber abdecken."""
    _seed_chunk(_reset_conn, "chk-empty", "src-empty", "ws-test",
                created_at="2026-01-01T00:00:04")

    class _NoEmbedCollection:
        def get(self, ids, include):
            return {"ids": ids, "embeddings": None}

    import src.api.routes.sync as _sync_mod
    monkeypatch.setattr(_sync_mod, "_get_chroma", lambda: _NoEmbedCollection())

    resp = client.get("/memory/changes",
                      params={"since": "2000-01-01T00:00:00", "workspace_id": "ws-test"})
    assert resp.status_code == 200
    chunks = resp.json()["chunks"]
    assert any(c["chunk_id"] == "chk-empty" and c["embedding"] is None for c in chunks)
