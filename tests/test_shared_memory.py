"""Tests for Shared Memory — visibility (public/org/private) on sources."""
from __future__ import annotations

import datetime
import pytest

from src.memory.db_adapter import DBAdapter
from src.memory.store import _init_schema, upsert_source, insert_chunk
from src.memory.schema import Source, Chunk


def _db() -> DBAdapter:
    adapter = DBAdapter.memory()
    _init_schema(adapter)
    return adapter


# ---------------------------------------------------------------------------
# Task 1 — Schema
# ---------------------------------------------------------------------------

def test_sources_table_has_visibility_column():
    db = _db()
    cols = db.get_columns("sources")
    assert "visibility" in cols
    assert "org_id" in cols


def test_upsert_source_persists_visibility():
    db = _db()
    src = Source(source_id="s1", source_type="note", repo="", path="x")
    upsert_source(db, src, workspace_id="ws1", visibility="public")
    row = db.execute("SELECT visibility, org_id FROM sources WHERE source_id = 's1'").fetchone()
    assert row["visibility"] == "public"
    assert row["org_id"] is None


def test_upsert_source_persists_org_id():
    db = _db()
    src = Source(source_id="s2", source_type="note", repo="", path="x")
    upsert_source(db, src, workspace_id="ws1", visibility="org", org_id="myorg")
    row = db.execute("SELECT visibility, org_id FROM sources WHERE source_id = 's2'").fetchone()
    assert row["visibility"] == "org"
    assert row["org_id"] == "myorg"


def test_migrate_schema_adds_visibility_to_existing_db(tmp_path):
    from src.memory.store import _migrate_schema
    db = DBAdapter.create(tmp_path / "m.db")
    _init_schema(db)
    _migrate_schema(db)  # must not raise
    cols = db.get_columns("sources")
    assert "visibility" in cols


# ---------------------------------------------------------------------------
# Task 2 — JWT / TokenInfo
# ---------------------------------------------------------------------------

from src.api.jwt_auth import TokenInfo


def test_token_info_has_org_id_field():
    ti = TokenInfo(workspace_id="ws1", scopes=("mcp:memory",), org_id="myorg")
    assert ti.org_id == "myorg"


def test_token_info_org_id_defaults_to_none():
    ti = TokenInfo(workspace_id="ws1", scopes=("mcp:memory",))
    assert ti.org_id is None


# ---------------------------------------------------------------------------
# Task 3 — Retrieval
# ---------------------------------------------------------------------------

def _insert_chunk(db: DBAdapter, source_id: str, workspace_id: str,
                  visibility: str = "private", org_id: str | None = None) -> str:
    src = Source(source_id=source_id, source_type="note", repo="r", path="p")
    upsert_source(db, src, workspace_id=workspace_id,
                  visibility=visibility, org_id=org_id)
    chunk_id = f"chk-{source_id}"
    now = datetime.datetime.utcnow().isoformat()
    chunk = Chunk(
        chunk_id=chunk_id, source_id=source_id, text="hello",
        text_hash="h1", dedup_key="d1", created_at=now,
        workspace_id=workspace_id,
    )
    insert_chunk(db, chunk)
    return chunk_id


def test_public_chunk_visible_to_any_workspace():
    from src.memory.retrieval import _scope_filter
    db = _db()
    cid = _insert_chunk(db, "pub-src", "ws-owner", visibility="public")
    results = _scope_filter(db, workspace_id="ws-other", org_id=None)
    assert cid in results


def test_org_chunk_visible_when_org_matches():
    from src.memory.retrieval import _scope_filter
    db = _db()
    cid = _insert_chunk(db, "org-src", "ws-owner", visibility="org", org_id="myorg")
    results = _scope_filter(db, workspace_id="ws-member", org_id="myorg")
    assert cid in results


def test_org_chunk_hidden_when_org_differs():
    from src.memory.retrieval import _scope_filter
    db = _db()
    cid = _insert_chunk(db, "org-src2", "ws-owner", visibility="org", org_id="myorg")
    results = _scope_filter(db, workspace_id="ws-other", org_id="otherorg")
    assert cid not in results


def test_private_chunk_workspace_isolated():
    from src.memory.retrieval import _scope_filter
    db = _db()
    cid = _insert_chunk(db, "priv-src", "ws-owner", visibility="private")
    results = _scope_filter(db, workspace_id="ws-other", org_id=None)
    assert cid not in results
    results_own = _scope_filter(db, workspace_id="ws-owner", org_id=None)
    assert cid in results_own


# ---------------------------------------------------------------------------
# Task 4 — Admin PATCH endpoint
# ---------------------------------------------------------------------------

def test_admin_can_change_visibility():
    from fastapi.testclient import TestClient
    from src.api.server import app
    from src.api.auth import get_workspace
    import src.api.dependencies as _deps

    app.dependency_overrides[get_workspace] = lambda: "test-ws"
    db = _db()
    src = Source(source_id="change-src", source_type="note", repo="", path="x")
    upsert_source(db, src, workspace_id="test-ws", visibility="private")
    _deps._conn = db

    client = TestClient(app)
    resp = client.patch(
        "/sources/change-src/visibility",
        json={"visibility": "public", "org_id": None},
    )
    assert resp.status_code == 200
    assert resp.json()["visibility"] == "public"

    row = db.execute("SELECT visibility FROM sources WHERE source_id = 'change-src'").fetchone()
    assert row["visibility"] == "public"

    app.dependency_overrides.clear()
    _deps._conn = None
