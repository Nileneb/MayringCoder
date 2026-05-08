"""Issue #138 fix: /memory/feedback accepts source_id slugs (Solution B).

Inject blocks expose source_id strings ("github-issue:repo:slug",
"conversation:foo:abc...") not the opaque chunk_id. The original API
demanded chunk_id, forcing a search round-trip. Now the route resolves
slug→chunks and applies the signal to every active chunk of that source.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from src.api import auth as auth_mod
from src.api.jwt_auth import TokenInfo
from src.memory.schema import Chunk, Source
from src.memory.store import init_memory_db, upsert_source, insert_chunk


@pytest.fixture
def db_with_source(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    conn = init_memory_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    src = Source(
        source_id="github-issue:demo:slug-feedback",
        source_type="github_issue",
        repo="demo",
        path="issue/1",
        captured_at=now,
    )
    upsert_source(conn, src, workspace_id="user-2")
    for ord_ in range(3):
        insert_chunk(
            conn,
            Chunk(
                chunk_id=f"chk_slug_{ord_}",
                source_id=src.source_id,
                chunk_level="block",
                ordinal=ord_,
                start_offset=0,
                end_offset=10,
                text=f"text-{ord_}",
                text_hash=f"sha256:t{ord_}",
                created_at=now,
                is_active=True,
            ),
            workspace_id="user-2",
        )
    conn.commit()

    # Patch the connection accessor used by the route + skip auth
    from src.api.routes import memory as memory_route
    monkeypatch.setattr(memory_route, "_get_conn", lambda: conn)

    async def _fake_token():
        return TokenInfo(workspace_id="user-2", scopes=())
    app.dependency_overrides[auth_mod.get_token_info] = _fake_token
    yield conn
    app.dependency_overrides.pop(auth_mod.get_token_info, None)
    conn.close()


def test_feedback_accepts_source_id_slug(db_with_source):
    client = TestClient(app)
    r = client.post(
        "/memory/feedback",
        json={"chunk_id": "github-issue:demo:slug-feedback", "signal": "positive"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["recorded"] is True
    assert body["applied_to"] == 3
    assert set(body["chunk_ids"]) == {"chk_slug_0", "chk_slug_1", "chk_slug_2"}


def test_feedback_with_real_chunk_id_still_works(db_with_source):
    client = TestClient(app)
    r = client.post(
        "/memory/feedback",
        json={"chunk_id": "chk_slug_1", "signal": "negative"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["chunk_id"] == "chk_slug_1"
    assert body["recorded"] is True


def test_feedback_unknown_slug_returns_404(db_with_source):
    client = TestClient(app)
    r = client.post(
        "/memory/feedback",
        json={"chunk_id": "github-issue:nope:never", "signal": "positive"},
    )
    assert r.status_code == 404
    assert "no active chunks" in r.json()["detail"]


def test_feedback_neutral_still_rejected_via_slug(db_with_source):
    """Neutral was killed across the board — slug path must not be a back door."""
    client = TestClient(app)
    r = client.post(
        "/memory/feedback",
        json={"chunk_id": "github-issue:demo:slug-feedback", "signal": "neutral"},
    )
    assert r.status_code in (400, 422)
