"""#252 — scope_key: typed logical sub-bucket within a workspace.

A Recherche search scoped to project A must never see project B's papers
(same workspace). Paper/agent_result ingests REQUIRE a scope (FAIL-loud,
422 — no silent workspace-global default).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from mayring_core.memory.schema import Source, is_valid_scope_key
from mayring_core.memory.store import (
    CURRENT_SCHEMA_VERSION,
    init_memory_db,
    insert_chunk,
    upsert_source,
)
from mayring_core.memory.retrieval import _scope_filter


# ---------------------------------------------------------------------------
# schema / validator
# ---------------------------------------------------------------------------

def test_schema_version_bumped_and_column_present(tmp_path: Path):
    c = init_memory_db(tmp_path / "m.db")
    assert CURRENT_SCHEMA_VERSION >= 2
    assert "scope_key" in c.get_columns("sources")
    assert c.execute("PRAGMA user_version").fetchone()[0] == CURRENT_SCHEMA_VERSION
    c.close()


@pytest.mark.parametrize("val,ok", [
    ("project:abc-123", True),
    ("repo:https://github.com/x/y", True),
    ("campaign:42", True),
    (None, True),                # None = workspace-global
    ("abc", False),              # untyped
    ("", False),
    ("project:", False),         # type but no id
    ("weird:thing", False),      # unknown type
])
def test_is_valid_scope_key(val, ok):
    assert is_valid_scope_key(val) is ok


def test_upsert_source_roundtrips_scope_key(tmp_path: Path):
    c = init_memory_db(tmp_path / "m.db")
    upsert_source(c, Source(source_id="paper:r1", source_type="agent_result",
                            repo="", path="", scope_key="project:p1"),
                  workspace_id="bene")
    row = c.execute("SELECT scope_key FROM sources WHERE source_id='paper:r1'").fetchone()
    assert row[0] == "project:p1"
    c.close()


# ---------------------------------------------------------------------------
# retrieval isolation
# ---------------------------------------------------------------------------

def _seed_paper(conn, *, source_id, scope_key, chunk_id, text, ws="bene",
               user_id: str | None = None):
    # WHY(tenancy-T7 migration): 3-value visibility model — seeds use 'private'
    # with an explicit user_id so scope_filter can match them.  'public' is used
    # when user_id is absent (scope_key-isolation test; no visibility assertion).
    vis = "private" if user_id else "public"
    upsert_source(conn, Source(source_id=source_id, source_type="agent_result",
                               repo="", path="", scope_key=scope_key,
                               visibility=vis, user_id=user_id,
                               content_hash="h:" + chunk_id), workspace_id=ws)
    from mayring_core.memory.schema import Chunk
    insert_chunk(conn, Chunk(chunk_id=chunk_id, source_id=source_id, text=text,
                             text_hash="th:" + chunk_id), workspace_id=ws)


def test_scope_filter_isolates_projects(tmp_path: Path):
    """scope_key restricts results to the matching logical sub-bucket.

    Chunks are seeded as public (no user_id needed) so _scope_filter returns them
    regardless of caller identity — the assertion under test is scope_key isolation,
    not visibility isolation (which is covered by test_scope_filter_visibility.py).
    """
    c = init_memory_db(tmp_path / "m.db")
    _seed_paper(c, source_id="paper:a", scope_key="project:p1", chunk_id="ck_a", text="alpha")
    _seed_paper(c, source_id="paper:b", scope_key="project:p2", chunk_id="ck_b", text="beta")
    _seed_paper(c, source_id="conv:x", scope_key=None,           chunk_id="ck_x", text="global")
    c.commit()

    # scope_key="project:p1" → only ck_a; user_id=None is fine for public chunks
    only_p1 = set(_scope_filter(c, workspace_id="bene", user_id=None, scope_key="project:p1"))
    assert only_p1 == {"ck_a"}, only_p1

    only_p2 = set(_scope_filter(c, workspace_id="bene", user_id=None, scope_key="project:p2"))
    assert only_p2 == {"ck_b"}, only_p2

    # no scope_key → all public chunks in workspace
    all_ws = set(_scope_filter(c, workspace_id="bene", user_id=None))
    assert all_ws == {"ck_a", "ck_b", "ck_x"}, all_ws
    c.close()


# ---------------------------------------------------------------------------
# /memory/put FAIL-loud
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from src.api import auth as auth_module
    from src.api import server as srv

    async def _fake_ws():
        return "bene"

    # WHY(tenancy phase B): these tests exercise scope_key validation, not the
    # role gate → run as super-admin so caller_can() never blocks the ingest.
    from src.api.jwt_auth import TokenInfo
    info = TokenInfo(workspace_id="bene", scopes=("admin",), sub="1")

    async def _fake_info():
        return info

    srv.app.dependency_overrides[auth_module.get_workspace] = _fake_ws
    srv.app.dependency_overrides[auth_module.get_token_info] = _fake_info
    yield TestClient(srv.app)
    srv.app.dependency_overrides.clear()


def _put(client, **kw):
    return client.post("/memory/put", json=kw, headers={"Authorization": "Bearer tst"})


def test_memory_put_rejects_paper_without_scope(client):
    r = _put(client, source_id="paper:x", source_type="paper", content="some text")
    assert r.status_code == 422
    assert "scope is required" in r.json()["detail"]


def test_memory_put_rejects_untyped_scope(client):
    r = _put(client, source_id="paper:x", source_type="agent_result", content="t", scope="not-prefixed")
    assert r.status_code == 422
    assert "type-prefixed" in r.json()["detail"]


def test_memory_put_rejects_unknown_visibility(client):
    """Regression (tenancy-audit): visibility='workspace' was accepted on write
    but the read scope_filter (private/org/public/user) dropped it → invisible to
    everyone incl. the owner. /memory/put must reject unknown visibility loudly."""
    r = _put(client, source_id="note:x", source_type="note", content="t", visibility="workspace")
    assert r.status_code == 422
    assert "visibility must be one of" in r.json()["detail"]


def test_memory_put_accepts_paper_with_scope(client):
    with patch("src.api.routes.memory._run_ingest", return_value={"source_id": "paper:x", "state": "new",
                                                                  "chunk_ids": [], "indexed": 0}):
        r = _put(client, source_id="paper:x", source_type="agent_result", content="t", scope="project:p1")
    assert r.status_code == 200, r.text


def test_memory_put_repo_file_does_not_need_scope(client):
    with patch("src.api.routes.memory._run_ingest", return_value={"source_id": "repo:x", "state": "new",
                                                                  "chunk_ids": [], "indexed": 0}):
        r = _put(client, source_id="repo:x:f.py", source_type="repo_file", content="code", repo="https://github.com/x/y", path="f.py")
    assert r.status_code == 200, r.text
