"""Task 6 — Allowlist 3 Werte + share/PATCH syncen Chroma-Metadata.

Allowlist: /memory/put AND PATCH /sources/{id}/visibility AND
POST /sources/{id}/share must all reject visibility values outside
{'private', 'org', 'public'}.  'user' and 'workspace' are legacy values
that must produce 422/400.

Chroma-Sync: after a PATCH or share, the chunk metadata in Chroma must be
updated.  We verify that _get_chroma().update() is called with the correct
new visibility — a lightweight mock instead of spinning up an actual Chroma
instance.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from src.api import auth as auth_module
    from src.api import server as srv

    async def _fake_ws():
        return "bene"

    class _FakeInfo:
        org_ids: list = []
        sub = "1"
        scopes: list = []
        workspace_id = "bene"

        def membership_name(self, _org_id):
            return ""

    async def _fake_info():
        return _FakeInfo()

    srv.app.dependency_overrides[auth_module.get_workspace] = _fake_ws
    srv.app.dependency_overrides[auth_module.get_token_info] = _fake_info
    yield TestClient(srv.app)
    srv.app.dependency_overrides.clear()


def _put(client, **kw):
    return client.post("/memory/put", json=kw, headers={"Authorization": "Bearer tst"})


# ---------------------------------------------------------------------------
# Task 6a: Allowlist on /memory/put — 'user' and 'workspace' must be 422
# ---------------------------------------------------------------------------

def test_put_rejects_user_visibility(client):
    r = _put(client, source_id="note:a", source_type="note", content="t", visibility="user")
    assert r.status_code == 422, r.text
    assert "visibility must be one of" in r.json()["detail"]


def test_put_rejects_workspace_visibility(client):
    r = _put(client, source_id="note:b", source_type="note", content="t", visibility="workspace")
    assert r.status_code == 422, r.text
    assert "visibility must be one of" in r.json()["detail"]


def test_put_accepts_private(client):
    """'private' must still be accepted (personal default after T9)."""
    with (
        patch("src.api.routes.memory._run_ingest",
              return_value={"source_id": "note:c", "state": "new",
                            "chunk_ids": [], "indexed": 0}),
        patch("mayring_core.identity.workspace_resolver.workspace_kind",
              return_value="user"),
    ):
        r = _put(client, source_id="note:c", source_type="note", content="t",
                 visibility="private")
    assert r.status_code == 200, r.text


def test_put_accepts_public(client):
    with (
        patch("src.api.routes.memory._run_ingest",
              return_value={"source_id": "note:d", "state": "new",
                            "chunk_ids": [], "indexed": 0}),
        patch("mayring_core.identity.workspace_resolver.workspace_kind",
              return_value="user"),
    ):
        r = _put(client, source_id="note:d", source_type="note", content="t",
                 visibility="public")
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# Task 6b: PATCH /sources/{id}/visibility — 'user' must be 400
# ---------------------------------------------------------------------------

def test_patch_rejects_user_visibility(client):
    r = client.patch(
        "/sources/note:x/visibility",
        json={"visibility": "user"},
        headers={"Authorization": "Bearer tst"},
    )
    assert r.status_code == 400, r.text
    assert "visibility must be" in r.json()["detail"]


def test_patch_rejects_workspace_visibility(client):
    r = client.patch(
        "/sources/note:x/visibility",
        json={"visibility": "workspace"},
        headers={"Authorization": "Bearer tst"},
    )
    assert r.status_code == 400, r.text


# ---------------------------------------------------------------------------
# Task 6c: PATCH syncs Chroma metadata (mocked Chroma + DB row)
# ---------------------------------------------------------------------------

def _make_mock_conn(*, src_ws="bene", src_user="1", chunk_ids=("ck_1", "ck_2")):
    """Build a minimal mock DB connection for the PATCH path."""
    conn = MagicMock()

    def _execute(sql, params=()):
        result = MagicMock()
        if "SELECT workspace_id" in sql:
            row = MagicMock()
            row.__iter__ = lambda self: iter([src_ws, src_user])
            row.keys = lambda: ["workspace_id", "user_id"]
            row.__getitem__ = lambda self, k: (src_ws if k in (0, "workspace_id") else src_user)
            result.fetchone.return_value = row
        elif "SELECT chunk_id FROM chunks" in sql:
            rows = []
            for cid in chunk_ids:
                r = MagicMock()
                r.__getitem__ = lambda self, k, _cid=cid: _cid
                r.keys = lambda: ["chunk_id"]
                rows.append(r)
            result.fetchall.return_value = rows
        else:
            result.fetchone.return_value = None
            result.fetchall.return_value = []
        return result

    conn.execute.side_effect = _execute
    return conn


def test_patch_syncs_chroma_metadata(client):
    """After PATCH, Chroma.update() must be called with the new visibility."""
    mock_conn = _make_mock_conn()
    mock_chroma = MagicMock()

    with (
        patch("src.api.routes.memory._get_conn", return_value=mock_conn),
        patch("src.api.routes.memory._get_chroma", return_value=mock_chroma),
    ):
        r = client.patch(
            "/sources/note:x/visibility",
            json={"visibility": "public"},
            headers={"Authorization": "Bearer tst"},
        )

    assert r.status_code == 200, r.text
    mock_chroma.update.assert_called_once()
    call_kwargs = mock_chroma.update.call_args
    ids_arg = call_kwargs.kwargs.get("ids") or call_kwargs.args[0] if call_kwargs.args else None
    metadatas_arg = (
        call_kwargs.kwargs.get("metadatas")
        or (call_kwargs.args[1] if len(call_kwargs.args or []) > 1 else None)
    )
    assert metadatas_arg is not None, "chroma.update must pass metadatas"
    assert all(m["visibility"] == "public" for m in metadatas_arg)
