"""Task 6 (tenancy phase B) — role→permission capping in POST /memory/put.

The 'write' permission gates any ingest; promoting a source to org/public
additionally requires share_org / share_public. A plain 'user' role (per
DEFAULT_MATRIX) has search+write but NOT share_* — so it may ingest a
private source but must be 403'd when it tries to write a public one.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.api.jwt_auth import Membership, TokenInfo


@pytest.fixture
def client_user():
    """Caller with a plain 'user' membership in workspace 'bene'."""
    from fastapi.testclient import TestClient
    from src.api import auth as auth_module
    from src.api import server as srv

    async def _fake_ws():
        return "bene"

    info = TokenInfo(
        workspace_id="bene",
        scopes=("mcp:memory",),
        sub="1",
        memberships=(Membership(id="bene", type="personal", role="viewer"),),
    )

    async def _fake_info():
        return info

    srv.app.dependency_overrides[auth_module.get_workspace] = _fake_ws
    srv.app.dependency_overrides[auth_module.get_token_info] = _fake_info
    yield TestClient(srv.app)
    srv.app.dependency_overrides.clear()


def _put(client, **kw):
    return client.post("/memory/put", json=kw, headers={"Authorization": "Bearer tst"})


def test_user_cannot_write_public(client_user):
    """A 'user' role lacks share_public → visibility='public' must be 403."""
    with patch(
        "src.api.routes.memory._run_ingest",
        return_value={"source_id": "note:p", "state": "new", "chunk_ids": [], "indexed": 0},
    ):
        r = _put(client_user, source_id="note:p", source_type="note",
                 content="t", visibility="public")
    assert r.status_code == 403, r.text
    assert "share_public" in r.json()["detail"]


def test_service_token_wildcard_can_write_public():
    """Self-lockout guard: the MCP_SERVICE_TOKEN scope='*' must bypass the role
    gate (info.is_admin) so system ingests (repo-events/ambient) keep working."""
    from fastapi.testclient import TestClient
    from src.api import auth as auth_module
    from src.api import server as srv

    info = TokenInfo(workspace_id="system", scopes=("*",), sub=None)

    async def _fake_ws():
        return "system"

    async def _fake_info():
        return info

    srv.app.dependency_overrides[auth_module.get_workspace] = _fake_ws
    srv.app.dependency_overrides[auth_module.get_token_info] = _fake_info
    try:
        with patch(
            "src.api.routes.memory._run_ingest",
            return_value={"source_id": "note:s", "state": "new", "chunk_ids": [], "indexed": 0},
        ):
            r = _put(TestClient(srv.app), source_id="note:s", source_type="note",
                     content="t", visibility="public")
        assert r.status_code == 200, r.text
    finally:
        srv.app.dependency_overrides.clear()


def test_user_can_write_private(client_user):
    """A 'user' role HAS write → visibility='private' must succeed (200)."""
    with (
        patch(
            "src.api.routes.memory._run_ingest",
            return_value={"source_id": "note:q", "state": "new", "chunk_ids": [], "indexed": 0},
        ),
        patch("mayring_core.identity.workspace_resolver.workspace_kind", return_value="user"),
    ):
        r = _put(client_user, source_id="note:q", source_type="note",
                 content="t", visibility="private")
    assert r.status_code == 200, r.text
