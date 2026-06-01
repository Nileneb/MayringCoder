"""Org-workspace chicken-egg: the FIRST write to an org workspace must derive
visibility='org' and provision the local team row — even though the local
`workspaces` table has no row yet (workspace_kind='unknown'). The JWT membership
is authoritative. Before the fix the default fell back to 'private', so
ensure_team_workspace never fired and team memory never worked.
"""
from __future__ import annotations

from unittest.mock import patch

from src.api.jwt_auth import Membership, TokenInfo

_ORG = "019e4cfd-6c98-7034-a0ea-4e466e3f9847"


def _run(info):
    from fastapi.testclient import TestClient
    from src.api import auth as auth_module
    from src.api import server as srv

    async def _fake_ws():
        return _ORG

    async def _fake_info():
        return info

    captured: dict = {}
    provisioned: list[str] = []

    def _fake_ingest(source_dict, *a, **k):
        captured.update(source_dict)
        return {"source_id": source_dict["source_id"], "state": "new", "chunk_ids": [], "indexed": 0}

    srv.app.dependency_overrides[auth_module.get_workspace] = _fake_ws
    srv.app.dependency_overrides[auth_module.get_token_info] = _fake_info
    try:
        with (
            patch("src.api.routes.memory._run_ingest", side_effect=_fake_ingest),
            patch("mayring_core.identity.workspace_resolver.workspace_kind", return_value="unknown"),
            patch("mayring_core.identity.workspace_resolver.ensure_team_workspace",
                  side_effect=lambda conn, oid, **k: provisioned.append(oid)),
            patch("src.api.authz_helpers.caller_can", return_value=True),
        ):
            r = TestClient(srv.app).post(
                "/memory/put",
                json={"source_id": "note:o", "source_type": "note", "content": "team note"},
                headers={"Authorization": "Bearer tst"},
            )
        return r, captured, provisioned
    finally:
        srv.app.dependency_overrides.clear()


def test_first_org_write_derives_org_and_provisions_team_row():
    info = TokenInfo(
        workspace_id=_ORG, scopes=("mcp:memory",), sub="1",
        memberships=(Membership(id=_ORG, type="organization", role="admin"),),
        active_workspace_kind="organization",
    )
    r, captured, provisioned = _run(info)

    assert r.status_code == 200, r.text
    assert captured["visibility"] == "org"
    assert captured["org_id"] == _ORG
    assert provisioned == [_ORG], "org workspace must be provisioned on first write"


def test_personal_write_stays_private_not_org():
    """Guard: a personal-membership caller must NOT be flipped to org."""
    info = TokenInfo(
        workspace_id=_ORG, scopes=("mcp:memory",), sub="1",
        memberships=(Membership(id="bene", type="personal", role="viewer"),),
        active_workspace_kind="personal",
    )
    r, captured, provisioned = _run(info)

    assert r.status_code == 200, r.text
    assert captured["visibility"] == "private"
    assert provisioned == []
