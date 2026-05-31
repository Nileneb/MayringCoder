"""Task 9 — Ingest-Default nach Workspace-Typ.

When visibility is omitted on /memory/put, the route must derive it from the
workspace kind: personal (kind='user') → private + user_id stamp,
org-artig (kind IN ('team','project','organization')) → org + org_id stamp.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures — mirror pattern from test_scope_key.py
# ---------------------------------------------------------------------------

@pytest.fixture
def client_personal_ws():
    """Personal (kind='user') workspace — expects visibility='private' default."""
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

        def membership_name(self, _org_id):  # noqa: D102
            return ""

    async def _fake_info():
        return _FakeInfo()

    srv.app.dependency_overrides[auth_module.get_workspace] = _fake_ws
    srv.app.dependency_overrides[auth_module.get_token_info] = _fake_info
    yield TestClient(srv.app)
    srv.app.dependency_overrides.clear()


@pytest.fixture
def client_team_ws():
    """Org-artig (kind='team') workspace — expects visibility='org' default."""
    from fastapi.testclient import TestClient
    from src.api import auth as auth_module
    from src.api import server as srv

    _ORG_ID = "org-team-001"

    async def _fake_ws():
        return _ORG_ID

    class _FakeInfo:
        org_ids: list = [_ORG_ID]
        sub = "2"
        scopes: list = []
        workspace_id = _ORG_ID

        def membership_name(self, _org_id):  # noqa: D102
            return ""

    async def _fake_info():
        return _FakeInfo()

    srv.app.dependency_overrides[auth_module.get_workspace] = _fake_ws
    srv.app.dependency_overrides[auth_module.get_token_info] = _fake_info
    yield TestClient(srv.app)
    srv.app.dependency_overrides.clear()


_INGEST_OK = {"source_id": "note:x", "state": "new", "chunk_ids": [], "indexed": 0}


# ---------------------------------------------------------------------------
# Task 9: personal workspace → private
# ---------------------------------------------------------------------------

def test_personal_ws_defaults_to_private(client_personal_ws):
    """No visibility in body → route stamps 'private' + user_id for personal WS."""
    captured: dict = {}

    def _fake_ingest(source_dict, *args, **kwargs):
        captured.update(source_dict)
        return _INGEST_OK

    with (
        patch("src.api.routes.memory._run_ingest", side_effect=_fake_ingest),
        patch(
            "mayring_core.identity.workspace_resolver.workspace_kind",
            return_value="user",
        ),
    ):
        r = client_personal_ws.post(
            "/memory/put",
            json={"source_id": "note:x", "source_type": "note", "content": "hello"},
            headers={"Authorization": "Bearer tst"},
        )

    assert r.status_code == 200, r.text
    assert captured.get("visibility") == "private", captured
    assert captured.get("user_id") == "1", captured


# ---------------------------------------------------------------------------
# Task 9: team workspace → org
# ---------------------------------------------------------------------------

def test_team_ws_defaults_to_org(client_team_ws):
    """No visibility in body → route stamps 'org' + org_id=workspace_id for team WS."""
    captured: dict = {}

    def _fake_ingest(source_dict, *args, **kwargs):
        captured.update(source_dict)
        return _INGEST_OK

    with (
        patch("src.api.routes.memory._run_ingest", side_effect=_fake_ingest),
        patch(
            "mayring_core.identity.workspace_resolver.workspace_kind",
            return_value="team",
        ),
    ):
        r = client_team_ws.post(
            "/memory/put",
            json={"source_id": "note:y", "source_type": "note", "content": "team note"},
            headers={"Authorization": "Bearer tst"},
        )

    assert r.status_code == 200, r.text
    assert captured.get("visibility") == "org", captured
    assert captured.get("org_id") == "org-team-001", captured


# ---------------------------------------------------------------------------
# Task 7: owner-fallback when info.sub is None (Service-Token ingests)
# ---------------------------------------------------------------------------

@pytest.fixture
def client_service_token_personal_ws():
    """Personal (kind='user') workspace, but info.sub=None (Service-Token caller)."""
    from fastapi.testclient import TestClient
    from src.api import auth as auth_module
    from src.api import server as srv

    async def _fake_ws():
        return "bene"

    class _FakeInfoNoSub:
        org_ids: list = []
        sub = None  # Service-Token: no user subject
        scopes: list = ["*"]
        workspace_id = "bene"

        def membership_name(self, _org_id):  # noqa: D102
            return ""

    async def _fake_info():
        return _FakeInfoNoSub()

    srv.app.dependency_overrides[auth_module.get_workspace] = _fake_ws
    srv.app.dependency_overrides[auth_module.get_token_info] = _fake_info
    yield TestClient(srv.app)
    srv.app.dependency_overrides.clear()


def test_service_token_personal_ws_uses_owner_fallback(client_service_token_personal_ws, monkeypatch):
    """No sub on Service-Token + personal-WS → user_id falls back to workspace owner.

    MAYRING_PERSONAL_OWNERS='{"bene":"owner-1"}' → user_id must be "owner-1".
    """
    captured: dict = {}

    def _fake_ingest(source_dict, *args, **kwargs):
        captured.update(source_dict)
        return _INGEST_OK

    monkeypatch.setenv("MAYRING_PERSONAL_OWNERS", '{"bene":"owner-1"}')

    with (
        patch("src.api.routes.memory._run_ingest", side_effect=_fake_ingest),
        patch(
            "mayring_core.identity.workspace_resolver.workspace_kind",
            return_value="user",
        ),
    ):
        r = client_service_token_personal_ws.post(
            "/memory/put",
            json={"source_id": "note:svc", "source_type": "note", "content": "ambient capture"},
            headers={"Authorization": "Bearer svc-token"},
        )

    assert r.status_code == 200, r.text
    assert captured.get("visibility") == "private", captured
    assert captured.get("user_id") == "owner-1", f"expected owner-1, got {captured.get('user_id')!r}"


def test_service_token_no_owner_map_user_id_is_none(client_service_token_personal_ws, monkeypatch):
    """Service-Token + personal-WS + no MAYRING_PERSONAL_OWNERS → user_id=None (best-effort)."""
    captured: dict = {}

    def _fake_ingest(source_dict, *args, **kwargs):
        captured.update(source_dict)
        return _INGEST_OK

    monkeypatch.delenv("MAYRING_PERSONAL_OWNERS", raising=False)

    with (
        patch("src.api.routes.memory._run_ingest", side_effect=_fake_ingest),
        patch(
            "mayring_core.identity.workspace_resolver.workspace_kind",
            return_value="user",
        ),
    ):
        r = client_service_token_personal_ws.post(
            "/memory/put",
            json={"source_id": "note:svc2", "source_type": "note", "content": "ambient capture"},
            headers={"Authorization": "Bearer svc-token"},
        )

    assert r.status_code == 200, r.text
    assert captured.get("visibility") == "private", captured
    assert captured.get("user_id") is None, captured


# ---------------------------------------------------------------------------
# Task 9: explicit visibility is preserved, not overwritten
# ---------------------------------------------------------------------------

def test_explicit_visibility_not_overwritten(client_personal_ws):
    """Explicit visibility='public' must pass through unchanged."""
    captured: dict = {}

    def _fake_ingest(source_dict, *args, **kwargs):
        captured.update(source_dict)
        return _INGEST_OK

    with (
        patch("src.api.routes.memory._run_ingest", side_effect=_fake_ingest),
        patch(
            "mayring_core.identity.workspace_resolver.workspace_kind",
            return_value="user",
        ),
    ):
        r = client_personal_ws.post(
            "/memory/put",
            json={"source_id": "note:z", "source_type": "note", "content": "t",
                  "visibility": "public"},
            headers={"Authorization": "Bearer tst"},
        )

    assert r.status_code == 200, r.text
    assert captured.get("visibility") == "public", captured
