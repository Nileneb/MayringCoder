"""Admin/service-only act-as identity override (V2 org-memory acceptance harness)."""
import asyncio
import os
from unittest.mock import patch

import pytest
from fastapi.security import HTTPAuthorizationCredentials

from src.api.auth import get_token_info
from src.api.jwt_auth import TokenInfo


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _creds(tok="svc"):
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=tok)


@pytest.fixture
def service_token(monkeypatch):
    monkeypatch.setattr("src.api.auth._SERVICE_TOKEN", "svc")
    monkeypatch.setenv("MAYRING_ALLOW_ACT_AS", "1")
    yield


def test_act_as_builds_synthetic_nonadmin_identity(service_token):
    info = _run(get_token_info(
        creds=_creds("svc"),
        x_act_as_sub="42",
        x_act_as_orgs="org-a,org-b",
        x_act_as_workspace="ws-alice",
    ))
    assert info.sub == "42"
    assert info.workspace_id == "ws-alice"
    assert set(info.org_ids) == {"org-a", "org-b"}
    assert info.is_admin is False
    assert "*" not in info.scopes


def test_act_as_ignored_when_flag_off(service_token, monkeypatch):
    monkeypatch.delenv("MAYRING_ALLOW_ACT_AS", raising=False)
    info = _run(get_token_info(creds=_creds("svc"), x_act_as_sub="42",
                               x_act_as_orgs="org-a", x_act_as_workspace="ws-alice"))
    assert info.sub != "42"
    assert "*" in info.scopes


def test_act_as_ignored_for_non_privileged_token(monkeypatch):
    monkeypatch.setattr("src.api.auth._SERVICE_TOKEN", "svc")
    monkeypatch.setenv("MAYRING_ALLOW_ACT_AS", "1")
    fake = TokenInfo(workspace_id="ws-bob", scopes=("mcp:memory",), sub="7")
    with patch("src.api.auth.validate_jwt_token", return_value=fake):
        info = _run(get_token_info(creds=_creds("user-jwt"), x_act_as_sub="42",
                                   x_act_as_orgs="org-a", x_act_as_workspace="ws-alice"))
    assert info.sub == "7"
    assert info.workspace_id == "ws-bob"


def test_no_act_as_headers_is_passthrough(service_token):
    info = _run(get_token_info(creds=_creds("svc")))
    assert "*" in info.scopes
