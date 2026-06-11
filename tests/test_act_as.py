"""Admin/service-only act-as identity override (V2 org-memory acceptance harness)."""
import asyncio


def _maybe_run(c):
    return asyncio.run(c) if asyncio.iscoroutine(c) else c
import os
from unittest.mock import patch

import pytest
from fastapi.security import HTTPAuthorizationCredentials

from src.api.auth import get_token_info
from src.api.jwt_auth import TokenInfo


def _run(coro):
    # _maybe_run() each call: robust to import side-effects that close the
    # default loop (a2a-sdk does this on import). Python 3.13 also raises from
    # get_event_loop() outside a running loop. Same pattern as test_dashboard_endpoints.
    return _maybe_run(coro)


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


def test_act_as_allowed_for_admin_jwt(monkeypatch):
    """The privileged caller may be an admin JWT (scope 'admin'), not only the
    service token ('*'). Both arms of _is_privileged must permit act-as."""
    monkeypatch.setattr("src.api.auth._SERVICE_TOKEN", "svc")
    monkeypatch.setenv("MAYRING_ALLOW_ACT_AS", "1")
    admin = TokenInfo(workspace_id="ws-admin", scopes=("mcp:memory", "admin"), sub="1")
    with patch("src.api.auth.validate_jwt_token", return_value=admin):
        info = _run(get_token_info(creds=_creds("admin-jwt"), x_act_as_sub="99",
                                   x_act_as_orgs="org-z", x_act_as_workspace="ws-z"))
    assert info.sub == "99"
    assert info.workspace_id == "ws-z"
    assert set(info.org_ids) == {"org-z"}
    assert info.is_admin is False  # synthetic identity is downgraded


def test_act_as_workspace_only_keeps_real_sub(service_token):
    """Spec: 'at least one header' triggers the override. With only a workspace
    header, sub falls back to the real caller's sub and orgs stay empty."""
    info = _run(get_token_info(creds=_creds("svc"), x_act_as_workspace="ws-only"))
    assert info.workspace_id == "ws-only"
    assert info.org_ids == ()
    assert "*" not in info.scopes  # still downgraded


def test_act_as_orgs_cannot_inject_privilege(service_token):
    """Defense-in-depth: org ids and scopes are orthogonal axes. A crafted
    X-Act-As-Orgs containing '*'/'admin' lands as literal org-bucket ids and
    must NEVER raise the synthetic identity's privilege."""
    info = _run(get_token_info(
        creds=_creds("svc"),
        x_act_as_sub="42",
        x_act_as_orgs="*,admin,org-real",
        x_act_as_workspace="ws-x",
    ))
    assert info.is_admin is False
    assert "*" not in info.scopes
    assert "admin" not in info.scopes
    # the literal strings are confined to org_ids, never privilege
    assert set(info.org_ids) == {"*", "admin", "org-real"}
