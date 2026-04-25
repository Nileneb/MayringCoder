"""RS256 JWT validation tests."""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock

import anyio
import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from src.api import jwt_auth


@pytest.fixture(scope="module")
def rsa_keys():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private_pem, public_pem


@pytest.fixture
def configured_env(rsa_keys, tmp_path: Path, monkeypatch):
    private_pem, public_pem = rsa_keys
    key_file = tmp_path / "jwt_public.pem"
    key_file.write_text(public_pem)
    monkeypatch.setenv("JWT_PUBLIC_KEY_PATH", str(key_file))
    monkeypatch.setenv("JWT_ISSUER", "https://app.linn.games")
    monkeypatch.setenv("JWT_AUDIENCE", "mayringcoder")
    jwt_auth.reset_public_key_cache()
    yield private_pem
    jwt_auth.reset_public_key_cache()


_MCP_SCOPE = ["mcp:memory"]


def _sign(private_pem: str, **claims) -> str:
    claims.setdefault("scope", _MCP_SCOPE)
    return pyjwt.encode(claims, private_pem, algorithm="RS256")


# ---------------------------------------------------------------------------
# validate_jwt_token()
# ---------------------------------------------------------------------------

def test_valid_tenant_token(configured_env):
    token = _sign(
        configured_env,
        iss="https://app.linn.games",
        aud="mayringcoder",
        exp=int(time.time()) + 300,
        workspace_id="bene-workspace",
    )
    info = jwt_auth.validate_jwt_token(token)
    assert info is not None
    assert info.workspace_id == "bene-workspace"
    assert info.is_admin is False


def test_admin_scope_list(configured_env):
    token = _sign(
        configured_env,
        iss="https://app.linn.games",
        aud="mayringcoder",
        exp=int(time.time()) + 300,
        workspace_id="bene-workspace",
        scope=["mcp:memory", "admin"],
    )
    info = jwt_auth.validate_jwt_token(token)
    assert info is not None
    assert info.is_admin is True


def test_missing_mcp_memory_scope_rejected(configured_env):
    token = _sign(
        configured_env,
        iss="https://app.linn.games",
        aud="mayringcoder",
        exp=int(time.time()) + 300,
        workspace_id="bene-workspace",
        scope=["paper-search:read"],
    )
    assert jwt_auth.validate_jwt_token(token) is None


def test_admin_scope_space_string(configured_env):
    token = _sign(
        configured_env,
        iss="https://app.linn.games",
        aud="mayringcoder",
        exp=int(time.time()) + 300,
        workspace_id="bene-workspace",
        scope="mcp:memory admin read",
    )
    info = jwt_auth.validate_jwt_token(token)
    assert info is not None
    assert info.is_admin is True
    assert "read" in info.scopes


def test_expired_token(configured_env):
    token = _sign(
        configured_env,
        iss="https://app.linn.games",
        aud="mayringcoder",
        exp=int(time.time()) - 1,
        workspace_id="bene-workspace",
    )
    assert jwt_auth.validate_jwt_token(token) is None


def test_missing_workspace_id(configured_env):
    token = _sign(
        configured_env,
        iss="https://app.linn.games",
        aud="mayringcoder",
        exp=int(time.time()) + 300,
    )
    assert jwt_auth.validate_jwt_token(token) is None


def test_empty_workspace_id(configured_env):
    token = _sign(
        configured_env,
        iss="https://app.linn.games",
        aud="mayringcoder",
        exp=int(time.time()) + 300,
        workspace_id="   ",
    )
    assert jwt_auth.validate_jwt_token(token) is None


def test_wrong_issuer(configured_env):
    token = _sign(
        configured_env,
        iss="https://evil.example",
        aud="mayringcoder",
        exp=int(time.time()) + 300,
        workspace_id="bene-workspace",
    )
    assert jwt_auth.validate_jwt_token(token) is None


def test_wrong_audience(configured_env):
    token = _sign(
        configured_env,
        iss="https://app.linn.games",
        aud="other-service",
        exp=int(time.time()) + 300,
        workspace_id="bene-workspace",
    )
    assert jwt_auth.validate_jwt_token(token) is None


def test_tampered_signature(configured_env):
    token = _sign(
        configured_env,
        iss="https://app.linn.games",
        aud="mayringcoder",
        exp=int(time.time()) + 300,
        workspace_id="bene-workspace",
    )
    tampered = token[:-4] + "AAAA"
    assert jwt_auth.validate_jwt_token(tampered) is None


def test_no_public_key_configured(monkeypatch):
    monkeypatch.delenv("JWT_PUBLIC_KEY_PATH", raising=False)
    jwt_auth.reset_public_key_cache()
    try:
        assert jwt_auth.validate_jwt_token("any.token.here") is None
    finally:
        jwt_auth.reset_public_key_cache()


def test_missing_exp_claim(configured_env):
    token = _sign(
        configured_env,
        iss="https://app.linn.games",
        aud="mayringcoder",
        workspace_id="bene-workspace",
    )
    assert jwt_auth.validate_jwt_token(token) is None


def test_empty_token_rejected(configured_env):
    assert jwt_auth.validate_jwt_token("") is None


def test_workspace_id_trimmed(configured_env):
    token = _sign(
        configured_env,
        iss="https://app.linn.games",
        aud="mayringcoder",
        exp=int(time.time()) + 300,
        workspace_id="  bene-workspace  ",
    )
    info = jwt_auth.validate_jwt_token(token)
    assert info is not None
    assert info.workspace_id == "bene-workspace"


# ---------------------------------------------------------------------------
# _JWTAuthMiddleware (HTTP ASGI)
# ---------------------------------------------------------------------------

def _make_scope(scope_type: str = "http", headers: dict[bytes, bytes] | None = None) -> dict:
    return {"type": scope_type, "headers": list((headers or {}).items())}


async def _drive(mw, scope_type="http", token=None, bearer=False):
    headers: dict[bytes, bytes] = {}
    if token is not None:
        if bearer:
            headers[b"authorization"] = f"Bearer {token}".encode()
        else:
            headers[b"x-auth-token"] = token.encode()
    scope = _make_scope(scope_type, headers)
    downstream = AsyncMock()
    mw._app = downstream
    sent: list[dict] = []

    async def _send(msg):
        sent.append(msg)

    await mw(scope, AsyncMock(), _send)
    return sent, downstream, scope


def _load_middleware(monkeypatch, enabled: bool):
    import src.api.mcp as mod
    import src.api.mcp_auth as auth_mod
    monkeypatch.setattr(mod, "_AUTH_ENABLED", enabled)
    monkeypatch.setattr(auth_mod, "_AUTH_ENABLED", enabled)
    from src.api.mcp import _JWTAuthMiddleware
    return _JWTAuthMiddleware(AsyncMock())


class TestMiddlewareEnabled:
    def test_valid_token_passes(self, configured_env, monkeypatch):
        mw = _load_middleware(monkeypatch, enabled=True)
        token = _sign(
            configured_env,
            iss="https://app.linn.games",
            aud="mayringcoder",
            exp=int(time.time()) + 300,
            workspace_id="ws_abc",
        )

        async def _run():
            sent, downstream, scope = await _drive(mw, token=token)
            assert sent == []
            downstream.assert_awaited_once()
            assert scope["workspace_id"] == "ws_abc"

        anyio.run(_run)

    def test_bearer_header(self, configured_env, monkeypatch):
        mw = _load_middleware(monkeypatch, enabled=True)
        token = _sign(
            configured_env,
            iss="https://app.linn.games",
            aud="mayringcoder",
            exp=int(time.time()) + 300,
            workspace_id="ws_bearer",
        )

        async def _run():
            sent, downstream, scope = await _drive(mw, token=token, bearer=True)
            assert sent == []
            assert scope["workspace_id"] == "ws_bearer"

        anyio.run(_run)

    def test_missing_token_401(self, configured_env, monkeypatch):
        mw = _load_middleware(monkeypatch, enabled=True)

        async def _run():
            sent, downstream, _ = await _drive(mw, token=None)
            downstream.assert_not_awaited()
            assert sent[0]["status"] == 401

        anyio.run(_run)

    def test_invalid_token_401(self, configured_env, monkeypatch):
        mw = _load_middleware(monkeypatch, enabled=True)

        async def _run():
            sent, downstream, _ = await _drive(mw, token="garbage")
            downstream.assert_not_awaited()
            assert sent[0]["status"] == 401

        anyio.run(_run)

    def test_non_http_passes(self, configured_env, monkeypatch):
        mw = _load_middleware(monkeypatch, enabled=True)

        async def _run():
            sent, downstream, _ = await _drive(mw, scope_type="websocket")
            assert sent == []
            downstream.assert_awaited_once()

        anyio.run(_run)


class TestMiddlewareDisabled:
    def test_no_auth_passes_all(self, monkeypatch):
        mw = _load_middleware(monkeypatch, enabled=False)

        async def _run():
            sent, downstream, _ = await _drive(mw, token=None)
            assert sent == []
            downstream.assert_awaited_once()

        anyio.run(_run)
