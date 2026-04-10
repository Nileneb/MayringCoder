"""Tests for JWT authentication middleware."""
from __future__ import annotations

import importlib
import os
import time
from unittest.mock import AsyncMock

import anyio
import jwt
import pytest

SECRET = "test-secret-key-for-jwt"


def _make_token(workspace_id="default", scope="repo", exp_offset=3600):
    return jwt.encode(
        {
            "workspace_id": workspace_id,
            "scope": scope,
            "exp": int(time.time()) + exp_offset,
        },
        SECRET,
        algorithm="HS256",
    )


def _make_scope(scope_type: str = "http", headers: dict[bytes, bytes] | None = None) -> dict:
    return {"type": scope_type, "headers": list((headers or {}).items())}


async def _drive_jwt(middleware, scope_type: str = "http", token: str | None = None, bearer: bool = False):
    """Drive JWT middleware; return (sent_messages, downstream_mock, scope)."""
    headers: dict[bytes, bytes] = {}
    if token is not None:
        if bearer:
            headers[b"authorization"] = f"Bearer {token}".encode()
        else:
            headers[b"x-auth-token"] = token.encode()

    scope = _make_scope(scope_type, headers)
    receive = AsyncMock()
    downstream = AsyncMock()
    middleware._app = downstream

    sent: list[dict] = []

    async def _send(msg):
        sent.append(msg)

    await middleware(scope, receive, _send)
    return sent, downstream, scope


# ---------------------------------------------------------------------------
# TestJWTMiddleware — JWT auth enabled
# ---------------------------------------------------------------------------

class TestJWTMiddleware:
    """When MCP_AUTH_ENABLED=true and MCP_AUTH_SECRET is set, JWT is validated."""

    def _get_middleware(self, monkeypatch):
        monkeypatch.setenv("MCP_AUTH_ENABLED", "true")
        monkeypatch.setenv("MCP_AUTH_SECRET", SECRET)
        monkeypatch.delenv("MCP_AUTH_TOKEN", raising=False)
        import src.mcp_server as mod
        monkeypatch.setattr(mod, "_AUTH_ENABLED", True)
        monkeypatch.setattr(mod, "_AUTH_SECRET", SECRET)
        monkeypatch.setattr(mod, "_AUTH_TOKEN", "")
        from src.mcp_server import _JWTAuthMiddleware
        return _JWTAuthMiddleware(AsyncMock())

    def test_valid_token_passes_through(self, monkeypatch):
        mw = self._get_middleware(monkeypatch)
        token = _make_token(workspace_id="ws_123")

        async def _run():
            sent, downstream, scope = await _drive_jwt(mw, token=token)
            assert sent == []
            downstream.assert_awaited_once()
            assert scope["workspace_id"] == "ws_123"

        anyio.run(_run)

    def test_bearer_header_also_accepted(self, monkeypatch):
        mw = self._get_middleware(monkeypatch)
        token = _make_token(workspace_id="ws_bearer")

        async def _run():
            sent, downstream, scope = await _drive_jwt(mw, token=token, bearer=True)
            assert sent == []
            downstream.assert_awaited_once()
            assert scope["workspace_id"] == "ws_bearer"

        anyio.run(_run)

    def test_missing_token_returns_401(self, monkeypatch):
        mw = self._get_middleware(monkeypatch)

        async def _run():
            sent, downstream, scope = await _drive_jwt(mw, token=None)
            downstream.assert_not_awaited()
            assert len(sent) == 2
            assert sent[0]["type"] == "http.response.start"
            assert sent[0]["status"] == 401

        anyio.run(_run)

    def test_expired_token_returns_401(self, monkeypatch):
        mw = self._get_middleware(monkeypatch)
        token = _make_token(exp_offset=-10)  # expired 10 seconds ago

        async def _run():
            sent, downstream, scope = await _drive_jwt(mw, token=token)
            downstream.assert_not_awaited()
            assert sent[0]["status"] == 401

        anyio.run(_run)

    def test_invalid_signature_returns_401(self, monkeypatch):
        mw = self._get_middleware(monkeypatch)
        token = _make_token()
        # Sign with a different secret
        bad_token = jwt.encode(
            {"workspace_id": "ws_evil", "exp": int(time.time()) + 3600},
            "wrong-secret",
            algorithm="HS256",
        )

        async def _run():
            sent, downstream, scope = await _drive_jwt(mw, token=bad_token)
            downstream.assert_not_awaited()
            assert sent[0]["status"] == 401

        anyio.run(_run)

    def test_non_http_scope_passes_through(self, monkeypatch):
        mw = self._get_middleware(monkeypatch)

        async def _run():
            sent, downstream, scope = await _drive_jwt(mw, scope_type="websocket")
            assert sent == []
            downstream.assert_awaited_once()

        anyio.run(_run)


# ---------------------------------------------------------------------------
# TestJWTMiddlewareDisabled — auth disabled
# ---------------------------------------------------------------------------

class TestJWTMiddlewareDisabled:
    """When MCP_AUTH_ENABLED=false, all requests pass and workspace_id='default'."""

    def _get_middleware(self, monkeypatch):
        import src.mcp_server as mod
        monkeypatch.setattr(mod, "_AUTH_ENABLED", False)
        monkeypatch.setattr(mod, "_AUTH_SECRET", "")
        monkeypatch.setattr(mod, "_AUTH_TOKEN", "")
        from src.mcp_server import _JWTAuthMiddleware
        return _JWTAuthMiddleware(AsyncMock())

    def test_disabled_auth_passes_all_requests(self, monkeypatch):
        mw = self._get_middleware(monkeypatch)

        async def _run():
            # No token at all — should pass
            sent, downstream, scope = await _drive_jwt(mw, token=None)
            assert sent == []
            downstream.assert_awaited_once()
            assert scope.get("workspace_id") == "default"

        anyio.run(_run)
