"""Tests for _XAuthMiddleware in src/mcp_server.py."""

from __future__ import annotations

from unittest.mock import AsyncMock

import anyio
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scope(scope_type: str = "http", headers: dict[bytes, bytes] | None = None) -> dict:
    """Build a minimal ASGI scope dict."""
    return {"type": scope_type, "headers": list((headers or {}).items())}


async def _drive(middleware, scope_type: str, token: bytes | None = None):
    """Drive middleware and return (sent_messages, downstream_mock)."""
    headers: dict[bytes, bytes] = {}
    if token is not None:
        headers[b"x-auth-token"] = token

    scope = _make_scope(scope_type, headers)
    receive = AsyncMock()
    downstream = AsyncMock()
    middleware._app = downstream

    sent: list[dict] = []

    async def _send(msg):
        sent.append(msg)

    await middleware(scope, receive, _send)
    return sent, downstream


# ---------------------------------------------------------------------------
# Tests — no token set (dev mode)
# ---------------------------------------------------------------------------

class TestXAuthMiddlewareNoToken:
    """When MCP_AUTH_TOKEN is empty, all requests pass through without auth."""

    def test_http_request_passes_without_auth(self, monkeypatch):
        import src.api.mcp as mod
        monkeypatch.setattr(mod, "_AUTH_TOKEN", "")
        from src.api.mcp import _XAuthMiddleware

        async def _run():
            mw = _XAuthMiddleware(AsyncMock())
            sent, downstream = await _drive(mw, "http")
            assert sent == []
            downstream.assert_awaited_once()

        anyio.run(_run)

    def test_non_http_scope_passes_without_auth(self, monkeypatch):
        import src.api.mcp as mod
        monkeypatch.setattr(mod, "_AUTH_TOKEN", "")
        from src.api.mcp import _XAuthMiddleware

        async def _run():
            mw = _XAuthMiddleware(AsyncMock())
            sent, downstream = await _drive(mw, "lifespan")
            assert sent == []
            downstream.assert_awaited_once()

        anyio.run(_run)


# ---------------------------------------------------------------------------
# Tests — token set
# ---------------------------------------------------------------------------

class TestXAuthMiddlewareWithToken:
    """When MCP_AUTH_TOKEN is set, the X-Auth-Token header is required."""

    TOKEN = "super-secret-42"

    def test_correct_token_passes(self, monkeypatch):
        import src.api.mcp as mod
        monkeypatch.setattr(mod, "_AUTH_TOKEN", self.TOKEN)
        from src.api.mcp import _XAuthMiddleware

        async def _run():
            mw = _XAuthMiddleware(AsyncMock())
            sent, downstream = await _drive(mw, "http", self.TOKEN.encode())
            assert sent == []
            downstream.assert_awaited_once()

        anyio.run(_run)

    def test_wrong_token_returns_401(self, monkeypatch):
        import src.api.mcp as mod
        monkeypatch.setattr(mod, "_AUTH_TOKEN", self.TOKEN)
        from src.api.mcp import _XAuthMiddleware

        async def _run():
            mw = _XAuthMiddleware(AsyncMock())
            sent, downstream = await _drive(mw, "http", b"wrong-token")
            downstream.assert_not_awaited()
            assert len(sent) == 2
            assert sent[0]["type"] == "http.response.start"
            assert sent[0]["status"] == 401
            assert sent[1]["type"] == "http.response.body"
            assert sent[1]["body"] == b"Unauthorized"

        anyio.run(_run)

    def test_missing_token_returns_401(self, monkeypatch):
        import src.api.mcp as mod
        monkeypatch.setattr(mod, "_AUTH_TOKEN", self.TOKEN)
        from src.api.mcp import _XAuthMiddleware

        async def _run():
            mw = _XAuthMiddleware(AsyncMock())
            sent, downstream = await _drive(mw, "http", token=None)  # no header
            downstream.assert_not_awaited()
            assert sent[0]["status"] == 401

        anyio.run(_run)

    def test_lifespan_scope_skips_auth(self, monkeypatch):
        """ASGI lifespan events must never be rejected by auth."""
        import src.api.mcp as mod
        monkeypatch.setattr(mod, "_AUTH_TOKEN", self.TOKEN)
        from src.api.mcp import _XAuthMiddleware

        async def _run():
            mw = _XAuthMiddleware(AsyncMock())
            # lifespan scope, no X-Auth-Token header at all
            sent, downstream = await _drive(mw, "lifespan", token=None)
            assert sent == []
            downstream.assert_awaited_once()

        anyio.run(_run)
