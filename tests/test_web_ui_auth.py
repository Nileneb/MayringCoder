"""Gradio WebUI JWT refresh helper tests."""
from __future__ import annotations

import httpx
import pytest

import src.api.web_ui as web_ui


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("LARAVEL_INTERNAL_URL", "http://laravel-test")


def _response(payload: dict | None, status: int) -> httpx.Response:
    req = httpx.Request("POST", "http://laravel-test/api/mayring/refresh-token")
    if payload is None:
        return httpx.Response(status, content=b"", request=req)
    import json
    return httpx.Response(status, content=json.dumps(payload).encode(), request=req)


def test_refresh_empty_token_returns_none():
    assert web_ui.refresh_jwt("") is None


def test_refresh_success(monkeypatch):
    captured = {}

    def _post(url, headers=None, timeout=None):
        captured["url"] = url
        captured["auth"] = (headers or {}).get("Authorization")
        return _response({"token": "fresh.jwt.token"}, 200)

    monkeypatch.setattr(web_ui._httpx, "post", _post)
    assert web_ui.refresh_jwt("old.jwt") == "fresh.jwt.token"
    assert captured["auth"] == "Bearer old.jwt"
    assert captured["url"].endswith("/api/mayring/refresh-token")


def test_refresh_401_returns_none(monkeypatch):
    monkeypatch.setattr(web_ui._httpx, "post", lambda *a, **kw: _response({"error": "nope"}, 401))
    assert web_ui.refresh_jwt("old.jwt") is None


def test_refresh_empty_payload_returns_none(monkeypatch):
    monkeypatch.setattr(web_ui._httpx, "post", lambda *a, **kw: _response({"token": ""}, 200))
    assert web_ui.refresh_jwt("old.jwt") is None


def test_refresh_network_error_returns_none(monkeypatch):
    def _boom(*a, **kw):
        raise httpx.ConnectError("boom")
    monkeypatch.setattr(web_ui._httpx, "post", _boom)
    assert web_ui.refresh_jwt("old.jwt") is None


def test_api_post_retries_on_401_with_fresh_token(monkeypatch):
    """_api_post should auto-refresh on 401 and retry transparently."""
    calls: list[dict] = []

    def _post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "auth": (headers or {}).get("Authorization")})
        if "/api/mayring/refresh-token" in url:
            return _response({"token": "new.jwt"}, 200)
        if (headers or {}).get("Authorization") == "Bearer old.jwt":
            return _response({"error": "unauth"}, 401)
        return _response({"ok": True}, 200)

    monkeypatch.setattr(web_ui._httpx, "post", _post)
    web_ui._api_url = "http://api-test"

    out = web_ui._api_post("memory/search", {"query": "x"}, "old.jwt")
    assert out["ok"] is True
    assert out["_refreshed_token"] == "new.jwt"
    # 3 calls: initial 401, refresh, retry with new token
    auth_values = [c["auth"] for c in calls]
    assert "Bearer old.jwt" in auth_values
    assert "Bearer new.jwt" in auth_values


def test_api_post_gives_up_when_refresh_fails(monkeypatch):
    def _post(url, json=None, headers=None, timeout=None):
        if "/api/mayring/refresh-token" in url:
            return _response({}, 401)  # refresh denied
        return _response({"error": "unauth"}, 401)

    monkeypatch.setattr(web_ui._httpx, "post", _post)
    web_ui._api_url = "http://api-test"

    out = web_ui._api_post("memory/search", {"query": "x"}, "old.jwt")
    assert "abgelaufen" in out["error"].lower() or "neu einloggen" in out["error"].lower()
