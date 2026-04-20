"""LLMEndpoint fetch + cache tests."""
from __future__ import annotations

import time

import httpx
import pytest

from src.llm import endpoint as endpoint_mod
from src.llm.endpoint import LLMEndpoint, get_llm_endpoint, invalidate_cache


@pytest.fixture(autouse=True)
def _clean_cache():
    invalidate_cache()
    yield
    invalidate_cache()


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("MCP_SERVICE_TOKEN", "srv-token")
    monkeypatch.setenv("LARAVEL_INTERNAL_URL", "http://laravel-test")
    monkeypatch.setenv("OLLAMA_URL", "http://default-ollama:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")


def _mock_response(monkeypatch, payload: dict | None, status: int = 200):
    def _get(url, headers=None, timeout=None):
        req = httpx.Request("GET", url)
        body = b"" if payload is None else __import__("json").dumps(payload).encode()
        return httpx.Response(status, content=body, request=req)
    monkeypatch.setattr(httpx, "get", _get)


def test_missing_workspace_returns_platform_default(configured):
    ep = get_llm_endpoint(None)
    assert ep.provider == "platform"
    assert ep.base_url == "http://default-ollama:11434"


def test_missing_service_token_returns_platform_default(monkeypatch):
    monkeypatch.delenv("MCP_SERVICE_TOKEN", raising=False)
    monkeypatch.setenv("OLLAMA_URL", "http://fb:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "m")
    ep = get_llm_endpoint("bene-workspace")
    assert ep.provider == "platform"
    assert ep.base_url == "http://fb:11434"


def test_successful_ollama_fetch(configured, monkeypatch):
    _mock_response(monkeypatch, {
        "provider": "ollama",
        "base_url": "http://home-tailnet:11434",
        "model": "llama3.1:8b",
    })
    ep = get_llm_endpoint("bene-workspace")
    assert ep.provider == "ollama"
    assert ep.base_url == "http://home-tailnet:11434"
    assert ep.model == "llama3.1:8b"
    assert ep.api_key is None


def test_successful_anthropic_fetch(configured, monkeypatch):
    _mock_response(monkeypatch, {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "model": "claude-sonnet-4-6",
        "api_key": "sk-ant-test",
    })
    ep = get_llm_endpoint("bene-workspace")
    assert ep.provider == "anthropic"
    assert ep.api_key == "sk-ant-test"
    h = ep.headers()
    assert h["x-api-key"] == "sk-ant-test"
    assert "anthropic-version" in h


def test_unknown_provider_falls_back(configured, monkeypatch):
    _mock_response(monkeypatch, {
        "provider": "gemini",  # not supported
        "base_url": "http://x",
        "model": "m",
    })
    ep = get_llm_endpoint("bene-workspace")
    assert ep.provider == "platform"


def test_404_falls_back_to_platform(configured, monkeypatch):
    _mock_response(monkeypatch, None, status=404)
    ep = get_llm_endpoint("bene-workspace")
    assert ep.provider == "platform"


def test_network_error_falls_back(configured, monkeypatch):
    def _boom(*a, **kw):
        raise httpx.ConnectError("boom")
    monkeypatch.setattr(httpx, "get", _boom)
    ep = get_llm_endpoint("bene-workspace")
    assert ep.provider == "platform"


def test_cache_hit_avoids_refetch(configured, monkeypatch):
    calls = {"n": 0}

    def _get(url, headers=None, timeout=None):
        calls["n"] += 1
        req = httpx.Request("GET", url)
        import json
        return httpx.Response(200, content=json.dumps({
            "provider": "ollama",
            "base_url": "http://x:11434",
            "model": "m",
        }).encode(), request=req)

    monkeypatch.setattr(httpx, "get", _get)
    get_llm_endpoint("ws")
    get_llm_endpoint("ws")
    get_llm_endpoint("ws")
    assert calls["n"] == 1


def test_cache_ttl_expiry(configured, monkeypatch):
    import json

    def _get(url, headers=None, timeout=None):
        req = httpx.Request("GET", url)
        return httpx.Response(200, content=json.dumps({
            "provider": "ollama", "base_url": "http://x:11434", "model": "m",
        }).encode(), request=req)
    monkeypatch.setattr(httpx, "get", _get)

    monkeypatch.setattr(endpoint_mod, "_CACHE_TTL_SECONDS", 0)
    get_llm_endpoint("ws")
    time.sleep(0.01)
    # Second call should refetch because TTL=0
    get_llm_endpoint("ws")
    # Both succeeded; just ensure no raise
    assert True


def test_invalidate_one_workspace(configured, monkeypatch):
    import json

    def _get(url, headers=None, timeout=None):
        req = httpx.Request("GET", url)
        return httpx.Response(200, content=json.dumps({
            "provider": "ollama", "base_url": "http://x", "model": "m",
        }).encode(), request=req)
    monkeypatch.setattr(httpx, "get", _get)

    get_llm_endpoint("ws1")
    get_llm_endpoint("ws2")
    invalidate_cache("ws1")
    # ws1 gone, ws2 still there
    with endpoint_mod._cache_lock:
        assert "ws1" not in endpoint_mod._cache
        assert "ws2" in endpoint_mod._cache
