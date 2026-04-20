"""Provider dispatch tests — verify correct backend is called."""
from __future__ import annotations

import json

import httpx
import pytest

from src.llm import dispatch as dispatch_mod
from src.llm.endpoint import LLMEndpoint


def test_ollama_dispatch_calls_ollama_client(monkeypatch):
    called = {}

    def _fake_ollama(url, model, prompt, **kw):
        called["url"] = url
        called["model"] = model
        called["prompt"] = prompt
        return "ollama-result"

    import src.ollama_client as oc
    monkeypatch.setattr(oc, "generate", _fake_ollama)

    ep = LLMEndpoint(provider="ollama", base_url="http://x:11434", model="m")
    out = dispatch_mod.generate(ep, "hello")
    assert out == "ollama-result"
    assert called["model"] == "m"


def test_platform_also_routes_to_ollama(monkeypatch):
    def _fake_ollama(url, model, prompt, **kw):
        return "platform-ollama"

    import src.ollama_client as oc
    monkeypatch.setattr(oc, "generate", _fake_ollama)

    ep = LLMEndpoint(provider="platform", base_url="http://x", model="m")
    assert dispatch_mod.generate(ep, "p") == "platform-ollama"


def test_anthropic_dispatch(monkeypatch):
    captured = {}

    def _post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        req = httpx.Request("POST", url)
        body = __import__("json").dumps({
            "content": [{"type": "text", "text": "hi from anthropic"}],
        }).encode()
        return httpx.Response(200, content=body, request=req)

    monkeypatch.setattr(httpx, "post", _post)

    ep = LLMEndpoint(
        provider="anthropic",
        base_url="https://api.anthropic.com",
        model="claude-sonnet-4-6",
        api_key="sk-ant",
    )
    out = dispatch_mod.generate(ep, "hello", system="you are helpful")
    assert out == "hi from anthropic"
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["json"]["model"] == "claude-sonnet-4-6"
    assert captured["json"]["system"] == "you are helpful"
    assert captured["headers"]["x-api-key"] == "sk-ant"


def test_openai_dispatch(monkeypatch):
    def _post(url, json=None, headers=None, timeout=None):
        req = httpx.Request("POST", url)
        body = __import__("json").dumps({
            "choices": [{"message": {"content": "hi from openai"}}],
        }).encode()
        return httpx.Response(200, content=body, request=req)

    monkeypatch.setattr(httpx, "post", _post)

    ep = LLMEndpoint(
        provider="openai",
        base_url="https://api.openai.com",
        model="gpt-5",
        api_key="sk-oai",
    )
    assert dispatch_mod.generate(ep, "x") == "hi from openai"


def test_unknown_provider_raises():
    ep = LLMEndpoint(provider="ollama", base_url="x", model="m")
    object.__setattr__(ep, "provider", "gibberish")  # bypass frozen
    with pytest.raises(ValueError):
        dispatch_mod.generate(ep, "x")


def test_anthropic_returns_empty_on_no_content(monkeypatch):
    def _post(url, json=None, headers=None, timeout=None):
        req = httpx.Request("POST", url)
        return httpx.Response(200, content=b'{"content": []}', request=req)

    monkeypatch.setattr(httpx, "post", _post)

    ep = LLMEndpoint(provider="anthropic", base_url="https://x", model="m", api_key="k")
    assert dispatch_mod.generate(ep, "x") == ""


def test_anthropic_error_propagates(monkeypatch):
    def _post(url, json=None, headers=None, timeout=None):
        req = httpx.Request("POST", url)
        return httpx.Response(500, content=b'{"error":"down"}', request=req)

    monkeypatch.setattr(httpx, "post", _post)

    ep = LLMEndpoint(provider="anthropic", base_url="https://x", model="m", api_key="k")
    with pytest.raises(httpx.HTTPStatusError):
        dispatch_mod.generate(ep, "x")
