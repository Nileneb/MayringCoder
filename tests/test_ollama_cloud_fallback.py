"""Cloud-Fallback in src/ollama_client.py::generate (Issue #192 B+).

Wenn lokales three.linn.games überlastet/unavailable ist, soll die
generate()-Funktion auf Ollama Cloud (OLLAMA_CLOUD_URL +
OLLAMA_CLOUD_API_KEY) ausweichen — am letzten Retry, nicht bei jedem
Connect-Error (das wäre teuer).

Trigger:
- httpx.ConnectError / httpx.TimeoutException am letzten retry
- HTTP 503/529/429 am letzten retry (overload/rate-limit)

NICHT-Trigger:
- 4xx (Client-Bug, Cloud würde gleich kaputt sein)
- frühere Retries (geben dem local server eine Chance)
- OLLAMA_CLOUD_API_KEY leer (kein Fallback konfiguriert)
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import httpx
import pytest


def _stream_resp(payload: list[dict]) -> MagicMock:
    """Mock for httpx.stream() returning newline-separated JSON."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.iter_lines = lambda: (json_dumps(p) for p in payload)
    cm = MagicMock()
    cm.__enter__ = lambda self: resp
    cm.__exit__ = lambda *a: False
    return cm


def json_dumps(d):
    import json
    return json.dumps(d)


def test_falls_back_to_cloud_on_final_connect_error(monkeypatch):
    """Local connect-error am letzten retry → cloud-call mit auth-header."""
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "sk-cloud-test")
    monkeypatch.setattr("src.ollama_client._CLOUD_API_KEY", "sk-cloud-test")
    monkeypatch.setattr("src.ollama_client._CLOUD_URL", "https://cloud.fake")

    cloud_resp = MagicMock()
    cloud_resp.status_code = 200
    cloud_resp.raise_for_status = MagicMock()
    cloud_resp.json = lambda: {"response": "from cloud"}

    captured: dict = {}

    def fake_post(url, json, headers=None, timeout=None, verify=True):
        captured["url"] = url
        captured["headers"] = headers or {}
        if "cloud.fake" in url:
            return cloud_resp
        raise httpx.ConnectError("local down")

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    from src.ollama_client import generate
    out = generate("http://local:11434", "mistral", "ping", stream=False, max_retries=2)
    assert out == "from cloud"
    assert "cloud.fake" in captured["url"]
    assert captured["headers"].get("Authorization") == "Bearer sk-cloud-test"


def test_no_cloud_fallback_when_api_key_missing(monkeypatch):
    """Ohne OLLAMA_CLOUD_API_KEY → nur normale Retries, kein cloud-call."""
    monkeypatch.setattr("src.ollama_client._CLOUD_API_KEY", "")
    monkeypatch.setattr("src.ollama_client._CLOUD_URL", "https://cloud.fake")

    call_log: list[str] = []

    def fake_post(url, **_):
        call_log.append(url)
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    from src.ollama_client import generate
    with pytest.raises(httpx.ConnectError):
        generate("http://local:11434", "m", "x", stream=False, max_retries=2)
    # Nur local URLs in call_log, kein cloud
    assert all("local" in u for u in call_log), call_log
    assert not any("cloud.fake" in u for u in call_log), call_log


def test_503_on_final_retry_falls_back_to_cloud(monkeypatch):
    """HTTP 503 (overload) am letzten retry → cloud-call."""
    monkeypatch.setattr("src.ollama_client._CLOUD_API_KEY", "sk-test")
    monkeypatch.setattr("src.ollama_client._CLOUD_URL", "https://cloud.fake")

    overload_resp = MagicMock()
    overload_resp.status_code = 503

    def overload_raise():
        raise httpx.HTTPStatusError("503", request=MagicMock(), response=overload_resp)
    overload_resp.raise_for_status = overload_raise

    cloud_resp = MagicMock()
    cloud_resp.status_code = 200
    cloud_resp.raise_for_status = MagicMock()
    cloud_resp.json = lambda: {"response": "cloud-after-503"}

    def fake_post(url, **_):
        if "cloud.fake" in url:
            return cloud_resp
        return overload_resp

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    from src.ollama_client import generate
    # max_retries=1 → 503 ist gleich der letzte retry → cloud-fallback
    out = generate("http://local:11434", "m", "x", stream=False, max_retries=1)
    assert out == "cloud-after-503"


def test_4xx_does_NOT_trigger_cloud_fallback(monkeypatch):
    """400/404 (Client-Bug) → kein Cloud-Fallback (wäre auch dort kaputt)."""
    monkeypatch.setattr("src.ollama_client._CLOUD_API_KEY", "sk-test")
    monkeypatch.setattr("src.ollama_client._CLOUD_URL", "https://cloud.fake")

    bad_resp = MagicMock()
    bad_resp.status_code = 404
    def bad_raise():
        raise httpx.HTTPStatusError("404", request=MagicMock(), response=bad_resp)
    bad_resp.raise_for_status = bad_raise

    call_log: list[str] = []

    def fake_post(url, **_):
        call_log.append(url)
        return bad_resp

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    from src.ollama_client import generate
    with pytest.raises(httpx.HTTPStatusError):
        generate("http://local:11434", "m", "x", stream=False, max_retries=1)
    # Cloud-URL darf NICHT aufgerufen werden bei 4xx
    assert not any("cloud.fake" in u for u in call_log)


def test_cloud_fallback_skipped_when_url_equals_local(monkeypatch):
    """Defense: wenn _CLOUD_URL == local URL (Fehlkonfig), kein Selbst-Loop."""
    monkeypatch.setattr("src.ollama_client._CLOUD_API_KEY", "sk-test")
    monkeypatch.setattr("src.ollama_client._CLOUD_URL", "http://local:11434")

    def fake_post(url, **_):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    from src.ollama_client import generate
    with pytest.raises(httpx.ConnectError):
        generate("http://local:11434", "m", "x", stream=False, max_retries=1)
