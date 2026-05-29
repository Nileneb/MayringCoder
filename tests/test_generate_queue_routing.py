"""Routing-Guards für #1 — Ingest-generate über die zentrale PiQueue (/pi/run).

WHY: ``src/provider_setup.py::_generate`` routet generate-Jobs über POST /pi/run,
wenn ``MAYRING_GENERATE_VIA_QUEUE=1`` (so sammelt der Ingest-Subprozess seine
dominante generate-Last an EINER bounded/observierbaren Stelle). Diese Tests
fixieren: (1) Flag aus → direkter Pfad unverändert; (2) Flag an → /pi/run mit
erhaltenem Determinismus (options + num_predict-Merge + response_format);
(3) Embeddings NIE über die Queue; (4) fail-soft auf direct bei /pi/run-Ausfall.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from mayring_core import providers


class _FakeResp:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self) -> None:  # noqa: D401
        return None

    def json(self) -> dict:
        return {"content": self._content, "workspace_id": "system"}


def test_flag_off_uses_direct_path(monkeypatch):
    monkeypatch.delenv("MAYRING_GENERATE_VIA_QUEUE", raising=False)
    with patch("httpx.post") as post, \
         patch("src.analysis.analyzer._ollama_generate", return_value="DIRECT") as direct:
        out = providers.generate_text("p", "http://localhost:11434", "m", "label")
    assert out == "DIRECT"
    direct.assert_called_once()
    post.assert_not_called()


def test_flag_on_routes_to_pi_run(monkeypatch):
    monkeypatch.setenv("MAYRING_GENERATE_VIA_QUEUE", "1")
    captured: dict = {}

    def _fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _FakeResp("QUEUED_RESULT")

    with patch("httpx.post", side_effect=_fake_post), \
         patch("src.analysis.analyzer._ollama_generate") as direct:
        out = providers.generate_text(
            "prompt", "http://localhost:11434", "mymodel", "label",
            options={"temperature": 0.0, "seed": 7}, response_format="json",
            num_predict=800,
        )
    assert out == "QUEUED_RESULT"
    direct.assert_not_called()
    assert captured["url"].endswith("/pi/run")
    body = captured["json"]
    assert body["kind"] == "categorize"        # non-pi-task → plain generate, no memory aug
    assert body["job_class"] == "background"    # darf Hook/Interaktiv-Lane nicht aushungern
    assert body["model"] == "mymodel"
    assert body["response_format"] == "json"
    # Determinismus erhalten + num_predict in options gemergt (überschreibt Handler-Cap 1024)
    assert body["options"]["temperature"] == 0.0
    assert body["options"]["seed"] == 7
    assert body["options"]["num_predict"] == 800


def test_caller_num_predict_default_merged(monkeypatch):
    """Ohne explizites num_predict erbt der Queue-Job den Generator-Default 4096 —
    sonst würde der Handler-Default 1024 lange Generierungen still truncaten."""
    monkeypatch.setenv("MAYRING_GENERATE_VIA_QUEUE", "1")
    captured: dict = {}
    with patch("httpx.post", side_effect=lambda url, json=None, **kw: captured.update(json or {}) or _FakeResp("x")):
        providers.generate_text("prompt", "http://localhost:11434", "m", "label")
    assert captured["options"]["num_predict"] == 4096


def test_embed_never_routes_through_queue(monkeypatch):
    """Embeddings MÜSSEN auf three.linn.games bleiben — nie über die generate-Queue."""
    monkeypatch.setenv("MAYRING_GENERATE_VIA_QUEUE", "1")
    sentinel = [[1.0, 2.0]]
    with patch("httpx.post") as post, \
         patch("src.analysis.context_rag._embed_texts", return_value=sentinel) as emb:
        out = providers.embed_texts(["hello"], "http://localhost:11434")
    assert out is sentinel
    emb.assert_called_once()
    post.assert_not_called()


def test_fail_soft_falls_back_to_direct(monkeypatch, caplog):
    """Bei /pi/run-Ausfall: 1× retry → direkter generate-Call MIT Warnung (kein silent swallow)."""
    monkeypatch.setenv("MAYRING_GENERATE_VIA_QUEUE", "1")
    with patch("httpx.post", side_effect=RuntimeError("connection refused")) as post, \
         patch("src.analysis.analyzer._ollama_generate", return_value="FALLBACK") as direct:
        with caplog.at_level("WARNING"):
            out = providers.generate_text("prompt", "http://localhost:11434", "m", "label")
    assert out == "FALLBACK"
    direct.assert_called_once()
    assert post.call_count == 2  # initial + 1 retry
    assert any("fail-soft" in r.message for r in caplog.records)
