"""Tests for GET /stats/admin/text-models + POST /stats/admin/text-model.

Pattern mirrors test_igio_lens.py: call endpoint functions directly,
monkeypatch module-level helpers, no TestClient.
"""
from __future__ import annotations

import asyncio
import pytest
from fastapi import HTTPException

import src.api.routes.text_models as tm
from src.api.jwt_auth import TokenInfo


def _run(c):
    return asyncio.run(c)


def _admin() -> TokenInfo:
    return TokenInfo(workspace_id="system", scopes=("*",))


def test_list_text_models(monkeypatch, tmp_path):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tm, "_ollama_tags", lambda: ["mistral:7b-instruct", "qwen3.5-mayring:2b"])
    res = _run(tm.list_text_models(info=_admin()))
    names = [m["name"] for m in res["models"]]
    assert "qwen3.5-mayring:2b" in names
    q = next(m for m in res["models"] if m["name"] == "qwen3.5-mayring:2b")
    assert q["cloud_available"] is False


def test_list_marks_cloud_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("OLLAMA_CLOUD_MODELS", "mistral:7b-instruct")
    monkeypatch.setattr(tm, "_ollama_tags", lambda: ["mistral:7b-instruct", "qwen3.5-mayring:2b"])
    res = _run(tm.list_text_models(info=_admin()))
    cloud = next(m for m in res["models"] if m["name"] == "mistral:7b-instruct")
    assert cloud["cloud_available"] is True


def test_set_text_model_writes_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tm, "_ollama_tags", lambda: ["mistral:7b-instruct", "qwen3.5-mayring:2b"])
    res = _run(tm.set_text_model(tm.TextModelReq(model="qwen3.5-mayring:2b"), info=_admin()))
    assert res["active"] == "qwen3.5-mayring:2b"
    from mayring_core.config import _resolve_cache_dir, BASE_DIR
    assert (_resolve_cache_dir(BASE_DIR) / "text_model.txt").read_text().strip() == "qwen3.5-mayring:2b"


def test_set_unknown_model_422(monkeypatch, tmp_path):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tm, "_ollama_tags", lambda: ["mistral:7b-instruct"])
    with pytest.raises(HTTPException) as exc:
        _run(tm.set_text_model(tm.TextModelReq(model="nope:1b"), info=_admin()))
    assert exc.value.status_code == 422


def test_set_model_requires_admin(monkeypatch, tmp_path):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tm, "_ollama_tags", lambda: ["mistral:7b-instruct"])
    non_admin = TokenInfo(workspace_id="ws-regular", scopes=("mcp:memory",))
    with pytest.raises(HTTPException) as exc:
        _run(tm.set_text_model(tm.TextModelReq(model="mistral:7b-instruct"), info=non_admin))
    assert exc.value.status_code == 403
