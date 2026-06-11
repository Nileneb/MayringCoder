"""Tests for the new reranker-active endpoints (Task 7).

Mirrors tests/test_igio_lens.py: asyncio.run + TokenInfo(workspace_id="system",
scopes=("*",)) for admin, monkeypatch MAYRING_CACHE_DIR.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

import src.api.routes.reranker_admin as ra
from src.api.jwt_auth import TokenInfo


def _run(c):
    return asyncio.run(c) if asyncio.iscoroutine(c) else c


def _admin() -> TokenInfo:
    return TokenInfo(workspace_id="system", scopes=("*",))


def test_set_and_get_active(monkeypatch, tmp_path):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    (tmp_path / "rerank_v3.json").write_text('{"weights":{}}')
    (tmp_path / "rerank_v4.json").write_text('{"weights":{}}')
    info = _admin()
    _run(ra.set_reranker_active(ra.RerankerActiveReq(versions=["v3", "v4"]), info=info))
    g = _run(ra.list_reranker_versions_endpoint(info=info))
    assert sorted(g["active"]) == ["v3", "v4"]


def test_single_active_version(monkeypatch, tmp_path):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    (tmp_path / "rerank_v2.json").write_text('{"weights":{}}')
    info = _admin()
    result = _run(ra.set_reranker_active(ra.RerankerActiveReq(versions=["v2"]), info=info))
    assert result == {"active": ["v2"]}
    g = _run(ra.list_reranker_versions_endpoint(info=info))
    assert g["active"] == ["v2"]


def test_third_version_422(monkeypatch, tmp_path):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    for v in ("v2", "v3", "v4"):
        (tmp_path / f"rerank_{v}.json").write_text('{"weights":{}}')
    info = _admin()
    with pytest.raises(HTTPException) as exc_info:
        _run(ra.set_reranker_active(ra.RerankerActiveReq(versions=["v2", "v3", "v4"]), info=info))
    assert exc_info.value.status_code == 422


def test_nonexistent_version_422(monkeypatch, tmp_path):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    info = _admin()
    with pytest.raises(HTTPException) as exc_info:
        _run(ra.set_reranker_active(ra.RerankerActiveReq(versions=["v99"]), info=info))
    assert exc_info.value.status_code == 422


def test_non_admin_forbidden(monkeypatch, tmp_path):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    info = TokenInfo(workspace_id="ws-user", scopes=())
    with pytest.raises(HTTPException) as exc_info:
        _run(ra.set_reranker_active(ra.RerankerActiveReq(versions=["v1"]), info=info))
    assert exc_info.value.status_code == 403


def test_autorollout_endpoints_removed():
    paths = [r.path for r in ra.router.routes]
    assert "/stats/admin/reranker-autorollout" not in paths


def test_versions_endpoint_returns_list(monkeypatch, tmp_path):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    info = _admin()
    result = _run(ra.list_reranker_versions_endpoint(info=info))
    assert isinstance(result["active"], list)
    assert "versions" in result
