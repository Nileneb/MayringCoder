"""Tests for the new reranker-active endpoints (Task 7).

Pattern: asyncio.run + TokenInfo(workspace_id="system", scopes=("*",)) for
admin, monkeypatch MAYRING_CACHE_DIR.
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


# --- Qualitäts-Gate (2026-06-20): ein Modell darf nur aktiv werden, wenn es auf
# der leakage-freien clean-eval ≥ v1-Baseline liegt. Verhindert, dass degenerierte/
# unterdurchschnittliche Modelle (v4–v7) je wieder in den Serving-A/B-Pool kriechen. ---

def test_gate_rejects_below_baseline(monkeypatch, tmp_path):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    (tmp_path / "rerank_v5.json").write_text('{"weights":{"v":0.3}}')
    monkeypatch.setattr(ra, "_clean_eval_scores", lambda k=5: {"v1": 0.3678, "v5": 0.3433})
    info = _admin()
    with pytest.raises(HTTPException) as exc:
        _run(ra.set_reranker_active(ra.RerankerActiveReq(versions=["v5"]), info=info))
    assert exc.value.status_code == 422
    assert "baseline" in str(exc.value.detail).lower()


def test_gate_allows_above_baseline(monkeypatch, tmp_path):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    (tmp_path / "rerank_v3.json").write_text('{"weights":{"v":1.1}}')
    monkeypatch.setattr(ra, "_clean_eval_scores", lambda k=5: {"v1": 0.3678, "v3": 0.3684})
    info = _admin()
    result = _run(ra.set_reranker_active(ra.RerankerActiveReq(versions=["v3"]), info=info))
    assert result == {"active": ["v3"]}


def test_gate_force_overrides_below_baseline(monkeypatch, tmp_path):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    (tmp_path / "rerank_v5.json").write_text('{"weights":{"v":0.3}}')
    monkeypatch.setattr(ra, "_clean_eval_scores", lambda k=5: {"v1": 0.3678, "v5": 0.3433})
    info = _admin()
    result = _run(ra.set_reranker_active(
        ra.RerankerActiveReq(versions=["v5"]), info=info, force=True))
    assert result["active"] == ["v5"]


def test_gate_skips_without_eval_data(monkeypatch, tmp_path):
    """No claude-prewarm labels yet → no quality evidence → don't block (fail-soft).
    The existence/format validation in write_active_versions still applies."""
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    (tmp_path / "rerank_v5.json").write_text('{"weights":{"v":0.3}}')
    monkeypatch.setattr(ra, "_clean_eval_scores", lambda k=5: {})
    info = _admin()
    result = _run(ra.set_reranker_active(ra.RerankerActiveReq(versions=["v5"]), info=info))
    assert result == {"active": ["v5"]}


def test_gate_v1_always_allowed(monkeypatch, tmp_path):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(ra, "_clean_eval_scores", lambda k=5: {"v1": 0.3678, "v5": 0.3433})
    info = _admin()
    result = _run(ra.set_reranker_active(ra.RerankerActiveReq(versions=["v1"]), info=info))
    assert result == {"active": ["v1"]}


def test_versions_active_flag_matches_serving_sot(monkeypatch, tmp_path):
    """The per-version active flag MUST reflect rerank_active.json (the serving SoT),
    not rerank_default.txt (legacy). Regression for the v4-active=True display lie."""
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    (tmp_path / "rerank_v3.json").write_text('{"weights":{"v":1.1}}')
    (tmp_path / "rerank_v4.json").write_text('{"weights":{"v":0.9}}')
    # legacy default points at v4, but the serving SoT will be v3
    (tmp_path / "rerank_default.txt").write_text("v4")
    (tmp_path / "rerank_active.json").write_text('["v3"]')
    info = _admin()
    result = _run(ra.list_reranker_versions_endpoint(info=info))
    by_ver = {v["version"]: v["active"] for v in result["versions"]}
    assert by_ver["v3"] is True
    assert by_ver["v4"] is False
    assert result["active"] == ["v3"]
