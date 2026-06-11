"""TTL-cache for read-only dashboard endpoints (api-saturation 2026-05-24).

Two concerns:
  1. functools.wraps must keep FastAPI's dependency resolution working on the
     decorated endpoints (else every dashboard call 500s).
  2. the cache must key by workspace_id so it never bleeds across tenants, and
     must expire.
"""
from __future__ import annotations

import asyncio


def _maybe_run(c):
    return asyncio.run(c) if asyncio.iscoroutine(c) else c

import pytest
from fastapi.testclient import TestClient

import src.api.dependencies as _deps
from src.api.auth import get_workspace
from src.api.routes import dashboard
from src.api.server import app


@pytest.fixture(autouse=True)
def _override_workspace():
    app.dependency_overrides[get_workspace] = lambda: "test-ws"
    dashboard._DASH_CACHE.clear()
    yield
    app.dependency_overrides.pop(get_workspace, None)
    dashboard._DASH_CACHE.clear()


@pytest.fixture(autouse=True)
def _reset_conn(tmp_path, monkeypatch):
    from mayring_core.memory.db_adapter import DBAdapter
    from mayring_core.memory.store import _init_schema
    adapter = DBAdapter.create(tmp_path / "test.db", check_same_thread=False)
    _init_schema(adapter)
    monkeypatch.setattr(_deps, "_conn", adapter)
    yield adapter
    monkeypatch.setattr(_deps, "_conn", None)


def test_decorated_endpoint_still_resolves_deps_and_returns_200():
    """functools.wraps preserved the signature → FastAPI resolved get_workspace.

    Uses /stats/triggers as the example: /stats/recent-ops is intentionally
    UNcached now (live ingest feed — see the smoke-fix WHY in dashboard.py).
    """
    client = TestClient(app)
    r = client.get("/stats/triggers")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["workspace_id"] == "test-ws"
    assert "triggers" in body
    # the call populated the cache
    assert any(k.startswith("triggers:") for k in dashboard._DASH_CACHE)


def _run(coro):
    return _maybe_run(coro) if asyncio.iscoroutine(coro) else coro


def test_cache_keys_by_workspace_no_cross_tenant_bleed():
    dashboard._DASH_CACHE.clear()
    calls: list[tuple] = []

    @dashboard._dashboard_ttl_cache
    def ep(*, workspace_id, limit=50):
        calls.append((workspace_id, limit))
        return {"ws": workspace_id}

    a1 = _run(ep(workspace_id="A", limit=50))
    a2 = _run(ep(workspace_id="A", limit=50))  # within TTL → cached
    assert a1 == a2 == {"ws": "A"}
    assert calls.count(("A", 50)) == 1, "producer should run once for repeated args"

    b1 = _run(ep(workspace_id="B", limit=50))  # different tenant → own entry
    assert b1 == {"ws": "B"}
    assert calls.count(("B", 50)) == 1
    # A's entry must be untouched by B (no bleed)
    assert _run(ep(workspace_id="A", limit=50)) == {"ws": "A"}
    assert calls.count(("A", 50)) == 1


def test_cache_expires_after_ttl():
    dashboard._DASH_CACHE.clear()
    calls: list[int] = []

    @dashboard._dashboard_ttl_cache
    def ep(*, workspace_id):
        calls.append(1)
        return {"n": len(calls)}

    _run(ep(workspace_id="A"))
    assert len(calls) == 1
    # force the stored entry stale (monotonic() is always > 0) → producer re-runs
    for k in list(dashboard._DASH_CACHE):
        _, val = dashboard._DASH_CACHE[k]
        dashboard._DASH_CACHE[k] = (0.0, val)
    _run(ep(workspace_id="A"))
    assert len(calls) == 2
