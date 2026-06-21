"""reference-doc-layer: /memory/search default-excludes reference, /reference/search
forces reference_only, include_reference opts through. Route-wiring only — the
retrieval behaviour itself is unit-tested in mayring-core/tests/test_reference_doc_layer.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import src.api.dependencies as _deps
import src.api.routes.memory as memory_route
from src.api.auth import get_token_info, get_workspace
from src.api.jwt_auth import TokenInfo
from src.api.server import app


@pytest.fixture(autouse=True)
def _override_workspace():
    app.dependency_overrides[get_workspace] = lambda: "test-ws"
    app.dependency_overrides[get_token_info] = lambda: TokenInfo(
        workspace_id="test-ws", scopes=("*",), sub="u-me")
    yield
    app.dependency_overrides.pop(get_workspace, None)
    app.dependency_overrides.pop(get_token_info, None)


@pytest.fixture(autouse=True)
def _reset_conn(tmp_path, monkeypatch):
    from mayring_core.memory.db_adapter import DBAdapter
    from mayring_core.memory.store import _init_schema
    adapter = DBAdapter.create(tmp_path / "test.db", check_same_thread=False)
    _init_schema(adapter)
    monkeypatch.setattr(_deps, "_conn", adapter)
    yield adapter
    monkeypatch.setattr(_deps, "_conn", None)


@pytest.fixture
def captured_opts(monkeypatch):
    """Stub the search + task-derivation so no embed/LLM/Chroma is needed; capture opts."""
    seen: dict = {}

    def _fake_run_search(query, conn, chroma, url, opts, char_budget):
        seen.update(opts)
        return {"results": [], "compressed": ""}

    monkeypatch.setattr(memory_route, "_run_search", _fake_run_search)
    # task-derivation makes an embed call — neutralise it.
    import mayring_core.memory.task_derivation as _td
    monkeypatch.setattr(_td, "derive_research_question_fast", lambda *a, **k: None)
    monkeypatch.setattr(_td, "derive_research_question_background", lambda *a, **k: None)
    return seen


def test_memory_search_default_no_reference_opts(captured_opts):
    r = TestClient(app).post("/memory/search", json={"query": "webgl tier"})
    assert r.status_code == 200
    assert "include_reference" not in captured_opts
    assert "reference_only" not in captured_opts


def test_memory_search_include_reference_passes_through(captured_opts):
    r = TestClient(app).post(
        "/memory/search", json={"query": "webgl tier", "include_reference": True})
    assert r.status_code == 200
    assert captured_opts.get("include_reference") is True


def test_reference_search_forces_reference_only(captured_opts):
    r = TestClient(app).post("/reference/search", json={"query": "webgl tier"})
    assert r.status_code == 200
    assert captured_opts.get("reference_only") is True
