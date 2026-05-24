"""POST /search (Laravel MayringMcpClient alias) must resolve auth itself.

WHY: search_alias calls memory_search() directly (plain Python call, not via
FastAPI), so memory_search's ``info=Depends(get_token_info)`` default was never
resolved — it stayed a Depends object, and ``info.sub`` inside the body raised →
/search 500'd for every call. search_alias must declare + forward ``info``.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

import src.api.dependencies as _deps
import src.api.routes.memory as _mod
from mayring_core.memory.db_adapter import DBAdapter
from mayring_core.memory.store import _init_schema
from src.api.auth import get_token_info, get_workspace
from src.api.jwt_auth import Membership, TokenInfo
from src.api.server import app


def test_search_alias_resolves_auth_and_forwards_opts(monkeypatch):
    db = DBAdapter.memory()
    _init_schema(db)
    monkeypatch.setattr(_deps, "_conn", db, raising=False)

    ti = TokenInfo(
        workspace_id="ws-bene", sub="42", scopes=("mcp:memory",),
        memberships=(Membership(id="org-acme", type="organization", role="editor"),),
    )
    app.dependency_overrides[get_token_info] = lambda: ti
    app.dependency_overrides[get_workspace] = lambda: "ws-bene"

    captured: dict = {}

    def _fake_search(query, conn, chroma, ollama_url, opts, char_budget):
        captured["opts"] = dict(opts)
        return {"results": [], "prompt_context": "", "diagnostics": {}}

    monkeypatch.setattr(_mod, "_run_search", _fake_search)

    try:
        client = TestClient(app)
        resp = client.post(
            "/search", json={"query": "x", "top_k": 3},
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200, resp.text
        # info was resolved (not a Depends object) → user_id + org_ids forwarded
        assert captured["opts"]["user_id"] == "42"
        assert "org-acme" in captured["opts"]["org_ids"]
    finally:
        app.dependency_overrides.clear()
        monkeypatch.setattr(_deps, "_conn", None, raising=False)
