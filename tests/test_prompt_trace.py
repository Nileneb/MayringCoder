"""GET /stats/prompt-trace — the per-prompt retrieval trace (Phase 5 observability).

Proves the endpoint surfaces query + per-chunk stage scores from
context_feedback_log, orders chunks by final score, and stays workspace-scoped.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import src.api.dependencies as _deps
from src.api.auth import get_workspace
from src.api.server import app


@pytest.fixture(autouse=True)
def _override_workspace():
    app.dependency_overrides[get_workspace] = lambda: "test-ws"
    yield
    app.dependency_overrides.pop(get_workspace, None)


@pytest.fixture
def conn(tmp_path, monkeypatch):
    from mayring_core.memory.db_adapter import DBAdapter
    from mayring_core.memory.store import _init_schema
    adapter = DBAdapter.create(tmp_path / "trace.db", check_same_thread=False)
    _init_schema(adapter)
    monkeypatch.setattr(_deps, "_conn", adapter)
    yield adapter
    monkeypatch.setattr(_deps, "_conn", None)


@pytest.fixture
def client():
    return TestClient(app)


def _insert(conn, *, query, ids, scores, ws="test-ws", referenced=0, reranker="v2"):
    conn.execute(
        "INSERT INTO context_feedback_log "
        "(trigger_ids, context_text, was_referenced, led_to_retrieval, "
        " relevance_score, captured_at, query, stage_scores, workspace_id, "
        " reranker_version) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (json.dumps(ids), "ctx", referenced, 0, 0.0,
         "2026-05-24T00:00:00+00:00", query, json.dumps(scores), ws, reranker),
    )
    conn.commit()


def test_trace_returns_query_and_chunks_sorted_by_final(conn, client):
    _insert(
        conn,
        query="how does workspace scoping work",
        ids=["chk_a", "chk_b"],
        scores={"chk_a": {"v": 0.2, "f": 0.4}, "chk_b": {"v": 0.9, "f": 0.8}},
        referenced=1,
    )
    r = client.get("/stats/prompt-trace")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["workspace_id"] == "test-ws"
    assert len(body["prompts"]) == 1
    p = body["prompts"][0]
    assert p["query"] == "how does workspace scoping work"
    assert p["reranker_version"] == "v2"
    assert p["was_referenced"] is True
    assert p["chunk_count"] == 2
    # ordered by final score desc → chk_b (0.8) before chk_a (0.4)
    assert [c["chunk_id"] for c in p["chunks"]] == ["chk_b", "chk_a"]
    assert p["chunks"][0]["scores"]["v"] == 0.9
    assert p["chunks"][0]["final"] == 0.8


def test_trace_is_workspace_scoped(conn, client):
    _insert(conn, query="mine", ids=["c1"], scores={"c1": {"f": 0.5}}, ws="test-ws")
    _insert(conn, query="foreign", ids=["c2"], scores={"c2": {"f": 0.5}}, ws="other-ws")
    body = client.get("/stats/prompt-trace").json()
    queries = [p["query"] for p in body["prompts"]]
    assert queries == ["mine"]  # foreign workspace row not leaked


def test_trace_skips_empty_query_rows(conn, client):
    # hook-less / legacy rows with empty query must not clutter the trace
    _insert(conn, query="", ids=["c1"], scores={}, ws="test-ws")
    _insert(conn, query="real prompt", ids=["c1"], scores={"c1": {"f": 0.3}}, ws="test-ws")
    body = client.get("/stats/prompt-trace").json()
    assert [p["query"] for p in body["prompts"]] == ["real prompt"]
