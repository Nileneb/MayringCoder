"""Tests for the offline span-relevance judge (tools/span_judge.py).

Covers the two properties the export pipeline relies on: graceful
degradation when Ollama is unreachable, and caching so retrains do not
re-hit Ollama for the same (query, chunk) pairs.
"""
from __future__ import annotations

import sqlite3

import pytest

from tools import span_judge


def _chunks_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, text TEXT)")
    conn.execute("INSERT INTO chunks VALUES ('chk_a', 'relevant text about X')")
    conn.commit()
    return conn


def test_judge_relevance_graceful_on_failure(monkeypatch):
    """Ollama down → {} (no crash, no silent swallow — logs a warning)."""
    import httpx

    def _boom(*a, **k):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", _boom)
    out = span_judge.judge_relevance(
        "query", [("chk_a", "some text")], model="test-model"
    )
    assert out == {}


def test_judge_relevance_parses_and_clamps(monkeypatch):
    """Parses the JSON-mode response and clamps scores into [0, 1]."""
    import httpx

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": '{"scores":{"chk_a":0.9,"chk_b":1.5}}'}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp())
    out = span_judge.judge_relevance(
        "query",
        [("chk_a", "text a"), ("chk_b", "text b")],
        model="test-model",
    )
    assert out["chk_a"] == pytest.approx(0.9)
    assert out["chk_b"] == pytest.approx(1.0)  # clamped


def test_scores_for_query_caches(monkeypatch):
    """Second call for the same (query, chunk) reads the cache — the
    Ollama judge is invoked at most once."""
    conn = _chunks_db()
    monkeypatch.setattr(span_judge, "_judge_model", lambda *a, **k: "m")

    calls: list[int] = []

    def _fake_judge(query, items, **kwargs):
        calls.append(1)
        return {cid: 0.8 for cid, _ in items}

    monkeypatch.setattr(span_judge, "judge_relevance", _fake_judge)

    r1 = span_judge.scores_for_query(conn, "q", ["chk_a"])
    r2 = span_judge.scores_for_query(conn, "q", ["chk_a"])
    assert r1 == {"chk_a": pytest.approx(0.8)}
    assert r2 == {"chk_a": pytest.approx(0.8)}
    assert len(calls) == 1  # second call served from span_judge_cache


def test_loads_lenient_raw_fenced_prose_and_broken():
    import json as _json
    assert span_judge._loads_lenient('{"scores":{"a":1}}') == {"scores": {"a": 1}}
    assert span_judge._loads_lenient('```json\n{"scores":{"a":1}}\n```') == {"scores": {"a": 1}}
    assert span_judge._loads_lenient('note: {"scores":{}} end') == {"scores": {}}
    with pytest.raises(_json.JSONDecodeError):
        span_judge._loads_lenient("not json")


def test_judge_relevance_parses_fenced_response(monkeypatch):
    """Defensive: a backend that wraps the JSON in ```json fences must still
    parse (the raw num_predict=600 truncation that silently skipped the whole
    retrain refinement is fixed separately by the higher num_predict)."""
    import httpx

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": '```json\n{"scores":{"chk_a":0.6}}\n```'}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp())
    out = span_judge.judge_relevance("q", [("chk_a", "t")], model="m")
    assert out["chk_a"] == pytest.approx(0.6)


def test_scores_for_query_empty_inputs():
    conn = _chunks_db()
    assert span_judge.scores_for_query(conn, "", ["chk_a"]) == {}
    assert span_judge.scores_for_query(conn, "q", []) == {}


def test_scores_for_query_survives_cache_write_lock(monkeypatch):
    """The export runs in-container next to the live API (shared memory.db).
    A 'database is locked' on the cache write must NOT crash the export — the
    fresh scores are still returned (cache is an optimization, not critical).
    Regression for the bug that made the in-prod span_judge run write no model."""
    conn = _chunks_db()
    monkeypatch.setattr(span_judge, "_judge_model", lambda *a, **k: "m")
    monkeypatch.setattr(span_judge, "judge_relevance",
                        lambda q, items, **k: {cid: 0.7 for cid, _ in items})

    def _locked(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(span_judge, "_write_cache", _locked)
    out = span_judge.scores_for_query(conn, "q", ["chk_a"])
    assert out == {"chk_a": pytest.approx(0.7)}
