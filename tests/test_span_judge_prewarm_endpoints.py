"""Tests for the server-side span-judge prewarm endpoints (Pfad A).

The pure dump/ingest roundtrip is covered by test_span_judge_prewarm.py; here
we only assert the HTTP layer: admin-guard + that the endpoints wire through to
span_judge_prewarm against a memory.db connection, and that the train trigger
threads span_judge_max_calls (the claude-only override that stops the weak
ministral judge from re-poisoning v).
"""
from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest
from fastapi import HTTPException

import src.api.routes.reranker_admin as ra
from src.api.jwt_auth import TokenInfo


def _run(c):
    return asyncio.run(c) if asyncio.iscoroutine(c) else c


def _admin() -> TokenInfo:
    return TokenInfo(workspace_id="system", scopes=("*",))


def _viewer() -> TokenInfo:
    return TokenInfo(workspace_id="ws", scopes=("read",))


class _KeepOpen(sqlite3.Connection):
    """In-memory conn whose close() is a no-op so the same db survives both
    endpoint calls (each endpoint closes its conn in a finally block)."""
    def close(self):  # noqa: D401
        pass


def _seeded_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", factory=_KeepOpen)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE context_feedback_log(id INTEGER PRIMARY KEY, query TEXT,"
        " trigger_ids TEXT, stage_scores TEXT, was_referenced INT,"
        " captured_at TEXT, workspace_id TEXT);"
        "CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, text TEXT);"
    )
    conn.execute("INSERT INTO chunks VALUES('chk_a','relevant'),('chk_b','noise')")
    conn.execute(
        "INSERT INTO context_feedback_log VALUES"
        "(1,'q one',?,'{\"chk_a\":{},\"chk_b\":{}}',1,datetime('now'),'ws')",
        (json.dumps(["chk_a", "chk_b"]),),
    )
    conn.commit()
    return conn


def test_uncached_pairs_requires_admin():
    with pytest.raises(HTTPException) as e:
        _run(ra.span_judge_uncached_pairs(info=_viewer()))
    assert e.value.status_code == 403


def test_ingest_requires_admin():
    with pytest.raises(HTTPException) as e:
        _run(ra.span_judge_ingest(ra.SpanJudgeIngestReq(scores=[]), info=_viewer()))
    assert e.value.status_code == 403


def test_dump_then_ingest_roundtrip(monkeypatch):
    conn = _seeded_conn()
    # one shared in-memory db for both endpoint calls (each call would otherwise
    # open + close its own; the seeded conn stands in for prod memory.db).
    monkeypatch.setattr(ra, "_memory_db_conn", lambda: conn)

    pairs = _run(ra.span_judge_uncached_pairs(days=30, limit=300, info=_admin()))
    assert pairs["pairs"] == 2
    assert pairs["data"][0]["query"] == "q one"

    written = _run(ra.span_judge_ingest(
        ra.SpanJudgeIngestReq(scores=[
            {"query": "q one", "scores": {"chk_a": 0.9, "chk_b": 0.0}}]),
        info=_admin()))
    assert written == {"ingested": 2}

    # now the pair is cached → no longer dumped as uncached
    again = _run(ra.span_judge_uncached_pairs(days=30, limit=300, info=_admin()))
    assert again["pairs"] == 0


def test_train_trigger_threads_max_calls(monkeypatch):
    async def _fake_sub(*a, **k):  # don't spawn a real subprocess in the test
        return None

    monkeypatch.setattr(ra, "_run_train_subprocess", _fake_sub)
    monkeypatch.setattr(ra, "_save_train_job", lambda job_id: None)

    out = _run(ra.trigger_train_reranker(
        info=_admin(), days=7, span_judge=True, span_judge_max_calls=0))
    assert out["span_judge"] is True
    assert out["span_judge_max_calls"] == 0
    # the override is persisted into the job record the subprocess reads
    assert ra._TRAIN_JOBS[out["job_id"]]["span_judge_max_calls"] == 0
