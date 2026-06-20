"""Tests for the live task-anchored search endpoint (/memory/task-search).

Focus: the clean finetune corpus is written (raw_query → task → questions →
chunks) on every call, and the loop is wired to the real search via retrieve_fn.
"""
from __future__ import annotations

import json
import sqlite3

import src.api.routes.memory as mem
from src.api.jwt_auth import TokenInfo
from src.api.routes.models import TaskSearchRequest


def _info():
    return TokenInfo(workspace_id="ws1", scopes=("*",), sub="u1")


def test_task_search_logs_corpus(monkeypatch):
    conn = sqlite3.connect(":memory:")
    monkeypatch.setattr(mem, "_get_conn", lambda: conn)
    monkeypatch.setattr(mem, "_get_chroma", lambda: object())
    # stub the real search: each query returns one chunk named after the query
    monkeypatch.setattr(mem, "_run_search",
                        lambda q, *a, **k: {"results": [{"chunk_id": f"c::{q}", "text": q}]})
    # stub gemma pieces
    import tools.sufficiency_gate as sg
    monkeypatch.setattr(sg, "derive_task", lambda prompt, url=None, **k: "clean task")
    monkeypatch.setattr(sg, "decompose_questions", lambda t, *a, **k: ["q1", "q2"])
    monkeypatch.setattr(sg, "is_answered", lambda q, ch, *a, **k: True)

    req = TaskSearchRequest(query="JAAAA mach das")
    out = mem._task_search_sync(req, "ws1", _info())

    assert out["task"] == "clean task"
    assert out["halted_by"] == "all_answered"
    assert len(out["chunks"]) >= 1

    rows = conn.execute(
        "SELECT raw_query, task, questions, halted_by, n_chunks FROM task_search_log"
    ).fetchall()
    assert len(rows) == 1
    raw_query, task, questions, halted_by, n_chunks = rows[0]
    assert raw_query == "JAAAA mach das"
    assert task == "clean task"
    assert "q1" in json.loads(questions)
    assert n_chunks >= 1


def test_already_task_skips_distillation(monkeypatch):
    conn = sqlite3.connect(":memory:")
    monkeypatch.setattr(mem, "_get_conn", lambda: conn)
    monkeypatch.setattr(mem, "_get_chroma", lambda: object())
    monkeypatch.setattr(mem, "_run_search",
                        lambda q, *a, **k: {"results": [{"chunk_id": "c1", "text": "t"}]})
    import tools.sufficiency_gate as sg

    def _boom(*a, **k):
        raise AssertionError("derive_task must NOT be called when already_task=True")

    monkeypatch.setattr(sg, "derive_task", _boom)
    monkeypatch.setattr(sg, "decompose_questions", lambda t, *a, **k: ["q1"])
    monkeypatch.setattr(sg, "is_answered", lambda q, ch, *a, **k: True)

    out = mem._task_search_sync(
        TaskSearchRequest(query="already a clean task", already_task=True), "ws1", _info())
    assert out["task"] == "already a clean task"


def test_corpus_log_failure_does_not_break_search(monkeypatch):
    """Corpus logging is best-effort — a DB error must not fail the search."""
    monkeypatch.setattr(mem, "_get_chroma", lambda: object())
    monkeypatch.setattr(mem, "_run_search",
                        lambda q, *a, **k: {"results": [{"chunk_id": "c1", "text": "t"}]})
    import tools.sufficiency_gate as sg
    monkeypatch.setattr(sg, "derive_task", lambda prompt, url=None, **k: "task")
    monkeypatch.setattr(sg, "decompose_questions", lambda t, *a, **k: ["q1"])
    monkeypatch.setattr(sg, "is_answered", lambda q, ch, *a, **k: True)

    class _BadConn:
        def execute(self, *a, **k): raise sqlite3.OperationalError("locked")
        def commit(self): ...
    monkeypatch.setattr(mem, "_get_conn", lambda: _BadConn())

    out = mem._task_search_sync(TaskSearchRequest(query="x"), "ws1", _info())
    assert "chunks" in out  # search still returns despite log failure
