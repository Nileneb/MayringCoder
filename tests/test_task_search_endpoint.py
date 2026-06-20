"""Tests for the live task-anchored search (memory_service.run_task_search),
shared by the REST endpoint (/memory/task-search) and the MCP tool.

Focus: the clean finetune corpus is written (raw_query → task → questions →
chunks) on every call, distillation is skippable, and corpus logging is
best-effort (never fails the search).
"""
from __future__ import annotations

import json
import sqlite3

import src.api.memory_service as ms


def _patch_pieces(monkeypatch, *, run_search=None, derive=None,
                  decompose=None, answered=None):
    monkeypatch.setattr(ms, "run_search",
                        run_search or (lambda q, *a, **k: {"results": [{"chunk_id": f"c::{q}", "text": q}]}))
    import tools.sufficiency_gate as sg
    monkeypatch.setattr(sg, "derive_task", derive or (lambda prompt, url=None, **k: "clean task"))
    monkeypatch.setattr(sg, "decompose_questions", decompose or (lambda t, *a, **k: ["q1", "q2"]))
    monkeypatch.setattr(sg, "is_answered", answered or (lambda q, ch, *a, **k: True))


def test_task_search_logs_corpus(monkeypatch):
    conn = sqlite3.connect(":memory:")
    _patch_pieces(monkeypatch)
    out = ms.run_task_search("JAAAA mach das", conn, object(), "http://ollama",
                             {"workspace_id": "ws1"})

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

    def _boom(*a, **k):
        raise AssertionError("derive_task must NOT be called when already_task=True")

    _patch_pieces(monkeypatch, derive=_boom)
    out = ms.run_task_search("already a clean task", conn, object(), "http://ollama",
                             {"workspace_id": "ws1"}, already_task=True)
    assert out["task"] == "already a clean task"


def test_corpus_log_failure_does_not_break_search(monkeypatch):
    """Corpus logging is best-effort — a DB error must not fail the search."""
    _patch_pieces(monkeypatch)

    class _BadConn:
        def execute(self, *a, **k):
            raise sqlite3.OperationalError("locked")
        def commit(self): ...

    out = ms.run_task_search("x", _BadConn(), object(), "http://ollama",
                             {"workspace_id": "ws1"})
    assert "chunks" in out  # search still returns despite log failure


def test_endpoint_delegates_to_run_task_search(monkeypatch):
    """The REST wrapper just builds opts and delegates — smoke that the wiring holds."""
    import src.api.routes.memory as mem
    from src.api.routes.models import TaskSearchRequest
    from src.api.jwt_auth import TokenInfo

    captured = {}

    def _fake(query, conn, chroma, url, opts, **kw):
        captured["query"] = query
        captured["opts"] = opts
        return {"task": "t", "questions": [], "halted_by": "all_answered",
                "loops": 0, "chunks": []}

    monkeypatch.setattr(mem, "_get_conn", lambda: object())
    monkeypatch.setattr(mem, "_get_chroma", lambda: object())
    monkeypatch.setattr("src.api.memory_service.run_task_search", _fake)

    info = TokenInfo(workspace_id="ws1", scopes=("*",), sub="u1")
    out = mem._task_search_sync(TaskSearchRequest(query="raw", project="p1"), "ws1", info)
    assert out["workspace_id"] == "ws1"
    assert captured["query"] == "raw"
    assert captured["opts"]["project_id"] == "p1"
