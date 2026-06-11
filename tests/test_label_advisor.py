"""Async label-advisor: refine LOW-cosine-confidence category_labels via an LLM
CONSTRAINED to the codebook (SubQ/SSA-style, central-queue-routed).

Guards the two properties that keep it from re-introducing the duplication the
category-consolidation removed: (1) only low-confidence chunks are touched,
(2) the LLM's labels are filtered to the active codebook (hallucinated names
dropped) so labels stay aligned with the FK/codebook SoT.
"""
from __future__ import annotations

import asyncio


def _maybe_run(c):
    return asyncio.run(c) if asyncio.iscoroutine(c) else c
import sqlite3

from src.api.routes import reranker_admin as ra


class _FakeQueue:
    def __init__(self, content: str):
        self._content = content

    async def enqueue(self, job):
        return {"content": self._content}


class _FakeRouter:
    def __init__(self, url):
        pass

    def resolve(self, task):
        return "mistral:7b-instruct"


def _conn_with_chunks() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE chunks (chunk_id TEXT, summary TEXT, text TEXT, "
                 "is_active INTEGER, category_source TEXT, category_labels TEXT)")
    conn.execute("CREATE TABLE chunk_categories (chunk_id TEXT, category_id INTEGER, confidence REAL)")
    conn.execute("CREATE TABLE categories (id INTEGER, name TEXT, status TEXT)")
    conn.execute("INSERT INTO chunks VALUES ('chk_low','auth login token','',1,'deductive-link','tools')")
    conn.execute("INSERT INTO chunks VALUES ('chk_high','x','',1,'deductive-link','api')")
    conn.execute("INSERT INTO chunk_categories VALUES ('chk_low',1,0.40)")   # weak cosine → eligible
    conn.execute("INSERT INTO chunk_categories VALUES ('chk_high',2,0.90)")  # strong → skipped
    conn.executemany("INSERT INTO categories VALUES (?,?,?)",
                     [(1, "auth", "active"), (2, "api", "active"), (3, "data_access", "active")])
    conn.commit()
    return conn


def _patch(monkeypatch, conn, llm_content: str):
    monkeypatch.setattr(ra, "_conn", lambda: conn)
    monkeypatch.setattr(ra, "_is_admin", lambda info: True)
    monkeypatch.setattr("mayring_pi_agent.pi_queue.get_pi_queue", lambda: _FakeQueue(llm_content))
    monkeypatch.setattr("mayring_core.model_router.ModelRouter", _FakeRouter)
    monkeypatch.setattr("mayring_core.memory.store.kv_get", lambda cid: None)
    monkeypatch.setattr("mayring_core.memory.store.kv_put", lambda cid, d: None)


def test_label_advisor_low_conf_only_and_codebook_constrained(monkeypatch):
    conn = _conn_with_chunks()
    # LLM returns one valid codebook label + one hallucinated (must be dropped).
    _patch(monkeypatch, conn, '{"chk_low": ["auth", "totally-made-up-category"]}')

    out = _maybe_run(ra.label_advisor(after=0, limit=10, confidence_threshold=0.62,
                                       info=None, workspace_id="ws"))
    assert out["processed"] == 1   # only chk_low (weak cosine); chk_high (0.90) skipped
    assert out["advised"] == 1

    low = conn.execute("SELECT category_labels, category_source FROM chunks WHERE chunk_id='chk_low'").fetchone()
    assert low[0] == "auth"            # hallucinated label dropped → only codebook name kept
    assert low[1] == "llm-advised"
    # strong-cosine chunk left untouched
    assert conn.execute("SELECT category_labels FROM chunks WHERE chunk_id='chk_high'").fetchone()[0] == "api"


def test_label_advisor_no_eligible_chunks(monkeypatch):
    conn = _conn_with_chunks()
    _patch(monkeypatch, conn, "{}")
    # threshold below the weak link → nothing eligible
    out = _maybe_run(ra.label_advisor(after=0, limit=10, confidence_threshold=0.1,
                                       info=None, workspace_id="ws"))
    assert out["processed"] == 0 and out["advised"] == 0 and out["has_more"] is False
