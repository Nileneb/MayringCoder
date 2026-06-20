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
    # full-loop path fuses derive+decompose into one call (~1.5s win)
    monkeypatch.setattr(sg, "derive_and_decompose",
                        lambda prompt, url=None, **k: ("clean task", ["q1", "q2"]))
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


def test_anchor_only_one_search_no_loop(monkeypatch):
    """anchor_only: derive_task + ONE search, NO decomposition — hot-path mode.
    decompose/is_answered must NOT be called; still logs the corpus row."""
    conn = sqlite3.connect(":memory:")
    searches = {"n": 0}

    def _search(q, *a, **k):
        searches["n"] += 1
        return {"results": [{"chunk_id": "c1", "text": "t"}]}

    import tools.sufficiency_gate as sg

    def _no_decompose(*a, **k):
        raise AssertionError("decompose must NOT run in anchor_only mode")

    monkeypatch.setattr(ms, "run_search", _search)
    monkeypatch.setattr(sg, "derive_task", lambda prompt, url=None, **k: "the task")
    monkeypatch.setattr(sg, "decompose_questions", _no_decompose)

    out = ms.run_task_search("raw prompt", conn, object(), "http://ollama",
                             {"workspace_id": "ws1"}, anchor_only=True)
    assert out["task"] == "the task"
    assert out["halted_by"] == "anchor_only"
    assert out["questions"] == ["the task"]
    assert searches["n"] == 1  # exactly one search, no fan-out
    rows = conn.execute("SELECT halted_by FROM task_search_log").fetchall()
    assert rows[0][0] == "anchor_only"


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

    # corpus-worthy query so the INSERT (and the simulated failure) actually fire
    out = ms.run_task_search("eine echte lange testfrage zum reranker",
                             _BadConn(), object(), "http://ollama",
                             {"workspace_id": "ws1"})
    assert "chunks" in out  # search still returns despite log failure


def test_corpus_endpoint_shows_rows(monkeypatch):
    import src.api.routes.memory as mem
    from src.api.jwt_auth import TokenInfo
    conn = sqlite3.connect(":memory:")
    ms.ensure_task_search_log(conn)
    conn.execute(
        "INSERT INTO task_search_log (workspace_id, raw_query, task, questions, "
        "halted_by, loops, n_chunks, chunk_ids, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        ("ws1", "JAAA mach das", "do the thing", '["q1","q2"]', "anchor_only", 0, 5, "[]", "2026-06-20"))
    conn.commit()
    monkeypatch.setattr(mem, "_get_conn", lambda: conn)
    out = mem.task_search_corpus(limit=10, workspace_id="ws1",
                                 info=TokenInfo(workspace_id="ws1", scopes=("*",), sub="u1"))
    assert out["total"] == 1
    assert out["recent"][0]["task"] == "do the thing"
    assert out["recent"][0]["n_questions"] == 2
    assert out["recent"][0]["halted_by"] == "anchor_only"

    # REGRESSION: the endpoint must NOT close the (thread-local, persistent) conn —
    # closing it poisoned the next request on that worker thread → "Cannot operate
    # on a closed database" 500s. The same conn must still be usable afterwards.
    still_alive = conn.execute("SELECT COUNT(*) FROM task_search_log").fetchone()[0]
    assert still_alive == 1


def test_corpus_endpoint_empty_before_first_search(monkeypatch):
    import src.api.routes.memory as mem
    from src.api.jwt_auth import TokenInfo
    conn = sqlite3.connect(":memory:")  # no table yet
    monkeypatch.setattr(mem, "_get_conn", lambda: conn)
    out = mem.task_search_corpus(limit=10, workspace_id="ws1",
                                 info=TokenInfo(workspace_id="ws1", scopes=("*",), sub="u1"))
    assert out["total"] == 0
    assert "hint" in out


def test_endpoint_delegates_to_run_task_search(monkeypatch):
    """The REST wrapper builds opts, forwards the bearer + internal URL, delegates."""
    import src.api.routes.memory as mem
    from src.api.routes.models import TaskSearchRequest
    from src.api.jwt_auth import TokenInfo

    captured = {}

    def _fake(query, conn, chroma, url, opts, **kw):
        captured["query"] = query
        captured["opts"] = opts
        captured["kw"] = kw
        return {"task": "t", "questions": [], "halted_by": "all_answered",
                "loops": 0, "chunks": []}

    monkeypatch.setattr(mem, "_get_conn", lambda: object())
    monkeypatch.setattr(mem, "_get_chroma", lambda: object())
    monkeypatch.setattr("src.api.memory_service.run_task_search", _fake)

    info = TokenInfo(workspace_id="ws1", scopes=("*",), sub="u1")
    out = mem._task_search_sync(TaskSearchRequest(query="raw", project="p1"),
                                "ws1", info, bearer="Bearer abc.def.ghi")
    assert out["workspace_id"] == "ws1"
    assert captured["query"] == "raw"
    assert captured["opts"]["project_id"] == "p1"
    # FALLE 2: the raw bearer + an internal URL reach run_task_search for the fanout
    assert captured["kw"]["bearer"] == "Bearer abc.def.ghi"
    assert "memory_service" not in captured["kw"]["retrieve_url"]  # is an http url
    assert captured["kw"]["retrieve_url"].startswith("http")


# --- HTTP-fanout (FALLE 1 + 2): the act-path loop searches via our own API -------

class _FakeResp:
    def __init__(self, payload):
        self._payload = payload
    def raise_for_status(self):
        pass
    def json(self):
        return self._payload


class _FakeClient:
    """Records POSTs; returns one chunk echoing the query so mapping is checkable."""
    calls: list[dict] = []

    def __init__(self, *a, **k):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def post(self, url, json=None, headers=None):
        _FakeClient.calls.append({"url": url, "json": json, "headers": headers})
        return _FakeResp({"results": [{"chunk_id": f"c::{json['query']}",
                                       "text": json["query"]}]})


def _patch_httpx(monkeypatch):
    import httpx
    _FakeClient.calls = []
    monkeypatch.setattr(httpx, "Client", _FakeClient)


def test_http_retrieve_fn_maps_and_authenticates(monkeypatch):
    _patch_httpx(monkeypatch)
    fn = ms._http_retrieve_fn(
        "http://mayring-api:8090/", "raw.jwt.token",
        {"top_k": 5, "scope_key": "project:x", "project_id": "p1",
         "session_id": "s1", "category_hint": ["auth"]}, char_budget=4000)
    chunks = fn("how does auth work")

    assert chunks == [{"chunk_id": "c::how does auth work", "text": "how does auth work"}]
    call = _FakeClient.calls[0]
    assert call["url"] == "http://mayring-api:8090/memory/search"
    # bearer normalised to "Bearer <token>" (FALLE 2), scope preserved
    assert call["headers"]["Authorization"] == "Bearer raw.jwt.token"
    assert call["json"]["query"] == "how does auth work"
    assert call["json"]["top_k"] == 5
    assert call["json"]["scope"] == "project:x"
    assert call["json"]["project"] == "p1"
    assert call["json"]["session_id"] == "s1"
    assert call["json"]["category_hint"] == ["auth"]
    assert call["json"]["llm_prefilter"] is False  # loop judges; skip PI-advisor


def test_http_retrieve_fn_keeps_existing_bearer_prefix(monkeypatch):
    _patch_httpx(monkeypatch)
    fn = ms._http_retrieve_fn("http://api", "Bearer already.prefixed", {}, 6000)
    fn("q")
    assert _FakeClient.calls[0]["headers"]["Authorization"] == "Bearer already.prefixed"


def test_run_task_search_uses_http_fanout_when_bearer_present(monkeypatch):
    """retrieve_url + bearer + act-path → sub-questions go via HTTP, NOT in-process
    run_search. The conservative thread pool is capped at _FANOUT_CAP."""
    conn = sqlite3.connect(":memory:")
    _patch_httpx(monkeypatch)

    def _no_inprocess(*a, **k):
        raise AssertionError("in-process run_search must NOT run when fanout is active")

    import tools.sufficiency_gate as sg
    monkeypatch.setattr(ms, "run_search", _no_inprocess)
    monkeypatch.setattr(sg, "derive_and_decompose",
                        lambda prompt, url=None, **k: ("the task", ["qa", "qb"]))
    monkeypatch.setattr(sg, "is_answered", lambda q, ch, *a, **k: True)

    out = ms.run_task_search(
        "raw emotional prompt about reranker", conn, object(), "http://ollama",
        {"workspace_id": "ws1", "top_k": 8}, conn_factory=lambda: conn,
        retrieve_url="http://mayring-api:8090", bearer="jwt.tok", parallelism=4)

    assert out["task"] == "the task"
    assert len(out["chunks"]) >= 1
    # every sub-question hit /memory/search over HTTP
    assert len(_FakeClient.calls) >= 1
    assert all(c["url"].endswith("/memory/search") for c in _FakeClient.calls)


def test_anchor_only_never_fans_out_even_with_bearer(monkeypatch):
    """Hot-path stays a single in-process search; the bearer/url must be ignored."""
    conn = sqlite3.connect(":memory:")
    _patch_httpx(monkeypatch)
    searches = {"n": 0}

    def _search(q, *a, **k):
        searches["n"] += 1
        return {"results": [{"chunk_id": "c1", "text": "t"}]}

    import tools.sufficiency_gate as sg
    monkeypatch.setattr(ms, "run_search", _search)
    monkeypatch.setattr(sg, "derive_task", lambda prompt, url=None, **k: "the task")

    ms.run_task_search("raw prompt", conn, object(), "http://ollama",
                       {"workspace_id": "ws1"}, anchor_only=True,
                       retrieve_url="http://mayring-api:8090", bearer="jwt.tok")
    assert searches["n"] == 1           # in-process single call
    assert _FakeClient.calls == []      # NO http fanout in anchor_only


def test_kill_switch_forces_in_process(monkeypatch):
    """TASK_SEARCH_FANOUT=0 → in-process even with bearer (instant prod rollback)."""
    conn = sqlite3.connect(":memory:")
    _patch_httpx(monkeypatch)
    monkeypatch.setenv("TASK_SEARCH_FANOUT", "0")
    seen = {"n": 0}

    def _search(q, *a, **k):
        seen["n"] += 1
        return {"results": [{"chunk_id": "c1", "text": "t"}]}

    import tools.sufficiency_gate as sg
    monkeypatch.setattr(ms, "run_search", _search)
    monkeypatch.setattr(sg, "derive_and_decompose",
                        lambda prompt, url=None, **k: ("task", ["qa"]))
    monkeypatch.setattr(sg, "is_answered", lambda q, ch, *a, **k: True)

    ms.run_task_search("raw prompt about x", conn, object(), "http://ollama",
                       {"workspace_id": "ws1"}, conn_factory=lambda: conn,
                       retrieve_url="http://mayring-api:8090", bearer="jwt.tok")
    assert seen["n"] >= 1               # in-process search ran
    assert _FakeClient.calls == []      # fanout suppressed by kill-switch


def test_http_retrieve_degrades_on_failure(monkeypatch):
    """A failed sub-question search returns [] (logged), never raises — the loop
    must survive a transient network/auth blip on one of N parallel calls."""
    import httpx
    _FakeClient.calls = []

    class _BoomClient(_FakeClient):
        def post(self, url, json=None, headers=None):
            raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx, "Client", _BoomClient)
    fn = ms._http_retrieve_fn("http://api", "jwt", {}, 6000)
    assert fn("q") == []  # degraded, not raised
