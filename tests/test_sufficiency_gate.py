"""Tests for the Mythos-style sufficiency gate (tools/sufficiency_gate.py).

The gate simulates OpenMythos' ACT halting on the ORCHESTRATION level: a small
thinking model (gemma4:e4b) judges whether the retrieved chunks suffice; if not,
its `requery` drives one more retrieval loop. The loop IS the "think until solved"
mechanism; the model's `sufficient` flag is the ACT halt scalar. These tests pin
the four halting criteria (sufficient / cap / budget / no-progress) and the
fail-safe behaviour so the loop can NEVER hang (the span_judge-hang lesson).
"""
from __future__ import annotations

import pytest

from tools import sufficiency_gate as sg


def _chunk(cid: str, text: str = "x") -> dict:
    return {"chunk_id": cid, "text": text}


# --- judge_sufficiency: parsing + fail-safe ---

def test_judge_parses_structured_verdict(monkeypatch):
    import httpx

    class _Resp:
        def raise_for_status(self): ...
        def json(self):
            return {"message": {"content":
                '{"sufficient": false, "missing": ["X"], "requery": "find X"}'}}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp())
    v = sg.judge_sufficiency("q", [_chunk("a")], "http://ollama")
    assert v["sufficient"] is False
    assert v["missing"] == ["X"]
    assert v["requery"] == "find X"


def test_judge_fail_safe_on_broken_json(monkeypatch):
    """A broken/empty judge response must NOT hang the loop — degrade to
    sufficient=True (use what we have) rather than loop forever. No silent
    swallow: it's the explicit 'judge unavailable → no gating' path."""
    import httpx

    class _Resp:
        def raise_for_status(self): ...
        def json(self):
            return {"message": {"content": "not json at all"}}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp())
    v = sg.judge_sufficiency("q", [_chunk("a")], "http://ollama")
    assert v["sufficient"] is True  # fail-safe
    assert v["missing"] == []


def test_judge_fail_safe_on_connection_error(monkeypatch):
    import httpx

    def _boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", _boom)
    v = sg.judge_sufficiency("q", [_chunk("a")], "http://ollama")
    assert v["sufficient"] is True


# --- run_sufficiency_loop: the four halting criteria ---

def _judge_seq(*verdicts):
    """A judge_fn that returns the given verdicts in order, then repeats the last."""
    calls = {"i": 0}
    seq = list(verdicts)

    def _fn(query, chunks, **kw):
        v = seq[min(calls["i"], len(seq) - 1)]
        calls["i"] += 1
        return v

    _fn.calls = calls
    return _fn


def test_halts_on_sufficient_without_retrieval():
    judge = _judge_seq({"sufficient": True, "missing": [], "requery": ""})
    retrieved = {"n": 0}

    def retrieve(q):
        retrieved["n"] += 1
        return [_chunk("new")]

    out = sg.run_sufficiency_loop("q", [_chunk("a")], retrieve,
                                  "http://ollama", judge_fn=judge)
    assert out["halted_by"] == "sufficient"
    assert out["loops"] == 0
    assert retrieved["n"] == 0  # never re-retrieved


def test_reretrieve_then_sufficient_grows_context():
    judge = _judge_seq(
        {"sufficient": False, "missing": ["X"], "requery": "find X"},
        {"sufficient": True, "missing": [], "requery": ""},
    )

    def retrieve(q):
        return [_chunk("b"), _chunk("c")]

    out = sg.run_sufficiency_loop("q", [_chunk("a")], retrieve,
                                  "http://ollama", judge_fn=judge, max_loops=3)
    assert out["halted_by"] == "sufficient"
    assert out["loops"] == 1
    ids = {c["chunk_id"] for c in out["final_chunks"]}
    assert ids == {"a", "b", "c"}  # merged fresh chunks


def test_halts_on_cap():
    judge = _judge_seq({"sufficient": False, "missing": ["X"], "requery": "more"})
    n = {"i": 0}

    def retrieve(q):
        n["i"] += 1
        return [_chunk(f"new{n['i']}")]  # always fresh

    out = sg.run_sufficiency_loop("q", [_chunk("a")], retrieve,
                                  "http://ollama", judge_fn=judge, max_loops=2)
    assert out["halted_by"] == "cap"
    assert out["loops"] == 2


def test_halts_on_no_progress():
    """Re-retrieval returns only chunks we already have → no new info → halt."""
    judge = _judge_seq({"sufficient": False, "missing": ["X"], "requery": "more"})

    def retrieve(q):
        return [_chunk("a")]  # duplicate of the seed → nothing fresh

    out = sg.run_sufficiency_loop("q", [_chunk("a")], retrieve,
                                  "http://ollama", judge_fn=judge, max_loops=5)
    assert out["halted_by"] == "no_progress"


def test_halts_on_budget():
    judge = _judge_seq({"sufficient": False, "missing": ["X"], "requery": "more"})
    ticks = iter([0.0, 999.0, 999.0, 999.0])

    def retrieve(q):
        return [_chunk("fresh")]

    out = sg.run_sufficiency_loop("q", [_chunk("a")], retrieve, "http://ollama",
                                  judge_fn=judge, max_loops=5, budget_s=20.0,
                                  clock=lambda: next(ticks))
    assert out["halted_by"] == "budget"


def test_halts_on_no_requery():
    """insufficient but the model gives no requery string → can't progress."""
    judge = _judge_seq({"sufficient": False, "missing": ["X"], "requery": ""})
    out = sg.run_sufficiency_loop("q", [_chunk("a")], lambda q: [_chunk("z")],
                                  "http://ollama", judge_fn=judge)
    assert out["halted_by"] == "no_requery"


def test_trace_records_each_verdict():
    judge = _judge_seq(
        {"sufficient": False, "missing": ["X"], "requery": "find X"},
        {"sufficient": True, "missing": [], "requery": ""},
    )
    out = sg.run_sufficiency_loop("q", [_chunk("a")], lambda q: [_chunk("b")],
                                  "http://ollama", judge_fn=judge, max_loops=3)
    assert len(out["trace"]) == 2
    assert out["trace"][0]["sufficient"] is False
    assert out["trace"][1]["sufficient"] is True
