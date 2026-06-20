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


# --- derive_task / decompose: fail-safe ---

def test_derive_task_falls_back_to_raw_prompt(monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (_ for _ in ()).throw(
        httpx.ConnectError("x")))
    assert sg.derive_task("JAAAA mach den reranker") == "JAAAA mach den reranker"


def test_derive_task_extracts(monkeypatch):
    import httpx

    class _R:
        def raise_for_status(self): ...
        def json(self): return {"message": {"content": '{"task": "Reranker aktivieren"}'}}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _R())
    assert sg.derive_task("JAAAA mach den reranker") == "Reranker aktivieren"


def test_derive_and_decompose_one_call(monkeypatch):
    """Fused distill+decompose: ONE gemma call returns task AND sub-questions."""
    import httpx
    calls = {"n": 0}

    class _R:
        def raise_for_status(self): ...
        def json(self):
            calls["n"] += 1
            return {"message": {"content":
                '{"task": "Reranker aktivieren", "questions": ["wie active sync", "wie rollback"]}'}}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _R())
    task, qs = sg.derive_and_decompose("JAAAA mach den reranker scharf")
    assert task == "Reranker aktivieren"
    assert qs == ["wie active sync", "wie rollback"]
    assert calls["n"] == 1  # the whole point: one call, not two


def test_derive_and_decompose_caps_max_q(monkeypatch):
    import httpx

    class _R:
        def raise_for_status(self): ...
        def json(self): return {"message": {"content":
            '{"task": "t", "questions": ["a", "b", "c", "d", "e"]}'}}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _R())
    _, qs = sg.derive_and_decompose("prompt", max_q=3)
    assert qs == ["a", "b", "c"]


def test_derive_and_decompose_fail_safe(monkeypatch):
    """On error → (raw prompt, []) so the loop seeds with the raw prompt, never worse."""
    import httpx
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (_ for _ in ()).throw(
        httpx.ConnectError("x")))
    task, qs = sg.derive_and_decompose("  the raw prompt  ")
    assert task == "the raw prompt"
    assert qs == []


def test_loop_uses_passed_questions_skips_decompose():
    """run_task_loop with questions=[...] uses them directly and never decomposes
    (the fused-call path). Task is still seeded as the first query."""
    seen = []

    def _decompose_boom(t):
        raise AssertionError("decompose must NOT run when questions are passed")

    out = sg.run_task_loop(
        "the task", lambda q: (seen.append(q), [_chunk(q)])[1],
        questions=["facet a", "facet b"],
        decompose_fn=_decompose_boom, parallelism=1,  # deterministic side-effect order
        answered_fn=lambda q, ch: True, max_loops=1)
    # seeded task + the two passed facets were all searched, in order
    assert seen == ["the task", "facet a", "facet b"]
    assert out["questions"] == ["the task", "facet a", "facet b"]


def test_loop_empty_questions_means_single_shot_on_task():
    """questions=[] (fused fail-safe) → loop runs single-shot on the seeded task."""
    seen = []
    out = sg.run_task_loop(
        "only the task", lambda q: (seen.append(q), [_chunk(q)])[1],
        questions=[], answered_fn=lambda q, ch: True, max_loops=1)
    assert seen == ["only the task"]
    assert out["questions"] == ["only the task"]


def test_judge_answered_batch_parses_index_map(monkeypatch):
    import httpx

    class _R:
        def raise_for_status(self): ...
        def json(self): return {"message": {"content":
            '{"answered": {"0": true, "1": false, "2": true}}'}}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _R())
    out = sg.judge_answered_batch(["q0", "q1", "q2"], [_chunk("a")])
    assert out == [True, False, True]


def test_judge_answered_batch_fail_safe(monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (_ for _ in ()).throw(
        httpx.ConnectError("x")))
    # error → all answered (never loop forever)
    assert sg.judge_answered_batch(["q0", "q1"], [_chunk("a")]) == [True, True]


def test_judge_answered_batch_missing_index_defaults_true(monkeypatch):
    import httpx

    class _R:
        def raise_for_status(self): ...
        def json(self): return {"message": {"content": '{"answered": {"0": false}}'}}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _R())
    # index 1 missing → defaults True (fail-safe)
    assert sg.judge_answered_batch(["q0", "q1"], [_chunk("a")]) == [False, True]


def test_loop_uses_batch_judge_when_no_answered_fn(monkeypatch):
    """Without an injected answered_fn the loop makes ONE batched judge call per
    round (not N) — the gemma-load reduction."""
    calls = {"n": 0}

    def _fake_batch(qs, ch, *a, **k):
        calls["n"] += 1
        return [True] * len(qs)  # all answered → halt after one round

    monkeypatch.setattr(sg, "judge_answered_batch", _fake_batch)
    out = sg.run_task_loop("the-task", lambda q: [_chunk(q)], "http://o",
                           decompose_fn=lambda t: ["qa", "qb"])  # no answered_fn
    assert out["halted_by"] == "all_answered"
    assert calls["n"] == 1  # ONE batched call for all 3 questions, not 3


def test_decompose_falls_back_to_task(monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (_ for _ in ()).throw(
        httpx.ConnectError("x")))
    assert sg.decompose_questions("Reranker aktivieren") == ["Reranker aktivieren"]


# --- run_task_loop: question-decomposition + semantic halt ---

def test_task_loop_all_answered():
    decompose = lambda t: ["q1", "q2"]
    answered = lambda q, ch: True  # both answered immediately
    out = sg.run_task_loop("task", lambda q: [_chunk(q)], "http://ollama",
                           decompose_fn=decompose, answered_fn=answered,
                           seed_with_task=False)
    assert out["halted_by"] == "all_answered"
    assert out["questions"] == ["q1", "q2"]
    # collected one chunk per question
    assert {c["chunk_id"] for c in out["final_chunks"]} == {"q1", "q2"}


def test_task_loop_seeds_with_task_query():
    """The task itself is the first query (primary anchor); sub-questions only
    broaden. Without this the loop discards the good task-anchor retrieval."""
    decompose = lambda t: ["q1"]
    answered = lambda q, ch: True
    out = sg.run_task_loop("the-task", lambda q: [_chunk(q)], "http://ollama",
                           decompose_fn=decompose, answered_fn=answered)
    assert out["questions"][0] == "the-task"
    assert "the-task" in {c["chunk_id"] for c in out["final_chunks"]}


def test_task_loop_collects_from_all_questions():
    """Each sub-question fans out its own retrieval → broader recall than one query."""
    decompose = lambda t: ["qa", "qb"]
    answered = lambda q, ch: True
    catalog = {"qa": [_chunk("a1"), _chunk("a2")], "qb": [_chunk("b1")]}
    out = sg.run_task_loop("task", lambda q: catalog[q], "http://ollama",
                           decompose_fn=decompose, answered_fn=answered,
                           seed_with_task=False)
    assert {c["chunk_id"] for c in out["final_chunks"]} == {"a1", "a2", "b1"}


def test_task_loop_halts_no_progress():
    decompose = lambda t: ["q1"]
    answered = lambda q, ch: False  # never answered
    out = sg.run_task_loop("task", lambda q: [_chunk("only")], "http://ollama",
                           decompose_fn=decompose, answered_fn=answered, max_loops=5)
    assert out["halted_by"] == "no_progress"  # round 2 brings nothing fresh


def test_task_loop_halts_cap():
    decompose = lambda t: ["q1"]
    answered = lambda q, ch: False
    n = {"i": 0}

    def retrieve(q):
        n["i"] += 1
        return [_chunk(f"c{n['i']}")]  # always fresh

    out = sg.run_task_loop("task", retrieve, "http://ollama",
                           decompose_fn=decompose, answered_fn=answered, max_loops=2)
    assert out["halted_by"] == "cap"
    assert out["loops"] == 2


def test_task_loop_parallel_is_faster_and_deterministic():
    """Parallel sub-question retrieval: 3 questions × 0.15s each cost ~0.15s
    parallel vs ~0.45s sequential — AND the merged chunk set is identical
    (order-preserving map + seen-set keep determinism)."""
    import time as _t
    decompose = lambda t: ["qa", "qb"]  # +seed → 3 questions
    answered = lambda q, ch: True
    catalog = {"the-task": [_chunk("t1")], "qa": [_chunk("a1")], "qb": [_chunk("b1")]}

    def slow_retrieve(q):
        _t.sleep(0.15)
        return catalog.get(q, [])

    t0 = _t.time()
    seq = sg.run_task_loop("the-task", slow_retrieve, "http://o",
                           decompose_fn=decompose, answered_fn=answered, parallelism=1)
    seq_dt = _t.time() - t0

    t0 = _t.time()
    par = sg.run_task_loop("the-task", slow_retrieve, "http://o",
                           decompose_fn=decompose, answered_fn=answered, parallelism=4)
    par_dt = _t.time() - t0

    # same result regardless of parallelism
    assert {c["chunk_id"] for c in seq["final_chunks"]} == {"t1", "a1", "b1"}
    assert {c["chunk_id"] for c in par["final_chunks"]} == {"t1", "a1", "b1"}
    # parallel meaningfully faster (3 questions: ~0.45s seq vs ~0.15s par)
    assert par_dt < seq_dt * 0.6, f"parallel {par_dt:.2f}s not < 0.6×seq {seq_dt:.2f}s"


def test_task_loop_halts_budget():
    decompose = lambda t: ["q1"]
    answered = lambda q, ch: False
    ticks = iter([0.0, 999.0, 999.0, 999.0])
    n = {"i": 0}

    def retrieve(q):
        n["i"] += 1
        return [_chunk(f"c{n['i']}")]

    out = sg.run_task_loop("task", retrieve, "http://ollama",
                           decompose_fn=decompose, answered_fn=answered,
                           max_loops=9, budget_s=20.0, clock=lambda: next(ticks))
    assert out["halted_by"] == "budget"
