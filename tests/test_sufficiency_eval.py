"""Tests for the sufficiency-gate outcome metric (_sufficiency_agreement).

The metric asks: does gemma's sufficient/insufficient verdict agree with the
Claude relevance labels? Ground-truth proxy = max chunk relevance ≥ threshold.
"""
from __future__ import annotations

import src.api.routes.reranker_admin as ra


def _eq(query, rels):
    return {"query": query, "chunks": [{"chunk_id": f"c{i}", "text": "t"}
                                       for i in range(len(rels))], "rels": rels}


def test_perfect_agreement():
    eqs = [_eq("a", [0.9]), _eq("b", [0.1, 0.2])]
    # gemma mirrors objective truth: a sufficient, b not
    judge = lambda q, ch: {"sufficient": q == "a"}
    r = ra._sufficiency_agreement(eqs, judge)
    assert r["agreement"] == 1.0
    assert r["false_pass_rate"] == 0.0
    assert r["false_loop_rate"] == 0.0


def test_false_pass_is_flagged():
    """gemma says sufficient on an objectively-insufficient query → dangerous."""
    eqs = [_eq("weak", [0.2, 0.3])]
    judge = lambda q, ch: {"sufficient": True}
    r = ra._sufficiency_agreement(eqs, judge)
    assert r["false_pass_rate"] == 1.0
    assert r["agreement"] == 0.0


def test_false_loop_is_flagged():
    """gemma says insufficient on an objectively-sufficient query → wasted loop."""
    eqs = [_eq("strong", [0.9])]
    judge = lambda q, ch: {"sufficient": False}
    r = ra._sufficiency_agreement(eqs, judge)
    assert r["false_loop_rate"] == 1.0


def test_threshold_boundary():
    eqs = [_eq("edge", [0.6])]  # exactly at threshold → objective sufficient
    judge = lambda q, ch: {"sufficient": True}
    assert ra._sufficiency_agreement(eqs, judge, rel_threshold=0.6)["agreement"] == 1.0
    # raise threshold above 0.6 → now objectively insufficient → gemma's True is a false_pass
    assert ra._sufficiency_agreement(eqs, judge, rel_threshold=0.7)["false_pass_rate"] == 1.0


def test_empty_returns_empty():
    assert ra._sufficiency_agreement([], lambda q, ch: {"sufficient": True}) == {}


def test_counts_reported():
    eqs = [_eq("a", [0.9]), _eq("b", [0.1]), _eq("c", [0.8])]
    judge = lambda q, ch: {"sufficient": True}  # always sufficient
    r = ra._sufficiency_agreement(eqs, judge)
    assert r["eval_queries"] == 3
    assert r["objective_sufficient"] == 2  # a, c
    assert r["gemma_sufficient"] == 3
