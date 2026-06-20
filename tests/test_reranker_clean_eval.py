"""Tests for the leakage-free reranker clean-eval (nDCG vs Claude labels)."""
from __future__ import annotations

import asyncio

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


def test_ndcg_perfect_ranking_is_1():
    # model ranked chunks in Claude's exact relevance order → nDCG 1.0
    assert ra._ndcg_at_k([1.0, 0.6, 0.0], k=5) == 1.0


def test_ndcg_worst_ranking_below_1():
    # reverse order → strictly worse
    assert ra._ndcg_at_k([0.0, 0.6, 1.0], k=5) < 1.0


def test_ndcg_zero_when_no_relevance():
    assert ra._ndcg_at_k([0.0, 0.0], k=5) == 0.0


def test_clean_eval_requires_admin():
    with pytest.raises(HTTPException) as e:
        _run(ra.reranker_clean_eval(info=_viewer()))
    assert e.value.status_code == 403
