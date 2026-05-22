"""Sanity checks for reranker_v2 model loader.

Real bug from production: training run produced
weights={v: -0.51, s: 2.71, sf: 8.77, ...} — sf alone (feedback) had a
17x stronger pull than every other feature combined, and v (vector)
ended up *negative*. Effect: vector-stage hits in /memory/search were
*demoted* below symbolic-only candidates. The smoke RAG check kept
seeing v=0 in top-5 because vector-positive chunks were sorted out.

Defence: refuse to load any v2 model that has negative vector or
symbolic weights — those are retrieval-positive signals by design.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_model(path: Path, weights: dict[str, float]) -> None:
    path.write_text(json.dumps({
        "version": "v2",
        "estimator": "logistic_regression",
        "intercept": 0.0,
        "weights": weights,
        "metrics": {"auc": 0.9},
    }))


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Each test gets its own cache dir + cleared module cache."""
    from mayring_core import config as _cfg
    monkeypatch.setattr(_cfg, "CACHE_DIR", tmp_path)
    from mayring_core.memory import reranker_v2
    reranker_v2.invalidate_v2_cache()
    yield
    reranker_v2.invalidate_v2_cache()


def test_healthy_weights_load(tmp_path):
    from mayring_core.memory import reranker_v2
    _write_model(tmp_path / "rerank_v2.json", {
        "v": 0.5, "s": 0.4, "r": 0.1, "a": 0.1, "sf": 0.3, "sl": 0.2,
    })
    model = reranker_v2._load_model()
    assert model is not None
    assert model["weights"]["v"] == 0.5


def test_negative_vector_weight_rejected(tmp_path):
    """Production failure: v=-0.51 made vector hits rank below noise."""
    from mayring_core.memory import reranker_v2
    _write_model(tmp_path / "rerank_v2.json", {
        "v": -0.51, "s": 2.71, "r": -2.51, "a": 0.0, "sf": 8.77, "sl": -0.78,
    })
    model = reranker_v2._load_model()
    assert model is None


def test_negative_symbolic_weight_rejected(tmp_path):
    """Symbolic token-overlap is also retrieval-positive — guard both."""
    from mayring_core.memory import reranker_v2
    _write_model(tmp_path / "rerank_v2.json", {
        "v": 0.5, "s": -0.1, "r": 0.0, "a": 0.0, "sf": 0.0, "sl": 0.0,
    })
    assert reranker_v2._load_model() is None


def test_negative_pt_or_re_weight_rejected(tmp_path):
    """Issue #187: pt (predicted-topic) und re (rationale-presence) sind
    retrieval-positive Features. Negative weights würden chunks mit predicted-
    topic-match oder rationale-edge AKTIV runter ranken — analog v/s-flip."""
    from mayring_core.memory import reranker_v2
    # pt negativ
    _write_model(tmp_path / "rerank_v2.json", {
        "v": 0.5, "s": 0.4, "r": 0.1, "a": 0.1,
        "pt": -0.1, "re": 0.1,
        "sf": 0.3, "sl": 0.2,
    })
    assert reranker_v2._load_model() is None

    reranker_v2.invalidate_v2_cache()

    # re negativ
    _write_model(tmp_path / "rerank_v2.json", {
        "v": 0.5, "s": 0.4, "r": 0.1, "a": 0.1,
        "pt": 0.1, "re": -0.05,
        "sf": 0.3, "sl": 0.2,
    })
    assert reranker_v2._load_model() is None


def test_zero_weights_pass(tmp_path):
    """Zero-weight on retrieval-positive feature is suspicious but legal —
    only NEGATIVE blocks. Catches the actual production bug, doesn't
    overclaim."""
    from mayring_core.memory import reranker_v2
    _write_model(tmp_path / "rerank_v2.json", {
        "v": 0.0, "s": 0.0, "r": 0.0, "a": 0.0, "sf": 0.5, "sl": 0.5,
    })
    assert reranker_v2._load_model() is not None


def test_get_active_falls_back_to_v1_on_degenerate_model(tmp_path, monkeypatch):
    """End-to-end: even with RERANKER_VERSION=v2, a degenerate model
    triggers the silent v1-fallback that already exists for missing files."""
    from mayring_core.memory import reranker_v2
    _write_model(tmp_path / "rerank_v2.json", {
        "v": -1.0, "s": 1.0, "r": 0.0, "a": 0.0, "sf": 0.0, "sl": 0.0,
    })
    monkeypatch.setenv("RERANKER_VERSION", "v2")
    version, model = reranker_v2.get_active_reranker()
    assert version == "v1"
    assert model is None
