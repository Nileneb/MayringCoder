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


def test_negative_symbolic_no_longer_gated(tmp_path):
    """2026-06-05: s (symbolic) ist kein v2-Feature mehr (kollinear zu v → gedroppt).
    Ein altes Modell mit negativem s darf NICHT mehr deshalb rejected werden —
    nur v (echter #180-Leak) hard-gated. v=+0.5 stark positiv → muss laden."""
    from mayring_core.memory import reranker_v2
    _write_model(tmp_path / "rerank_v2.json", {
        "v": 0.5, "s": -0.1, "r": 0.0, "a": 0.0, "sf": 0.0, "sl": 0.0,
    })
    assert reranker_v2._load_model() is not None  # lädt trotz s<0


def test_negative_pt_weight_rejected(tmp_path):
    """Issue #187: pt (predicted-topic) ist ein retrieval-positives Feature.
    Negatives Gewicht würde chunks mit predicted-topic-match AKTIV runter ranken
    — analog v/s-flip → reject."""
    from mayring_core.memory import reranker_v2
    _write_model(tmp_path / "rerank_v2.json", {
        "v": 0.5, "s": 0.4, "r": 0.1, "a": 0.1,
        "pt": -0.1,
        "sf": 0.3, "sl": 0.2,
    })
    assert reranker_v2._load_model() is None


def test_negative_re_weight_no_longer_gated(tmp_path):
    """2026-06-05: re (rationale_edge) ist KEIN Feature mehr (am Inferenz-Pfad
    nicht lieferbar). Ein altes Modell mit negativem re darf NICHT mehr deshalb
    rejected werden — sonst bliebe v2 ewig tot (re=-0.67 war genau der Grund)."""
    from mayring_core.memory import reranker_v2
    _write_model(tmp_path / "rerank_v2.json", {
        "v": 0.5, "s": 0.4, "r": 0.1, "a": 0.1,
        "pt": 0.1, "re": -0.67,
        "sf": 0.3, "sl": 0.2,
    })
    assert reranker_v2._load_model() is not None  # lädt trotz re<0


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


def test_delete_reranker_version_guards_active_and_v1(tmp_path):
    """Delete-Button-Backend: trainierte v<N> löschbar; v1 + aktives Modell geschützt."""
    from mayring_core.memory import reranker_v2
    _write_model(tmp_path / "rerank_v2.json", {"v": 0.9, "r": 0.1, "pt": 0.0})
    _write_model(tmp_path / "rerank_v3.json", {"v": 0.8, "r": 0.1, "pt": 0.0})
    reranker_v2.write_runtime_default("v3")          # v3 aktiv
    assert reranker_v2.delete_reranker_version("v2") is True      # nicht-aktiv → weg
    assert not (tmp_path / "rerank_v2.json").exists()
    with pytest.raises(ValueError):                  # aktiv → geschützt
        reranker_v2.delete_reranker_version("v3")
    with pytest.raises(ValueError):                  # v1 baseline → geschützt
        reranker_v2.delete_reranker_version("v1")


def test_autorollout_flag_toggle(tmp_path):
    """A/B-Gate abstellbar: default an, off → aus, on → wieder an."""
    from mayring_core.memory import reranker_v2
    assert reranker_v2.read_autorollout_enabled() is True   # default
    reranker_v2.write_autorollout_enabled(False)
    assert reranker_v2.read_autorollout_enabled() is False
    reranker_v2.write_autorollout_enabled(True)
    assert reranker_v2.read_autorollout_enabled() is True
