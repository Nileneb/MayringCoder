import json

import pytest

from mayring_core.memory import reranker_v2 as rk


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "_cache_dir", lambda: tmp_path)
    rk.invalidate_v2_cache()
    return tmp_path


def _write_model(cache, version, weights=None, metrics=None):
    (cache / f"rerank_{version}.json").write_text(json.dumps({
        "version": version,
        "trained_at": f"2026-05-30T0{version[-1]}:00:00Z",
        "n_train": 400, "n_test": 100,
        "metrics": metrics or {"auc": 0.71, "p@1": 0.5, "ndcg@5": 0.58},
        "weights": weights or {"v": 1.2, "s": 0.4, "r": 0.05, "a": 0.18, "cat_match": 0.0},
        "intercept": -0.3,
    }))


def test_list_versions_includes_baseline_and_files(cache):
    _write_model(cache, "v2")
    _write_model(cache, "v3")
    (cache / "rerank_default.txt").write_text("v3")
    versions = {v["version"]: v for v in rk.list_reranker_versions()}
    assert "v1" in versions and versions["v1"]["baseline"] is True
    assert "v2" in versions and "v3" in versions
    assert versions["v3"]["active"] is True
    assert versions["v2"]["active"] is False
    assert versions["v3"]["metrics"]["ndcg@5"] == 0.58
    assert versions["v3"]["n_train"] == 400


def test_write_runtime_default_accepts_existing_vN(cache):
    _write_model(cache, "v3")
    assert rk.write_runtime_default("v3") == "v3"
    assert rk._read_runtime_default() == "v3"
    assert rk.write_runtime_default("v1") == "v1"
    with pytest.raises(ValueError):
        rk.write_runtime_default("v9")  # no rerank_v9.json


def test_get_active_reranker_loads_vN(cache):
    _write_model(cache, "v3", weights={"v": 2.0, "s": 0.5, "r": 0.1, "a": 0.2, "cat_match": 0.0})
    (cache / "rerank_default.txt").write_text("v3")
    version, model = rk.get_active_reranker(query_hint="q")
    assert version == "v3"
    assert model is not None and model["weights"]["v"] == 2.0
