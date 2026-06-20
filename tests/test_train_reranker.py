"""Tests for tools/train_reranker.py — focus on the trainer-side sanity
gate that prevents degenerate models from ever hitting disk.

Loader-side gate (src/memory/reranker_v2.py::_load_model) catches
degenerate models at runtime, but if the trainer keeps writing them
the cron success-rate stays at 100% while the runtime silently falls
back to v1. The trainer must reject upstream so failures are loud.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_dataset(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _row(v: float, s: float, label: int) -> dict:
    """One training row in the v3-feature schema."""
    feats = {"v": v, "s": s, "r": 0.5, "a": 0.0, "pt": 0.0}
    return {"chunk_id": f"c_{v}_{s}", "features": feats, "label": label}


def test_healthy_data_writes_model(tmp_path):
    """Sanity baseline: data with strong positive v, s in label=1
    rows trains a model with positive v, s weights and writes the
    file."""
    from tools.train_reranker import train

    rows = []
    # 60 positives where high v + high s → strong positive signal
    for i in range(60):
        rows.append(_row(0.8, 0.7, 1))
    # 60 negatives where low v + low s
    for i in range(60):
        rows.append(_row(0.1, 0.1, 0))
    in_path = tmp_path / "ds.jsonl"
    out_path = tmp_path / "model.json"
    _write_dataset(in_path, rows)

    rc = train(in_path, out_path)
    assert rc == 0
    assert out_path.exists()
    model = json.loads(out_path.read_text())
    assert model["weights"]["v"] >= 0
    # WHY(2026-06-20): s (symbolic) wieder lernbares Feature — Clean-Eval zeigte,
    # das s-lose Feature-Set fiel UNTER die Vektor-Baseline; v3 (mit s) gewann.
    assert model["weights"]["s"] >= 0


def test_rejects_negative_vector_weight(tmp_path):
    """Construct data where v negatively correlates with label —
    trainer must REFUSE TO WRITE the file. Real Issue-#180 scenario:
    the data made v=-0.51 in production, runtime then silent-fallbacks
    to v1 forever while the cron reports green."""
    from tools.train_reranker import train

    rows = []
    # Pattern: high v ↔ label=0, low v ↔ label=1 (inverted signal)
    for i in range(60):
        rows.append(_row(0.1, 0.7, 1))
    for i in range(60):
        rows.append(_row(0.9, 0.1, 0))
    in_path = tmp_path / "ds.jsonl"
    out_path = tmp_path / "model.json"
    _write_dataset(in_path, rows)

    rc = train(in_path, out_path)
    assert rc == 3, "trainer should return 3 for rejected models"
    assert not out_path.exists(), \
        "rejected model must NOT touch disk (else loader still reads it)"


def test_rejection_preserves_existing_model(tmp_path):
    """If the cron rejects today's training, yesterday's model file
    should remain untouched. Without this guarantee a single bad
    training run would wipe a working model."""
    from tools.train_reranker import train

    out_path = tmp_path / "model.json"
    out_path.write_text(json.dumps({"version": "v2", "weights":
                                    {"v": 0.3, "s": 0.2}}))
    original_mtime = out_path.stat().st_mtime

    # Bad data → reject
    rows = [_row(0.1, 0.1, 1) for _ in range(60)]
    rows += [_row(0.9, 0.9, 0) for _ in range(60)]
    in_path = tmp_path / "ds.jsonl"
    _write_dataset(in_path, rows)

    rc = train(in_path, out_path)
    assert rc == 3
    # File untouched
    assert out_path.stat().st_mtime == original_mtime
    saved = json.loads(out_path.read_text())
    assert saved["weights"]["v"] == 0.3  # original kept


def test_neutralizes_negative_pt_so_model_loads():
    """Root cause of 'v2 never live' (2026-05-28): the trainer wrote models with
    a negative pt (and the now-removed re) that the LOADER then silently rejected
    → v1 fallback. Negative pt must be clamped to 0 (loader tolerates 0); v/s and
    the non-gated r/a weights stay untouched. re ist seit 2026-06-05 kein
    Feature mehr → nicht mehr geclampt."""
    from tools.train_reranker import _neutralize_weak_negatives

    w = {"v": 0.5, "s": 1.0, "pt": -0.3,
         "r": -0.2847, "a": -1.0}
    out = _neutralize_weak_negatives(w)
    # loader-gated weak feature neutralized
    assert out["pt"] == 0.0
    # NOT loader-gated → learned sign preserved
    assert out["r"] == -0.2847
    assert out["a"] == -1.0
    # v/s never clamped here (their negative is a hard-reject elsewhere)
    assert out["v"] == 0.5 and out["s"] == 1.0


def test_positive_pt_untouched():
    from tools.train_reranker import _neutralize_weak_negatives
    w = {"v": 0.5, "s": 1.0, "pt": 0.4}
    assert _neutralize_weak_negatives(w) == w
