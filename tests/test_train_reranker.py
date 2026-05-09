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


def _row(v: float, s: float, igio: str, label: int) -> dict:
    """One training row in the v3-feature schema."""
    feats = {
        "v": v, "s": s, "r": 0.5, "a": 0.0,
        "igio_issue": 1.0 if igio == "issue" else 0.0,
        "igio_goal": 1.0 if igio == "goal" else 0.0,
        "igio_intervention": 1.0 if igio == "intervention" else 0.0,
        "igio_outcome": 1.0 if igio == "outcome" else 0.0,
        "igio_unknown": 1.0 if igio == "unknown" else 0.0,
    }
    return {"chunk_id": f"c_{v}_{s}", "features": feats, "label": label}


def test_healthy_data_writes_model(tmp_path):
    """Sanity baseline: data with strong positive v, s in label=1
    rows trains a model with positive v, s weights and writes the
    file."""
    from tools.train_reranker import train

    rows = []
    # 60 positives where high v + high s → strong positive signal
    for i in range(60):
        rows.append(_row(0.8, 0.7, "outcome", 1))
    # 60 negatives where low v + low s
    for i in range(60):
        rows.append(_row(0.1, 0.1, "intervention", 0))
    in_path = tmp_path / "ds.jsonl"
    out_path = tmp_path / "model.json"
    _write_dataset(in_path, rows)

    rc = train(in_path, out_path)
    assert rc == 0
    assert out_path.exists()
    model = json.loads(out_path.read_text())
    assert model["weights"]["v"] >= 0
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
        rows.append(_row(0.1, 0.7, "outcome", 1))
    for i in range(60):
        rows.append(_row(0.9, 0.1, "intervention", 0))
    in_path = tmp_path / "ds.jsonl"
    out_path = tmp_path / "model.json"
    _write_dataset(in_path, rows)

    rc = train(in_path, out_path)
    assert rc == 3, "trainer should return 3 for rejected models"
    assert not out_path.exists(), \
        "rejected model must NOT touch disk (else loader still reads it)"


def test_rejects_negative_symbolic_weight(tmp_path):
    """Same as above but symbolic flipped. Symbolic is also a
    retrieval-positive signal; negative s makes the model rank
    token-overlap-LESS chunks higher, which is incoherent."""
    from tools.train_reranker import train

    rows = []
    for i in range(60):
        rows.append(_row(0.7, 0.1, "outcome", 1))
    for i in range(60):
        rows.append(_row(0.1, 0.9, "intervention", 0))
    in_path = tmp_path / "ds.jsonl"
    out_path = tmp_path / "model.json"
    _write_dataset(in_path, rows)

    rc = train(in_path, out_path)
    assert rc == 3
    assert not out_path.exists()


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
    rows = [_row(0.1, 0.1, "outcome", 1) for _ in range(60)]
    rows += [_row(0.9, 0.9, "intervention", 0) for _ in range(60)]
    in_path = tmp_path / "ds.jsonl"
    _write_dataset(in_path, rows)

    rc = train(in_path, out_path)
    assert rc == 3
    # File untouched
    assert out_path.stat().st_mtime == original_mtime
    saved = json.loads(out_path.read_text())
    assert saved["weights"]["v"] == 0.3  # original kept
