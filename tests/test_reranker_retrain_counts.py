"""Regression: training_data_counts.new_rows_since_train must count rows logged
AFTER the model trained — not max(0, windowed_count − all_time_trainset_size).

The old formula subtracted the model's all-time train size (n_train+n_test) from a
`days`-window row count. Once the trainset exceeds the window, the delta goes
negative → clamped to 0 forever → ready_to_retrain never fires → the reranker
never retrains despite hundreds of new injections/day (the stalled-loop bug
found live 2026-05-28: window=6718 vs trainset=18336 → new_rows=0).
"""
from __future__ import annotations

import asyncio


def _maybe_run(c):
    return asyncio.run(c) if asyncio.iscoroutine(c) else c
import sqlite3

from src.api.routes import reranker_admin as ra
import mayring_core.memory.reranker_v2 as rv2


def _conn_with_rows() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE context_feedback_log (captured_at TEXT, query TEXT, stage_scores TEXT)")
    conn.execute("CREATE TABLE chunk_feedback (created_at TEXT, signal TEXT)")
    # 3 rows BEFORE the train time, 5 rows AFTER it — all training-eligible.
    for ts in ("2026-05-28T10:00:00.000000",) * 3:
        conn.execute("INSERT INTO context_feedback_log VALUES (?, 'q', '{\"v\":1}')", (ts,))
    for m in range(5):
        conn.execute(
            "INSERT INTO context_feedback_log VALUES (?, 'q', '{\"v\":1}')",
            (f"2026-05-28T17:{m:02d}:00.000000",),
        )
    conn.commit()
    return conn


def test_new_rows_since_train_counts_rows_after_trained_at(monkeypatch):
    conn = _conn_with_rows()
    monkeypatch.setattr(ra, "_conn", lambda: conn)
    monkeypatch.setattr(ra, "_is_admin", lambda info: True)
    # All-time trainset (18336) FAR larger than the windowed count → the old
    # max(0, count − 18336) would have been 0.
    monkeypatch.setattr(rv2, "_load_model", lambda: {
        "trained_at": "2026-05-28T16:53:20.512801+00:00",  # ISO + tz vs no-tz rows
        "n_train": 16000, "n_test": 2336, "metrics": {},
    })

    out = _maybe_run(ra.training_data_counts(info=None, days=3650))

    assert out["new_rows_since_train"] == 5      # the 5 rows after 16:53 — NOT 0
    assert out["n_rows_at_last_train"] == 18336  # still surfaced, just not used for the delta


def test_new_rows_since_train_falls_back_to_log_count_without_model(monkeypatch):
    conn = _conn_with_rows()
    monkeypatch.setattr(ra, "_conn", lambda: conn)
    monkeypatch.setattr(ra, "_is_admin", lambda info: True)
    monkeypatch.setattr(rv2, "_load_model", lambda: None)  # never trained

    out = _maybe_run(ra.training_data_counts(info=None, days=3650))

    # No model → new_rows == full windowed count (8 rows, all training-eligible).
    assert out["new_rows_since_train"] == out["retrieval_log_with_features"] == 8


def test_counts_fall_back_to_latest_versioned_model(monkeypatch):
    """Seit dem Versioning akkumulieren Trainings als rerank_v3..vN — das Legacy-
    rerank_v2.json existiert nicht mehr. _load_model() (default v2) liefert dann
    None; die Counts müssen aufs NEUESTE trainierte Modell zurückfallen statt
    'nie trainiert' zu zeigen (Dashboard-Regression 2026-06-13)."""
    conn = _conn_with_rows()
    monkeypatch.setattr(ra, "_conn", lambda: conn)
    monkeypatch.setattr(ra, "_is_admin", lambda info: True)
    monkeypatch.setattr(rv2, "_load_model", lambda *a, **k: None)
    monkeypatch.setattr(rv2, "list_reranker_versions", lambda: [
        {"version": "v1", "baseline": True, "trained_at": None},
        {"version": "v5", "trained_at": "2026-06-08T19:03:15+00:00",
         "n_train": 20677, "n_test": 6410, "metrics": {"auc": 0.83}},
        {"version": "v7", "trained_at": "2026-06-12T21:14:18+00:00",
         "n_train": 21764, "n_test": 5937, "metrics": {"auc": 0.6}},
    ])

    out = _maybe_run(ra.training_data_counts(info=None, days=3650))
    assert out["last_trained_at"] == "2026-06-12T21:14:18+00:00"
    assert out["n_rows_at_last_train"] == 21764 + 5937
    assert out["last_metrics"] == {"auc": 0.6}
