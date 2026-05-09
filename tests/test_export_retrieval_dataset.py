"""Tests for the reranker training-data exporter.

The exporter is what got Issue #180's degenerate model trained — the
two killer bugs were:

  1. Chunk-global label aggregation made 9 chunks responsible for 42%
     of all positives → model learned chunk-id leaderboard.
  2. sf as feature + chunk_feedback as label source → target leakage,
     sf ended up with weight 8.77 (17× bigger than every other feature).

Plus 32% of training events were smoke-test / task-notification noise
that pulled the weights toward synthetic-pattern equilibrium.

These tests guard against regression on each of those.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


def _build_db(path: Path) -> sqlite3.Connection:
    """Minimal schema needed by export_retrieval_dataset.py."""
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            igio_axis TEXT
        );
        CREATE TABLE context_feedback_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            trigger_ids TEXT,
            stage_scores TEXT,
            was_referenced INTEGER,
            captured_at TEXT,
            workspace_id TEXT
        );
        CREATE TABLE chunk_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT,
            signal TEXT,
            metadata TEXT,
            created_at TEXT
        );
    """)
    return conn


def _insert_event(conn, query, chunks, scores_per_chunk, was_referenced=1):
    conn.execute(
        "INSERT INTO context_feedback_log "
        "(query, trigger_ids, stage_scores, was_referenced, captured_at, workspace_id) "
        "VALUES (?,?,?,?,datetime('now'),?)",
        (query, json.dumps(chunks),
         json.dumps(scores_per_chunk), was_referenced, "bene"),
    )


def test_noise_queries_are_filtered_out(tmp_path):
    """smoke/marker/task-notification queries pollute training — exporter
    must drop them. Production data: 32% of 1133 events were noise."""
    from tools.export_retrieval_dataset import export

    conn = _build_db(tmp_path / "memory.db")
    # 1 real event + 4 noise events
    _insert_event(conn, "real user question", ["chk_a"],
                  {"chk_a": {"v": 0.5, "s": 0.3, "r": 0.8, "a": 0.0}})
    _insert_event(conn, "smoke watcher 1234", ["chk_a"],
                  {"chk_a": {"v": 0.5, "s": 0.3, "r": 0.8, "a": 0.0}})
    _insert_event(conn, "marker token 5678", ["chk_a"],
                  {"chk_a": {"v": 0.5, "s": 0.3, "r": 0.8, "a": 0.0}})
    _insert_event(conn, "<task-notification>...", ["chk_a"],
                  {"chk_a": {"v": 0.5, "s": 0.3, "r": 0.8, "a": 0.0}})
    _insert_event(conn, "fix bug", ["chk_a"],
                  {"chk_a": {"v": 0.5, "s": 0.3, "r": 0.8, "a": 0.0}})
    conn.commit(); conn.close()

    out = tmp_path / "ds.jsonl"
    n = export(tmp_path / "memory.db", out, days=30, negative_mode="unlabeled")

    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert n == len(rows) == 1
    assert rows[0]["query"] == "real user question"


def test_features_drop_sf_and_sl(tmp_path):
    """sf and sl were the target-leakage culprits in v2 — they MUST not
    appear in exported feature dicts. (sf is computed from chunk_feedback
    which is also where the label comes from.)"""
    from tools.export_retrieval_dataset import export

    conn = _build_db(tmp_path / "memory.db")
    _insert_event(
        conn, "user question", ["chk_x"],
        {"chk_x": {"v": 0.5, "s": 0.3, "r": 0.8, "a": 0.0,
                   "sf": 0.9, "sl": 0.7}},
    )
    conn.commit(); conn.close()

    out = tmp_path / "ds.jsonl"
    export(tmp_path / "memory.db", out, days=30, negative_mode="unlabeled")
    feats = json.loads(out.read_text().splitlines()[0])["features"]
    assert "sf" not in feats
    assert "sl" not in feats
    # but the retrieval features ARE there
    assert {"v", "s", "r", "a"} <= set(feats)


def test_igio_axis_one_hot_in_features(tmp_path):
    """IGIO axis is the new training signal (Issue #180). Each chunk
    must contribute its axis as one-hot, with 'unknown' as the
    fallback for unclassified chunks."""
    from tools.export_retrieval_dataset import export

    conn = _build_db(tmp_path / "memory.db")
    conn.execute("INSERT INTO chunks(chunk_id, igio_axis) VALUES('chk_classified','outcome')")
    # chk_other not in chunks → fallback 'unknown'
    _insert_event(
        conn, "real q", ["chk_classified", "chk_other"],
        {"chk_classified": {"v": 0.5, "s": 0.3, "r": 0.8, "a": 0.0},
         "chk_other":      {"v": 0.6, "s": 0.4, "r": 0.7, "a": 0.0}},
    )
    conn.commit(); conn.close()

    out = tmp_path / "ds.jsonl"
    export(tmp_path / "memory.db", out, days=30, negative_mode="unlabeled")
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    by_id = {r["chunk_id"]: r for r in rows}
    assert by_id["chk_classified"]["features"]["igio_outcome"] == 1.0
    assert by_id["chk_classified"]["features"]["igio_unknown"] == 0.0
    assert by_id["chk_other"]["features"]["igio_unknown"] == 1.0
    assert by_id["chk_other"]["features"]["igio_outcome"] == 0.0


def test_label_uses_event_was_referenced_not_chunk_global(tmp_path):
    """Critical regression: the OLD label_map aggregated chunk_feedback
    rows into a per-chunk-id label, which made 9 chunks dominate the
    positive class (42% of v2's positives came from one chunk). The
    new exporter uses event-level was_referenced so two events
    referencing the same chunk_id can produce different labels."""
    from tools.export_retrieval_dataset import export

    conn = _build_db(tmp_path / "memory.db")
    # Same chunk in two events, one referenced, one not
    _insert_event(
        conn, "q1", ["chk_x"],
        {"chk_x": {"v": 0.5, "s": 0.3, "r": 0.8, "a": 0.0}},
        was_referenced=1,
    )
    _insert_event(
        conn, "q2", ["chk_x"],
        {"chk_x": {"v": 0.5, "s": 0.3, "r": 0.8, "a": 0.0}},
        was_referenced=0,
    )
    conn.commit(); conn.close()

    out = tmp_path / "ds.jsonl"
    export(tmp_path / "memory.db", out, days=30, negative_mode="unlabeled")
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    labels_by_query = {r["query"]: r["label"] for r in rows}
    assert labels_by_query == {"q1": 1, "q2": 0}
