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
            chunk_id TEXT PRIMARY KEY
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


# ── Tests for #209 rating-weighted samples ────────────────────────

def _insert_rating(conn, chunk_id: str, rating: str) -> None:
    conn.execute(
        "INSERT INTO chunk_feedback (chunk_id, signal, metadata, created_at) "
        "VALUES (?, ?, '{}', datetime('now'))",
        (chunk_id, rating),
    )


def _export_to_jsonl(tmp_path, conn) -> list[dict]:
    """Helper: run the exporter on the test DB, return parsed rows."""
    from tools.export_retrieval_dataset import export
    out_path = tmp_path / "ds.jsonl"
    db_path = tmp_path / "test.db"
    # The conn passed in already points to db_path
    export(db_path, out_path, days=30, negative_mode="unlabeled")
    return [json.loads(line) for line in out_path.read_text().splitlines()]


def test_rating_5_sets_label_1_with_high_weight(tmp_path):
    db_path = tmp_path / "test.db"
    conn = _build_db(db_path)
    conn.execute("INSERT INTO chunks (chunk_id) VALUES (?)", ("chk_top",))
    _insert_event(conn, "real query", ["chk_top"],
                  {"chk_top": {"v": 0.7, "s": 0.5, "r": 0.3, "a": 0.0,
                               "pt": 0.0, "re": 0.0}},
                  was_referenced=0)  # was_referenced=0, but user rated 5★
    _insert_rating(conn, "chk_top", "5")
    conn.commit()
    conn.close()

    rows = _export_to_jsonl(tmp_path, None)
    assert len(rows) == 1
    # Rating 5★ überstimmt das was_referenced=0 — chunk wird positiv.
    assert rows[0]["label"] == 1
    assert rows[0]["sample_weight"] >= 1.0


def test_rating_1_sets_label_0_with_high_weight(tmp_path):
    db_path = tmp_path / "test.db"
    conn = _build_db(db_path)
    conn.execute("INSERT INTO chunks (chunk_id) VALUES (?)", ("chk_bad",))
    _insert_event(conn, "real query", ["chk_bad"],
                  {"chk_bad": {"v": 0.6, "s": 0.4, "r": 0.5, "a": 0.0,
                               "pt": 0.0, "re": 0.0}},
                  was_referenced=1)  # referenced but rated 1★ later
    _insert_rating(conn, "chk_bad", "1")
    conn.commit()
    conn.close()

    rows = _export_to_jsonl(tmp_path, None)
    assert rows[0]["label"] == 0
    assert rows[0]["sample_weight"] >= 1.0


def test_rating_3_is_omitted_no_signal(tmp_path):
    """Neutral 3★ ratings should NOT influence label — fall back to was_referenced + default weight."""
    db_path = tmp_path / "test.db"
    conn = _build_db(db_path)
    conn.execute("INSERT INTO chunks (chunk_id) VALUES (?)", ("chk_mid",))
    _insert_event(conn, "real query", ["chk_mid"],
                  {"chk_mid": {"v": 0.5, "s": 0.5, "r": 0.5, "a": 0.0,
                               "pt": 0.0, "re": 0.0}},
                  was_referenced=1)
    _insert_rating(conn, "chk_mid", "3")
    conn.commit()
    conn.close()

    rows = _export_to_jsonl(tmp_path, None)
    # was_referenced bleibt das label, weight ist default 1.0
    assert rows[0]["label"] == 1
    assert rows[0]["sample_weight"] == 1.0


def test_no_rating_uses_was_referenced_default_weight(tmp_path):
    db_path = tmp_path / "test.db"
    conn = _build_db(db_path)
    conn.execute("INSERT INTO chunks (chunk_id) VALUES (?)", ("chk_no_fb",))
    _insert_event(conn, "real query", ["chk_no_fb"],
                  {"chk_no_fb": {"v": 0.5, "s": 0.4, "r": 0.3, "a": 0.0,
                                 "pt": 0.0, "re": 0.0}},
                  was_referenced=1)
    conn.commit()
    conn.close()

    rows = _export_to_jsonl(tmp_path, None)
    assert rows[0]["label"] == 1
    assert rows[0]["sample_weight"] == 1.0


def test_multiple_ratings_get_n_bonus(tmp_path):
    """3× 5★ feedback sollte weight > 1× 5★ haben (mehr confidence)."""
    db_path = tmp_path / "test.db"
    conn = _build_db(db_path)
    conn.execute("INSERT INTO chunks (chunk_id) VALUES (?)", ("chk_consensus",))
    _insert_event(conn, "real query", ["chk_consensus"],
                  {"chk_consensus": {"v": 0.7, "s": 0.5, "r": 0.5, "a": 0.0,
                                     "pt": 0.0, "re": 0.0}},
                  was_referenced=1)
    for _ in range(3):
        _insert_rating(conn, "chk_consensus", "5")
    conn.commit()
    conn.close()

    rows = _export_to_jsonl(tmp_path, None)
    # n=3 → weight = 1.0 * (1 + 0.1 * 2) = 1.2
    assert rows[0]["sample_weight"] == pytest.approx(1.2)


def test_legacy_positive_signal_maps_to_rating_4(tmp_path):
    """Pre-rating-migration callers used 'positive'/'negative' — accept those too."""
    db_path = tmp_path / "test.db"
    conn = _build_db(db_path)
    conn.execute("INSERT INTO chunks (chunk_id) VALUES (?)", ("chk_legacy",))
    _insert_event(conn, "real query", ["chk_legacy"],
                  {"chk_legacy": {"v": 0.4, "s": 0.4, "r": 0.4, "a": 0.0,
                                  "pt": 0.0, "re": 0.0}},
                  was_referenced=0)  # not referenced, but legacy 'positive'
    _insert_rating(conn, "chk_legacy", "positive")
    conn.commit()
    conn.close()

    rows = _export_to_jsonl(tmp_path, None)
    # positive → rating-equiv 4 → weak positive (label=1, weight=0.5)
    assert rows[0]["label"] == 1
    assert rows[0]["sample_weight"] == pytest.approx(0.5)


# ── Tests für Span-Judge Offline-Teacher (SSA) ───────────────────────
# Der Judge wird als fake span_judge_fn injiziert (kein echtes Ollama):
#   span_judge_fn(conn, query, chunk_ids) -> {chunk_id: score 0..1}

def _fake_judge(score_by_chunk: dict[str, float]):
    def _fn(conn, query, chunk_ids):
        return {cid: score_by_chunk[cid]
                for cid in chunk_ids if cid in score_by_chunk}
    return _fn


def _run_export(tmp_path, db_path, span_judge_fn, negative_mode="unlabeled"):
    from tools.export_retrieval_dataset import export
    out = tmp_path / "ds.jsonl"
    export(db_path, out, days=30, negative_mode=negative_mode,
           span_judge_fn=span_judge_fn)
    return [json.loads(line) for line in out.read_text().splitlines()]


def test_span_judge_downweights_false_positive(tmp_path):
    """was_referenced=1 aber Judge sagt irrelevant → Label bleibt 1,
    Sample-Weight wird auf 0.3 gedrückt (vermutlicher False-Positive)."""
    db_path = tmp_path / "memory.db"
    conn = _build_db(db_path)
    _insert_event(conn, "real query", ["chk_fp"],
                  {"chk_fp": {"v": 0.5, "s": 0.3, "r": 0.8, "a": 0.0}},
                  was_referenced=1)
    conn.commit(); conn.close()

    rows = _run_export(tmp_path, db_path, _fake_judge({"chk_fp": 0.1}))
    assert rows[0]["label"] == 1
    assert rows[0]["sample_weight"] == pytest.approx(0.3)


def test_span_judge_promotes_false_negative(tmp_path):
    """was_referenced=0 aber Judge sagt hoch-relevant → Label auf 1
    angehoben mit moderatem Gewicht 0.5 (vermutlicher False-Negative)."""
    db_path = tmp_path / "memory.db"
    conn = _build_db(db_path)
    _insert_event(conn, "real query", ["chk_fn"],
                  {"chk_fn": {"v": 0.6, "s": 0.4, "r": 0.7, "a": 0.0}},
                  was_referenced=0)
    conn.commit(); conn.close()

    rows = _run_export(tmp_path, db_path, _fake_judge({"chk_fn": 0.9}))
    assert rows[0]["label"] == 1
    assert rows[0]["sample_weight"] == pytest.approx(0.5)


def test_span_judge_mid_score_keeps_label(tmp_path):
    """Score zwischen den Schwellen (0.25..0.75) → keine Änderung."""
    db_path = tmp_path / "memory.db"
    conn = _build_db(db_path)
    _insert_event(conn, "real query", ["chk_mid"],
                  {"chk_mid": {"v": 0.5, "s": 0.5, "r": 0.5, "a": 0.0}},
                  was_referenced=1)
    conn.commit(); conn.close()

    rows = _run_export(tmp_path, db_path, _fake_judge({"chk_mid": 0.5}))
    assert rows[0]["label"] == 1
    assert rows[0]["sample_weight"] == pytest.approx(1.0)


def test_explicit_rating_overrides_span_judge(tmp_path):
    """Human-Rating bleibt Ground-Truth: 5★ überstimmt einen niedrigen
    Judge-Score — kein Down-Weight."""
    db_path = tmp_path / "memory.db"
    conn = _build_db(db_path)
    conn.execute("INSERT INTO chunks (chunk_id) VALUES (?)",
                 ("chk_rated",))
    _insert_event(conn, "real query", ["chk_rated"],
                  {"chk_rated": {"v": 0.5, "s": 0.3, "r": 0.8, "a": 0.0}},
                  was_referenced=1)
    _insert_rating(conn, "chk_rated", "5")
    conn.commit(); conn.close()

    rows = _run_export(tmp_path, db_path, _fake_judge({"chk_rated": 0.1}))
    # Rating gewinnt: label 1, weight 1.0 (NICHT 0.3 vom Judge)
    assert rows[0]["label"] == 1
    assert rows[0]["sample_weight"] == pytest.approx(1.0)


def test_span_judge_unavailable_falls_back(tmp_path):
    """Judge nicht erreichbar (leeres Dict) → reines was_referenced,
    keine Refinement, kein Crash."""
    db_path = tmp_path / "memory.db"
    conn = _build_db(db_path)
    _insert_event(conn, "real query", ["chk_x"],
                  {"chk_x": {"v": 0.5, "s": 0.3, "r": 0.8, "a": 0.0}},
                  was_referenced=1)
    conn.commit(); conn.close()

    rows = _run_export(tmp_path, db_path, _fake_judge({}))  # nichts gescort
    assert rows[0]["label"] == 1
    assert rows[0]["sample_weight"] == pytest.approx(1.0)
    assert rows[0]["span_score"] is None


def test_span_score_never_leaks_into_features(tmp_path):
    """span_score ist Top-Level (Analyse), NIE in features — sonst würde
    der Trainer ein Gewicht lernen, das die Laufzeit nie liefern kann."""
    db_path = tmp_path / "memory.db"
    conn = _build_db(db_path)
    _insert_event(conn, "real query", ["chk_x"],
                  {"chk_x": {"v": 0.5, "s": 0.3, "r": 0.8, "a": 0.0}},
                  was_referenced=1)
    conn.commit(); conn.close()

    rows = _run_export(tmp_path, db_path, _fake_judge({"chk_x": 0.9}))
    assert "span_score" in rows[0]
    assert rows[0]["span_score"] == pytest.approx(0.9)
    assert "span_score" not in rows[0]["features"]
