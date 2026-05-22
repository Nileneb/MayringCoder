"""Tests for tools/export_judge_relevance_dataset.py (#260)."""
import json
import sqlite3
from pathlib import Path

from export_judge_relevance_dataset import export


def _build_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            text TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE chunk_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT NOT NULL,
            signal TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            workspace_id TEXT NOT NULL DEFAULT 'default'
        );
        """
    )
    conn.executemany(
        "INSERT INTO chunks VALUES (?,?)",
        [("chk_hi", "highly relevant"), ("chk_lo", "irrelevant"),
         ("chk_q", "queried chunk"), ("chk_neutral", "meh")],
    )
    fb = [
        # chk_hi: two 5★ → avg 5 → score 1.0
        ("chk_hi", "5", "{}", "2026-05-01T00:00:00Z", "ws1"),
        ("chk_hi", "5", "{}", "2026-05-01T00:00:00Z", "ws1"),
        # chk_lo: one 1★ → score 0.0
        ("chk_lo", "1", "{}", "2026-05-01T00:00:00Z", "ws1"),
        # chk_q: 3★ with a query_context → score 0.5, query set
        ("chk_q", "3", json.dumps({"query_context": "wie funktioniert auth?"}),
         "2026-05-01T00:00:00Z", "ws1"),
        # chk_neutral: 'neutral' signal → no rating value → excluded
        ("chk_neutral", "neutral", "{}", "2026-05-01T00:00:00Z", "ws1"),
    ]
    conn.executemany(
        "INSERT INTO chunk_feedback (chunk_id, signal, metadata, created_at, workspace_id) "
        "VALUES (?,?,?,?,?)", fb,
    )
    conn.commit()
    conn.close()


def _read(path: Path) -> dict:
    return {json.loads(l)["chunk_id"]: json.loads(l)
            for l in path.read_text().splitlines()}


def test_score_normalization_and_neutral_skip(tmp_path):
    db = tmp_path / "memory.db"
    _build_db(db)
    out = tmp_path / "ds.jsonl"

    n = export(db, out, days=3650, require_query=False)

    rows = _read(out)
    assert "chk_neutral" not in rows          # neutral carries no signal
    assert rows["chk_hi"]["output"] == 1.0     # 5★ → 1.0
    assert rows["chk_hi"]["n_ratings"] == 2
    assert rows["chk_lo"]["output"] == 0.0     # 1★ → 0.0
    assert rows["chk_q"]["output"] == 0.5      # 3★ → 0.5
    assert n == 3


def test_query_context_extracted(tmp_path):
    db = tmp_path / "memory.db"
    _build_db(db)
    out = tmp_path / "ds.jsonl"

    export(db, out, days=3650, require_query=False)
    rows = _read(out)

    assert rows["chk_q"]["input"]["query"] == "wie funktioniert auth?"
    assert rows["chk_q"]["input"]["chunk_text"] == "queried chunk"
    assert rows["chk_hi"]["input"]["query"] == ""   # no query_context


def test_require_query_filters_rows_without_query(tmp_path):
    db = tmp_path / "memory.db"
    _build_db(db)
    out = tmp_path / "ds.jsonl"

    n = export(db, out, days=3650, require_query=True)

    rows = _read(out)
    assert set(rows) == {"chk_q"}   # only the row with a query_context survives
    assert n == 1
