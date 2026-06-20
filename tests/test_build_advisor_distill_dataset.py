"""Tests for the Pfad-B advisor-distillation dataset builder."""
from __future__ import annotations

import json
import sqlite3

from tools import span_judge, build_advisor_distill_dataset as bd


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE context_feedback_log(id INTEGER PRIMARY KEY, query TEXT);"
        "CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, text TEXT);"
        "CREATE TABLE span_judge_cache(query_hash TEXT, chunk_id TEXT,"
        " score REAL, model TEXT, computed_at TEXT,"
        " PRIMARY KEY(query_hash, chunk_id));"
    )
    conn.execute("INSERT INTO context_feedback_log(query) VALUES"
                 "('how does auth work'),('other q')")
    conn.execute("INSERT INTO chunks VALUES('chk_a','jwt validation flow'),('chk_b','noise')")
    qh = span_judge.query_hash("how does auth work")
    qh2 = span_judge.query_hash("other q")
    conn.execute("INSERT INTO span_judge_cache VALUES(?,?,?,?,?)",
                 (qh, "chk_a", 0.9, "claude-prewarm", "t"))
    conn.execute("INSERT INTO span_judge_cache VALUES(?,?,?,?,?)",
                 (qh, "chk_b", 0.0, "claude-prewarm", "t"))
    # an ollama-judged pair (different query) must NOT be harvested as Claude data
    conn.execute("INSERT INTO span_judge_cache VALUES(?,?,?,?,?)",
                 (qh2, "chk_a", 0.5, "ministral-3:3b", "t"))
    conn.commit()
    return conn


def test_builds_query_text_score_records():
    recs = bd.build_records(_db())
    assert len(recs) == 2
    by_chunk = {r["chunk_id"]: r for r in recs}
    assert by_chunk["chk_a"]["query"] == "how does auth work"
    assert by_chunk["chk_a"]["text"] == "jwt validation flow"
    assert by_chunk["chk_a"]["score"] == 0.9
    assert by_chunk["chk_b"]["score"] == 0.0


def test_only_claude_prewarm_model():
    # default model filter drops the ministral row → exactly 2 claude rows
    recs = bd.build_records(_db(), model="claude-prewarm")
    assert all(r["chunk_id"] in {"chk_a", "chk_b"} for r in recs)
    assert len(recs) == 2
    # filtering for the ollama model yields its single row
    assert len(bd.build_records(_db(), model="ministral-3:3b")) == 1


def test_skips_pairs_with_unknown_query_hash():
    conn = _db()
    conn.execute("INSERT INTO span_judge_cache VALUES('deadbeef','chk_a',0.7,'claude-prewarm','t')")
    conn.commit()
    recs = bd.build_records(conn)
    # the orphan hash (no context_feedback_log query) is skipped, not crashed
    assert all(r["query"] == "how does auth work" for r in recs)
    assert len(recs) == 2
