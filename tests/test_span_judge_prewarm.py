"""Tests for the Claude span-judge cache pre-warm (tools/span_judge_prewarm.py).

Covers the property the GPU-relief relies on: pairs Claude judges and ingests
become span_judge_cache hits, so the subsequent export skips the model call.
"""
from __future__ import annotations

import json
import sqlite3

from tools import span_judge, span_judge_prewarm as pw


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE context_feedback_log(id INTEGER PRIMARY KEY, query TEXT,"
        " trigger_ids TEXT, stage_scores TEXT, was_referenced INT,"
        " captured_at TEXT, workspace_id TEXT);"
        "CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, text TEXT);"
    )
    conn.execute("INSERT INTO chunks VALUES('chk_a','relevant'),('chk_b','noise')")
    conn.execute(
        "INSERT INTO context_feedback_log VALUES"
        "(1,'q one',?,'{\"chk_a\":{},\"chk_b\":{}}',1,datetime('now'),'ws')",
        (json.dumps(["chk_a", "chk_b"]),),
    )
    conn.commit()
    return conn


def test_dump_emits_uncached_pairs_with_text():
    conn = _db()
    batches = pw.dump(conn, days=30, limit=300)
    assert len(batches) == 1
    b = batches[0]
    assert b["query"] == "q one"
    assert {c["chunk_id"] for c in b["chunks"]} == {"chk_a", "chk_b"}
    assert all(c["text"] for c in b["chunks"])


def test_dump_respects_limit():
    conn = _db()
    batches = pw.dump(conn, days=30, limit=1)
    assert sum(len(b["chunks"]) for b in batches) == 1


def test_ingest_makes_export_a_cache_hit():
    conn = _db()
    doc = [{"query": "q one", "scores": {"chk_a": 0.9, "chk_b": 0.0}}]
    assert pw.ingest(conn, doc) == 2
    # cache-first scores_for_query returns Claude's scores without a model call
    got = span_judge.scores_for_query(conn, "q one", ["chk_a", "chk_b"])
    assert got == {"chk_a": 0.9, "chk_b": 0.0}
    # and the pair is no longer dumped as uncached
    assert pw.dump(conn, days=30, limit=300) == []


def test_ingest_clamps_and_tags_model():
    conn = _db()
    pw.ingest(conn, [{"query": "q one", "scores": {"chk_a": 1.7, "chk_b": -3}}])
    rows = dict(conn.execute(
        "SELECT chunk_id, score FROM span_judge_cache").fetchall())
    assert rows == {"chk_a": 1.0, "chk_b": 0.0}
    model = conn.execute("SELECT DISTINCT model FROM span_judge_cache").fetchone()[0]
    assert model == "claude-prewarm"
