"""Issue #87 acceptance: unit test for memory_context_generator.

Pipeline 2 of #87 generates (prompt, completion) JSONL training pairs
from chunks + chunk_feedback, with hierarchical level-sorting (file →
class → function → block) and feedback-score weighting within each
level.

These tests use an in-memory SQLite DB (no production data, no LLM).
Coverage:
  - generate_auto_feedback writes 1–5 star ratings derived from chunk
    properties (no feedback events double-written)
  - _build_context sorts hierarchically by chunk_level then by
    feedback_score
  - generate_pairs produces shape with prompt + completion + metadata
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.training.memory_context_generator import (
    _auto_rating,
    _build_context,
    _signal_to_score,
    generate_auto_feedback,
    generate_pairs,
)


def _make_chunk(
    chunk_id: str,
    *,
    level: str = "file",
    labels: str = "",
    quality: float = 0.0,
    text: str = "",
    source_id: str = "test-src",
    ordinal: int = 0,
    created_at: str = "2026-05-08T00:00:00+00:00",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "chunk_level": level,
        "category_labels": labels,
        "quality_score": quality,
        "text": text,
        "source_id": source_id,
        "ordinal": ordinal,
        "created_at": created_at,
    }


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("""
        CREATE TABLE chunks (
            chunk_id        TEXT PRIMARY KEY,
            chunk_level     TEXT,
            category_labels TEXT,
            quality_score   REAL,
            text            TEXT,
            source_id       TEXT,
            ordinal         INTEGER,
            workspace_id    TEXT NOT NULL DEFAULT 'default',
            is_active       INTEGER NOT NULL DEFAULT 1,
            created_at      TEXT NOT NULL DEFAULT ''
        )
    """)
    c.execute("""
        CREATE TABLE chunk_feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id    TEXT NOT NULL,
            signal      TEXT NOT NULL,
            metadata    TEXT NOT NULL DEFAULT '{}',
            created_at  TEXT NOT NULL
        )
    """)
    return c


def test_auto_rating_5stars_for_classified_specific_chunk(conn):
    chunk = _make_chunk("c1", level="function",
                        labels="api,data_access", quality=0.9, text="def foo():")
    rating = _auto_rating(_RowLike(chunk))
    assert rating == "5", f"classified+specific+text → 5★, got {rating}"


def test_auto_rating_1star_for_empty_text(conn):
    chunk = _make_chunk("c1", level="function", text="")
    rating = _auto_rating(_RowLike(chunk))
    assert rating == "1"


def test_auto_rating_2star_for_file_overview_no_labels(conn):
    chunk = _make_chunk("c1", level="file", labels="", text="some content")
    rating = _auto_rating(_RowLike(chunk))
    assert rating == "2"


def test_signal_to_score_binary_legacy():
    assert _signal_to_score("positive") == pytest.approx(1.0)
    assert _signal_to_score("negative") == pytest.approx(-1.0)


def test_signal_to_score_star_ratings():
    assert _signal_to_score("5") == pytest.approx(1.0)
    assert _signal_to_score("3") == pytest.approx(0.0)
    assert _signal_to_score("1") == pytest.approx(-1.0)


def test_generate_auto_feedback_writes_no_double_entries(conn):
    now = datetime.now(tz=timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO chunks (chunk_id, chunk_level, category_labels, "
        "quality_score, text, source_id, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        ("c1", "function", "api", 0.9, "def x():", "src1", now),
    )
    conn.commit()
    n1 = generate_auto_feedback(conn, workspace_id="default")
    n2 = generate_auto_feedback(conn, workspace_id="default")
    assert n1 == 1, "first call writes 1 row"
    assert n2 == 0, "second call writes 0 — auto_feedback only fills empty slots"


def test_build_context_hierarchical_order(conn):
    """File chunks come before class chunks before function chunks
    even when ordinal/score say otherwise."""
    chunks = [
        _RowLike(_make_chunk("f1", level="function",
                             text="def foo():\n    pass", ordinal=2)),
        _RowLike(_make_chunk("c1", level="class",
                             text="class Bar:", ordinal=1)),
        _RowLike(_make_chunk("file1", level="file",
                             text="# Module overview", ordinal=0)),
    ]
    out = _build_context(chunks, scores={}, char_budget=2000)
    file_pos = out.index("[file]")
    class_pos = out.index("[class]")
    function_pos = out.index("[function]")
    assert file_pos < class_pos < function_pos


def test_generate_pairs_shape(conn):
    now = datetime.now(tz=timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO chunks (chunk_id, chunk_level, category_labels, "
        "quality_score, text, source_id, ordinal, created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        ("c1", "file", "api", 0.5, "module overview text", "src/a.py", 0, now),
    )
    conn.execute(
        "INSERT INTO chunks (chunk_id, chunk_level, category_labels, "
        "quality_score, text, source_id, ordinal, created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        ("c2", "function", "api", 0.7, "def foo(): return 1", "src/a.py", 1, now),
    )
    conn.commit()
    pairs = generate_pairs(conn, workspace_id="default", limit=10)
    assert len(pairs) >= 1
    p = pairs[0]
    needed = {"source_id", "prompt", "completion", "chunk_count",
              "labels", "avg_feedback_score"}
    assert needed <= set(p.keys()), f"missing: {needed - set(p.keys())}"
    assert "<memory_context>" in p["prompt"]
    assert "</memory_context>" in p["prompt"]


class _RowLike(dict):
    """sqlite3.Row-compatible dict — supports both .keys() and []."""
    def __getitem__(self, key):
        return super().__getitem__(key)
    def keys(self):
        return list(super().keys())
    def get(self, key, default=None):
        return super().get(key, default)
