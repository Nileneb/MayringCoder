"""Tests for adaptive ambient snapshot scoring + dynamic budget."""
from __future__ import annotations
from datetime import datetime, timedelta

from src.memory.ambient import _score_entry, _score_snapshot_entries, build_context
from src.memory.store import init_memory_db, _now_iso


def _iso(days_ago: float) -> str:
    return (datetime.utcnow() - timedelta(days=days_ago)).isoformat()


def test_score_recency_boost_under_24h(tmp_path):
    conn = init_memory_db(tmp_path / "t.db")
    recent = _score_entry("recent item", _iso(0.5), conn)
    old = _score_entry("old item", _iso(30), conn)
    assert recent > old
    assert recent > 0.5


def test_score_decay_halves_at_half_life(tmp_path):
    conn = init_memory_db(tmp_path / "t.db")
    fresh = _score_entry("item", _iso(0.01), conn, half_life_days=14.0)
    aged = _score_entry("item", _iso(14.0), conn, half_life_days=14.0)
    assert 0.10 < aged < 0.25


def test_score_sticky_has_minimum_floor(tmp_path):
    conn = init_memory_db(tmp_path / "t.db")
    sticky = _score_entry("[sticky] architectural decision", _iso(1000), conn)
    regular = _score_entry("regular decision", _iso(1000), conn)
    assert sticky >= 0.9
    assert regular < 0.05


def test_score_feedback_boost(tmp_path):
    conn = init_memory_db(tmp_path / "t.db")
    entry_text = "credit service is central to the architecture of the system"
    prefix = entry_text.strip()[:40]
    for _ in range(2):
        conn.execute(
            """INSERT INTO context_feedback_log
               (trigger_ids, context_text, was_referenced, led_to_retrieval, relevance_score, captured_at)
               VALUES (?, ?, 1, 0, 0.9, ?)""",
            ("[]", prefix + " extended text in db", datetime.utcnow().isoformat()),
        )
    conn.commit()
    boosted = _score_entry(entry_text, _iso(0.5), conn)
    unboosted = _score_entry("unrelated random text longer than forty chars here", _iso(0.5), conn)
    assert boosted > unboosted + 0.3


def test_score_snapshot_entries_sorted_desc(tmp_path):
    conn = init_memory_db(tmp_path / "t.db")
    entries = [
        ("old entry", _iso(30)),
        ("[sticky] architecture", _iso(30)),
        ("fresh entry", _iso(0.1)),
    ]
    ranked = _score_snapshot_entries(conn, entries)
    texts = [t for t, _s in ranked]
    assert texts[0] == "[sticky] architecture"
    assert "old" in texts[-1]


def test_score_invalid_timestamp_does_not_crash(tmp_path):
    conn = init_memory_db(tmp_path / "t.db")
    score = _score_entry("item", "not-a-date", conn)
    assert 0.0 <= score <= 1.0


def test_dynamic_budget_respects_trigger_length(tmp_path, monkeypatch):
    """When trigger hint is long, build_context should produce a compressed result."""
    conn = init_memory_db(tmp_path / "t.db")

    long_text = "SNAPSHOT " * 600  # ~5400 chars
    src_id = "ambient:global:snapshot"
    now = _now_iso()
    conn.execute(
        """INSERT INTO sources
           (source_id, source_type, repo, path, branch, "commit", content_hash, captured_at)
           VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
        (src_id, "ambient_snapshot", "", "ambient/snapshot", "local", "", "sha256:a", now),
    )
    conn.execute(
        """INSERT INTO chunks
           (chunk_id, source_id, chunk_level, ordinal, start_offset, end_offset,
            text, text_hash, dedup_key, created_at, is_active)
           VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
        ("c1", src_id, "ambient_snapshot", 0, 0, len(long_text),
         long_text, "h1", "d1", now),
    )
    conn.commit()

    import src.memory.ambient as _ambient

    monkeypatch.setattr(
        _ambient,
        "trigger_scan",
        lambda *a, **kw: _ambient.TriggerResult(context="tiny", trigger_ids=[]),
    )
    short_ctx = build_context("test", conn, "", repo_slug="")

    monkeypatch.setattr(
        _ambient,
        "trigger_scan",
        lambda *a, **kw: _ambient.TriggerResult(context="X" * 400, trigger_ids=[]),
    )
    long_ctx = build_context("test", conn, "", repo_slug="")

    assert len(long_ctx) < len(short_ctx)
