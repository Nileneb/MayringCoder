"""Tests für task_derivation.py + retrieval task-boost integration."""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import pytest

from src.memory.store import init_memory_db
from src.memory.task_derivation import (
    derive_task,
    get_task_boost_for_chunks,
    link_chunk_to_task,
)


@pytest.fixture
def conn(tmp_path):
    return init_memory_db(tmp_path / "memory.db")


def _seed_chunk(conn, chunk_id: str = "chk_test_001") -> None:
    conn.execute(
        """INSERT OR IGNORE INTO sources (source_id, source_type, captured_at)
           VALUES ('src_test', 'repo_file', ?)""",
        (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),),
    )
    conn.execute(
        """INSERT OR IGNORE INTO chunks (chunk_id, source_id, text, text_hash, created_at)
           VALUES (?, 'src_test', 'sample text', 'h', ?)""",
        (chunk_id, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
    )


def test_link_chunk_to_task_new_row(conn):
    _seed_chunk(conn)
    conn.execute(
        """INSERT INTO task_categories (task_id, title, embedding_id, first_seen_at, last_used_at, workspace_id)
           VALUES ('task_x', 'Test Task', '[]', ?, ?, 'default')""",
        ("2026-05-11T00:00:00Z", "2026-05-11T00:00:00Z"),
    )

    link_chunk_to_task(conn, "task_x", "chk_test_001", relevance_score=1.0)
    row = conn.execute(
        "SELECT relevance_score FROM task_chunk_links WHERE task_id='task_x'"
    ).fetchone()
    assert row is not None
    assert row[0] == pytest.approx(1.0)


def test_link_chunk_to_task_is_idempotent_and_increments(conn):
    _seed_chunk(conn)
    conn.execute(
        """INSERT INTO task_categories (task_id, title, embedding_id, first_seen_at, last_used_at, workspace_id)
           VALUES ('task_x', 'Test Task', '[]', ?, ?, 'default')""",
        ("2026-05-11T00:00:00Z", "2026-05-11T00:00:00Z"),
    )

    link_chunk_to_task(conn, "task_x", "chk_test_001", relevance_score=0.5)
    link_chunk_to_task(conn, "task_x", "chk_test_001", relevance_score=1.0)
    link_chunk_to_task(conn, "task_x", "chk_test_001", relevance_score=2.0)

    row = conn.execute(
        "SELECT relevance_score FROM task_chunk_links WHERE task_id='task_x' AND chunk_id='chk_test_001'"
    ).fetchone()
    # increments clipped to 5.0
    assert row[0] == pytest.approx(3.5)


def test_link_chunk_relevance_clipped_at_5(conn):
    _seed_chunk(conn)
    conn.execute(
        """INSERT INTO task_categories (task_id, title, embedding_id, first_seen_at, last_used_at, workspace_id)
           VALUES ('task_x', 'Test Task', '[]', ?, ?, 'default')""",
        ("2026-05-11T00:00:00Z", "2026-05-11T00:00:00Z"),
    )
    for _ in range(20):
        link_chunk_to_task(conn, "task_x", "chk_test_001", relevance_score=1.0)

    row = conn.execute(
        "SELECT relevance_score FROM task_chunk_links WHERE task_id='task_x'"
    ).fetchone()
    assert row[0] == pytest.approx(5.0)


def test_get_task_boost_for_chunks_returns_only_matches(conn):
    _seed_chunk(conn, "chk_match")
    _seed_chunk(conn, "chk_unmatched")
    conn.execute(
        """INSERT INTO task_categories (task_id, title, embedding_id, first_seen_at, last_used_at, workspace_id)
           VALUES ('task_x', 'Test', '[]', ?, ?, 'default')""",
        ("2026-05-11T00:00:00Z", "2026-05-11T00:00:00Z"),
    )
    link_chunk_to_task(conn, "task_x", "chk_match", relevance_score=2.5)

    boost = get_task_boost_for_chunks(conn, "task_x", ["chk_match", "chk_unmatched"])
    assert boost == {"chk_match": pytest.approx(2.5)}


def test_derive_task_reuses_existing_via_embedding_similarity(conn):
    """When embed returns same vector for both prompts, cascade should reuse."""
    existing_emb = [0.1] * 768
    conn.execute(
        """INSERT INTO task_categories (task_id, title, embedding_id, first_seen_at, last_used_at, workspace_id)
           VALUES ('task_existing', 'JWT-Token-Validierung implementieren', ?, ?, ?, 'default')""",
        (
            json.dumps(existing_emb),
            "2026-05-11T00:00:00Z",
            "2026-05-11T00:00:00Z",
        ),
    )

    with patch("src.memory.task_derivation._embed_text", return_value=existing_emb):
        result = derive_task("how do I validate JWT", conn, "http://ollama:11434", "default")

    assert result is not None
    assert result["task_id"] == "task_existing"
    assert result["reused"] is True
    # occurrence_count incremented
    count = conn.execute(
        "SELECT occurrence_count FROM task_categories WHERE task_id='task_existing'"
    ).fetchone()[0]
    assert count == 2


def test_derive_task_creates_new_when_below_similarity_threshold(conn):
    """When no similar task exists, mistral-cascade creates a new row."""
    prompt_emb = [0.5] * 768
    title_emb = [0.6] * 768

    with patch("src.memory.task_derivation._embed_text", side_effect=[prompt_emb, title_emb]):
        with patch(
            "src.memory.task_derivation._call_mistral_for_task",
            return_value="Deploy-Fehler diagnostizieren und beheben",
        ):
            result = derive_task(
                "der deploy ist kaputt was tun?",
                conn, "http://ollama:11434", "default",
            )

    assert result is not None
    assert result["reused"] is False
    assert result["title"] == "Deploy-Fehler diagnostizieren und beheben"

    row = conn.execute(
        "SELECT title, occurrence_count, workspace_id FROM task_categories WHERE task_id = ?",
        (result["task_id"],),
    ).fetchone()
    assert row[0] == "Deploy-Fehler diagnostizieren und beheben"
    assert row[1] == 1
    assert row[2] == "default"


def test_derive_task_short_prompt_returns_none(conn):
    result = derive_task("hi", conn, "http://ollama:11434", "default")
    assert result is None


def test_derive_task_workspace_isolation(conn):
    """Same prompt in different workspaces should not collide."""
    emb_a = [0.7] * 768
    emb_b = [0.7] * 768

    # Pre-existing task in workspace 'alice'
    conn.execute(
        """INSERT INTO task_categories (task_id, title, embedding_id, first_seen_at, last_used_at, workspace_id)
           VALUES ('task_alice', 'Alice task', ?, ?, ?, 'alice')""",
        (json.dumps(emb_a), "2026-05-11T00:00:00Z", "2026-05-11T00:00:00Z"),
    )

    # Bob in different workspace — should NOT find alice's task (workspace filter)
    with patch("src.memory.task_derivation._embed_text", side_effect=[emb_b, emb_b]):
        with patch(
            "src.memory.task_derivation._call_mistral_for_task",
            return_value="Bob's distinct task",
        ):
            result = derive_task(
                "irgendein prompt",
                conn, "http://ollama:11434", workspace_id="bob",
            )

    assert result is not None
    assert result["task_id"] != "task_alice"
    # Verify alice's task was not modified
    alice_count = conn.execute(
        "SELECT occurrence_count FROM task_categories WHERE task_id='task_alice'"
    ).fetchone()[0]
    assert alice_count == 1


def test_add_feedback_creates_task_link_when_rating_high(conn):
    """rating>=4 + metadata.task_id should create a task_chunk_link."""
    from src.memory.store import add_feedback

    _seed_chunk(conn, "chk_fb_test")
    conn.execute(
        """INSERT INTO task_categories (task_id, title, embedding_id, first_seen_at, last_used_at, workspace_id)
           VALUES ('task_link_test', 'Test', '[]', ?, ?, 'default')""",
        ("2026-05-11T00:00:00Z", "2026-05-11T00:00:00Z"),
    )

    add_feedback(conn, "chk_fb_test", "5", metadata={"task_id": "task_link_test"})

    row = conn.execute(
        "SELECT relevance_score FROM task_chunk_links WHERE task_id='task_link_test' AND chunk_id='chk_fb_test'"
    ).fetchone()
    assert row is not None
    # rating 5 → score (5-3)*0.5 = 1.0
    assert row[0] == pytest.approx(1.0)


def test_add_feedback_no_link_for_low_rating(conn):
    """rating < 4 should NOT create task-chunk-link (we only learn from positives)."""
    from src.memory.store import add_feedback

    _seed_chunk(conn, "chk_low")
    conn.execute(
        """INSERT INTO task_categories (task_id, title, embedding_id, first_seen_at, last_used_at, workspace_id)
           VALUES ('task_low', 'Test', '[]', ?, ?, 'default')""",
        ("2026-05-11T00:00:00Z", "2026-05-11T00:00:00Z"),
    )

    add_feedback(conn, "chk_low", "2", metadata={"task_id": "task_low"})

    row = conn.execute(
        "SELECT 1 FROM task_chunk_links WHERE task_id='task_low'"
    ).fetchone()
    assert row is None


def test_add_feedback_no_link_without_task_id_metadata(conn):
    """Legacy callers (no task_id in metadata) should not break feedback."""
    from src.memory.store import add_feedback

    _seed_chunk(conn, "chk_legacy")
    # No task_categories row, no metadata — should still succeed
    add_feedback(conn, "chk_legacy", "5", metadata={})
    add_feedback(conn, "chk_legacy", "4", metadata=None)

    # No task_chunk_links should have been created
    rows = conn.execute("SELECT 1 FROM task_chunk_links").fetchall()
    assert rows == []
