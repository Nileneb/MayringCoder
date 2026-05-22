"""Tests für task_derivation.py + retrieval task-boost integration."""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import pytest

from src.memory.store import init_memory_db
from src.memory.task_derivation import (
    derive_research_question,
    get_research_question_boost,
    link_chunk_to_research_question,
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
        """INSERT INTO research_questions (research_question_id, title, embedding_id, first_seen_at, last_used_at, workspace_id)
           VALUES ('task_x', 'Test Task', '[]', ?, ?, 'default')""",
        ("2026-05-11T00:00:00Z", "2026-05-11T00:00:00Z"),
    )

    link_chunk_to_research_question(conn, "task_x", "chk_test_001", relevance_score=1.0)
    row = conn.execute(
        "SELECT relevance_score FROM research_question_chunk_links WHERE research_question_id='task_x'"
    ).fetchone()
    assert row is not None
    assert row[0] == pytest.approx(1.0)


def test_link_chunk_to_task_is_idempotent_and_increments(conn):
    _seed_chunk(conn)
    conn.execute(
        """INSERT INTO research_questions (research_question_id, title, embedding_id, first_seen_at, last_used_at, workspace_id)
           VALUES ('task_x', 'Test Task', '[]', ?, ?, 'default')""",
        ("2026-05-11T00:00:00Z", "2026-05-11T00:00:00Z"),
    )

    link_chunk_to_research_question(conn, "task_x", "chk_test_001", relevance_score=0.5)
    link_chunk_to_research_question(conn, "task_x", "chk_test_001", relevance_score=1.0)
    link_chunk_to_research_question(conn, "task_x", "chk_test_001", relevance_score=2.0)

    row = conn.execute(
        "SELECT relevance_score FROM research_question_chunk_links WHERE research_question_id='task_x' AND chunk_id='chk_test_001'"
    ).fetchone()
    # increments clipped to 5.0
    assert row[0] == pytest.approx(3.5)


def test_link_chunk_relevance_clipped_at_5(conn):
    _seed_chunk(conn)
    conn.execute(
        """INSERT INTO research_questions (research_question_id, title, embedding_id, first_seen_at, last_used_at, workspace_id)
           VALUES ('task_x', 'Test Task', '[]', ?, ?, 'default')""",
        ("2026-05-11T00:00:00Z", "2026-05-11T00:00:00Z"),
    )
    for _ in range(20):
        link_chunk_to_research_question(conn, "task_x", "chk_test_001", relevance_score=1.0)

    row = conn.execute(
        "SELECT relevance_score FROM research_question_chunk_links WHERE research_question_id='task_x'"
    ).fetchone()
    assert row[0] == pytest.approx(5.0)


def test_get_task_boost_for_chunks_returns_only_matches(conn):
    _seed_chunk(conn, "chk_match")
    _seed_chunk(conn, "chk_unmatched")
    conn.execute(
        """INSERT INTO research_questions (research_question_id, title, embedding_id, first_seen_at, last_used_at, workspace_id)
           VALUES ('task_x', 'Test', '[]', ?, ?, 'default')""",
        ("2026-05-11T00:00:00Z", "2026-05-11T00:00:00Z"),
    )
    link_chunk_to_research_question(conn, "task_x", "chk_match", relevance_score=2.5)

    boost = get_research_question_boost(conn, "task_x", ["chk_match", "chk_unmatched"])
    assert boost == {"chk_match": pytest.approx(2.5)}


def test_derive_task_reuses_existing_via_embedding_similarity(conn):
    """When embed returns same vector for both prompts, cascade should reuse."""
    existing_emb = [0.1] * 768
    conn.execute(
        """INSERT INTO research_questions (research_question_id, title, embedding_id, first_seen_at, last_used_at, workspace_id)
           VALUES ('task_existing', 'JWT-Token-Validierung implementieren', ?, ?, ?, 'default')""",
        (
            json.dumps(existing_emb),
            "2026-05-11T00:00:00Z",
            "2026-05-11T00:00:00Z",
        ),
    )

    with patch("src.memory.task_derivation._embed_text", return_value=existing_emb):
        result = derive_research_question("how do I validate JWT", conn, "http://ollama:11434", "default")

    assert result is not None
    assert result["research_question_id"] == "task_existing"
    assert result["reused"] is True
    # occurrence_count incremented
    count = conn.execute(
        "SELECT occurrence_count FROM research_questions WHERE research_question_id='task_existing'"
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
            result = derive_research_question(
                "der deploy ist kaputt was tun?",
                conn, "http://ollama:11434", "default",
            )

    assert result is not None
    assert result["reused"] is False
    assert result["title"] == "Deploy-Fehler diagnostizieren und beheben"

    row = conn.execute(
        "SELECT title, occurrence_count, workspace_id FROM research_questions WHERE research_question_id = ?",
        (result["research_question_id"],),
    ).fetchone()
    assert row[0] == "Deploy-Fehler diagnostizieren und beheben"
    assert row[1] == 1
    assert row[2] == "default"


def test_derive_task_short_prompt_returns_none(conn):
    result = derive_research_question("hi", conn, "http://ollama:11434", "default")
    assert result is None


def test_derive_task_workspace_isolation(conn):
    """Same prompt in different workspaces should not collide."""
    emb_a = [0.7] * 768
    emb_b = [0.7] * 768

    # Pre-existing task in workspace 'alice'
    conn.execute(
        """INSERT INTO research_questions (research_question_id, title, embedding_id, first_seen_at, last_used_at, workspace_id)
           VALUES ('task_alice', 'Alice task', ?, ?, ?, 'alice')""",
        (json.dumps(emb_a), "2026-05-11T00:00:00Z", "2026-05-11T00:00:00Z"),
    )

    # Bob in different workspace — should NOT find alice's task (workspace filter)
    with patch("src.memory.task_derivation._embed_text", side_effect=[emb_b, emb_b]):
        with patch(
            "src.memory.task_derivation._call_mistral_for_task",
            return_value="Bob's distinct task",
        ):
            result = derive_research_question(
                "irgendein prompt",
                conn, "http://ollama:11434", workspace_id="bob",
            )

    assert result is not None
    assert result["research_question_id"] != "task_alice"
    # Verify alice's research_question was not modified
    alice_count = conn.execute(
        "SELECT occurrence_count FROM research_questions WHERE research_question_id='task_alice'"
    ).fetchone()[0]
    assert alice_count == 1


def test_add_feedback_creates_task_link_when_rating_high(conn):
    """rating>=4 + metadata.task_id should create a task_chunk_link."""
    from src.memory.store import add_feedback

    _seed_chunk(conn, "chk_fb_test")
    conn.execute(
        """INSERT INTO research_questions (research_question_id, title, embedding_id, first_seen_at, last_used_at, workspace_id)
           VALUES ('task_link_test', 'Test', '[]', ?, ?, 'default')""",
        ("2026-05-11T00:00:00Z", "2026-05-11T00:00:00Z"),
    )

    add_feedback(conn, "chk_fb_test", "5", metadata={"task_id": "task_link_test"})

    row = conn.execute(
        "SELECT relevance_score FROM research_question_chunk_links WHERE research_question_id='task_link_test' AND chunk_id='chk_fb_test'"
    ).fetchone()
    assert row is not None
    # rating 5 → score (5-3)*0.5 = 1.0
    assert row[0] == pytest.approx(1.0)


def test_add_feedback_no_link_for_low_rating(conn):
    """rating < 4 should NOT create task-chunk-link (we only learn from positives)."""
    from src.memory.store import add_feedback

    _seed_chunk(conn, "chk_low")
    conn.execute(
        """INSERT INTO research_questions (research_question_id, title, embedding_id, first_seen_at, last_used_at, workspace_id)
           VALUES ('task_low', 'Test', '[]', ?, ?, 'default')""",
        ("2026-05-11T00:00:00Z", "2026-05-11T00:00:00Z"),
    )

    add_feedback(conn, "chk_low", "2", metadata={"task_id": "task_low"})

    row = conn.execute(
        "SELECT 1 FROM research_question_chunk_links WHERE research_question_id='task_low'"
    ).fetchone()
    assert row is None


def test_add_feedback_no_link_without_task_id_metadata(conn):
    """Legacy callers (no task_id in metadata) should not break feedback."""
    from src.memory.store import add_feedback

    _seed_chunk(conn, "chk_legacy")
    # No research_questions row, no metadata — should still succeed
    add_feedback(conn, "chk_legacy", "5", metadata={})
    add_feedback(conn, "chk_legacy", "4", metadata=None)

    # No research_question_chunk_links should have been created
    rows = conn.execute("SELECT 1 FROM research_question_chunk_links").fetchall()
    assert rows == []


# ── derive_task_fast (hot-path, no mistral) ──────────────────────

def test_derive_task_fast_returns_existing_match(conn):
    """derive_task_fast: existing task with high sim → return it, no mistral."""
    emb = [0.42] * 768
    conn.execute(
        """INSERT INTO research_questions (research_question_id, title, embedding_id, first_seen_at, last_used_at, workspace_id)
           VALUES ('task_known', 'Known Task', ?, ?, ?, 'default')""",
        (json.dumps(emb), "2026-05-11T00:00:00Z", "2026-05-11T00:00:00Z"),
    )
    from src.memory.task_derivation import derive_research_question_fast
    with patch("src.memory.task_derivation._embed_text", return_value=emb):
        # _call_mistral must NEVER be called in the fast path
        with patch("src.memory.task_derivation._call_mistral_for_task") as mock_mistral:
            result = derive_research_question_fast("any prompt", conn, "http://ollama:11434", "default")
    assert result is not None
    assert result["research_question_id"] == "task_known"
    assert result["reused"] is True
    mock_mistral.assert_not_called()


def test_derive_task_fast_returns_none_when_no_match(conn):
    """derive_task_fast: no existing task → None (does NOT create one)."""
    from src.memory.task_derivation import derive_research_question_fast
    with patch("src.memory.task_derivation._embed_text", return_value=[0.9] * 768):
        with patch("src.memory.task_derivation._call_mistral_for_task") as mock_mistral:
            result = derive_research_question_fast("brand new topic", conn, "http://ollama:11434", "default")
    assert result is None
    mock_mistral.assert_not_called()
    # No new row created
    count = conn.execute("SELECT COUNT(*) FROM research_questions").fetchone()[0]
    assert count == 0


def test_derive_task_fast_increments_occurrence_on_match(conn):
    emb = [0.33] * 768
    conn.execute(
        """INSERT INTO research_questions (research_question_id, title, embedding_id, occurrence_count, first_seen_at, last_used_at, workspace_id)
           VALUES ('task_c', 'T', ?, 5, ?, ?, 'default')""",
        (json.dumps(emb), "2026-05-11T00:00:00Z", "2026-05-11T00:00:00Z"),
    )
    from src.memory.task_derivation import derive_research_question_fast
    with patch("src.memory.task_derivation._embed_text", return_value=emb):
        derive_research_question_fast("prompt text", conn, "http://ollama:11434", "default")
    n = conn.execute("SELECT occurrence_count FROM research_questions WHERE research_question_id='task_c'").fetchone()[0]
    assert n == 6


def test_migration_renames_task_categories_to_research_questions(tmp_path):
    from src.memory.db_adapter import DBAdapter
    from src.memory.store import _migrate_schema
    db = DBAdapter.create(tmp_path / "legacy.db")
    db.execute("CREATE TABLE task_categories (task_id TEXT PRIMARY KEY, title TEXT, workspace_id TEXT)")
    db.execute("CREATE TABLE task_chunk_links (task_id TEXT, chunk_id TEXT, relevance_score REAL, created_at TEXT, PRIMARY KEY(task_id, chunk_id))")
    db.commit()
    _migrate_schema(db)
    names = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "research_questions" in names
    assert "research_question_chunk_links" in names
    assert "task_categories" not in names
    assert "research_question_id" in db.get_columns("research_question_chunk_links")


def test_derive_research_question_clusters_near_identical_prompts(conn):
    """Near-identical prompts must cluster into ONE research_question row.

    Reproduces the apples-vs-oranges bug: on the current (broken) code,
    the create path calls _embed_text(task_title) and stores a TITLE embedding,
    while the match path always uses _embed_text(prompt). With different vectors
    the second prompt scores below 0.85 → creates a second row instead of
    reusing. The fix stores the prompt embedding in the row so the match path
    compares prompt-vs-prompt consistently.
    """
    import src.memory.task_derivation as td

    p_emb = [1.0] * 384 + [0.0] * 384   # prompt embedding
    t_emb = [0.0] * 384 + [1.0] * 384  # title embedding — orthogonal to p_emb (cosine=0.0)

    # side_effect: first call is _embed_text(prompt) → p_emb,
    #              second call is _embed_text(task_title) → t_emb
    with patch.object(td, "_embed_text", side_effect=[p_emb, t_emb]), \
         patch.object(td, "_call_mistral_for_task",
                      return_value="Smoke-Test nach #226 bestätigen"):
        first = td.derive_research_question(
            "confirm smoke test passed after #226", conn, "http://x", "ws1"
        )
    assert first is not None and first["reused"] is False

    # Second near-identical prompt: _embed_text(prompt2) → p_emb again.
    # On buggy code the stored embedding is t_emb, so cosine(p_emb, t_emb) = 0.0 < 0.85
    # → falls through to mistral path → creates a second row.
    # After fix: stored embedding is p_emb, cosine(p_emb, p_emb) = 1.0 → reuse, no mistral.
    with patch.object(td, "_embed_text", return_value=p_emb), \
         patch.object(td, "_call_mistral_for_task",
                      return_value="Smoke-Test nach #226 bestätigen") as m:
        second = td.derive_research_question(
            "please confirm the smoke test after #226 is green", conn, "http://x", "ws1"
        )
    assert second is not None and second["reused"] is True
    m.assert_not_called()

    cnt = conn.execute(
        "SELECT COUNT(*) FROM research_questions WHERE workspace_id='ws1'"
    ).fetchone()[0]
    assert cnt == 1

    occ = conn.execute(
        "SELECT occurrence_count FROM research_questions WHERE workspace_id='ws1'"
    ).fetchone()[0]
    assert occ == 2


def test_init_schema_renames_legacy_task_categories_with_data(tmp_path):
    from src.memory.db_adapter import DBAdapter
    from src.memory.store import _init_schema
    db = DBAdapter.create(tmp_path / "legacy2.db")
    db.execute("CREATE TABLE task_categories (task_id TEXT PRIMARY KEY, title TEXT, "
               "embedding_id TEXT, parent_task_id TEXT, occurrence_count INTEGER, "
               "first_seen_at TEXT, last_used_at TEXT, workspace_id TEXT)")
    db.execute("INSERT INTO task_categories (task_id, title, first_seen_at, last_used_at, workspace_id) "
               "VALUES ('t1','q','2026-01-01','2026-01-01','ws')")
    db.execute("CREATE TABLE task_chunk_links (task_id TEXT, chunk_id TEXT, relevance_score REAL, "
               "created_at TEXT, PRIMARY KEY(task_id, chunk_id))")
    db.commit()
    _init_schema(db)
    names = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "task_categories" not in names           # renamed, not orphaned
    assert "research_questions" in names
    # the real data survived the rename
    row = db.execute("SELECT title FROM research_questions WHERE research_question_id='t1'").fetchone()
    assert row[0] == "q"
