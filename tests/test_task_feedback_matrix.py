import json
from src.memory.db_adapter import DBAdapter
from src.memory.store import _init_schema, add_feedback
from src.wiki_v2.store import get_task_feedback_matrix


def _db():
    db = DBAdapter.memory()
    _init_schema(db)
    return db


def _insert_chunk(db, chunk_id, source_id="src/api/auth.py"):
    db.execute(
        "INSERT OR IGNORE INTO sources(source_id, source_type, repo, path, content_hash, captured_at)"
        " VALUES (?, 'repo_file', 'test', ?, 'h', '2026-01-01')",
        (source_id, source_id),
    )
    db.execute(
        "INSERT OR IGNORE INTO chunks(chunk_id, source_id, text, text_hash, dedup_key, created_at, is_active)"
        " VALUES (?, ?, 'text', 'h', 'd', '2026-01-01', 1)",
        (chunk_id, source_id),
    )
    db.commit()


def test_task_feedback_matrix_groups_by_query():
    db = _db()
    _insert_chunk(db, "chk_a", "src/api/auth.py")
    _insert_chunk(db, "chk_b", "src/api/jwt.py")
    add_feedback(db, "chk_a", "positive", {"query_context": "fix auth bug"})
    add_feedback(db, "chk_b", "negative", {"query_context": "fix auth bug"})
    add_feedback(db, "chk_a", "positive", {"query_context": "optimize search"})

    result = get_task_feedback_matrix(db, limit=50)
    queries = [t["query"] for t in result]
    assert "fix auth bug" in queries
    assert "optimize search" in queries
    auth_task = next(t for t in result if t["query"] == "fix auth bug")
    assert len(auth_task["chunks"]) == 2


def test_task_feedback_matrix_skips_no_query():
    db = _db()
    _insert_chunk(db, "chk_c")
    add_feedback(db, "chk_c", "positive", {})
    result = get_task_feedback_matrix(db, limit=50)
    assert all(t["query"] for t in result)


def test_task_feedback_matrix_query_filter():
    db = _db()
    _insert_chunk(db, "chk_d")
    add_feedback(db, "chk_d", "positive", {"query_context": "memory search performance"})
    result = get_task_feedback_matrix(db, limit=50, query_filter="memory")
    assert len(result) >= 1
    assert all("memory" in t["query"].lower() for t in result)
