"""Tests for ambient context layer (src/memory/ambient.py)."""
import sqlite3
import pytest
from pathlib import Path

from src.memory.ambient import (
    _load_recent_conversations,
    _load_recent_issues,
    _load_wiki_top_connections,
    load_ambient_snapshot,
    generate_ambient_snapshot,
)


def _init_test_db() -> sqlite3.Connection:
    """Create an in-memory test DB with full schema."""
    from src.memory.store import _init_schema
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _init_schema(conn)
    return conn


class TestLoadRecentConversations:
    """Test _load_recent_conversations()."""

    def test_empty_database(self):
        """Empty DB returns empty list."""
        conn = _init_test_db()
        result = _load_recent_conversations(conn, "myrepo")
        assert result == []
        conn.close()

    def test_filters_by_repo(self):
        """Only returns conversations for matching repo."""
        conn = _init_test_db()

        # Insert source with repo="myrepo"
        conn.execute(
            "INSERT INTO sources (source_id, source_type, repo, path, branch, \"commit\", content_hash, captured_at) VALUES (?,?,?,?,?,?,?,?)",
            ("conv:myrepo:1", "conversation_summary", "myrepo", "conv/1", "main", "", "sha256:abc", "2026-01-01T00:00:00")
        )
        conn.execute(
            "INSERT INTO chunks (chunk_id, source_id, text, chunk_level, is_active, created_at) VALUES (?,?,?,?,?,?)",
            ("chk1", "conv:myrepo:1", "This is conversation 1", "section", 1, "2026-01-01T00:00:00")
        )

        # Insert source with repo="otherrepo"
        conn.execute(
            "INSERT INTO sources (source_id, source_type, repo, path, branch, \"commit\", content_hash, captured_at) VALUES (?,?,?,?,?,?,?,?)",
            ("conv:otherrepo:1", "conversation_summary", "otherrepo", "conv/1", "main", "", "sha256:def", "2026-01-01T00:00:00")
        )
        conn.execute(
            "INSERT INTO chunks (chunk_id, source_id, text, chunk_level, is_active, created_at) VALUES (?,?,?,?,?,?)",
            ("chk2", "conv:otherrepo:1", "This is conversation 2", "section", 1, "2026-01-01T00:00:00")
        )
        conn.commit()

        result = _load_recent_conversations(conn, "myrepo")
        assert len(result) == 1
        assert "conversation 1" in result[0]
        conn.close()


class TestLoadRecentIssues:
    """Test _load_recent_issues()."""

    def test_empty_database(self):
        """Empty DB returns empty list."""
        conn = _init_test_db()
        result = _load_recent_issues(conn, "myrepo")
        assert result == []
        conn.close()

    def test_loads_issue_summaries(self):
        """Loads issue summaries from DB."""
        conn = _init_test_db()

        conn.execute(
            "INSERT INTO sources (source_id, source_type, repo, path, branch, \"commit\", content_hash, captured_at) VALUES (?,?,?,?,?,?,?,?)",
            ("issue:myrepo:1", "github_issue", "myrepo", "issues/1", "main", "", "sha256:abc", "2026-01-01T00:00:00")
        )
        conn.execute(
            "INSERT INTO chunks (chunk_id, source_id, text, chunk_level, is_active, created_at) VALUES (?,?,?,?,?,?)",
            ("chk1", "issue:myrepo:1", "Issue about login bug", "section", 1, "2026-01-01T00:00:00")
        )
        conn.commit()

        result = _load_recent_issues(conn, "myrepo")
        assert len(result) == 1
        assert "login bug" in result[0]
        conn.close()


class TestLoadWikiTopConnections:
    """Test _load_wiki_top_connections()."""

    def test_no_wiki_file(self, tmp_path, monkeypatch):
        """No wiki file returns default message."""
        monkeypatch.chdir(tmp_path)
        result = _load_wiki_top_connections("myrepo")
        assert result == "(kein Wiki vorhanden)"


class TestLoadAmbientSnapshot:
    """Test load_ambient_snapshot()."""

    def test_no_snapshot_returns_none(self):
        """No snapshot in DB returns None."""
        conn = _init_test_db()
        result = load_ambient_snapshot(conn, "myrepo")
        assert result is None
        conn.close()

    def test_returns_snapshot_content(self):
        """Loads and returns existing snapshot."""
        conn = _init_test_db()
        source_id = "ambient:myrepo:snapshot"

        # Insert source
        conn.execute(
            "INSERT INTO sources (source_id, source_type, repo, path, branch, \"commit\", content_hash, captured_at) VALUES (?,?,?,?,?,?,?,?)",
            (source_id, "ambient_snapshot", "myrepo", "ambient/snapshot", "local", "", "sha256:abc", "2026-01-01T00:00:00")
        )

        # Insert chunk with snapshot content
        conn.execute(
            "INSERT INTO chunks (chunk_id, source_id, text, chunk_level, is_active, created_at) VALUES (?,?,?,?,?,?)",
            ("chk1", source_id, "Mein Snapshot-Text", "ambient_snapshot", 1, "2026-01-01T00:00:00")
        )
        conn.commit()

        result = load_ambient_snapshot(conn, "myrepo")
        assert result == "Mein Snapshot-Text"
        conn.close()


def test_generate_ambient_snapshot_returns_none_on_empty_model():
    """generate_ambient_snapshot returns None immediately when model is empty."""
    conn = _init_test_db()
    result = generate_ambient_snapshot(conn, "http://localhost:11434", "", "myrepo")
    assert result is None
    conn.close()


def test_generate_ambient_snapshot_returns_none_on_llm_error(monkeypatch):
    """generate_ambient_snapshot returns None when LLM raises."""
    conn = _init_test_db()

    def _fake_generate(*args, **kwargs):
        raise RuntimeError("Ollama not reachable")

    monkeypatch.setattr("src.analysis.analyzer._ollama_generate", _fake_generate)
    result = generate_ambient_snapshot(conn, "http://localhost:11434", "llama3", "myrepo")
    assert result is None
    conn.close()
