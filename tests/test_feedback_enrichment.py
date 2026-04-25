"""Tests for feedback() auto-enrichment from context_feedback_log (Issue #86)."""
from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock, patch


def _make_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE context_feedback_log (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            trigger_ids      TEXT NOT NULL,
            context_text     TEXT NOT NULL,
            was_referenced   INTEGER NOT NULL DEFAULT 0,
            led_to_retrieval INTEGER NOT NULL DEFAULT 0,
            relevance_score  REAL NOT NULL DEFAULT 0.0,
            captured_at      TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE chunk_feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id    TEXT NOT NULL,
            signal      TEXT NOT NULL,
            metadata    TEXT NOT NULL DEFAULT '{}',
            created_at  TEXT NOT NULL DEFAULT ''
        );
    """)
    return conn


def _register_and_call_feedback(conn, chunk_id, signal, metadata=None):
    """Instantiate the FastMCP tool and call feedback() with the given args."""
    from mcp.server.fastmcp import FastMCP
    from src.api.mcp_memory_tools import register_memory_tools

    mcp = FastMCP("test")
    with (
        patch("src.api.mcp_memory_tools._get_conn", return_value=conn),
        patch("src.api.mcp_memory_tools._get_chroma", return_value=None),
        patch("src.api.mcp_memory_tools._enforce_tenant", side_effect=lambda x: x or "default"),
        patch("src.api.mcp_memory_tools.add_feedback") as mock_add_fb,
        patch("src.api.mcp_memory_tools.invalidate_query_cache"),
    ):
        register_memory_tools(mcp)
        tools = {t.name: t for t in mcp._tool_manager.list_tools()}
        result = tools["feedback"].fn(chunk_id=chunk_id, signal=signal, metadata=metadata)
        return result, mock_add_fb


def test_feedback_enriches_query_context_from_log():
    conn = _make_db()
    conn.execute(
        "INSERT INTO context_feedback_log (trigger_ids,context_text,was_referenced,captured_at)"
        " VALUES (?,?,0,'2026-01-01')",
        (json.dumps(["chk_abc", "chk_xyz"]), "Relevant chunk about memory retrieval."),
    )
    conn.commit()

    result, mock_add_fb = _register_and_call_feedback(conn, "chk_abc", "positive")

    assert result.get("recorded") is True
    _, _, _, meta = mock_add_fb.call_args.args
    assert "query_context" in meta
    assert "memory retrieval" in meta["query_context"]


def test_feedback_marks_was_referenced_in_log():
    conn = _make_db()
    conn.execute(
        "INSERT INTO context_feedback_log (trigger_ids,context_text,was_referenced,captured_at)"
        " VALUES (?,?,0,'2026-01-01')",
        (json.dumps(["chk_def"]), "Some context text."),
    )
    conn.commit()

    _register_and_call_feedback(conn, "chk_def", "positive")

    row = conn.execute("SELECT was_referenced FROM context_feedback_log WHERE id=1").fetchone()
    assert row[0] == 1


def test_feedback_no_log_entry_still_records():
    conn = _make_db()
    result, mock_add_fb = _register_and_call_feedback(conn, "chk_unknown", "negative")

    assert result.get("recorded") is True
    _, _, _, meta = mock_add_fb.call_args.args
    assert "query_context" not in meta


def test_feedback_preserves_existing_metadata():
    conn = _make_db()
    conn.execute(
        "INSERT INTO context_feedback_log (trigger_ids,context_text,was_referenced,captured_at)"
        " VALUES (?,?,0,'2026-01-01')",
        (json.dumps(["chk_ghi"]), "Auto context."),
    )
    conn.commit()

    result, mock_add_fb = _register_and_call_feedback(
        conn, "chk_ghi", "positive", {"task": "my task"}
    )

    _, _, _, meta = mock_add_fb.call_args.args
    assert meta.get("task") == "my task"
    assert "query_context" in meta
