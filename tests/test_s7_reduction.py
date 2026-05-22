"""Tests for Mayring S3 task-aware injection and S7 on-demand reduction."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mayring_core.memory.ingestion.categorization import (
    _task_relevant_categories,
    reduce_categories,
)
from mayring_core.memory.store import init_memory_db, update_chunk_category_labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db_with_chunks(tmp_path, labels_per_chunk: list[list[str]]):
    conn = init_memory_db(tmp_path / "m.db")
    now = "2026-05-14T00:00:00+00:00"
    conn.execute(
        "INSERT OR IGNORE INTO sources (source_id, source_type, captured_at, workspace_id) "
        "VALUES (?,?,?,?)",
        ("paper:p1", "paper", now, "ws"),
    )
    chunk_ids = []
    for i, labels in enumerate(labels_per_chunk):
        cid = f"chk_{i:04d}"
        conn.execute(
            "INSERT INTO chunks (chunk_id, source_id, text, category_labels, "
            "category_source, is_active, workspace_id, created_at) "
            "VALUES (?,?,?,?,?,1,?,?)",
            (cid, "paper:p1", f"text {i}", ",".join(labels), "hybrid", "ws", now),
        )
        chunk_ids.append(cid)
    conn.commit()
    return conn, chunk_ids


# ---------------------------------------------------------------------------
# S3 task-aware injection
# ---------------------------------------------------------------------------

def test_task_injection_no_chroma_returns_empty():
    """No chroma → empty list, no crash."""
    result = _task_relevant_categories(
        task="Patientenautonomie",
        conn=None,
        chroma_collection=None,
        ollama_url="http://x",
        model="m",
        workspace_id="ws",
    )
    assert result == []


def test_task_injection_no_task_returns_empty():
    """Empty task → no search, empty list."""
    result = _task_relevant_categories(
        task="",
        conn=None,
        chroma_collection=MagicMock(),
        ollama_url="http://x",
        model="m",
        workspace_id="ws",
    )
    assert result == []


def test_task_injection_extracts_labels_from_chroma():
    """Mock ChromaDB returns metadatas with labels → they are returned."""
    mock_chroma = MagicMock()
    mock_chroma.query.return_value = {
        "metadatas": [[
            {"category_labels": "patientenautonomie,informed-consent"},
            {"category_labels": "[neu]shared-decision,informed-consent"},
        ]]
    }
    with patch("src.analysis.context_rag._embed_texts", return_value=[[0.1, 0.2]]):
        result = _task_relevant_categories(
            task="Patientenautonomie",
            conn=None,
            chroma_collection=mock_chroma,
            ollama_url="http://x",
            model="m",
            workspace_id="ws",
        )
    assert "patientenautonomie" in result
    assert "informed-consent" in result
    assert "shared-decision" in result
    # [neu] prefix should be stripped
    assert all(not l.startswith("[neu]") for l in result)


def test_task_injection_forces_hybrid_mode(tmp_path):
    """When task_labels are injected, mode becomes 'hybrid' regardless of input."""
    from mayring_core.memory.schema import Chunk
    conn, _ = _make_db_with_chunks(tmp_path, [["auth"]] * 3)

    chunk = Chunk(
        chunk_id="test-chunk-001",
        source_id="paper:p1",
        text="This is about authentication flows.",
        category_labels=[],
    )

    mock_chroma = MagicMock()
    mock_chroma.query.return_value = {
        "metadatas": [[{"category_labels": "auth,security"}]]
    }

    with (
        patch("src.analysis.context_rag._embed_texts", return_value=[[0.1, 0.2]]),
        patch("src.analysis.analyzer._ollama_generate", return_value="auth") as mock_gen,
    ):
        from mayring_core.memory.ingestion.categorization import mayring_categorize
        result_chunks = mayring_categorize(
            [chunk],
            ollama_url="http://x",
            model="m",
            mode="inductive",   # explicitly inductive
            conn=conn,
            workspace_id="ws",
            task="Authentication",
            chroma_collection=mock_chroma,
        )

    # chunk.category_source should be "hybrid" because task_labels forced it
    assert result_chunks[0].category_source == "hybrid"


# ---------------------------------------------------------------------------
# S7 reduce_categories
# ---------------------------------------------------------------------------

def test_threshold_skip(tmp_path):
    """Fewer unique labels than threshold → skipped."""
    conn, ids = _make_db_with_chunks(tmp_path, [
        ["auth", "api"], ["auth", "tests"], ["config"]
    ])
    result = reduce_categories(ids, conn, None, "http://x", "m", threshold=10)
    assert result["skipped"] is True
    # reduce_categories counts DISTINCT labels: {auth, api, tests, config} = 4
    # (auth appears twice). The threshold compares against the distinct count.
    assert "unique_labels=4" in result["reason"]


def test_mapping_applied_to_sqlite(tmp_path):
    """Valid LLM mapping → SQL UPDATE applied correctly."""
    # 20 chunks with auth variants to exceed threshold
    labels = [["auth-check"], ["auth-validation"], ["auth-middleware"]]
    labels += [["data-access"]] * 17
    conn, ids = _make_db_with_chunks(tmp_path, labels)

    mock_mapping = {
        "auth-check": "auth", "auth-validation": "auth",
        "auth-middleware": "auth", "data-access": "data-access",
    }
    with patch("src.analysis.analyzer._ollama_generate",
               return_value=json.dumps(mock_mapping)):
        result = reduce_categories(ids, conn, None, "http://x", "m", threshold=3)

    assert result["skipped"] is False
    assert result["chunks_updated"] > 0
    row = conn.execute(
        "SELECT category_labels FROM chunks WHERE chunk_id = ?", (ids[0],)
    ).fetchone()
    assert row[0] == "auth"


def test_malformed_llm_json_skips(tmp_path):
    """LLM returns garbage → skipped with reason, no crash."""
    # Need >threshold distinct labels so we get PAST the threshold-skip and
    # actually reach the LLM call whose malformed output we're testing.
    labels = [["auth-check"], ["auth-validation"], ["auth-middleware"], ["data-access"]]
    labels += [["auth-check"]] * 16
    conn, ids = _make_db_with_chunks(tmp_path, labels)
    with patch("src.analysis.analyzer._ollama_generate",
               return_value="I cannot help with that"):
        result = reduce_categories(ids, conn, None, "http://x", "m", threshold=3)
    assert result["skipped"] is True
    assert result["reason"] == "empty mapping"


def test_update_chunk_category_labels_strips_neu(tmp_path):
    """[neu] prefix is stripped before mapping lookup."""
    conn = init_memory_db(tmp_path / "m.db")
    now = "2026-05-14T00:00:00+00:00"
    conn.execute(
        "INSERT OR IGNORE INTO sources (source_id, source_type, captured_at, workspace_id) "
        "VALUES (?,?,?,?)", ("s1", "note", now, "ws")
    )
    conn.execute(
        "INSERT INTO chunks (chunk_id, source_id, text, category_labels, "
        "category_source, is_active, workspace_id, created_at) "
        "VALUES (?,?,?,?,?,1,?,?)",
        ("c1", "s1", "t", "[neu]auth-check,data-access", "hybrid", "ws", now),
    )
    conn.commit()

    mapping = {"auth-check": "auth", "data-access": "data-access"}
    n = update_chunk_category_labels(conn, ["c1"], mapping)
    assert n == 1
    row = conn.execute("SELECT category_labels FROM chunks WHERE chunk_id='c1'").fetchone()
    assert row[0] == "auth,data-access"
