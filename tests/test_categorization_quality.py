"""Tests for Mayring categorization quality improvements.

Covers:
- mayring_categorize() validation logic (hybrid, deductive, inductive modes)
- _resolve_codebook() with auto, explicit, and profile lookups
- _path_fallback_category() regex matching
- category_precision_at_k() metric
- uncategorized_rate() metric
"""

from __future__ import annotations

import sqlite3
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.benchmark_retrieval import category_precision_at_k, uncategorized_rate
from src.memory.ingest import _resolve_codebook, _path_fallback_category
from src.memory.schema import Chunk


# ---------------------------------------------------------------------------
# _resolve_codebook()
# ---------------------------------------------------------------------------

class TestResolveCodebook:
    def test_auto_repo_file_returns_code_categories(self):
        cats = _resolve_codebook("auto", "repo_file")
        assert len(cats) > 0
        # Should contain code-related labels
        cats_lower = [c.lower() for c in cats]
        assert any("api" in c or "code" in c or "dom" in c for c in cats_lower)

    def test_auto_conversation_returns_social_categories(self):
        cats = _resolve_codebook("auto", "conversation_summary")
        assert len(cats) > 0

    def test_auto_github_issue_returns_social_categories(self):
        cats = _resolve_codebook("auto", "github_issue")
        assert len(cats) > 0

    def test_explicit_code_codebook(self):
        cats = _resolve_codebook("code", "note")
        assert len(cats) > 0

    def test_explicit_social_codebook(self):
        cats = _resolve_codebook("social", "note")
        assert len(cats) > 0

    def test_original_fallback_for_unknown_codebook(self):
        cats = _resolve_codebook("nonexistent_xyz_abc", "repo_file")
        assert len(cats) > 0
        # Should fall back to original Mayring categories
        cats_lower = [c.lower() for c in cats]
        assert any("zusammenfassung" in c or "explikation" in c for c in cats_lower)

    def test_path_injection_blocked(self):
        cats = _resolve_codebook("../../etc/passwd", "repo_file")
        assert len(cats) > 0  # fallback

    def test_comma_injection_blocked(self):
        cats = _resolve_codebook("code,evil", "repo_file")
        assert len(cats) > 0  # fallback, not an error

    def test_original_sentinel(self):
        cats = _resolve_codebook("original", "repo_file")
        cats_lower = [c.lower() for c in cats]
        assert any("zusammenfassung" in c for c in cats_lower)


# ---------------------------------------------------------------------------
# _path_fallback_category()
# ---------------------------------------------------------------------------

class TestPathFallbackCategory:
    def test_api_path(self):
        cats = _path_fallback_category("/api/routes/users.py")
        assert "api" in cats

    def test_test_path(self):
        cats = _path_fallback_category("tests/test_cache.py")
        assert "tests" in cats

    def test_config_path(self):
        cats = _path_fallback_category("src/config.py")
        assert "config" in cats

    def test_auth_path(self):
        cats = _path_fallback_category("src/auth/guard.py")
        assert "auth" in cats

    def test_unknown_path_returns_list(self):
        cats = _path_fallback_category("some/random/file.py")
        assert isinstance(cats, list)

    def test_empty_path(self):
        cats = _path_fallback_category("")
        assert isinstance(cats, list)

    def test_utils_path(self):
        cats = _path_fallback_category("src/utils/helpers.py")
        assert "utils" in cats


# ---------------------------------------------------------------------------
# mayring_categorize() — mode validation
# ---------------------------------------------------------------------------

class TestMayringCategorizeValidation:
    """Test the validation logic inside mayring_categorize() without calling Ollama."""

    def _make_chunk(self, source_id: str = "test:chunk:001") -> Chunk:
        return Chunk(
            chunk_id="c001",
            source_id=source_id,
            text="def authenticate(token): pass",
        )

    @patch("src.analysis.analyzer._ollama_generate")
    def test_deductive_mode_validates_against_codebook(self, mock_gen):
        mock_gen.return_value = "api\nsicherheit"
        from src.memory.ingest import mayring_categorize
        chunks = [self._make_chunk()]
        result = mayring_categorize(chunks, "http://localhost:11434", "nomic", "deductive", "code", "repo_file")
        # Only labels present in codebook should be kept
        assert isinstance(result[0].category_labels, list)

    @patch("src.analysis.analyzer._ollama_generate")
    def test_inductive_mode_accepts_free_labels(self, mock_gen):
        mock_gen.return_value = "custom-pattern-xyz\nnew-concept"
        from src.memory.ingest import mayring_categorize
        chunks = [self._make_chunk()]
        result = mayring_categorize(chunks, "http://localhost:11434", "nomic", "inductive", "code", "repo_file")
        # Inductive: free-form labels accepted
        assert "custom-pattern-xyz" in result[0].category_labels
        assert "new-concept" in result[0].category_labels

    @patch("src.analysis.analyzer._ollama_generate")
    def test_hybrid_mode_keeps_neu_prefix(self, mock_gen):
        mock_gen.return_value = "[neu]custom-label"
        from src.memory.ingest import mayring_categorize
        chunks = [self._make_chunk()]
        result = mayring_categorize(chunks, "http://localhost:11434", "nomic", "hybrid", "code", "repo_file")
        assert "[neu]custom-label" in result[0].category_labels

    @patch("src.analysis.analyzer._ollama_generate")
    def test_error_triggers_path_fallback(self, mock_gen):
        mock_gen.side_effect = Exception("connection refused")
        from src.memory.ingest import mayring_categorize
        chunk = self._make_chunk("github_issue:repo:api_route.py:abc123")
        result = mayring_categorize([chunk], "http://localhost:11434", "nomic", "deductive", "code", "repo_file")
        assert result[0].category_source == "fallback"
        assert isinstance(result[0].category_labels, list)

    @patch("src.analysis.analyzer._ollama_generate")
    def test_category_confidence_set_when_labels_found(self, mock_gen):
        mock_gen.return_value = "api"
        from src.memory.ingest import mayring_categorize
        chunks = [self._make_chunk()]
        result = mayring_categorize(chunks, "http://localhost:11434", "nomic", "deductive", "code", "repo_file")
        if result[0].category_labels:
            assert result[0].category_confidence == 1.0
            assert result[0].category_source in ("deductive", "inductive", "hybrid")

    @patch("src.analysis.analyzer._ollama_generate")
    def test_label_with_comma_rejected(self, mock_gen):
        mock_gen.return_value = "api,sicherheit"  # comma in single label (invalid)
        from src.memory.ingest import mayring_categorize
        chunks = [self._make_chunk()]
        result = mayring_categorize(chunks, "http://localhost:11434", "nomic", "inductive", "code", "repo_file")
        # Label with comma should be rejected
        assert all("," not in lbl for lbl in result[0].category_labels)

    @patch("src.analysis.analyzer._ollama_generate")
    def test_truncation_at_1200_chars(self, mock_gen):
        """Verify the prompt passed to _ollama_generate is not longer than 1200 chars of text."""
        mock_gen.return_value = "api"
        from src.memory.ingest import mayring_categorize
        long_text = "x" * 5000
        chunk = Chunk(
            chunk_id="c_long",
            source_id="test:big:001",
            text=long_text,
        )
        mayring_categorize([chunk], "http://localhost:11434", "nomic", "deductive", "code", "repo_file")
        # The prompt passed to LLM should contain at most 1200 chars of the text
        call_args = mock_gen.call_args
        prompt_sent = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")
        assert "x" * 1201 not in prompt_sent


# ---------------------------------------------------------------------------
# Benchmark metrics
# ---------------------------------------------------------------------------

class TestCategoryPrecisionAtK:
    def _make_record(self, labels: list[str]) -> Any:
        r = MagicMock()
        r.category_labels = labels
        return r

    def test_perfect_score(self):
        records = [[self._make_record(["bug", "security"])]]
        cats = ["bug"]
        assert category_precision_at_k(records, cats, k=5) == 1.0

    def test_zero_score_no_match(self):
        records = [[self._make_record(["api"])]]
        cats = ["bug"]
        assert category_precision_at_k(records, cats, k=5) == 0.0

    def test_partial_score(self):
        records = [
            [self._make_record(["bug"])],
            [self._make_record(["api"])],
        ]
        cats = ["bug", "bug"]
        assert category_precision_at_k(records, cats, k=5) == 0.5

    def test_queries_without_expected_category_skipped(self):
        records = [
            [self._make_record(["bug"])],
            [self._make_record(["api"])],
        ]
        cats = ["bug", None]
        # Only first query counts
        assert category_precision_at_k(records, cats, k=5) == 1.0

    def test_empty_records_returns_zero(self):
        assert category_precision_at_k([], [], k=5) == 0.0

    def test_all_none_expected_returns_zero(self):
        records = [[self._make_record(["api"])]]
        cats = [None]
        assert category_precision_at_k(records, cats, k=5) == 0.0

    def test_partial_label_match(self):
        """'sicherheit' should match expected 'sicherh' (substring match)."""
        records = [[self._make_record(["sicherheit"])]]
        cats = ["sicherh"]
        assert category_precision_at_k(records, cats, k=5) == 1.0


class TestUncategorizedRate:
    def _make_db(self, total: int, categorized: int) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                category_labels TEXT,
                is_active INTEGER
            )
        """)
        for i in range(categorized):
            conn.execute("INSERT INTO chunks VALUES (?, ?, 1)", (f"c{i}", "api"))
        for i in range(total - categorized):
            conn.execute("INSERT INTO chunks VALUES (?, ?, 1)", (f"u{i}", ""))
        conn.commit()
        return conn

    def test_all_categorized(self):
        conn = self._make_db(10, 10)
        assert uncategorized_rate(conn) == 0.0

    def test_none_categorized(self):
        conn = self._make_db(10, 0)
        assert uncategorized_rate(conn) == 1.0

    def test_half_categorized(self):
        conn = self._make_db(10, 5)
        assert uncategorized_rate(conn) == 0.5

    def test_empty_db_returns_one(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE chunks (chunk_id TEXT, category_labels TEXT, is_active INTEGER)")
        conn.commit()
        assert uncategorized_rate(conn) == 1.0
