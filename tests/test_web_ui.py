"""Tests for src/ollama_status.py and src/web_ui.py.

Mock strategy:
  - unittest.mock.patch on src.ollama_status.check_ollama
  - unittest.mock.patch on src.memory_retrieval.search
  - unittest.mock.patch on subprocess.run for subprocess-path tests
  - unittest.mock.patch on httpx.get for HTTP-fallback tests
"""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test A: check_ollama — Ollama not reachable
# ---------------------------------------------------------------------------

class TestCheckOllamaUnavailable:
    """check_ollama returns (False, []) when nothing responds."""

    def test_both_paths_fail(self):
        """subprocess fails + httpx fails → (False, [])."""
        from src.ollama_client import check_ollama

        with (
            patch("subprocess.run", side_effect=FileNotFoundError("ollama not found")),
            patch("httpx.get", side_effect=Exception("connection refused")),
        ):
            ok, models = check_ollama("http://localhost:11434")

        assert ok is False
        assert models == []

    def test_no_exception_raised(self):
        """Must not propagate any exception."""
        from src.ollama_client import check_ollama

        with (
            patch("subprocess.run", side_effect=OSError("no such file")),
            patch("httpx.get", side_effect=RuntimeError("network error")),
        ):
            # Should not raise
            result = check_ollama("http://localhost:11434")

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_timeout_in_subprocess(self):
        """subprocess timeout → httpx path → also fails → (False, [])."""
        import subprocess
        from src.ollama_client import check_ollama

        with (
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ollama", 3)),
            patch("httpx.get", side_effect=Exception("timeout")),
        ):
            ok, models = check_ollama("http://localhost:11434")

        assert ok is False
        assert models == []


# ---------------------------------------------------------------------------
# Test B: check_ollama — subprocess fails, httpx fallback
# ---------------------------------------------------------------------------

class TestCheckOllamaSubprocessFallback:
    """When subprocess fails, check_ollama falls back to httpx /api/tags."""

    def test_subprocess_fails_httpx_succeeds(self):
        """subprocess returncode != 0 → httpx returns model list."""
        from src.ollama_client import check_ollama

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "mistral:7b"},
                {"name": "nomic-embed-text"},
            ]
        }

        with (
            patch("subprocess.run", side_effect=FileNotFoundError("ollama not found")),
            patch("httpx.get", return_value=mock_resp),
        ):
            ok, models = check_ollama("http://localhost:11434")

        assert ok is True
        assert "mistral:7b" in models
        assert "nomic-embed-text" in models

    def test_subprocess_fails_httpx_status_error(self):
        """subprocess fails + httpx returns 500 → (False, [])."""
        from src.ollama_client import check_ollama

        mock_resp = MagicMock()
        mock_resp.status_code = 500

        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch("httpx.get", return_value=mock_resp),
        ):
            ok, models = check_ollama("http://localhost:11434")

        assert ok is False
        assert models == []

    def test_subprocess_success_no_httpx_call(self):
        """If subprocess succeeds, httpx must not be called."""
        from src.ollama_client import check_ollama

        proc_mock = MagicMock()
        proc_mock.returncode = 0
        proc_mock.stdout = "NAME               ID\nmistral:7b         abc123\n"

        with (
            patch("subprocess.run", return_value=proc_mock) as mock_run,
            patch("httpx.get") as mock_get,
        ):
            ok, models = check_ollama("http://localhost:11434")

        assert ok is True
        assert "mistral:7b" in models
        mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# Test C: Ingest tab — warning when Ollama unavailable
# ---------------------------------------------------------------------------

class TestIngestTabWithoutOllama:
    """_do_ingest() returns warning when ollama_available=False."""

    def test_warning_in_output_when_ollama_down(self, tmp_path):
        """Ingest proceeds but result contains warning key."""
        import src.web_ui as web_ui

        # Patch memory modules so we can test without real DB
        mock_ingest_result = {
            "source_id": "repo:ui:test.txt",
            "chunk_ids": ["chk_aabbccdd"],
            "indexed": False,
            "deduped": 0,
            "superseded": 0,
        }

        fake_conn = MagicMock(spec=sqlite3.Connection)

        with (
            patch.object(web_ui, "_MEMORY_READY", True),
            patch.object(web_ui, "_conn", fake_conn),
            patch("src.web_ui._get_conn", return_value=fake_conn),
            patch("src.web_ui._get_chroma", return_value=None),
            patch("src.web_ui.ingest", return_value=mock_ingest_result),
            patch("src.web_ui.Source") as MockSource,
            patch("src.web_ui.hashlib") as mock_hashlib,
        ):
            # Setup hashlib mock
            mock_hash = MagicMock()
            mock_hash.hexdigest.return_value = "a" * 64
            mock_hashlib.sha256.return_value = mock_hash

            # Setup Source mock
            mock_src_instance = MagicMock()
            MockSource.return_value = mock_src_instance
            MockSource.make_id.return_value = "repo:ui:test.txt"

            raw = web_ui._do_ingest(
                text_input="def foo(): pass",
                file_upload=None,
                source_path="test.txt",
                repo="",
                categorize=False,
                mode="hybrid",
                codebook="auto",
                model="",
                ollama_available=False,
            )

        result = json.loads(raw)
        assert "warning" in result
        assert "Ollama" in result["warning"]

    def test_no_content_returns_error(self):
        """Empty text + no file → error JSON."""
        import src.web_ui as web_ui

        with patch.object(web_ui, "_MEMORY_READY", True):
            raw = web_ui._do_ingest(
                text_input="",
                file_upload=None,
                source_path="",
                repo="",
                categorize=False,
                mode="hybrid",
                codebook="auto",
                model="",
                ollama_available=False,
            )

        result = json.loads(raw)
        assert "error" in result


# ---------------------------------------------------------------------------
# Test D: Search — symbolic fallback without Ollama
# ---------------------------------------------------------------------------

class TestSearchFallbackSymbolic:
    """search() is called and returns a list even without vector embeddings."""

    def test_search_returns_list_without_embedding(self):
        """_do_search returns (status, rows) when ollama_available=False."""
        import src.web_ui as web_ui
        from src.memory_schema import RetrievalRecord

        fake_record = RetrievalRecord(
            chunk_id="chk_aabbccdd112233",
            score_final=0.42,
            score_symbolic=0.42,
            source_id="repo:test:foo.py",
            text="def foo(): pass",
            category_labels=["utility"],
            reasons=["token_overlap"],
        )

        fake_conn = MagicMock(spec=sqlite3.Connection)

        with (
            patch.object(web_ui, "_MEMORY_READY", True),
            patch("src.web_ui._get_conn", return_value=fake_conn),
            patch("src.web_ui._get_chroma", return_value=None),
            patch("src.web_ui.search", return_value=[fake_record]),
        ):
            status, rows = web_ui._do_search(
                query="foo",
                top_k=5,
                ollama_available=False,
            )

        assert "1" in status or "symbolisch" in status
        assert len(rows) == 1
        assert rows[0][0] == "chk_aabbccdd"  # truncated to 12 chars

    def test_search_empty_query_returns_hint(self):
        """Empty query → hint message, no rows."""
        import src.web_ui as web_ui

        with patch.object(web_ui, "_MEMORY_READY", True):
            status, rows = web_ui._do_search("", 8, False)

        assert "eingeben" in status.lower() or "suchbegriff" in status.lower()
        assert rows == []

    def test_search_no_memory_ready(self):
        """_MEMORY_READY=False → error message, no rows."""
        import src.web_ui as web_ui

        with patch.object(web_ui, "_MEMORY_READY", False):
            status, rows = web_ui._do_search("something", 8, True)

        assert "nicht geladen" in status.lower() or "memory" in status.lower()
        assert rows == []


# ---------------------------------------------------------------------------
# Test E: Feedback — add_feedback correctly called
# ---------------------------------------------------------------------------

class TestFeedbackWrite:
    """_do_feedback() calls add_feedback() with correct arguments."""

    def test_feedback_positive(self):
        """Positive signal without label → add_feedback called with empty metadata."""
        import src.web_ui as web_ui

        fake_conn = MagicMock(spec=sqlite3.Connection)

        with (
            patch.object(web_ui, "_MEMORY_READY", True),
            patch("src.web_ui._get_conn", return_value=fake_conn),
            patch("src.web_ui.add_feedback") as mock_add_feedback,
        ):
            result = web_ui._do_feedback(
                chunk_id="chk_aabbccdd",
                signal="positive",
                label="",
            )

        mock_add_feedback.assert_called_once_with(
            fake_conn, "chk_aabbccdd", "positive", {}
        )
        assert "gespeichert" in result.lower()

    def test_feedback_with_label(self):
        """Label is passed in metadata dict."""
        import src.web_ui as web_ui

        fake_conn = MagicMock(spec=sqlite3.Connection)

        with (
            patch.object(web_ui, "_MEMORY_READY", True),
            patch("src.web_ui._get_conn", return_value=fake_conn),
            patch("src.web_ui.add_feedback") as mock_add_feedback,
        ):
            web_ui._do_feedback(
                chunk_id="chk_test",
                signal="negative",
                label="irrelevant duplicate",
            )

        mock_add_feedback.assert_called_once_with(
            fake_conn, "chk_test", "negative", {"label": "irrelevant duplicate"}
        )

    def test_feedback_empty_chunk_id(self):
        """Empty chunk_id → error message without calling add_feedback."""
        import src.web_ui as web_ui

        with (
            patch.object(web_ui, "_MEMORY_READY", True),
            patch("src.web_ui.add_feedback") as mock_add_feedback,
        ):
            result = web_ui._do_feedback(
                chunk_id="",
                signal="positive",
                label="",
            )

        mock_add_feedback.assert_not_called()
        assert "chunk" in result.lower() or "id" in result.lower()

    def test_feedback_memory_not_ready(self):
        """_MEMORY_READY=False → error message."""
        import src.web_ui as web_ui

        with (
            patch.object(web_ui, "_MEMORY_READY", False),
            patch("src.web_ui.add_feedback") as mock_add_feedback,
        ):
            result = web_ui._do_feedback("chk_x", "positive", "")

        mock_add_feedback.assert_not_called()
        assert "nicht geladen" in result.lower() or "memory" in result.lower()


# ---------------------------------------------------------------------------
# Test F: Model selector — _do_ingest() uses provided model
# ---------------------------------------------------------------------------

class TestModelSelector:
    """Tests für Modell-Auswahl in der UI."""

    def test_do_ingest_uses_provided_model(self) -> None:
        """_do_ingest() übergibt model an ingest()."""
        from src.web_ui import _do_ingest
        captured: list[str] = []

        def fake_ingest(source, content, conn, chroma_collection, ollama_url, model, opts=None):
            captured.append(model)
            return {"source_id": "x", "chunk_ids": [], "indexed": False, "deduped": 0, "superseded": 0}

        with patch("src.web_ui.ingest", side_effect=fake_ingest), \
             patch("src.web_ui._get_conn", return_value=MagicMock()), \
             patch("src.web_ui._get_chroma", return_value=None), \
             patch("src.web_ui._MEMORY_READY", True):
            _do_ingest("hello world", None, "test.txt", "owner/repo",
                       categorize=False, mode="hybrid", codebook="auto",
                       model="mistral:7b", ollama_available=True)

        assert captured == ["mistral:7b"]

    def test_do_ingest_passes_mode_and_codebook_in_opts(self) -> None:
        """_do_ingest() schreibt mode + codebook in opts."""
        from src.web_ui import _do_ingest
        captured_opts: list[dict] = []

        def fake_ingest(source, content, conn, chroma_collection, ollama_url, model, opts=None):
            captured_opts.append(opts or {})
            return {"source_id": "x", "chunk_ids": [], "indexed": False, "deduped": 0, "superseded": 0}

        with patch("src.web_ui.ingest", side_effect=fake_ingest), \
             patch("src.web_ui._get_conn", return_value=MagicMock()), \
             patch("src.web_ui._get_chroma", return_value=None), \
             patch("src.web_ui._MEMORY_READY", True):
            _do_ingest("hello world", None, "test.txt", "owner/repo",
                       categorize=True, mode="deductive", codebook="social",
                       model="llama3", ollama_available=True)

        assert captured_opts[0]["mode"] == "deductive"
        assert captured_opts[0]["codebook"] == "social"
        assert captured_opts[0]["categorize"] is True


# ---------------------------------------------------------------------------
# Test G: Conversation tab — _do_ingest_conversation()
# ---------------------------------------------------------------------------

class TestConversationTab:
    """Tests für Conversation-Summary Ingestion via UI."""

    def test_do_ingest_conversation_calls_ingest_conversation_summary(self) -> None:
        from src.web_ui import _do_ingest_conversation
        captured: list[dict] = []

        def fake_ingest_conv(summary_text, conn, chroma_collection, ollama_url, model,
                             session_id=None, run_id=None):
            captured.append({"session_id": session_id, "run_id": run_id, "text": summary_text})
            return {"source_id": "conv:x", "chunk_ids": ["c1"], "indexed": False, "deduped": 0, "superseded": 0}

        with patch("src.web_ui.ingest_conversation_summary", side_effect=fake_ingest_conv), \
             patch("src.web_ui._get_conn", return_value=MagicMock()), \
             patch("src.web_ui._get_chroma", return_value=None), \
             patch("src.web_ui._MEMORY_READY", True):
            result = _do_ingest_conversation(
                summary_text="## Summary\n\nWas wir gemacht haben.",
                session_id="sess-abc",
                run_id="run-xyz",
                model="mistral:7b",
                ollama_available=True,
            )

        assert captured[0]["session_id"] == "sess-abc"
        assert captured[0]["run_id"] == "run-xyz"
        assert "conv:x" in result

    def test_do_ingest_conversation_empty_text_returns_error(self) -> None:
        from src.web_ui import _do_ingest_conversation
        with patch("src.web_ui._MEMORY_READY", True):
            result = _do_ingest_conversation("", "sess-1", "", "model", True)
        assert "error" in result.lower() or "Kein" in result


# ---------------------------------------------------------------------------
# Test H: E2E Analysis Flow — ingest → search → feedback roundtrip
# ---------------------------------------------------------------------------

class TestE2EAnalysisFlow:
    """End-to-end: ingest → search → feedback roundtrip."""

    def test_ingest_then_search_returns_results(self):
        """Ingest text, then search for it — full roundtrip."""
        import src.web_ui as web_ui
        fake_conn = MagicMock(spec=sqlite3.Connection)
        ingest_result = {"source_id": "repo:test:e2e.py", "chunk_ids": ["chk_e2e001"], "indexed": True, "deduped": 0, "superseded": 0}
        from src.memory_schema import RetrievalRecord
        search_result = RetrievalRecord(chunk_id="chk_e2e001", score_final=0.85, score_symbolic=0.6, source_id="repo:test:e2e.py", text="def authenticate(user): pass", category_labels=["auth"], reasons=["token_overlap", "embedding_similarity"])

        with (
            patch.object(web_ui, "_MEMORY_READY", True),
            patch("src.web_ui._get_conn", return_value=fake_conn),
            patch("src.web_ui._get_chroma", return_value=None),
            patch("src.web_ui.ingest", return_value=ingest_result),
            patch("src.web_ui.Source") as MockSource,
            patch("src.web_ui.hashlib") as mock_hashlib,
            patch("src.web_ui.search", return_value=[search_result]),
        ):
            mock_hash = MagicMock()
            mock_hash.hexdigest.return_value = "b" * 64
            mock_hashlib.sha256.return_value = mock_hash
            MockSource.return_value = MagicMock()
            MockSource.make_id.return_value = "repo:test:e2e.py"

            raw = web_ui._do_ingest(text_input="def authenticate(user): pass", file_upload=None, source_path="e2e.py", repo="test", categorize=False, mode="hybrid", codebook="auto", model="", ollama_available=False)
            result = json.loads(raw)
            assert result["source_id"] == "repo:test:e2e.py"

            status, rows = web_ui._do_search("authenticate", 5, False)
            assert len(rows) == 1
            assert "auth" in str(rows[0])

    def test_feedback_after_search(self):
        import src.web_ui as web_ui
        fake_conn = MagicMock(spec=sqlite3.Connection)
        with (
            patch.object(web_ui, "_MEMORY_READY", True),
            patch("src.web_ui._get_conn", return_value=fake_conn),
            patch("src.web_ui.add_feedback") as mock_fb,
        ):
            result = web_ui._do_feedback("chk_e2e001", "positive", "relevant")
        mock_fb.assert_called_once_with(fake_conn, "chk_e2e001", "positive", {"label": "relevant"})
        assert "gespeichert" in result.lower()


# ---------------------------------------------------------------------------
# Test I: E2E Conversation Flow — conversation summary ingestion + search
# ---------------------------------------------------------------------------

class TestE2EConversationFlow:
    """End-to-end: conversation summary ingestion + search."""

    def test_conversation_ingest_then_search(self):
        import src.web_ui as web_ui
        fake_conn = MagicMock(spec=sqlite3.Connection)
        conv_result = {"source_id": "conv:sess-e2e", "chunk_ids": ["chk_conv_001"], "indexed": True, "deduped": 0, "superseded": 0}
        from src.memory_schema import RetrievalRecord
        search_hit = RetrievalRecord(chunk_id="chk_conv_001", score_final=0.72, source_id="conv:sess-e2e", text="Wir haben HTTP-Transport implementiert.", category_labels=["Zusammenfassung"], reasons=["token_overlap"])

        with (
            patch.object(web_ui, "_MEMORY_READY", True),
            patch("src.web_ui._get_conn", return_value=fake_conn),
            patch("src.web_ui._get_chroma", return_value=None),
            patch("src.web_ui.ingest_conversation_summary", return_value=conv_result),
            patch("src.web_ui.search", return_value=[search_hit]),
        ):
            raw = web_ui._do_ingest_conversation(summary_text="## Summary\nWir haben HTTP-Transport implementiert.", session_id="sess-e2e", run_id="run-001", model="", ollama_available=False)
            assert "conv:sess-e2e" in raw

            status, rows = web_ui._do_search("HTTP Transport", 5, False)
            assert len(rows) == 1


# ---------------------------------------------------------------------------
# Test J: E2E Error Cases — error handling
# ---------------------------------------------------------------------------

class TestE2EErrorCases:
    """E2E error handling."""

    def test_ingest_with_no_content_and_no_file(self):
        import src.web_ui as web_ui
        with patch.object(web_ui, "_MEMORY_READY", True):
            raw = web_ui._do_ingest("", None, "", "", False, "hybrid", "auto", "", False)
        result = json.loads(raw)
        assert "error" in result

    def test_search_with_memory_not_loaded(self):
        import src.web_ui as web_ui
        with patch.object(web_ui, "_MEMORY_READY", False):
            status, rows = web_ui._do_search("test", 5, True)
        assert rows == []
        assert "memory" in status.lower() or "nicht" in status.lower()

    def test_feedback_on_empty_chunk_id(self):
        import src.web_ui as web_ui
        with patch.object(web_ui, "_MEMORY_READY", True):
            result = web_ui._do_feedback("", "positive", "")
        assert "chunk" in result.lower() or "id" in result.lower()
