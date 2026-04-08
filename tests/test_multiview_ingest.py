"""Tests for generate_multiview_chunks() and multiview integration in ingest()."""
from unittest.mock import MagicMock, patch
import json
import pytest

from src.memory_ingest import generate_multiview_chunks


SAMPLE_ISSUE = """# Login schlägt fehl nach Token-Erneuerung

Benutzer können sich nach einer Token-Erneuerung nicht mehr einloggen.
Der Fehler tritt auf Android 12 mit App-Version 2.3.1 auf.

## Entscheidung
Wir verwenden Ansatz B: Silent Token Refresh im Hintergrund.

## Betroffene Komponenten
AuthService, TokenManager, LoginActivity"""


class TestGenerateMultiviewChunks:

    def _mock_ollama_response(self, fact="Fakten.", decision="Entscheidung.", entities="auth, token"):
        return json.dumps({
            "fact_summary": fact,
            "decision_summary": decision,
            "entities_keywords": entities,
        })

    def test_empty_model_returns_only_view_full(self):
        chunks = generate_multiview_chunks("src::test", SAMPLE_ISSUE, "http://localhost:11434", "")
        assert len(chunks) == 1
        assert chunks[0].chunk_level == "view_full"

    def test_ollama_failure_returns_only_view_full(self):
        with patch("src.analyzer._ollama_generate", side_effect=Exception("connection refused")):
            chunks = generate_multiview_chunks(
                "src::test", SAMPLE_ISSUE, "http://localhost:11434", "mistral"
            )
        assert len(chunks) == 1
        assert chunks[0].chunk_level == "view_full"
        assert SAMPLE_ISSUE[:50] in chunks[0].text

    def test_successful_call_returns_4_chunks(self):
        with patch("src.analyzer._ollama_generate", return_value=self._mock_ollama_response()):
            chunks = generate_multiview_chunks(
                "src::test", SAMPLE_ISSUE, "http://localhost:11434", "mistral"
            )
        levels = {c.chunk_level for c in chunks}
        assert "view_fact" in levels
        assert "view_decision" in levels
        assert "view_entities" in levels
        assert "view_full" in levels
        assert len(chunks) == 4

    def test_chunk_ids_are_unique(self):
        with patch("src.analyzer._ollama_generate", return_value=self._mock_ollama_response()):
            chunks = generate_multiview_chunks(
                "src::test", SAMPLE_ISSUE, "http://localhost:11434", "mistral"
            )
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_view_full_contains_original_content(self):
        with patch("src.analyzer._ollama_generate", return_value=self._mock_ollama_response()):
            chunks = generate_multiview_chunks(
                "src::test", SAMPLE_ISSUE, "http://localhost:11434", "mistral"
            )
        full = next(c for c in chunks if c.chunk_level == "view_full")
        assert full.text == SAMPLE_ISSUE

    def test_empty_decision_field_skips_that_view(self):
        """If decision_summary is empty, view_decision chunk is omitted."""
        response = json.dumps({
            "fact_summary": "Fakten.",
            "decision_summary": "",
            "entities_keywords": "auth",
        })
        with patch("src.analyzer._ollama_generate", return_value=response):
            chunks = generate_multiview_chunks(
                "src::test", SAMPLE_ISSUE, "http://localhost:11434", "mistral"
            )
        levels = [c.chunk_level for c in chunks]
        assert "view_decision" not in levels

    def test_source_id_is_set_on_all_chunks(self):
        with patch("src.analyzer._ollama_generate", return_value=self._mock_ollama_response()):
            chunks = generate_multiview_chunks(
                "src::test123", SAMPLE_ISSUE, "http://localhost:11434", "mistral"
            )
        assert all(c.source_id == "src::test123" for c in chunks)

    def test_markdown_fenced_json_is_handled(self):
        """LLM sometimes wraps JSON in ```json ... ``` — must be parsed correctly."""
        fenced = "```json\n" + self._mock_ollama_response() + "\n```"
        with patch("src.analyzer._ollama_generate", return_value=fenced):
            chunks = generate_multiview_chunks(
                "src::test", SAMPLE_ISSUE, "http://localhost:11434", "mistral"
            )
        assert len(chunks) == 4


class TestIngestMultiviewIntegration:
    """Integration test: ingest() with opts={'multiview': True}."""

    def test_ingest_multiview_true_calls_generate_multiview(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from src.memory_ingest import ingest
        from src.memory_store import init_memory_db, upsert_source
        from src.memory_schema import Source

        conn = init_memory_db(tmp_path / "memory.db")
        source = Source(
            source_id="gh::test::issue/1",
            source_type="github_issue",
            repo="test/repo",
            path="issue/1",
            content_hash="sha256:abc",
        )

        mock_chunks = []  # generate_multiview_chunks returns empty → ingest returns 0 chunks

        with (
            patch("src.memory_ingest.generate_multiview_chunks", return_value=mock_chunks) as mock_gen,
            patch("src.memory_ingest.structural_chunk") as mock_struct,
        ):
            ingest(source, SAMPLE_ISSUE, conn, None, "http://localhost:11434", "mistral",
                   opts={"multiview": True})

        mock_gen.assert_called_once()
        mock_struct.assert_not_called()

    def test_ingest_multiview_false_uses_structural_chunk(self, tmp_path):
        from unittest.mock import patch
        from src.memory_ingest import ingest
        from src.memory_store import init_memory_db
        from src.memory_schema import Source

        conn = init_memory_db(tmp_path / "memory.db")
        source = Source(
            source_id="gh::test::issue/2",
            source_type="github_issue",
            repo="test/repo",
            path="issue/2",
            content_hash="sha256:def",
        )

        with (
            patch("src.memory_ingest.generate_multiview_chunks") as mock_gen,
            patch("src.memory_ingest.structural_chunk", return_value=[]) as mock_struct,
        ):
            ingest(source, SAMPLE_ISSUE, conn, None, "http://localhost:11434", "mistral",
                   opts={"multiview": False})

        mock_gen.assert_not_called()
        mock_struct.assert_called_once()
