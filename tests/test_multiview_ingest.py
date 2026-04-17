"""Tests for generate_multiview_chunks() and multiview integration in ingest()."""
from unittest.mock import MagicMock, patch
import json
import pytest

from src.memory.ingest import generate_multiview_chunks


SAMPLE_ISSUE = """# Login schlägt fehl nach Token-Erneuerung

Benutzer können sich nach einer Token-Erneuerung nicht mehr einloggen.
Der Fehler tritt auf Android 12 mit App-Version 2.3.1 auf.

## Entscheidung
Wir verwenden Ansatz B: Silent Token Refresh im Hintergrund.

## Betroffene Komponenten
AuthService, TokenManager, LoginActivity"""


class TestGenerateMultiviewChunks:

    def _mock_ollama_response(self, fact="Fakten.", impl="Module A, Service B.", decision="Entscheidung.", entities="auth, token"):
        return json.dumps({
            "fact_summary": fact,
            "impl_summary": impl,
            "decision_summary": decision,
            "entities_keywords": entities,
        })

    def test_empty_model_returns_only_view_full(self):
        chunks = generate_multiview_chunks("src::test", SAMPLE_ISSUE, "http://localhost:11434", "")
        assert len(chunks) == 1
        assert chunks[0].chunk_level == "view_full"

    def test_ollama_failure_returns_only_view_full(self):
        with patch("src.analysis.analyzer._ollama_generate", side_effect=Exception("connection refused")):
            chunks = generate_multiview_chunks(
                "src::test", SAMPLE_ISSUE, "http://localhost:11434", "mistral"
            )
        assert len(chunks) == 1
        assert chunks[0].chunk_level == "view_full"
        assert SAMPLE_ISSUE[:50] in chunks[0].text

    def test_successful_call_returns_5_chunks(self):
        with patch("src.analysis.analyzer._ollama_generate", return_value=self._mock_ollama_response()):
            chunks = generate_multiview_chunks(
                "src::test", SAMPLE_ISSUE, "http://localhost:11434", "mistral"
            )
        levels = {c.chunk_level for c in chunks}
        assert "view_fact" in levels
        assert "view_impl" in levels
        assert "view_decision" in levels
        assert "view_entities" in levels
        assert "view_full" in levels
        assert len(chunks) == 5

    def test_chunk_ids_are_unique(self):
        with patch("src.analysis.analyzer._ollama_generate", return_value=self._mock_ollama_response()):
            chunks = generate_multiview_chunks(
                "src::test", SAMPLE_ISSUE, "http://localhost:11434", "mistral"
            )
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_view_full_contains_original_content(self):
        with patch("src.analysis.analyzer._ollama_generate", return_value=self._mock_ollama_response()):
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
        with patch("src.analysis.analyzer._ollama_generate", return_value=response):
            chunks = generate_multiview_chunks(
                "src::test", SAMPLE_ISSUE, "http://localhost:11434", "mistral"
            )
        levels = [c.chunk_level for c in chunks]
        assert "view_decision" not in levels

    def test_source_id_is_set_on_all_chunks(self):
        with patch("src.analysis.analyzer._ollama_generate", return_value=self._mock_ollama_response()):
            chunks = generate_multiview_chunks(
                "src::test123", SAMPLE_ISSUE, "http://localhost:11434", "mistral"
            )
        assert all(c.source_id == "src::test123" for c in chunks)

    def test_markdown_fenced_json_is_handled(self):
        """LLM sometimes wraps JSON in ```json ... ``` — must be parsed correctly."""
        fenced = "```json\n" + self._mock_ollama_response() + "\n```"
        with patch("src.analysis.analyzer._ollama_generate", return_value=fenced):
            chunks = generate_multiview_chunks(
                "src::test", SAMPLE_ISSUE, "http://localhost:11434", "mistral"
            )
        assert len(chunks) == 5

    def test_llm_returns_list_instead_of_dict(self):
        """If LLM returns a JSON array instead of object, fall back to view_full only."""
        with patch("src.analysis.analyzer._ollama_generate", return_value='["fact", "decision"]'):
            chunks = generate_multiview_chunks(
                "src::test", SAMPLE_ISSUE, "http://localhost:11434", "mistral"
            )
        assert len(chunks) == 1
        assert chunks[0].chunk_level == "view_full"

    def test_entities_keywords_as_list_is_joined(self):
        """LLM returns entities_keywords as a list — must be joined to string."""
        response = json.dumps({
            "fact_summary": "Bug gefunden.",
            "decision_summary": "Ansatz B gewählt.",
            "entities_keywords": ["AuthService", "TokenManager", "LoginActivity"],
        })
        with patch("src.analysis.analyzer._ollama_generate", return_value=response):
            chunks = generate_multiview_chunks(
                "src::test", SAMPLE_ISSUE, "http://localhost:11434", "mistral"
            )
        entities_chunk = next((c for c in chunks if c.chunk_level == "view_entities"), None)
        assert entities_chunk is not None
        assert "AuthService" in entities_chunk.text
        assert "TokenManager" in entities_chunk.text

    def test_entities_keywords_as_empty_list_skips_view(self):
        """Empty list for entities_keywords should skip that view."""
        response = json.dumps({
            "fact_summary": "Bug gefunden.",
            "decision_summary": "Ansatz B gewählt.",
            "entities_keywords": [],
        })
        with patch("src.analysis.analyzer._ollama_generate", return_value=response):
            chunks = generate_multiview_chunks(
                "src::test", SAMPLE_ISSUE, "http://localhost:11434", "mistral"
            )
        levels = [c.chunk_level for c in chunks]
        assert "view_entities" not in levels

    def test_field_value_as_integer_is_coerced(self):
        """Integer field values must not raise AttributeError."""
        response = json.dumps({
            "fact_summary": 42,
            "decision_summary": "OK.",
            "entities_keywords": "auth",
        })
        with patch("src.analysis.analyzer._ollama_generate", return_value=response):
            chunks = generate_multiview_chunks(
                "src::test", SAMPLE_ISSUE, "http://localhost:11434", "mistral"
            )
        fact_chunk = next((c for c in chunks if c.chunk_level == "view_fact"), None)
        assert fact_chunk is not None
        assert fact_chunk.text == "42"


class TestIngestMultiviewIntegration:
    """Integration test: ingest() with opts={'multiview': True}."""

    def test_ingest_multiview_true_calls_generate_multiview(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from src.memory.ingest import ingest
        from src.memory.store import init_memory_db, upsert_source
        from src.memory.schema import Source

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
            patch("src.memory.ingest.generate_multiview_chunks", return_value=mock_chunks) as mock_gen,
            patch("src.memory.ingest.structural_chunk") as mock_struct,
        ):
            ingest(source, SAMPLE_ISSUE, conn, None, "http://localhost:11434", "mistral",
                   opts={"multiview": True})

        mock_gen.assert_called_once()
        mock_struct.assert_not_called()

    def test_ingest_multiview_false_uses_structural_chunk(self, tmp_path):
        from unittest.mock import patch
        from src.memory.ingest import ingest
        from src.memory.store import init_memory_db
        from src.memory.schema import Source

        conn = init_memory_db(tmp_path / "memory.db")
        source = Source(
            source_id="gh::test::issue/2",
            source_type="github_issue",
            repo="test/repo",
            path="issue/2",
            content_hash="sha256:def",
        )

        with (
            patch("src.memory.ingest.generate_multiview_chunks") as mock_gen,
            patch("src.memory.ingest.structural_chunk", return_value=[]) as mock_struct,
        ):
            ingest(source, SAMPLE_ISSUE, conn, None, "http://localhost:11434", "mistral",
                   opts={"multiview": False})

        mock_gen.assert_not_called()
        mock_struct.assert_called_once()
