"""Unit tests for finding-reactive RAG enrichment (Issue #18).

Tests for:
- _build_rag_query(): finding type → semantic query mapping
- enrich_findings_with_rag(): enrichment flow with mocked ChromaDB
- Second opinion / adversarial with RAG context injection
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.analysis.context import _build_rag_query, enrich_findings_with_rag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finding(ftype: str = "redundanz", evidence: str = "def save_user()", **kw) -> dict:
    base = {
        "type": ftype,
        "severity": "warning",
        "confidence": "medium",
        "line_hint": "~10",
        "evidence_excerpt": evidence,
        "fix_suggestion": "Refactor.",
    }
    base.update(kw)
    return base


def _result(fn: str = "a.py", findings: list[dict] | None = None) -> dict:
    return {
        "filename": fn,
        "category": "domain",
        "potential_smells": findings if findings is not None else [_finding()],
    }


# ---------------------------------------------------------------------------
# TestBuildRagQuery
# ---------------------------------------------------------------------------

class TestBuildRagQuery:

    def test_redundanz_query_contains_fn_name(self):
        f = _finding("redundanz", evidence="def calculate_total(): pass")
        q = _build_rag_query(f, "billing.py", "domain")
        assert "calculate_total" in q
        assert "Implementierung" in q

    def test_sicherheit_query_contains_category(self):
        f = _finding("sicherheit", evidence="validate(input)")
        q = _build_rag_query(f, "auth.py", "api")
        assert "api" in q
        assert "Auth" in q or "Validation" in q

    def test_zombie_code_query_contains_fn_name(self):
        f = _finding("zombie_code", evidence="def unused_helper(): pass")
        q = _build_rag_query(f, "utils.py", "utility")
        assert "unused_helper" in q
        assert "Referenz" in q

    def test_inkonsistenz_query_contains_category(self):
        f = _finding("inkonsistenz", evidence="mixed error handling")
        q = _build_rag_query(f, "service.py", "data_access")
        assert "data_access" in q
        assert "Fehlerbehandlung" in q

    def test_fehlerbehandlung_query_contains_pattern(self):
        f = _finding("fehlerbehandlung", evidence="no try except")
        q = _build_rag_query(f, "handler.py", "api")
        assert "Exception" in q or "Try-Catch" in q

    def test_overengineering_query_contains_category(self):
        f = _finding("overengineering", evidence="AbstractFactoryFactory")
        q = _build_rag_query(f, "factory.py", "domain")
        assert "domain" in q
        assert "Abstraktion" in q or "Vereinfachung" in q

    def test_unknown_type_fallback_uses_category_and_filename(self):
        f = _finding("freitext", evidence="some text here")
        q = _build_rag_query(f, "misc.py", "utility")
        assert "utility" in q
        assert "misc.py" in q

    def test_no_evidence_uses_filename_stem(self):
        f = _finding("redundanz", evidence="")
        q = _build_rag_query(f, "src/services/UserService.php", "domain")
        assert "UserService" in q

    def test_all_known_types_return_nonempty_string(self):
        known_types = [
            "redundanz", "sicherheit", "zombie_code",
            "inkonsistenz", "fehlerbehandlung", "overengineering", "unklar",
        ]
        for t in known_types:
            q = _build_rag_query(_finding(t), "file.py", "domain")
            assert len(q.strip()) > 0, f"Empty query for type={t}"


# ---------------------------------------------------------------------------
# TestEnrichFindingsWithRag
# ---------------------------------------------------------------------------

class TestEnrichFindingsWithRag:

    @patch("src.analysis.context._HAS_CHROMADB", True)
    @patch("src.analysis.context._embed_texts")
    @patch("src.analysis.context.chromadb")
    def test_stores_rag_context_and_query(self, mock_chromadb, mock_embed):
        # Setup mocks
        mock_embed.return_value = [[0.1] * 384]

        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "documents": [["[domain] user.py: User management"]],
        }
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        results = [_result("a.py", [_finding("redundanz")])]

        with patch("src.analysis.context._chroma_dir") as mock_dir:
            mock_dir.return_value = MagicMock(exists=MagicMock(return_value=True))
            enriched = enrich_findings_with_rag(results, "https://github.com/t/r", "http://localhost:11434")

        finding = enriched[0]["potential_smells"][0]
        assert "_rag_query" in finding
        assert "_rag_context" in finding
        assert "user.py" in finding["_rag_context"]

    @patch("src.analysis.context._HAS_CHROMADB", True)
    @patch("src.analysis.context._embed_texts")
    @patch("src.analysis.context.chromadb")
    def test_empty_results_noop(self, mock_chromadb, mock_embed):
        results = [{"filename": "a.py", "category": "domain", "potential_smells": []}]

        with patch("src.analysis.context._chroma_dir") as mock_dir:
            mock_dir.return_value = MagicMock(exists=MagicMock(return_value=True))
            mock_client = MagicMock()
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_collection = MagicMock()
            mock_collection.count.return_value = 5
            mock_client.get_collection.return_value = mock_collection

            enriched = enrich_findings_with_rag(results, "https://github.com/t/r", "http://localhost:11434")

        mock_embed.assert_not_called()
        assert enriched[0]["potential_smells"] == []

    @patch("src.analysis.context._HAS_CHROMADB", False)
    def test_no_chromadb_returns_unchanged(self):
        results = [_result()]
        enriched = enrich_findings_with_rag(results, "https://github.com/t/r", "http://localhost:11434")
        finding = enriched[0]["potential_smells"][0]
        assert "_rag_context" not in finding

    @patch("src.analysis.context._HAS_CHROMADB", True)
    @patch("src.analysis.context._embed_texts")
    @patch("src.analysis.context.chromadb")
    def test_multiple_findings_batch_embedded(self, mock_chromadb, mock_embed):
        mock_embed.return_value = [[0.1] * 384, [0.2] * 384, [0.3] * 384]

        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "documents": [["[api] controller.py: request handler"]],
        }
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        findings = [
            _finding("redundanz"),
            _finding("sicherheit"),
            _finding("zombie_code"),
        ]
        results = [_result("a.py", findings)]

        with patch("src.analysis.context._chroma_dir") as mock_dir:
            mock_dir.return_value = MagicMock(exists=MagicMock(return_value=True))
            enrich_findings_with_rag(results, "https://github.com/t/r", "http://localhost:11434")

        # _embed_texts should be called once with all 3 queries
        mock_embed.assert_called_once()
        queries_arg = mock_embed.call_args[0][0]
        assert len(queries_arg) == 3

    @patch("src.analysis.context._HAS_CHROMADB", True)
    @patch("src.analysis.context._embed_texts")
    @patch("src.analysis.context.chromadb")
    def test_error_in_results_skipped(self, mock_chromadb, mock_embed):
        """Results with 'error' key should be skipped entirely."""
        mock_embed.return_value = []

        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        results = [{"filename": "a.py", "error": "Timeout"}]

        with patch("src.analysis.context._chroma_dir") as mock_dir:
            mock_dir.return_value = MagicMock(exists=MagicMock(return_value=True))
            enriched = enrich_findings_with_rag(results, "https://github.com/t/r", "http://localhost:11434")

        mock_embed.assert_not_called()
        assert enriched == results


# ---------------------------------------------------------------------------
# TestSecondOpinionWithRagContext
# ---------------------------------------------------------------------------

class TestSecondOpinionWithRagContext:

    def test_rag_context_injected_into_prompt(self):
        """When finding has _rag_context, it should appear in the second opinion prompt."""
        from src.analysis.extractor import second_opinion_validate

        finding = {
            "_filename": "a.py",
            "type": "redundanz",
            "severity": "warning",
            "confidence": "medium",
            "line_hint": "~10",
            "evidence_excerpt": "duplicate logic",
            "fix_suggestion": "Remove.",
            "_rag_context": "## Projektkontext\n- [domain] user.py: User management",
        }
        files = [{"filename": "a.py", "content": "def foo(): pass"}]

        captured_prompt = []
        def _capture_generate(prompt, *args, **kwargs):
            captured_prompt.append(prompt)
            return json.dumps({"verdict": "BESTÄTIGT", "reasoning": "ok"})

        with patch("src.analysis.analyzer._ollama_generate", side_effect=_capture_generate):
            second_opinion_validate([finding], files, "http://localhost:11434", "model")

        assert len(captured_prompt) == 1
        assert "user.py" in captured_prompt[0]
        assert "Projektkontext" in captured_prompt[0]

    def test_no_rag_context_uses_default_text(self):
        """When finding has no _rag_context, prompt should say no context available."""
        from src.analysis.extractor import second_opinion_validate

        finding = {
            "_filename": "a.py",
            "type": "redundanz",
            "severity": "warning",
            "confidence": "medium",
            "line_hint": "",
            "evidence_excerpt": "issue here",
            "fix_suggestion": "Fix it.",
        }
        files = [{"filename": "a.py", "content": "def bar(): pass"}]

        captured_prompt = []
        def _capture_generate(prompt, *args, **kwargs):
            captured_prompt.append(prompt)
            return json.dumps({"verdict": "BESTÄTIGT", "reasoning": "ok"})

        with patch("src.analysis.analyzer._ollama_generate", side_effect=_capture_generate):
            second_opinion_validate([finding], files, "http://localhost:11434", "model")

        assert len(captured_prompt) == 1
        assert "kein zusätzlicher Projektkontext" in captured_prompt[0]


# ---------------------------------------------------------------------------
# TestAdversarialWithRagContext
# ---------------------------------------------------------------------------

class TestAdversarialWithRagContext:

    def test_rag_context_appended_to_adversarial_prompt(self):
        """When finding has _rag_context, it should be appended to the adversarial prompt."""
        from src.analysis.extractor import validate_findings

        finding = {
            "_filename": "a.py",
            "type": "redundanz",
            "severity": "warning",
            "confidence": "medium",
            "line_hint": "~10",
            "evidence_excerpt": "duplicate logic",
            "fix_suggestion": "Remove.",
            "_rag_context": "## Projektkontext\n- [api] service.py: API handlers",
        }
        files = [{"filename": "a.py", "content": "def baz(): pass"}]

        captured_prompt = []
        def _capture_generate(prompt, *args, **kwargs):
            captured_prompt.append(prompt)
            return "BESTÄTIGT: Correct finding."

        with patch("src.analysis.analyzer._ollama_generate", side_effect=_capture_generate):
            validate_findings([finding], files, "http://localhost:11434", "model")

        assert len(captured_prompt) == 1
        assert "PROJEKTKONTEXT" in captured_prompt[0]
        assert "service.py" in captured_prompt[0]
