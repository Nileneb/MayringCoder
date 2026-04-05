"""Unit tests for src.extractor."""

import json
import pytest
from src.extractor import (
    extract_python_signatures,
    parse_freetext_findings,
    parse_llm_extraction,
    EXTRACT_PROMPT,
)


# ---------------------------------------------------------------------------
# parse_freetext_findings  (pure regex — no network)
# ---------------------------------------------------------------------------

class TestParseFreetextFindings:
    def test_empty_string_returns_empty(self):
        assert parse_freetext_findings("", "f.py") == []

    def test_no_smell_keywords_returns_empty(self):
        assert parse_freetext_findings("Everything looks fine here.", "f.py") == []

    def test_detects_smell_keyword(self):
        raw = "- zombie_code: This function is never called anywhere in the codebase."
        result = parse_freetext_findings(raw, "f.py")
        assert len(result) == 1
        assert result[0]["source"] == "regex_extraction"

    def test_extracts_line_hint(self):
        raw = "- Redundanz in Zeile ~42: duplicate logic found."
        result = parse_freetext_findings(raw, "f.py")
        assert result[0]["line_hint"] == "~42"

    def test_no_line_hint_gives_empty_string(self):
        raw = "- security issue: password stored in plain text everywhere."
        result = parse_freetext_findings(raw, "f.py")
        assert result[0]["line_hint"] == ""

    def test_caps_at_10_findings(self):
        lines = [f"- zombie_code finding number {i} in the code." for i in range(20)]
        raw = "\n".join(lines)
        result = parse_freetext_findings(raw, "f.py")
        assert len(result) <= 10

    def test_short_chunks_skipped(self):
        # Chunks shorter than 30 chars are ignored even with keywords
        raw = "- bug"
        result = parse_freetext_findings(raw, "f.py")
        assert result == []

    def test_no_ollama_call_made(self, monkeypatch):
        """parse_freetext_findings must never call Ollama."""
        import src.extractor as mod
        monkeypatch.setattr(mod, "_regex_extract_findings",
                            lambda raw, fn: [{"type": "freitext", "line_hint": "",
                                              "evidence_excerpt": raw[:50],
                                              "fix_suggestion": "", "confidence": "low",
                                              "severity": "info", "source": "regex_extraction"}])
        # If an HTTP call were made, it would raise ConnectionRefusedError
        # (no Ollama running in CI). The test passing proves no network call occurs.
        result = parse_freetext_findings("some smell keyword here for testing", "f.py")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# parse_llm_extraction  (pure JSON parsing — no network)
# ---------------------------------------------------------------------------

class TestParseLlmExtraction:
    def _make_response(self, findings: list[dict]) -> str:
        return json.dumps({"findings": findings})

    def test_empty_string_returns_empty(self):
        assert parse_llm_extraction("", "f.py") == []

    def test_invalid_json_returns_empty(self):
        assert parse_llm_extraction("not json at all", "f.py") == []

    def test_valid_finding_parsed(self):
        raw = self._make_response([{
            "datei": "src/foo.py",
            "zeile": "~10",
            "typ": "redundanz",
            "begründung": "Duplicate logic found here.",
            "empfehlung": "Extract to shared helper.",
        }])
        result = parse_llm_extraction(raw, "f.py")
        assert len(result) == 1
        assert result[0]["type"] == "redundanz"
        assert result[0]["line_hint"] == "~10"
        assert result[0]["source"] == "freetext_extraction"

    def test_missing_mandatory_field_skipped(self):
        raw = self._make_response([{
            "datei": "src/foo.py",
            # missing "typ", "begründung", "empfehlung"
        }])
        result = parse_llm_extraction(raw, "f.py")
        assert result == []

    def test_evidence_excerpt_truncated_at_200(self):
        long_text = "X" * 500
        raw = self._make_response([{
            "datei": "f.py", "zeile": "", "typ": "freitext",
            "begründung": long_text, "empfehlung": "fix it",
        }])
        result = parse_llm_extraction(raw, "f.py")
        assert len(result[0]["evidence_excerpt"]) <= 200

    def test_json_inside_markdown_fences(self):
        inner = json.dumps({"findings": [{
            "datei": "a.py", "zeile": "~5", "typ": "sicherheit",
            "begründung": "SQL injection risk.", "empfehlung": "Use parameterized queries.",
        }]})
        raw = f"```json\n{inner}\n```"
        result = parse_llm_extraction(raw, "a.py")
        assert len(result) == 1

    def test_multiple_findings_all_parsed(self):
        raw = self._make_response([
            {"datei": "a.py", "zeile": "", "typ": "redundanz",
             "begründung": "Dup logic.", "empfehlung": "Merge."},
            {"datei": "b.py", "zeile": "~3", "typ": "sicherheit",
             "begründung": "Auth bypass.", "empfehlung": "Add guard."},
        ])
        result = parse_llm_extraction(raw, "f.py")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# EXTRACT_PROMPT is publicly accessible (no import error)
# ---------------------------------------------------------------------------

class TestExtractPromptExported:
    def test_extract_prompt_is_string(self):
        assert isinstance(EXTRACT_PROMPT, str)
        assert len(EXTRACT_PROMPT) > 50


# ---------------------------------------------------------------------------
# extract_python_signatures
# ---------------------------------------------------------------------------

class TestExtractPythonSignatures:
    def test_functions_extracted(self):
        code = """
def foo():
    pass
async def bar(x):
    pass
class Baz:
    pass
"""
        sig = extract_python_signatures(code)
        assert "foo" in sig["functions"]
        assert "bar" in sig["functions"]

    def test_classes_extracted(self):
        code = """
class UserController:
    pass
class Service:
    pass
"""
        sig = extract_python_signatures(code)
        assert "UserController" in sig["classes"]
        assert "Service" in sig["classes"]

    def test_imports_from(self):
        code = "from django.http import JsonResponse\nfrom rest_framework import views"
        sig = extract_python_signatures(code)
        assert "django.http" in sig["imports"]
        assert "rest_framework" in sig["imports"]

    def test_imports_simple(self):
        code = "import logging\nimport os, sys"
        sig = extract_python_signatures(code)
        assert "logging" in sig["imports"]
        assert "os" in sig["imports"]
        assert "sys" in sig["imports"]

    def test_imports_with_alias(self):
        code = "from os import path as p\nimport collections as col"
        sig = extract_python_signatures(code)
        assert "os" in sig["imports"]
        assert "collections" in sig["imports"]

    def test_empty_code(self):
        sig = extract_python_signatures("")
        assert sig["functions"] == []
        assert sig["classes"] == []
        assert sig["imports"] == []

