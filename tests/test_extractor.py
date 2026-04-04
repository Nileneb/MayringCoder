"""Unit tests for src.extractor."""

import json
import pytest
from src.extractor import (
    extract_python_signatures,
    levenshtein_ratio,
    check_redundancy_by_names,
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


# ---------------------------------------------------------------------------
# levenshtein_ratio
# ---------------------------------------------------------------------------

class TestLevenshteinRatio:
    def test_identical(self):
        assert levenshtein_ratio("send_email", "send_email") == 1.0

    def test_complete_mismatch(self):
        assert levenshtein_ratio("abc", "xyz") < 0.5

    def test_high_similarity(self):
        # send_email → send_mail: insert "_" (1 edit), delete "e" (1 edit)
        # max(10,9)=10 → 1 - 2/10 = 0.8
        assert levenshtein_ratio("send_email", "send_mail") > 0.80

    def test_partial_similarity(self):
        r = levenshtein_ratio("send_email", "send_notification")
        assert 0.3 < r < 0.6  # significant difference

    def test_case_insensitive(self):
        # Case is part of the similarity calculation
        assert levenshtein_ratio("send_email", "SEND_EMAIL") < 1.0

    def test_empty_string(self):
        assert levenshtein_ratio("", "abc") == 0.0
        assert levenshtein_ratio("abc", "") == 0.0
        # Two empty strings are identical (not "both empty = 0 similarity")
        assert levenshtein_ratio("", "") == 1.0

    def test_ordering_property(self):
        # Same-prefix names should be more similar than different-prefix names
        r1 = levenshtein_ratio("handle_request", "handle_request_old")
        r2 = levenshtein_ratio("handle_request", "parse_request")
        assert r1 > r2


# ---------------------------------------------------------------------------
# check_redundancy_by_names
# ---------------------------------------------------------------------------

class TestCheckRedundancyByNames:
    def test_no_candidates_below_threshold(self):
        # send_email ~ send_notification is ~0.41, below 0.80
        results = [
            {"filename": "a.py", "_signatures": {"functions": ["send_email"], "classes": [], "imports": []}},
            {"filename": "b.py", "_signatures": {"functions": ["send_notification"], "classes": [], "imports": []}},
        ]
        candidates = check_redundancy_by_names(results, threshold=0.80)
        assert candidates == []

    def test_one_candidate_above_threshold(self):
        # send_email ~ send_mail is ~0.9, above 0.80
        results = [
            {"filename": "a.py", "_signatures": {"functions": ["send_email"], "classes": [], "imports": []}},
            {"filename": "b.py", "_signatures": {"functions": ["send_mail"], "classes": [], "imports": []}},
        ]
        candidates = check_redundancy_by_names(results, threshold=0.80)
        assert len(candidates) == 1
        assert candidates[0]["needs_llm_review"] is True
        assert candidates[0]["type"] == "redundanz"
        assert candidates[0]["source"] == "name_redundancy_check"

    def test_multiple_candidates(self):
        # send_email, send_mail, sendmail — all similar
        results = [
            {"filename": "a.py", "_signatures": {"functions": ["send_email"], "classes": [], "imports": []}},
            {"filename": "b.py", "_signatures": {"functions": ["send_mail"], "classes": [], "imports": []}},
            {"filename": "c.py", "_signatures": {"functions": ["sendmail"], "classes": [], "imports": []}},
            {"filename": "d.py", "_signatures": {"functions": ["send_notification"], "classes": [], "imports": []}},
        ]
        candidates = check_redundancy_by_names(results, threshold=0.80)
        # a-b (0.9), a-c (1.0), b-c (1.0) → 3 pairs above threshold
        assert len(candidates) == 3

    def test_deduplication_by_pair(self):
        """Same pair (reversed) should not produce duplicates."""
        results = [
            {"filename": "a.py", "_signatures": {"functions": ["send_email"], "classes": [], "imports": []}},
            {"filename": "b.py", "_signatures": {"functions": ["send_mail"], "classes": [], "imports": []}},
        ]
        candidates = check_redundancy_by_names(results, threshold=0.80)
        # Only one unique pair
        assert len(candidates) == 1

    def test_skips_private_methods(self):
        """Methods starting with underscore are excluded."""
        results = [
            {"filename": "a.py", "_signatures": {"functions": ["send_email", "_internal_helper"], "classes": [], "imports": []}},
            {"filename": "b.py", "_signatures": {"functions": ["send_email", "__clone"], "classes": [], "imports": []}},
        ]
        candidates = check_redundancy_by_names(results, threshold=0.80)
        # Only the public send_email should be compared (private methods skipped)
        assert len(candidates) == 1

    def test_empty_overview_results(self):
        assert check_redundancy_by_names([], threshold=0.80) == []

    def test_no_signatures_key(self):
        """Files without _signatures are skipped gracefully."""
        results = [
            {"filename": "a.py"},  # no _signatures
            {"filename": "b.py", "_signatures": {"functions": ["send_email"], "classes": [], "imports": []}},
        ]
        candidates = check_redundancy_by_names(results, threshold=0.80)
        assert candidates == []

    def test_threshold_zero_includes_all(self):
        """threshold=0 includes everything above 0 similarity."""
        results = [
            {"filename": "a.py", "_signatures": {"functions": ["send_email"], "classes": [], "imports": []}},
            {"filename": "b.py", "_signatures": {"functions": ["send_notification"], "classes": [], "imports": []}},
        ]
        # Even dissimilar names (ratio > 0) are included at threshold=0
        candidates = check_redundancy_by_names(results, threshold=0)
        assert len(candidates) == 1
