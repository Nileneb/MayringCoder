"""Unit tests for src.extractor."""

import pytest
from src.extractor import (
    extract_python_signatures,
    levenshtein_ratio,
    check_redundancy_by_names,
)


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
