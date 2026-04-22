"""Extractor for the new Mayring hybrid-prompt output format.

The revised prompt asks the model to reason in three steps (paraphrase →
generalise → reduce) and emit only ``Kategorien: a, b, c``. This extractor
must (1) find that line and return only its payload, (2) fall back to the
raw response for prompts that still return a bare comma list.
"""
from __future__ import annotations

from src.memory.ingestion.categorization import _extract_label_line


class TestKategorienPrefix:
    def test_plain_line(self):
        assert _extract_label_line("Kategorien: api, data_access, auth") == "api, data_access, auth"

    def test_case_insensitive(self):
        assert _extract_label_line("KATEGORIEN: api, tests") == "api, tests"

    def test_english_alias(self):
        assert _extract_label_line("Categories: logging, error_handling") == "logging, error_handling"

    def test_buried_after_reasoning(self):
        resp = """Dieser Code holt einen User aus der DB.
Funktion: Datenzugriff in der Persistenz-Schicht.
Kategorien: data_access, api, [neu]persistence"""
        assert _extract_label_line(resp) == "data_access, api, [neu]persistence"

    def test_dash_separator(self):
        assert _extract_label_line("Kategorien - api, auth") == "api, auth"

    def test_leading_whitespace(self):
        assert _extract_label_line("   Kategorien: tests") == "tests"


class TestFallback:
    def test_bare_comma_list_falls_through(self):
        assert _extract_label_line("api, data_access") == "api, data_access"

    def test_empty_string(self):
        assert _extract_label_line("") == ""

    def test_none(self):
        assert _extract_label_line(None) == ""  # type: ignore[arg-type]

    def test_multiline_no_prefix(self):
        # Old deductive prompt may return bare lines — keep as-is for the caller's split logic
        assert _extract_label_line("api\ntests\nauth") == "api\ntests\nauth"
