"""Two quality-refinements for mayring_categorize (follow-up from first prod smoke):

1. `[neu]X` normalisiert sich zu `X`, wenn X bereits im Anker-Set ist
   (Modell-Bug: ~8% der Config-Calls kamen als `[neu]config` zurück
   obwohl `config` ein Anker war).

2. Test-Dateien bekommen automatisch das `tests`-Label, wenn der Pfad
   wie ein Test-File aussieht (pytest, jest, rspec). Das Modell klassi-
   fiziert nach Inhalt — `test_user_service.py` wird dann nur
   `domain, validation` gelabelt statt `tests, domain, validation`.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.memory.ingestion.categorization import (
    _looks_like_test_path,
    mayring_categorize,
)
from src.memory.schema import Chunk


class TestPathDetection:
    @pytest.mark.parametrize("path", [
        "tests/test_user.py",
        "src/app/tests/test_service.py",
        "tests/conftest.py",
        "tests/__init__.py",  # not a test file itself but in tests/
        "spec/user_spec.rb",
        "user_test.py",
        "user_test.js",
        "src/user.test.ts",
        "src/user.spec.ts",
        "__tests__/index.test.js",
    ])
    def test_recognises_test_paths(self, path):
        assert _looks_like_test_path(path), path

    @pytest.mark.parametrize("path", [
        "src/api/user.py",
        "src/user_service.py",
        "src/attestation.py",  # contains "test" but not as test marker
        "docs/latest.md",
        "",
    ])
    def test_rejects_non_test_paths(self, path):
        assert not _looks_like_test_path(path), path


def _make_chunk(source_id: str, text: str = "def x(): pass") -> Chunk:
    return Chunk(
        chunk_id=f"c:{source_id}",
        source_id=source_id,
        text=text,
        text_hash=Chunk.compute_text_hash(text),
        chunk_level="function",
        ordinal=0,
        dedup_key="k",
        category_labels=[],
        created_at="2026-04-22T00:00:00Z",
    )


class TestNeuPrefixNormalization:
    def test_neu_prefix_on_anchor_is_stripped(self):
        chunk = _make_chunk("src/app/config.py")

        def fake_generate(**kwargs):
            return "Kategorien: [neu]config, [neu]api, domain"

        with patch("src.analysis.analyzer._ollama_generate", side_effect=fake_generate):
            out = mayring_categorize(
                [chunk], "http://x", "model-x",
                mode="hybrid", codebook="code", source_type="repo_file",
            )
        labels = out[0].category_labels
        assert "config" in labels
        assert "api" in labels
        assert "[neu]config" not in labels
        assert "[neu]api" not in labels

    def test_genuinely_new_neu_label_is_kept(self):
        chunk = _make_chunk("src/app/weird.py")

        def fake_generate(**kwargs):
            return "Kategorien: utils, [neu]quantum-cryptography"

        with patch("src.analysis.analyzer._ollama_generate", side_effect=fake_generate):
            out = mayring_categorize(
                [chunk], "http://x", "model-x",
                mode="hybrid", codebook="code", source_type="repo_file",
            )
        labels = out[0].category_labels
        assert "[neu]quantum-cryptography" in labels
        assert "utils" in labels


class TestPathOverride:
    def test_test_file_gets_tests_label_even_if_llm_forgot(self):
        chunk = _make_chunk("tests/test_data_access.py")

        def fake_generate(**kwargs):
            return "Kategorien: data_access, validation"

        with patch("src.analysis.analyzer._ollama_generate", side_effect=fake_generate):
            out = mayring_categorize(
                [chunk], "http://x", "model-x",
                mode="hybrid", codebook="code", source_type="repo_file",
            )
        labels = out[0].category_labels
        assert "tests" in labels
        assert "data_access" in labels

    def test_non_test_file_not_tagged_tests(self):
        chunk = _make_chunk("src/app/user_service.py")

        def fake_generate(**kwargs):
            return "Kategorien: domain, data_access"

        with patch("src.analysis.analyzer._ollama_generate", side_effect=fake_generate):
            out = mayring_categorize(
                [chunk], "http://x", "model-x",
                mode="hybrid", codebook="code", source_type="repo_file",
            )
        labels = out[0].category_labels
        assert "tests" not in labels

    def test_test_file_already_tagged_not_duplicated(self):
        chunk = _make_chunk("tests/test_foo.py")

        def fake_generate(**kwargs):
            return "Kategorien: tests, utils"

        with patch("src.analysis.analyzer._ollama_generate", side_effect=fake_generate):
            out = mayring_categorize(
                [chunk], "http://x", "model-x",
                mode="hybrid", codebook="code", source_type="repo_file",
            )
        labels = out[0].category_labels
        assert labels.count("tests") == 1
