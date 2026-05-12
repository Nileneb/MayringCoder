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


class TestSourceTypeCodebookRouting:
    """WHY(2026-05-12): codebook is ALWAYS "auto" in the ingest path; the
    source_type → codebook mapping lives in _SOURCE_TYPE_TO_CODEBOOK. Lock
    the routing logic + the FAIL-loud behaviour for unmapped types."""

    def test_agent_result_uses_social_anchors_not_code(self):
        from src.memory.ingestion.categorization import _resolve_codebook

        cats = _resolve_codebook("auto", "agent_result")
        assert "argumentation" in cats or "methodik" in cats
        assert "auth" not in cats and "api" not in cats and "data_access" not in cats

    def test_paper_uses_social_anchors(self):
        from src.memory.ingestion.categorization import _resolve_codebook

        cats = _resolve_codebook("auto", "paper")
        assert "auth" not in cats and "api" not in cats

    def test_github_issue_uses_code_anchors_not_social(self):
        """GitHub issues are technical bug reports — code anchors, not prose."""
        from src.memory.ingestion.categorization import _resolve_codebook

        cats = _resolve_codebook("auto", "github_issue")
        assert "api" in cats and "data_access" in cats
        assert "argumentation" not in cats

    def test_repo_file_still_uses_code_anchors(self):
        from src.memory.ingestion.categorization import _resolve_codebook

        cats = _resolve_codebook("auto", "repo_file")
        assert "api" in cats and "data_access" in cats

    def test_unmapped_source_type_fails_loudly(self):
        """No silent fallback — unmapped source_type → a FAIL_ marker label."""
        from src.memory.ingestion.categorization import _resolve_codebook

        cats = _resolve_codebook("auto", "some_future_type")
        assert cats == ["FAIL_unmapped_source_type:some_future_type"]

    def test_empty_source_type_fails_loudly(self):
        from src.memory.ingestion.categorization import _resolve_codebook

        cats = _resolve_codebook("auto", "")
        assert cats == ["FAIL_unmapped_source_type:EMPTY"]

    def test_ingest_defaults_codebook_is_always_auto(self):
        from src.memory.ingestion.categorization import _INGEST_DEFAULTS, _INGEST_DEFAULT_FALLBACK

        for st, settings in _INGEST_DEFAULTS.items():
            assert settings["codebook"] == "auto", f"{st} must be auto"
        assert _INGEST_DEFAULT_FALLBACK["codebook"] == "auto"


class TestWorkspaceAnchorLabels:
    """#244 TODO2: hybrid mode augments anchors with categories already used
    in the workspace, so labels get reused across texts instead of re-minted."""

    def _db(self, tmp_path):
        from src.memory.store import init_memory_db
        c = init_memory_db(tmp_path / "m.db")
        now = "2026-05-12T00:00:00+00:00"
        c.execute("INSERT INTO sources (source_id, source_type, captured_at, workspace_id) VALUES (?,?,?,?)",
                  ("paper:p1", "agent_result", now, "bene"))
        rows = [
            ("c1", "argumentation,patientenautonomie,methodik"),
            ("c2", "patientenautonomie,[neu]shared-decision-making"),
            ("c3", "patientenautonomie,ergebnis"),
            ("c4", "[neu]shared-decision-making,argumentation"),
            ("c5", "x,y"),                       # too short → filtered
            ("c6", "aaaaaaaaaa"),                # gibberish (repetition) → filtered
        ]
        for cid, lbl in rows:
            c.execute("INSERT INTO chunks (chunk_id,source_id,text,category_labels,category_source,is_active,workspace_id,created_at) "
                      "VALUES (?,?,?,?,?,1,?,?)", (cid, "paper:p1", "t", lbl, "hybrid", "bene", now))
        c.commit()
        return c

    def test_ranks_by_frequency_folds_neu_excludes_static(self, tmp_path):
        from src.memory.ingestion.categorization import _workspace_anchor_labels
        c = self._db(tmp_path)
        anchors = _workspace_anchor_labels(c, "bene", exclude={"argumentation", "methodik", "ergebnis"}, limit=10)
        assert anchors == ["patientenautonomie", "shared-decision-making"]
        c.close()

    def test_empty_for_unknown_workspace(self, tmp_path):
        from src.memory.ingestion.categorization import _workspace_anchor_labels
        c = self._db(tmp_path)
        assert _workspace_anchor_labels(c, "nonexistent", exclude=set()) == []
        c.close()

    def test_limit_caps_output(self, tmp_path):
        from src.memory.ingestion.categorization import _workspace_anchor_labels
        c = self._db(tmp_path)
        assert len(_workspace_anchor_labels(c, "bene", exclude=set(), limit=1)) == 1
        c.close()

    def test_bad_conn_returns_empty(self):
        from src.memory.ingestion.categorization import _workspace_anchor_labels

        class _Boom:
            def execute(self, *a, **k):
                raise RuntimeError("db gone")
        assert _workspace_anchor_labels(_Boom(), "bene", exclude=set()) == []
