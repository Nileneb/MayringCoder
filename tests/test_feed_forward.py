"""Tests for the feed-forward pipeline (Issue #17): hot-zone context injection and tier filtering."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# _load_turbulence_cache
# ---------------------------------------------------------------------------

class TestLoadTurbulenceCache:
    @pytest.fixture
    def turb_cache(self, tmp_path: Path):
        """Create a mock turbulence cache JSON."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        report = {
            "summary": {"total_files": 3, "critical": 1, "medium": 1, "stable": 1},
            "critical_files": [
                {
                    "path": "app/Services/PaymentService.php",
                    "turbulence": 0.75,
                    "hot_zones": [
                        {
                            "start_line": 45,
                            "end_line": 90,
                            "peak_score": 0.85,
                            "categories": ["Sicherheit", "Daten"],
                            "affected_functions": [
                                {"name": "processPayment", "inputs": ["$amount"], "outputs": ["bool"], "calls": ["Stripe::charge"]},
                            ],
                        }
                    ],
                    "findings": [],
                },
            ],
            "all_files": [
                {
                    "path": "app/Services/PaymentService.php",
                    "tier": "deep",
                    "turbulence": 0.75,
                    "hot_zones": [
                        {
                            "start_line": 45,
                            "end_line": 90,
                            "peak_score": 0.85,
                            "categories": ["Sicherheit", "Daten"],
                            "affected_functions": [
                                {"name": "processPayment", "inputs": ["$amount"], "outputs": ["bool"], "calls": ["Stripe::charge"]},
                            ],
                        }
                    ],
                    "findings": [],
                },
                {
                    "path": "app/Http/Controllers/UserController.php",
                    "tier": "light",
                    "turbulence": 0.35,
                    "hot_zones": [],
                    "findings": [],
                },
                {
                    "path": "app/Models/User.php",
                    "tier": "stable",
                    "turbulence": 0.0,
                    "hot_zones": [],
                    "findings": [],
                },
            ],
            "redundancies": [],
        }

        # repo_slug("https://example.com/example/repo.git") -> "example-repo"
        slug = "example-repo"
        cache_path = cache_dir / f"{slug}_turbulence.json"
        cache_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

        repo_url = "https://example.com/example/repo.git"
        with patch("src.config.CACHE_DIR", cache_dir):
            yield {"cache_dir": cache_dir, "repo_url": repo_url, "report": report}

    def test_returns_hot_zone_map_and_tier_map(self, turb_cache):
        from checker import _load_turbulence_cache

        with patch("src.config.CACHE_DIR", turb_cache["cache_dir"]):
            hz_map, tier_map = _load_turbulence_cache(turb_cache["repo_url"])

        assert hz_map is not None
        assert tier_map is not None

    def test_tier_map_has_correct_tiers(self, turb_cache):
        from checker import _load_turbulence_cache

        with patch("src.config.CACHE_DIR", turb_cache["cache_dir"]):
            _, tier_map = _load_turbulence_cache(turb_cache["repo_url"])

        assert tier_map["app/Services/PaymentService.php"] == "deep"
        assert tier_map["app/Http/Controllers/UserController.php"] == "light"
        assert tier_map["app/Models/User.php"] == "stable"

    def test_hot_zone_context_includes_line_info(self, turb_cache):
        from checker import _load_turbulence_cache

        with patch("src.config.CACHE_DIR", turb_cache["cache_dir"]):
            hz_map, _ = _load_turbulence_cache(turb_cache["repo_url"])

        ctx = hz_map.get("app/Services/PaymentService.php", "")
        assert "45" in ctx
        assert "90" in ctx
        assert "Hot-Zone" in ctx

    def test_hot_zone_context_includes_affected_functions(self, turb_cache):
        from checker import _load_turbulence_cache

        with patch("src.config.CACHE_DIR", turb_cache["cache_dir"]):
            hz_map, _ = _load_turbulence_cache(turb_cache["repo_url"])

        ctx = hz_map.get("app/Services/PaymentService.php", "")
        assert "processPayment" in ctx
        assert "Stripe::charge" in ctx

    def test_files_without_hot_zones_have_no_context(self, turb_cache):
        from checker import _load_turbulence_cache

        with patch("src.config.CACHE_DIR", turb_cache["cache_dir"]):
            hz_map, _ = _load_turbulence_cache(turb_cache["repo_url"])

        assert "app/Http/Controllers/UserController.php" not in hz_map or \
               not hz_map.get("app/Http/Controllers/UserController.php")

    def test_nonexistent_cache_returns_none(self, tmp_path):
        from checker import _load_turbulence_cache

        with patch("src.config.CACHE_DIR", tmp_path / "nonexistent"):
            result = _load_turbulence_cache("https://example.com/no/repo.git")
        assert result == (None, None)


# ---------------------------------------------------------------------------
# analyze_file — hot_zone_context injection
# ---------------------------------------------------------------------------

class TestAnalyzeFileHotZoneContext:
    def test_hot_zone_context_injected_into_prompt(self):
        """Verify that hot_zone_context appears in the prompt sent to LLM."""
        from src.analyzer import analyze_file

        captured_prompt = {}

        def mock_generate(prompt, ollama_url, model, label, *, system_prompt=None):
            captured_prompt["value"] = prompt
            return '{"file_summary": "test", "potential_smells": []}'

        file = {"filename": "test.php", "content": "<?php echo 1;", "category": "domain"}
        template = "Analyze this file."
        hz_context = "ACHTUNG: Hot-Zone bei Zeile 10-20 (Sicherheit × Daten)"

        with patch("src.analyzer._ollama_generate", side_effect=mock_generate):
            analyze_file(file, template, "http://localhost", "test-model",
                         hot_zone_context=hz_context)

        assert "Hot-Zone" in captured_prompt["value"]
        assert "Zeile 10-20" in captured_prompt["value"]

    def test_no_hot_zone_context_by_default(self):
        """When hot_zone_context is None, prompt should not contain Hot-Zone."""
        from src.analyzer import analyze_file

        captured_prompt = {}

        def mock_generate(prompt, ollama_url, model, label, *, system_prompt=None):
            captured_prompt["value"] = prompt
            return '{"file_summary": "test", "potential_smells": []}'

        file = {"filename": "test.php", "content": "<?php echo 1;", "category": "domain"}

        with patch("src.analyzer._ollama_generate", side_effect=mock_generate):
            analyze_file(file, "Analyze.", "http://localhost", "test-model")

        assert "Hot-Zone" not in captured_prompt["value"]


# ---------------------------------------------------------------------------
# analyze_files — hot_zone_context_map passthrough
# ---------------------------------------------------------------------------

class TestAnalyzeFilesHotZoneMap:
    def test_passes_hot_zone_context_per_file(self, tmp_path):
        """Verify that hot_zone_context_map values are passed to each file's analysis."""
        from src.analyzer import analyze_files

        captured_contexts = []

        def mock_generate(prompt, ollama_url, model, label, *, system_prompt=None):
            captured_contexts.append(prompt)
            return '{"file_summary": "test", "potential_smells": []}'

        prompt_file = tmp_path / "test_prompt.md"
        prompt_file.write_text("Analyze this file.", encoding="utf-8")

        files = [
            {"filename": "a.php", "content": "<?php // a", "category": "domain"},
            {"filename": "b.php", "content": "<?php // b", "category": "api"},
        ]
        hz_map = {
            "a.php": "ACHTUNG: Hot-Zone A",
        }

        with patch("src.analyzer._ollama_generate", side_effect=mock_generate):
            results, budget_hit = analyze_files(
                files, ["a.php", "b.php"], prompt_file, "http://localhost", "m",
                hot_zone_context_map=hz_map,
            )

        assert len(results) == 2
        assert budget_hit is False
        # a.php should have hot-zone context
        assert "Hot-Zone A" in captured_contexts[0]
        # b.php should NOT have hot-zone context
        assert "Hot-Zone" not in captured_contexts[1]


class TestTimeBudget:
    def _make_prompt(self, tmp_path):
        p = tmp_path / "prompt.md"
        p.write_text("Analyze.", encoding="utf-8")
        return p

    def _files(self, n=5):
        return [{"filename": f"f{i}.py", "content": f"x = {i}", "category": "domain"}
                for i in range(n)]

    def test_no_budget_returns_all_files(self, tmp_path):
        from src.analyzer import analyze_files
        files = self._files(3)
        with patch("src.analyzer._ollama_generate",
                   return_value='{"potential_smells": []}'):
            results, hit = analyze_files(
                files, [f["filename"] for f in files],
                self._make_prompt(tmp_path), "http://localhost", "m",
            )
        assert len(results) == 3
        assert hit is False

    def test_expired_budget_stops_after_first_file(self, tmp_path):
        """Budget of 0 seconds expires immediately — only first file is processed."""
        from src.analyzer import analyze_files
        files = self._files(5)
        with patch("src.analyzer._ollama_generate",
                   return_value='{"potential_smells": []}'):
            results, hit = analyze_files(
                files, [f["filename"] for f in files],
                self._make_prompt(tmp_path), "http://localhost", "m",
                time_budget=0,
            )
        assert len(results) == 1
        assert hit is True

    def test_sufficient_budget_processes_all(self, tmp_path):
        """Large budget → all files processed, budget_hit=False."""
        from src.analyzer import analyze_files
        files = self._files(3)
        with patch("src.analyzer._ollama_generate",
                   return_value='{"potential_smells": []}'):
            results, hit = analyze_files(
                files, [f["filename"] for f in files],
                self._make_prompt(tmp_path), "http://localhost", "m",
                time_budget=9999,
            )
        assert len(results) == 3
        assert hit is False

    def test_overview_files_budget(self, tmp_path):
        """overview_files also respects time_budget."""
        from src.analyzer import overview_files
        files = self._files(5)
        with patch("src.analyzer._ollama_generate",
                   return_value='{"file_summary": "ok"}'):
            results, hit = overview_files(
                files, [f["filename"] for f in files],
                self._make_prompt(tmp_path), "http://localhost", "m",
                time_budget=0,
            )
        assert len(results) == 1
        assert hit is True


# ---------------------------------------------------------------------------
# Turbulence analyzer — overview_cache parameter
# ---------------------------------------------------------------------------

class TestTurbulenceOverviewCache:
    def test_overview_cache_sets_chunk_category(self, tmp_path):
        """When overview_cache provides a category, chunks should use it."""
        from src.turbulence_analyzer import analyze_repo

        # Create a minimal PHP file
        php_dir = tmp_path / "app" / "Services"
        php_dir.mkdir(parents=True)
        php_file = php_dir / "PaymentService.php"
        # Write enough content for MIN_CHUNKS_FOR_TRIAGE (at least ~30 lines)
        lines = ["<?php\n", "namespace App\\Services;\n", "\n"]
        lines += [f"// line {i}\n" for i in range(50)]
        lines += ["function processPayment($amount) {\n", "  return true;\n", "}\n"]
        php_file.write_text("".join(lines))

        overview_cache = {
            "app/Services/PaymentService.php": {
                "category": "domain",
                "functions": [{"name": "processPayment", "inputs": ["$amount"], "outputs": ["bool"], "calls": []}],
            }
        }

        report = analyze_repo(str(tmp_path), use_llm=False, overview_cache=overview_cache)
        assert "critical_files" in report

    def test_without_overview_cache_falls_back_to_heuristic(self, tmp_path):
        """Without cache, categorization should still work via heuristic."""
        from src.turbulence_analyzer import analyze_repo

        php_dir = tmp_path / "app"
        php_dir.mkdir()
        php_file = php_dir / "Test.php"
        lines = ["<?php\n"] + [f"// line {i}\n" for i in range(30)]
        php_file.write_text("".join(lines))

        report = analyze_repo(str(tmp_path), use_llm=False, overview_cache=None)
        assert "critical_files" in report


# ---------------------------------------------------------------------------
# build_report — all_files output (Issue #17 Gap)
# ---------------------------------------------------------------------------

class TestBuildReportAllFiles:
    def test_all_files_contains_every_tier(self):
        from src.turbulence_calculator import FileAnalysis
        from src.turbulence_report import build_report

        analyses = [
            FileAnalysis(path="deep.php", total_lines=100,
                         turbulence_score=0.75, tier="deep"),
            FileAnalysis(path="light.php", total_lines=50,
                         turbulence_score=0.35, tier="light"),
            FileAnalysis(path="skip.php", total_lines=30,
                         turbulence_score=0.1, tier="skip"),
            FileAnalysis(path="stable.php", total_lines=10,
                         turbulence_score=0.0, tier="stable"),
        ]
        report = build_report(analyses, [])

        assert "all_files" in report
        assert len(report["all_files"]) == 4

        tiers = {f["path"]: f["tier"] for f in report["all_files"]}
        assert tiers["deep.php"] == "deep"
        assert tiers["light.php"] == "light"
        assert tiers["skip.php"] == "skip"
        assert tiers["stable.php"] == "stable"

    def test_critical_files_still_only_deep(self):
        from src.turbulence_calculator import FileAnalysis
        from src.turbulence_report import build_report

        analyses = [
            FileAnalysis(path="deep.php", total_lines=100,
                         turbulence_score=0.75, tier="deep"),
            FileAnalysis(path="light.php", total_lines=50,
                         turbulence_score=0.35, tier="light"),
        ]
        report = build_report(analyses, [])

        assert len(report["critical_files"]) == 1
        assert report["critical_files"][0]["path"] == "deep.php"

    def test_all_files_entries_have_tier_field(self):
        from src.turbulence_calculator import FileAnalysis
        from src.turbulence_report import build_report

        analyses = [
            FileAnalysis(path="a.php", total_lines=10,
                         turbulence_score=0.0, tier="stable"),
        ]
        report = build_report(analyses, [])

        for entry in report["all_files"]:
            assert "tier" in entry
            assert "turbulence" in entry
