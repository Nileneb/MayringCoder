"""Unit tests for src.report (Issue #14).

All tests run without mocking global config state — generate_report()
now accepts max_chars_per_file as a parameter so tests can be fully
self-contained.
"""

import json
from pathlib import Path

import pytest

from src.report import generate_report, generate_overview_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_diff(**overrides) -> dict:
    base = {
        "changed": [], "added": ["a.py"], "removed": [],
        "unchanged": [], "unanalyzed": ["a.py"],
        "selected": ["a.py"], "skipped": [],
        "snapshot_id": 1,
    }
    base.update(overrides)
    return base


def _minimal_result(filename: str = "a.py", **overrides) -> dict:
    base = {
        "filename": filename,
        "category": "domain",
        "truncated": False,
        "file_summary": "A test file.",
        "_parse_error": False,
        "potential_smells": [],
    }
    base.update(overrides)
    return base


def _minimal_aggregation(**overrides) -> dict:
    base = {
        "total_findings": 0,
        "by_severity": {"critical": 0, "warning": 0, "info": 0},
        "top_findings": [],
        "needs_explikation": [],
        "next_steps": [],
        "parse_errors": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# generate_report — basic structure
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_returns_path_string(self, tmp_path):
        from src import report as mod
        mod.REPORTS_DIR = tmp_path
        path = generate_report(
            repo_url="https://github.com/test/repo",
            model="llama3",
            results=[_minimal_result()],
            aggregation=_minimal_aggregation(),
            diff=_minimal_diff(),
            timing=1.5,
        )
        assert isinstance(path, str)
        assert Path(path).exists()

    def test_report_file_contains_repo_url(self, tmp_path):
        from src import report as mod
        mod.REPORTS_DIR = tmp_path
        path = generate_report(
            repo_url="https://github.com/test/repo",
            model="llama3",
            results=[],
            aggregation=_minimal_aggregation(),
            diff=_minimal_diff(added=[], selected=[]),
            timing=2.0,
        )
        content = Path(path).read_text(encoding="utf-8")
        assert "https://github.com/test/repo" in content

    def test_report_contains_model_name(self, tmp_path):
        from src import report as mod
        mod.REPORTS_DIR = tmp_path
        path = generate_report(
            repo_url="https://github.com/test/repo",
            model="qwen2.5-coder",
            results=[],
            aggregation=_minimal_aggregation(),
            diff=_minimal_diff(added=[], selected=[]),
            timing=1.0,
        )
        content = Path(path).read_text(encoding="utf-8")
        assert "qwen2.5-coder" in content

    def test_run_id_appears_in_report(self, tmp_path):
        from src import report as mod
        mod.REPORTS_DIR = tmp_path
        path = generate_report(
            repo_url="https://github.com/test/repo",
            model="m",
            results=[],
            aggregation=_minimal_aggregation(),
            diff=_minimal_diff(added=[], selected=[]),
            timing=1.0,
            run_id="my-run-42",
        )
        content = Path(path).read_text(encoding="utf-8")
        assert "my-run-42" in content

    def test_default_run_id_is_default(self, tmp_path):
        from src import report as mod
        mod.REPORTS_DIR = tmp_path
        path = generate_report(
            repo_url="https://github.com/test/repo",
            model="m",
            results=[],
            aggregation=_minimal_aggregation(),
            diff=_minimal_diff(added=[], selected=[]),
            timing=1.0,
        )
        content = Path(path).read_text(encoding="utf-8")
        assert "run_id: default" in content

    def test_meta_json_written(self, tmp_path):
        from src import report as mod
        mod.REPORTS_DIR = tmp_path
        generate_report(
            repo_url="https://github.com/test/repo",
            model="m",
            results=[],
            aggregation=_minimal_aggregation(),
            diff=_minimal_diff(added=[], selected=[]),
            timing=1.0,
        )
        meta_files = list(tmp_path.glob("*_meta.json"))
        assert len(meta_files) == 1
        meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
        assert meta["repo"] == "https://github.com/test/repo"
        assert "diff_stats" in meta
        assert "run_time_s" in meta

    def test_findings_appear_in_report(self, tmp_path):
        from src import report as mod
        mod.REPORTS_DIR = tmp_path
        result = _minimal_result(
            potential_smells=[{
                "type": "zombie_code",
                "severity": "warning",
                "confidence": "high",
                "line_hint": "~10",
                "evidence_excerpt": "dead function here",
                "fix_suggestion": "Remove it.",
            }]
        )
        agg = _minimal_aggregation(
            total_findings=1,
            by_severity={"critical": 0, "warning": 1, "info": 0},
            top_findings=[{
                "type": "zombie_code",
                "severity": "warning",
                "confidence": "high",
                "_filename": "a.py",
                "evidence_excerpt": "dead function here",
                "fix_suggestion": "Remove it.",
            }],
        )
        path = generate_report(
            repo_url="https://github.com/test/repo",
            model="m",
            results=[result],
            aggregation=agg,
            diff=_minimal_diff(),
            timing=1.0,
        )
        content = Path(path).read_text(encoding="utf-8")
        assert "zombie_code" in content

    def test_truncated_file_uses_max_chars_parameter(self, tmp_path):
        """max_chars_per_file parameter — no global config call needed."""
        from src import report as mod
        mod.REPORTS_DIR = tmp_path
        result = _minimal_result(truncated=True)
        path = generate_report(
            repo_url="https://github.com/test/repo",
            model="m",
            results=[result],
            aggregation=_minimal_aggregation(),
            diff=_minimal_diff(),
            timing=1.0,
            max_chars_per_file=5000,
        )
        content = Path(path).read_text(encoding="utf-8")
        assert "5000 Zeichen" in content

    def test_truncated_message_absent_when_not_truncated(self, tmp_path):
        from src import report as mod
        mod.REPORTS_DIR = tmp_path
        result = _minimal_result(truncated=False)
        path = generate_report(
            repo_url="https://github.com/test/repo",
            model="m",
            results=[result],
            aggregation=_minimal_aggregation(),
            diff=_minimal_diff(),
            timing=1.0,
            max_chars_per_file=5000,
        )
        content = Path(path).read_text(encoding="utf-8")
        assert "Zeichen" not in content

    def test_embedding_prefilter_meta_in_report(self, tmp_path):
        from src import report as mod
        mod.REPORTS_DIR = tmp_path
        ep = {
            "model": "nomic-embed-text",
            "top_k": 10,
            "threshold": 0.3,
            "query": "security bugs",
            "files_before": 50,
            "files_after": 10,
            "filtered_out": ["x.py"],
        }
        path = generate_report(
            repo_url="https://github.com/test/repo",
            model="m",
            results=[],
            aggregation=_minimal_aggregation(),
            diff=_minimal_diff(added=[], selected=[]),
            timing=1.0,
            embedding_prefilter_meta=ep,
        )
        content = Path(path).read_text(encoding="utf-8")
        assert "nomic-embed-text" in content
        assert "security bugs" in content

    def test_embedding_prefilter_meta_in_json(self, tmp_path):
        from src import report as mod
        mod.REPORTS_DIR = tmp_path
        ep = {
            "model": "nomic-embed-text",
            "top_k": 10,
            "threshold": None,
            "query": "q",
            "files_before": 5,
            "files_after": 3,
            "filtered_out": [],
        }
        generate_report(
            repo_url="https://github.com/test/repo",
            model="m",
            results=[],
            aggregation=_minimal_aggregation(),
            diff=_minimal_diff(added=[], selected=[]),
            timing=1.0,
            embedding_prefilter_meta=ep,
        )
        meta_files = list(tmp_path.glob("*_meta.json"))
        meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
        assert meta["embedding_prefilter"]["model"] == "nomic-embed-text"

    def test_error_result_rendered(self, tmp_path):
        from src import report as mod
        mod.REPORTS_DIR = tmp_path
        result = {"filename": "bad.py", "error": "Timeout after 240s"}
        path = generate_report(
            repo_url="https://github.com/test/repo",
            model="m",
            results=[result],
            aggregation=_minimal_aggregation(),
            diff=_minimal_diff(added=["bad.py"], selected=["bad.py"]),
            timing=1.0,
        )
        content = Path(path).read_text(encoding="utf-8")
        assert "Timeout after 240s" in content


# ---------------------------------------------------------------------------
# generate_overview_report
# ---------------------------------------------------------------------------

class TestGenerateOverviewReport:
    def test_returns_path_string(self, tmp_path):
        from src import report as mod
        mod.REPORTS_DIR = tmp_path
        path = generate_overview_report(
            repo_url="https://github.com/test/repo",
            model="m",
            results=[],
            diff=_minimal_diff(added=[], selected=[]),
            timing=1.0,
        )
        assert isinstance(path, str)
        assert Path(path).exists()

    def test_file_summary_appears(self, tmp_path):
        from src import report as mod
        mod.REPORTS_DIR = tmp_path
        results = [_minimal_result(file_summary="Does important things.")]
        path = generate_overview_report(
            repo_url="https://github.com/test/repo",
            model="m",
            results=results,
            diff=_minimal_diff(),
            timing=1.0,
        )
        content = Path(path).read_text(encoding="utf-8")
        assert "Does important things." in content

    def test_error_result_rendered(self, tmp_path):
        from src import report as mod
        mod.REPORTS_DIR = tmp_path
        results = [{"filename": "fail.py", "category": "domain", "error": "Connection refused"}]
        path = generate_overview_report(
            repo_url="https://github.com/test/repo",
            model="m",
            results=results,
            diff=_minimal_diff(added=["fail.py"], selected=["fail.py"]),
            timing=1.0,
        )
        content = Path(path).read_text(encoding="utf-8")
        assert "Connection refused" in content

    def test_grouped_by_category(self, tmp_path):
        from src import report as mod
        mod.REPORTS_DIR = tmp_path
        results = [
            _minimal_result("a.py", category="api"),
            _minimal_result("b.py", category="domain"),
        ]
        path = generate_overview_report(
            repo_url="https://github.com/test/repo",
            model="m",
            results=results,
            diff=_minimal_diff(added=["a.py", "b.py"], selected=["a.py", "b.py"]),
            timing=1.0,
        )
        content = Path(path).read_text(encoding="utf-8")
        assert "## api" in content
        assert "## domain" in content

    def test_run_id_in_report(self, tmp_path):
        from src import report as mod
        mod.REPORTS_DIR = tmp_path
        path = generate_overview_report(
            repo_url="https://github.com/test/repo",
            model="m",
            results=[],
            diff=_minimal_diff(added=[], selected=[]),
            timing=1.0,
            run_id="overview-run",
        )
        content = Path(path).read_text(encoding="utf-8")
        assert "overview-run" in content


# ---------------------------------------------------------------------------
# full_scan metadata (Issue #20)
# ---------------------------------------------------------------------------

class TestFullScanMetadata:
    def test_full_scan_in_analyze_report(self, tmp_path):
        from src import report as mod
        mod.REPORTS_DIR = tmp_path
        path = generate_report(
            repo_url="https://github.com/test/repo",
            model="m",
            results=[_minimal_result()],
            aggregation=_minimal_aggregation(),
            diff=_minimal_diff(),
            timing=1.0,
            full_scan=True,
        )
        content = Path(path).read_text(encoding="utf-8")
        assert "full_scan: true" in content
        assert "Full Scan" in content

    def test_full_scan_in_meta_json(self, tmp_path):
        from src import report as mod
        mod.REPORTS_DIR = tmp_path
        generate_report(
            repo_url="https://github.com/test/repo",
            model="m",
            results=[],
            aggregation=_minimal_aggregation(),
            diff=_minimal_diff(added=[], selected=[]),
            timing=1.0,
            full_scan=True,
        )
        meta_files = list(tmp_path.glob("*_meta.json"))
        meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
        assert meta["full_scan"] is True

    def test_full_scan_absent_by_default(self, tmp_path):
        from src import report as mod
        mod.REPORTS_DIR = tmp_path
        path = generate_report(
            repo_url="https://github.com/test/repo",
            model="m",
            results=[_minimal_result()],
            aggregation=_minimal_aggregation(),
            diff=_minimal_diff(),
            timing=1.0,
        )
        content = Path(path).read_text(encoding="utf-8")
        assert "full_scan" not in content

    def test_full_scan_in_overview_report(self, tmp_path):
        from src import report as mod
        mod.REPORTS_DIR = tmp_path
        path = generate_overview_report(
            repo_url="https://github.com/test/repo",
            model="m",
            results=[],
            diff=_minimal_diff(added=[], selected=[]),
            timing=1.0,
            full_scan=True,
        )
        content = Path(path).read_text(encoding="utf-8")
        assert "full_scan: true" in content
        assert "Full Scan" in content
