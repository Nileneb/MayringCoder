"""Tests for benchmark_summary.py — metric computation and table rendering."""

import json
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run(
    run_id="bench_qwen",
    model="qwen2.5-coder:7b",
    files_checked=10,
    timing=300.0,
    total_findings=20,
    parse_errors=None,
    by_sev=None,
    smells=None,
    time_budget_hit=False,
) -> dict:
    results = []
    for i in range(files_checked):
        s = (smells or [{"type": "zombie_code", "severity": "warning", "confidence": "high"}])
        results.append({"filename": f"f{i}.php", "potential_smells": s})
    parse_err_files = [f"err{i}.php" for i in range(parse_errors or 0)]
    return {
        "run_id": run_id,
        "model": model,
        "timestamp": "2026-04-05T12:00:00Z",
        "files_checked": files_checked,
        "timing_seconds": timing,
        "time_budget_hit": time_budget_hit,
        "results": results,
        "aggregation": {
            "total_findings": total_findings,
            "by_severity": by_sev or {"critical": 2, "warning": 10, "info": 8},
            "parse_errors": parse_err_files,
        },
    }


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_basic_fields(self):
        from benchmark_summary import _compute_metrics
        run = _make_run(files_checked=10, timing=300.0, total_findings=20)
        m = _compute_metrics(run)
        assert m["model"] == "qwen2.5-coder:7b"
        assert m["files"] == 10
        assert m["findings"] == 20
        assert m["timing_s"] == 300.0

    def test_parse_error_rate_zero_when_no_errors(self):
        from benchmark_summary import _compute_metrics
        run = _make_run(parse_errors=0)
        m = _compute_metrics(run)
        assert m["parse_errors"] == 0

    def test_parse_errors_counted(self):
        from benchmark_summary import _compute_metrics
        run = _make_run(files_checked=10, parse_errors=3)
        m = _compute_metrics(run)
        assert m["parse_errors"] == 3

    def test_high_conf_ratio_all_high(self):
        from benchmark_summary import _compute_metrics
        run = _make_run(
            files_checked=2,
            smells=[{"type": "x", "severity": "warning", "confidence": "high"}],
        )
        m = _compute_metrics(run)
        assert m["high_conf_ratio"] == 1.0

    def test_high_conf_ratio_none_high(self):
        from benchmark_summary import _compute_metrics
        run = _make_run(
            files_checked=2,
            smells=[{"type": "x", "severity": "warning", "confidence": "low"}],
        )
        m = _compute_metrics(run)
        assert m["high_conf_ratio"] == 0.0

    def test_time_budget_hit_stored(self):
        from benchmark_summary import _compute_metrics
        run = _make_run(time_budget_hit=True)
        assert _compute_metrics(run)["time_budget_hit"] is True

    def test_no_results_no_crash(self):
        from benchmark_summary import _compute_metrics
        run = _make_run(files_checked=0, total_findings=0)
        run["results"] = []
        m = _compute_metrics(run)
        assert m["files"] == 0
        assert m["raw_score"] == 0.0


# ---------------------------------------------------------------------------
# _normalize_scores
# ---------------------------------------------------------------------------

class TestNormalizeScores:
    def test_best_run_gets_100(self):
        from benchmark_summary import _normalize_scores
        rows = [{"raw_score": 0.5}, {"raw_score": 0.25}, {"raw_score": 0.1}]
        result = _normalize_scores(rows)
        assert result[0]["score"] == 100

    def test_all_zero_scores(self):
        from benchmark_summary import _normalize_scores
        rows = [{"raw_score": 0.0}, {"raw_score": 0.0}]
        result = _normalize_scores(rows)
        assert all(r["score"] == 0 for r in result)

    def test_scores_relative(self):
        from benchmark_summary import _normalize_scores
        rows = [{"raw_score": 1.0}, {"raw_score": 0.5}]
        result = _normalize_scores(rows)
        assert result[0]["score"] == 100
        assert result[1]["score"] == 50


# ---------------------------------------------------------------------------
# _render_table
# ---------------------------------------------------------------------------

class TestRenderTable:
    def _row(self, **kwargs):
        defaults = {
            "model": "test-model", "run_id": "bench_test",
            "timestamp": "2026-04-05 12:00",
            "files": 10, "findings": 20, "critical": 2, "warning": 10,
            "parse_errors": 1, "timing_s": 300.0, "time_budget_hit": False,
            "high_conf_ratio": 0.8, "raw_score": 0.5, "score": 75,
        }
        defaults.update(kwargs)
        return defaults

    def test_returns_string(self):
        from benchmark_summary import _render_table
        assert isinstance(_render_table([self._row()]), str)

    def test_contains_model_name(self):
        from benchmark_summary import _render_table
        t = _render_table([self._row(model="qwen2.5-coder:7b")])
        assert "qwen2.5-coder:7b" in t

    def test_empty_returns_no_runs_message(self):
        from benchmark_summary import _render_table
        assert "Keine" in _render_table([])

    def test_budget_hit_marker_shown(self):
        from benchmark_summary import _render_table
        t = _render_table([self._row(time_budget_hit=True)])
        assert "⏱" in t

    def test_no_budget_marker_when_not_hit(self):
        from benchmark_summary import _render_table
        t = _render_table([self._row(time_budget_hit=False)])
        lines = [l for l in t.splitlines() if "test-model" in l]
        assert len(lines) == 1
        assert "⏱" not in lines[0]


# ---------------------------------------------------------------------------
# _load_bench_runs (filesystem)
# ---------------------------------------------------------------------------

class TestLoadBenchRuns:
    def test_loads_matching_prefix(self, tmp_path):
        from benchmark_summary import _load_bench_runs
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        (runs_dir / "bench_model_a.json").write_text(
            json.dumps({"run_id": "bench_model_a", "model": "a"}), encoding="utf-8"
        )
        (runs_dir / "other_run.json").write_text(
            json.dumps({"run_id": "other_run", "model": "b"}), encoding="utf-8"
        )
        result = _load_bench_runs([runs_dir], "bench_")
        assert len(result) == 1
        assert result[0]["run_id"] == "bench_model_a"

    def test_skips_invalid_json(self, tmp_path):
        from benchmark_summary import _load_bench_runs
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        (runs_dir / "bench_bad.json").write_text("not json", encoding="utf-8")
        result = _load_bench_runs([runs_dir], "bench_")
        assert result == []
