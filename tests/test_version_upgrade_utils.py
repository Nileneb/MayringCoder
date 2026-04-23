import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import tempfile
from pathlib import Path
from benchmarks.version_upgrade_utils import (
    filter_python_files, summarize_run_for_prompt, compute_metrics, get_shown_files
)


def test_filter_python_files():
    files = ["requests/models.py", "requests/sessions.py", "README.md", "setup.cfg", "docs/conf.py"]
    result = filter_python_files(files)
    assert result == {"requests/models.py", "requests/sessions.py"}


def test_filter_python_files_empty():
    assert filter_python_files([]) == set()


def test_summarize_run_for_prompt_includes_smells():
    run_data = {
        "results": [
            {"filename": "requests/sessions.py", "category": "networking",
             "file_summary": "HTTP session management", "potential_smells": ["unsafe redirect handling"]},
            {"filename": "requests/models.py", "category": "data_model",
             "file_summary": "Request/Response models", "potential_smells": []},
            {"filename": "setup.py", "category": "config",
             "file_summary": "Package setup", "potential_smells": []},
        ]
    }
    result = summarize_run_for_prompt(run_data, max_files=10)
    assert "requests/sessions.py" in result
    assert "unsafe redirect handling" in result
    assert "requests/models.py" in result
    assert "setup.py" not in result


def test_summarize_run_for_prompt_max_files():
    run_data = {"results": [
        {"filename": f"src/file{i}.py", "category": "logic",
         "file_summary": f"Summary {i}", "potential_smells": []}
        for i in range(20)
    ]}
    result = summarize_run_for_prompt(run_data, max_files=5)
    assert result.count("src/file") == 5


def test_compute_metrics_perfect_recall():
    gt = {"requests/models.py", "requests/sessions.py"}
    suggested = {"requests/models.py", "requests/sessions.py", "requests/auth.py"}
    m = compute_metrics(gt, suggested)
    assert m["recall"] == 1.0
    assert round(m["precision"], 3) == round(2/3, 3)
    assert m["tp"] == 2


def test_compute_metrics_zero_recall():
    gt = {"requests/models.py"}
    suggested = {"requests/utils.py"}
    m = compute_metrics(gt, suggested)
    assert m["recall"] == 0.0
    assert m["precision"] == 0.0
    assert m["f1"] == 0.0


def test_compute_metrics_empty_gt():
    m = compute_metrics(set(), {"requests/models.py"})
    assert m["recall"] == 0.0


def test_get_shown_files_respects_max_files():
    run_data = {"results": [
        {"filename": f"src/file{i}.py", "category": "logic",
         "file_summary": f"S{i}", "potential_smells": []}
        for i in range(50)
    ]}
    shown = get_shown_files(run_data, max_files=5)
    assert len(shown) == 5
    assert shown[0] == "src/file0.py"


def test_get_shown_files_excludes_skip_dirs():
    run_data = {"results": [
        {"filename": "src/main.py", "category": "core", "file_summary": "", "potential_smells": []},
        {"filename": "docs/conf.py", "category": "docs", "file_summary": "", "potential_smells": []},
    ]}
    shown = get_shown_files(run_data)
    assert shown == ["src/main.py"]


def test_compute_metrics_with_shown_files():
    gt = {"a/b.py", "a/c.py", "a/d.py"}
    suggested = {"a/b.py", "a/c.py"}
    shown = ["a/b.py", "a/c.py"]  # a/d.py war nicht im Prompt
    m = compute_metrics(gt, suggested, shown_files=shown)
    assert m["recall"] == round(2/3, 4)
    assert m["findable_recall"] == 1.0
    assert m["findable_gt_count"] == 2
    assert m["findable_tp"] == 2
    assert "findable_f1" in m


def test_compute_metrics_no_shown_files_backward_compat():
    m = compute_metrics({"a/b.py"}, {"a/b.py"})
    assert "findable_recall" not in m


def test_compute_metrics_empty_gt_with_shown():
    m = compute_metrics(set(), {"a/b.py"}, shown_files=["a/b.py"])
    assert m["recall"] == 0.0
    assert m["findable_recall"] == 0.0


def test_load_contexts(tmp_path):
    from benchmarks.version_upgrade_metrics import load_contexts
    ctx = {"repo": "psf/requests", "old_tag": "v1.2.3", "new_tag": "v2.0.0", "gt_files": ["requests/models.py"]}
    (tmp_path / "context_psf__requests_v1.2.3_ts.json").write_text(json.dumps(ctx))
    (tmp_path / "other_file.json").write_text("{}")
    result = load_contexts(str(tmp_path))
    assert "psf/requests" in result
    assert result["psf/requests"]["old_tag"] == "v1.2.3"
    assert "other_file" not in result
