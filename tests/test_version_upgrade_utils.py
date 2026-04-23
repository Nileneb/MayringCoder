import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import tempfile
from pathlib import Path
from benchmarks.version_upgrade_utils import (
    filter_python_files, summarize_run_for_prompt, compute_metrics
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


def test_load_contexts(tmp_path):
    from benchmarks.version_upgrade_metrics import load_contexts
    ctx = {"repo": "psf/requests", "old_tag": "v1.2.3", "new_tag": "v2.0.0", "gt_files": ["requests/models.py"]}
    (tmp_path / "context_psf__requests_v1.2.3_ts.json").write_text(json.dumps(ctx))
    (tmp_path / "other_file.json").write_text("{}")
    result = load_contexts(str(tmp_path))
    assert "psf/requests" in result
    assert result["psf/requests"]["old_tag"] == "v1.2.3"
    assert "other_file" not in result
