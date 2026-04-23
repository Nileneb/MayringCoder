import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from benchmarks.swebench_runner import extract_findings, run_mayringcoder


def test_extract_findings_returns_filenames():
    run_data = {
        "results": [
            {"filename": "django/views/base.py", "category": "bug"},
            {"filename": "django/utils/http.py", "category": "bug"},
        ]
    }
    result = extract_findings(run_data)
    assert result == {"django/views/base.py", "django/utils/http.py"}


def test_extract_findings_empty_results():
    assert extract_findings({}) == set()
    assert extract_findings({"results": []}) == set()


def test_run_mayringcoder_returns_filenames_set(tmp_path):
    workspace_id = "swebench_test"
    run_id = "bench_test_instance"
    run_data = {"results": [{"filename": "src/auth.py"}]}

    cache_path = tmp_path / workspace_id / "runs"
    cache_path.mkdir(parents=True)
    (cache_path / f"{run_id}.json").write_text(json.dumps(run_data))

    with patch("benchmarks.swebench_runner.subprocess.run") as mock_run, \
         patch("benchmarks.swebench_runner.CACHE_DIR", str(tmp_path)):
        mock_run.return_value = MagicMock(returncode=0)
        result = run_mayringcoder(
            repo_path=str(tmp_path / "repo"),
            run_id=run_id,
            workspace_id=workspace_id,
        )

    assert result == {"src/auth.py"}
