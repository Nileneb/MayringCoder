import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import json
from benchmarks.swebench_output import write_csv, write_json, print_summary


SAMPLE_ROWS = [
    {
        "instance_id": "django__django-11099",
        "repo": "django/django",
        "base_commit": "abc123",
        "gt_files": "views/generic.py",
        "mc_files": "views/generic.py,other.py",
        "match": "TP",
        "findings_count": 2,
        "runtime_s": 45.2,
    },
    {
        "instance_id": "psf__requests-4356",
        "repo": "psf/requests",
        "base_commit": "def456",
        "gt_files": "auth.py",
        "mc_files": "",
        "match": "FN",
        "findings_count": 0,
        "runtime_s": 60.1,
    },
]


def test_write_csv_creates_file_with_correct_headers(tmp_path):
    out = tmp_path / "results.csv"
    write_csv(SAMPLE_ROWS, str(out))
    assert out.exists()
    with open(out) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["instance_id"] == "django__django-11099"
    assert rows[0]["match"] == "TP"


def test_write_json_creates_valid_json(tmp_path):
    out = tmp_path / "details.json"
    details = [{"instance_id": "x", "findings": []}]
    write_json(details, str(out))
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded == details


def test_print_summary_runs_without_error(capsys):
    print_summary(SAMPLE_ROWS, total_runtime_s=105.3)
    captured = capsys.readouterr()
    assert "Recall" in captured.out
    assert "1/2" in captured.out
