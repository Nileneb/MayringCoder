# SWEbench Ground-Truth Validierung — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Standalone-Skript `benchmarks/swebench_eval.py` das MayringCoders Bug-Erkennungs-Recall gegen SWEbench-Lite Ground-Truth misst (File-Level TP/FN).

**Architecture:** SWEbench-Lite Dataset (HuggingFace) → git clone buggy commit → blocking subprocess call to `python -m src.pipeline` → load results from cache → compare found files against patch-ground-truth → CSV + JSON output.

**Tech Stack:** Python 3.11+, `datasets` (HuggingFace), `subprocess` (git + MayringCoder), `argparse`, `csv`, `json`

---

## File Structure

| Datei | Verantwortung |
|---|---|
| `benchmarks/swebench_eval.py` | Haupt-Skript: CLI, Orchestrierung, Output |
| `benchmarks/swebench_utils.py` | Pure-Funktionen: `patched_files()`, `determine_match()`, `load_instances()`, `clone_at_commit()` |
| `benchmarks/swebench_runner.py` | MayringCoder-Subprocess-Aufruf + Results laden |
| `tests/test_swebench_utils.py` | Unit-Tests für die Pure-Funktionen |
| `requirements.txt` | `datasets>=2.0` hinzufügen |

---

## Task 1: Dependency hinzufügen

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: `datasets>=2.0` in requirements.txt eintragen**

Aktuelle letzte Zeile der Datei prüfen, dann anhängen:
```
datasets>=2.0
```

- [ ] **Step 2: Installation prüfen**

```bash
pip install datasets>=2.0
```
Expected: Installed successfully (oder bereits vorhanden).

- [ ] **Step 3: Import-Sanity-Check**

```bash
python -c "from datasets import load_dataset; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git -C /home/nileneb/Desktop/MayringCoder add requirements.txt
git -C /home/nileneb/Desktop/MayringCoder commit -m "deps: add datasets>=2.0 for swebench eval"
```

---

## Task 2: Pure-Utility-Funktionen + Tests

**Files:**
- Create: `benchmarks/swebench_utils.py`
- Create: `tests/test_swebench_utils.py`

- [ ] **Step 1: Failing tests schreiben**

Erstelle `tests/test_swebench_utils.py`:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from benchmarks.swebench_utils import patched_files, determine_match


SAMPLE_PATCH = """\
diff --git a/django/views/generic/base.py b/django/views/generic/base.py
--- a/django/views/generic/base.py
+++ b/django/views/generic/base.py
@@ -1,3 +1,4 @@
 import logging
+import warnings
 from functools import wraps
diff --git a/django/utils/decorators.py b/django/utils/decorators.py
--- a/django/utils/decorators.py
+++ b/django/utils/decorators.py
@@ -10,2 +10,3 @@
 pass
"""


def test_patched_files_extracts_correct_paths():
    result = patched_files(SAMPLE_PATCH)
    assert result == {"django/views/generic/base.py", "django/utils/decorators.py"}


def test_patched_files_empty_patch():
    assert patched_files("") == set()


def test_patched_files_no_minus_a_lines():
    patch = "diff --git a/foo.py b/foo.py\n+++ b/foo.py\n"
    assert patched_files(patch) == set()


def test_determine_match_tp():
    mc = {"django/views/generic/base.py", "some/other.py"}
    gt = {"django/views/generic/base.py"}
    assert determine_match(mc, gt) == "TP"


def test_determine_match_fn():
    mc = {"completely/different.py"}
    gt = {"django/views/generic/base.py"}
    assert determine_match(mc, gt) == "FN"


def test_determine_match_empty_mc():
    assert determine_match(set(), {"django/views/generic/base.py"}) == "FN"


def test_determine_match_empty_gt():
    assert determine_match({"some/file.py"}, set()) == "FN"
```

- [ ] **Step 2: Tests laufen lassen — müssen FEHLSCHLAGEN**

```bash
cd /home/nileneb/Desktop/MayringCoder && python -m pytest tests/test_swebench_utils.py -v
```
Expected: `ImportError` oder `ModuleNotFoundError` — `benchmarks.swebench_utils` existiert noch nicht.

- [ ] **Step 3: `benchmarks/swebench_utils.py` implementieren**

```python
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def patched_files(patch: str) -> set[str]:
    return {line[6:] for line in patch.splitlines() if line.startswith("--- a/")}


def determine_match(mc_files: set[str], gt_files: set[str]) -> str:
    if not gt_files:
        return "FN"
    return "TP" if mc_files & gt_files else "FN"


def load_instances(n: int = 10, seed: int = 42) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    import random
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    return [ds[i] for i in sorted(indices)]


def clone_at_commit(repo: str, commit: str, dest: str) -> None:
    subprocess.run(
        ["git", "clone", "--depth", "50", f"https://github.com/{repo}", dest],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", commit],
        cwd=dest,
        check=True,
        capture_output=True,
    )
```

- [ ] **Step 4: Tests laufen lassen — müssen PASSEN**

```bash
cd /home/nileneb/Desktop/MayringCoder && python -m pytest tests/test_swebench_utils.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git -C /home/nileneb/Desktop/MayringCoder add benchmarks/swebench_utils.py tests/test_swebench_utils.py
git -C /home/nileneb/Desktop/MayringCoder commit -m "feat(swebench): pure utility functions + tests"
```

---

## Task 3: MayringCoder Subprocess-Runner

**Files:**
- Create: `benchmarks/swebench_runner.py`
- Create: `tests/test_swebench_runner.py`

- [ ] **Step 1: Failing tests schreiben**

Erstelle `tests/test_swebench_runner.py`:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

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
```

- [ ] **Step 2: Tests laufen lassen — müssen FEHLSCHLAGEN**

```bash
cd /home/nileneb/Desktop/MayringCoder && python -m pytest tests/test_swebench_runner.py -v
```
Expected: `ImportError` — `benchmarks.swebench_runner` existiert noch nicht.

- [ ] **Step 3: `benchmarks/swebench_runner.py` implementieren**

```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

CACHE_DIR = str(Path(__file__).parent.parent / "cache")


def extract_findings(run_data: dict) -> set[str]:
    return {r["filename"] for r in run_data.get("results", [])}


def run_mayringcoder(
    repo_path: str,
    run_id: str,
    workspace_id: str = "swebench",
    budget: int = 50,
    time_budget: int = 120,
    model: str | None = None,
) -> set[str]:
    cmd = [
        sys.executable, "-m", "src.pipeline",
        "--repo", repo_path,
        "--workspace-id", workspace_id,
        "--run-id", run_id,
        "--mode", "analyze",
        "--no-limit",
        "--budget", str(budget),
        "--time-budget", str(time_budget),
    ]
    if model:
        cmd += ["--model", model]

    subprocess.run(cmd, check=False, cwd=Path(__file__).parent.parent)

    run_file = Path(CACHE_DIR) / workspace_id / "runs" / f"{run_id}.json"
    if not run_file.exists():
        return set()

    data = json.loads(run_file.read_text())
    return extract_findings(data)
```

- [ ] **Step 4: Tests laufen lassen — müssen PASSEN**

```bash
cd /home/nileneb/Desktop/MayringCoder && python -m pytest tests/test_swebench_runner.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git -C /home/nileneb/Desktop/MayringCoder add benchmarks/swebench_runner.py tests/test_swebench_runner.py
git -C /home/nileneb/Desktop/MayringCoder commit -m "feat(swebench): mayringcoder subprocess runner"
```

---

## Task 4: Output-Funktionen (CSV + Konsole)

**Files:**
- Create: `benchmarks/swebench_output.py`
- Create: `tests/test_swebench_output.py`

- [ ] **Step 1: Failing tests schreiben**

Erstelle `tests/test_swebench_output.py`:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import json
import tempfile
from pathlib import Path
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
```

- [ ] **Step 2: Tests laufen lassen — müssen FEHLSCHLAGEN**

```bash
cd /home/nileneb/Desktop/MayringCoder && python -m pytest tests/test_swebench_output.py -v
```
Expected: `ImportError`.

- [ ] **Step 3: `benchmarks/swebench_output.py` implementieren**

```python
from __future__ import annotations

import csv
import json


FIELDNAMES = [
    "instance_id", "repo", "base_commit",
    "gt_files", "mc_files", "match",
    "findings_count", "runtime_s",
]


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_json(details: list[dict], path: str) -> None:
    with open(path, "w") as f:
        json.dump(details, f, indent=2)


def print_summary(rows: list[dict], total_runtime_s: float) -> None:
    tp = sum(1 for r in rows if r["match"] == "TP")
    total = len(rows)
    avg_findings = sum(r["findings_count"] for r in rows) / total if total else 0
    minutes, seconds = divmod(int(total_runtime_s), 60)

    print()
    print(f"{'instance_id':<35} {'repo':<20} {'gt_files':<30} {'hit':<5} {'match'}")
    print("-" * 100)
    for r in rows:
        hit = "YES" if r["match"] == "TP" else "NO"
        gt = r["gt_files"][:28] if r["gt_files"] else "-"
        print(f"{r['instance_id']:<35} {r['repo']:<20} {gt:<30} {hit:<5} {r['match']}")
    print("-" * 100)
    print(f"Recall:  {tp}/{total} ({tp/total*100:.1f}%)" if total else "Recall: n/a")
    print(f"Avg findings/instance: {avg_findings:.1f}")
    print(f"Runtime: {minutes}m {seconds:02d}s")
```

- [ ] **Step 4: Tests laufen lassen — müssen PASSEN**

```bash
cd /home/nileneb/Desktop/MayringCoder && python -m pytest tests/test_swebench_output.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git -C /home/nileneb/Desktop/MayringCoder add benchmarks/swebench_output.py tests/test_swebench_output.py
git -C /home/nileneb/Desktop/MayringCoder commit -m "feat(swebench): csv/json output + console summary"
```

---

## Task 5: Haupt-Skript `swebench_eval.py`

**Files:**
- Create: `benchmarks/swebench_eval.py`
- Create: `benchmarks/__init__.py` (falls nicht vorhanden)

- [ ] **Step 1: `benchmarks/__init__.py` sicherstellen**

```bash
touch /home/nileneb/Desktop/MayringCoder/benchmarks/__init__.py
```

- [ ] **Step 2: `benchmarks/swebench_eval.py` schreiben**

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.swebench_output import print_summary, write_csv, write_json
from benchmarks.swebench_runner import run_mayringcoder
from benchmarks.swebench_utils import clone_at_commit, determine_match, load_instances, patched_files


def check_mayringcoder_available() -> None:
    try:
        import src.pipeline  # noqa: F401
    except ImportError as e:
        sys.exit(f"MayringCoder src.pipeline not importable: {e}\nRun from MayringCoder root directory.")


def main() -> None:
    parser = argparse.ArgumentParser(description="SWEbench Ground-Truth Eval for MayringCoder")
    parser.add_argument("--n", type=int, default=10, help="Number of instances (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--model", type=str, default=None, help="Ollama model override")
    parser.add_argument("--time-budget", type=int, default=120, help="Seconds per instance (default: 120)")
    parser.add_argument("--budget", type=int, default=50, help="LLM call budget per instance (default: 50)")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "benchmarks" / "swebench_results"))
    args = parser.parse_args()

    check_mayringcoder_available()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    csv_path = output_dir / f"run_{run_ts}.csv"
    json_path = output_dir / f"run_{run_ts}_details.json"
    workspace_id = f"swebench_{run_ts}"

    print(f"Loading {args.n} SWEbench-Lite instances (seed={args.seed})...")
    instances = load_instances(n=args.n, seed=args.seed)
    print(f"Loaded {len(instances)} instances.\n")

    rows: list[dict] = []
    details: list[dict] = []
    total_start = time.time()

    for i, inst in enumerate(instances, 1):
        instance_id = inst["instance_id"]
        repo = inst["repo"]
        commit = inst["base_commit"]
        patch = inst["patch"]

        print(f"[{i}/{len(instances)}] {instance_id}")
        gt = patched_files(patch)
        run_id = f"bench_{instance_id.replace('/', '_').replace('-', '_')}"

        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                clone_at_commit(repo, commit, tmp)
            except Exception as e:
                print(f"  SKIP — clone failed: {e}")
                rows.append(_make_row(instance_id, repo, commit, gt, set(), "FN", 0, 0.0))
                continue

            mc_files = run_mayringcoder(
                repo_path=tmp,
                run_id=run_id,
                workspace_id=workspace_id,
                budget=args.budget,
                time_budget=args.time_budget,
                model=args.model,
            )

        runtime = time.time() - t0
        match = determine_match(mc_files, gt)
        print(f"  GT={sorted(gt)}  MC_HIT={bool(mc_files & gt)}  match={match}  ({runtime:.0f}s)")

        rows.append(_make_row(instance_id, repo, commit, gt, mc_files, match, len(mc_files), runtime))
        details.append({"instance_id": instance_id, "gt_files": list(gt), "mc_files": list(mc_files), "match": match})

    total_runtime = time.time() - total_start
    print_summary(rows, total_runtime)

    write_csv(rows, str(csv_path))
    write_json(details, str(json_path))
    print(f"\nCSV:  {csv_path}")
    print(f"JSON: {json_path}")


def _make_row(
    instance_id: str,
    repo: str,
    commit: str,
    gt: set[str],
    mc: set[str],
    match: str,
    findings_count: int,
    runtime_s: float,
) -> dict:
    return {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": commit,
        "gt_files": ",".join(sorted(gt)),
        "mc_files": ",".join(sorted(mc)),
        "match": match,
        "findings_count": findings_count,
        "runtime_s": round(runtime_s, 1),
    }


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Script ausführbar machen**

```bash
chmod +x /home/nileneb/Desktop/MayringCoder/benchmarks/swebench_eval.py
```

- [ ] **Step 4: Smoke-Test (Dry-Run — kein echter Download)**

```bash
cd /home/nileneb/Desktop/MayringCoder && python benchmarks/swebench_eval.py --help
```
Expected: Argparse help output, kein Crash.

- [ ] **Step 5: `.gitignore` für results/**

```bash
grep -q "swebench_results" /home/nileneb/Desktop/MayringCoder/.gitignore || echo "benchmarks/swebench_results/" >> /home/nileneb/Desktop/MayringCoder/.gitignore
```

- [ ] **Step 6: Commit**

```bash
git -C /home/nileneb/Desktop/MayringCoder add benchmarks/swebench_eval.py benchmarks/__init__.py .gitignore
git -C /home/nileneb/Desktop/MayringCoder commit -m "feat(swebench): main eval script + gitignore results/"
```

---

## Task 6: Echten Testlauf durchführen (1 Instance)

> Voraussetzung: MayringCoder-Umgebung konfiguriert, Ollama unter `three.linn.games` erreichbar, `.env` gesetzt.

- [ ] **Step 1: Einzelne Instance laufen lassen**

```bash
cd /home/nileneb/Desktop/MayringCoder && python benchmarks/swebench_eval.py --n 1 --seed 42 --time-budget 120
```
Expected: Ausgabe wie
```
[1/1] django__django-XXXXX
  GT=['django/path/to/file.py']  MC_HIT=True/False  match=TP/FN  (NNs)
Recall: 1/1 (100.0%) oder 0/1 (0.0%)
...
CSV: benchmarks/swebench_results/run_2026-04-23_HHMMSS.csv
```

- [ ] **Step 2: Alle neuen Tests laufen lassen — müssen PASSEN**

```bash
cd /home/nileneb/Desktop/MayringCoder && python -m pytest tests/test_swebench_utils.py tests/test_swebench_runner.py tests/test_swebench_output.py -v
```
Expected: 13 passed.

- [ ] **Step 3: Final Commit + Push**

```bash
git -C /home/nileneb/Desktop/MayringCoder add -A
git -C /home/nileneb/Desktop/MayringCoder commit -m "feat(swebench): ground-truth eval complete — all tests pass"
git -C /home/nileneb/Desktop/MayringCoder push origin main
```

---

## Verwendung

```bash
# 10 Instances mit Default-Settings
python benchmarks/swebench_eval.py

# Nur 3 Instances, schnellerer Budget
python benchmarks/swebench_eval.py --n 3 --time-budget 60 --budget 30

# Spezifisches Modell
python benchmarks/swebench_eval.py --n 5 --model "qwen2.5-coder:14b"

# Ausgabe in anderem Verzeichnis
python benchmarks/swebench_eval.py --output-dir /tmp/bench_results
```
