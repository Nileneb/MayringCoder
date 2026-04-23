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
