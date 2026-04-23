from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

CACHE_DIR = str(Path(__file__).parent.parent / "cache")


def extract_findings(run_data: dict) -> set[str]:
    return {r["filename"] for r in run_data.get("results", []) if r.get("potential_smells")}


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

    matches = list(Path(CACHE_DIR).glob(f"{workspace_id}/**/runs/{run_id}.json"))
    if not matches:
        return set()

    data = json.loads(matches[0].read_text())
    return extract_findings(data)
