from __future__ import annotations

import json
import re
from pathlib import Path

from benchmarks.swebench_runner import CACHE_DIR, run_mayringcoder


def _safe_id(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "_", s.lower())


def run_mayringcoder_on_version(
    repo_path: str,
    repo: str,
    tag: str,
    workspace_id: str,
    budget: int = 100,
    time_budget: int = 600,
    model: str | None = None,
) -> tuple[set[str], dict]:
    run_id = f"upgrade_{_safe_id(repo)}_{_safe_id(tag)}"
    mc_files = run_mayringcoder(
        repo_path=repo_path,
        run_id=run_id,
        workspace_id=workspace_id,
        budget=budget,
        time_budget=time_budget,
        model=model,
    )
    matches = list(Path(CACHE_DIR).glob(f"{workspace_id}/**/runs/{run_id}.json"))
    run_data = json.loads(matches[0].read_text()) if matches else {}
    return mc_files, run_data
