from __future__ import annotations

import subprocess


def patched_files(patch: str) -> set[str]:
    return {line[6:] for line in patch.splitlines() if line.startswith("--- a/")}


def determine_match(mc_files: set[str], gt_files: set[str]) -> str:
    if not gt_files:
        return "FN"
    return "TP" if mc_files & gt_files else "FN"


def load_instances(n: int = 10, seed: int = 42, repo_filter: set[str] | None = None) -> list[dict]:
    from datasets import load_dataset
    import random
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    pool = [inst for inst in ds if repo_filter is None or inst["repo"] in repo_filter]
    rng = random.Random(seed)
    selected = rng.sample(pool, min(n, len(pool)))
    return sorted(selected, key=lambda x: x["instance_id"])


def clone_at_commit(repo: str, commit: str, dest: str) -> None:
    subprocess.run(["git", "init", dest], check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", f"https://github.com/{repo}"],
        cwd=dest, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "fetch", "--depth", "1", "origin", commit],
        cwd=dest, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "FETCH_HEAD"],
        cwd=dest, check=True, capture_output=True,
    )
