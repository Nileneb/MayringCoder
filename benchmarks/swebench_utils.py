from __future__ import annotations

import subprocess


def patched_files(patch: str) -> set[str]:
    return {line[6:] for line in patch.splitlines() if line.startswith("--- a/")}


def determine_match(mc_files: set[str], gt_files: set[str]) -> str:
    if not gt_files:
        return "FN"
    return "TP" if mc_files & gt_files else "FN"


def load_instances(n: int = 10, seed: int = 42) -> list[dict]:
    from datasets import load_dataset
    import random
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
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
