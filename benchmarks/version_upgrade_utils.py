from __future__ import annotations

import re
import subprocess

_SRC_PATTERN = re.compile(r"\.py$")
_SKIP_DIRS = {"docs", "examples", "doc", "test", "tests", "benchmarks"}


def filter_python_files(files: list[str]) -> set[str]:
    result = set()
    for f in files:
        if not _SRC_PATTERN.search(f):
            continue
        parts = f.split("/")
        if len(parts) < 2:
            continue  # root-level files (setup.py, conftest.py, etc.)
        if any(p in _SKIP_DIRS for p in parts[:-1]):
            continue
        result.add(f)
    return result


def clone_at_tag(repo: str, tag: str, dest: str) -> None:
    subprocess.run(["git", "init", dest], check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", f"https://github.com/{repo}"],
        cwd=dest, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "fetch", "--depth", "1", "origin", f"refs/tags/{tag}"],
        cwd=dest, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "FETCH_HEAD"],
        cwd=dest, check=True, capture_output=True,
    )


def get_gt_diff_files(repo: str, old_tag: str, new_tag: str, dest: str) -> set[str]:
    subprocess.run(["git", "init", dest], check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", f"https://github.com/{repo}"],
        cwd=dest, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "fetch", "origin",
         f"refs/tags/{old_tag}:refs/tags/{old_tag}",
         f"refs/tags/{new_tag}:refs/tags/{new_tag}"],
        cwd=dest, check=True, capture_output=True,
    )
    result = subprocess.run(
        ["git", "diff", old_tag, new_tag, "--name-only"],
        cwd=dest, capture_output=True, text=True, check=True,
    )
    return filter_python_files(result.stdout.splitlines())


def summarize_run_for_prompt(run_data: dict, max_files: int = 30) -> str:
    lines = []
    results = run_data.get("results", [])
    src_results = [
        r for r in results
        if filter_python_files([r.get("filename", "")])
    ]
    for r in src_results[:max_files]:
        fname = r["filename"]
        cat = r.get("category", "unknown")
        summary = (r.get("file_summary") or "")[:200]
        smells = r.get("potential_smells") or []
        line = f"- {fname} [{cat}]: {summary}"
        if smells:
            line += f"\n  ⚠ {'; '.join(str(s) for s in smells[:3])}"
        lines.append(line)
    return "\n".join(lines)


def compute_metrics(gt_files: set[str], suggested_files: set[str]) -> dict:
    if not gt_files:
        return {
            "recall": 0.0, "precision": 0.0, "f1": 0.0,
            "tp": 0, "gt_count": 0, "suggested_count": len(suggested_files),
        }
    tp = len(gt_files & suggested_files)
    recall = tp / len(gt_files)
    precision = tp / len(suggested_files) if suggested_files else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "gt_count": len(gt_files),
        "suggested_count": len(suggested_files),
    }
