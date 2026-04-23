from __future__ import annotations

import re
import subprocess

_SRC_PATTERN = re.compile(r"\.py$")
_SKIP_DIRS = {"docs", "examples", "doc", "test", "tests", "benchmarks"}

PROMPT_MAX_FILES: int = 40


def filter_python_files(files: list[str]) -> set[str]:
    result = set()
    for f in files:
        if not _SRC_PATTERN.search(f):
            continue
        parts = f.split("/")
        if len(parts) < 2:
            continue
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


def _shown_src_results(run_data: dict, max_files: int = PROMPT_MAX_FILES) -> list[dict]:
    results = run_data.get("results", [])
    src_results = [
        r for r in results
        if filter_python_files([r.get("filename", "")])
    ]
    return src_results[:max_files]


def get_shown_files(run_data: dict, max_files: int = PROMPT_MAX_FILES) -> list[str]:
    """Returns the filenames that actually appear in the prompt summary (respects max_files)."""
    return [r["filename"] for r in _shown_src_results(run_data, max_files)]


def summarize_run_for_prompt(run_data: dict, max_files: int = PROMPT_MAX_FILES) -> str:
    lines = []
    for r in _shown_src_results(run_data, max_files):
        fname = r["filename"]
        cat = r.get("category", "unknown")
        summary = (r.get("file_summary") or "")[:200]
        smells = r.get("potential_smells") or []
        line = f"- {fname} [{cat}]: {summary}"
        if smells:
            line += f"\n  ⚠ {'; '.join(str(s) for s in smells[:3])}"
        lines.append(line)
    return "\n".join(lines)


def compute_metrics(
    gt_files: set[str],
    suggested_files: set[str],
    shown_files: list[str] | None = None,
) -> dict:
    if not gt_files:
        result = {
            "recall": 0.0, "precision": 0.0, "f1": 0.0,
            "tp": 0, "gt_count": 0, "suggested_count": len(suggested_files),
        }
        if shown_files is not None:
            result.update({
                "findable_gt_count": 0, "findable_tp": 0,
                "findable_recall": 0.0, "findable_f1": 0.0,
            })
        return result

    tp = len(gt_files & suggested_files)
    recall = tp / len(gt_files)
    precision = tp / len(suggested_files) if suggested_files else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    result = {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "gt_count": len(gt_files),
        "suggested_count": len(suggested_files),
    }

    if shown_files is not None:
        shown_set = set(shown_files)
        findable_gt = gt_files & shown_set
        findable_tp = len(findable_gt & suggested_files)
        findable_recall = findable_tp / len(findable_gt) if findable_gt else 0.0
        findable_f1 = (
            2 * (precision * findable_recall) / (precision + findable_recall)
            if (precision + findable_recall) > 0 else 0.0
        )
        result.update({
            "findable_gt_count": len(findable_gt),
            "findable_tp": findable_tp,
            "findable_recall": round(findable_recall, 4),
            "findable_f1": round(findable_f1, 4),
        })

    return result
