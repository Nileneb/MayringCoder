#!/usr/bin/env python3
"""Standalone ingestion of well-known Python packages into a MayringCoder workspace.

Usage:
    python benchmarks/ingest_packages.py --packages "psf/requests,pallets/flask"
    python benchmarks/ingest_packages.py --all-swebench
    python benchmarks/ingest_packages.py --all-swebench --workspace my_ecosystem
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.swebench_utils import clone_at_commit

SWEBENCH_REPOS = [
    "pallets/flask",
    "psf/requests",
    "mwaskom/seaborn",
    "pydata/xarray",
    "astropy/astropy",
    "pylint-dev/pylint",
    "sphinx-doc/sphinx",
    "pytest-dev/pytest",
    "matplotlib/matplotlib",
    "scikit-learn/scikit-learn",
    "django/django",
    "sympy/sympy",
]


def get_latest_tag(repo_path: str) -> str | None:
    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"],
        cwd=repo_path, capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def checkout_latest_tag(repo_path: str) -> str:
    tag = get_latest_tag(repo_path)
    if tag:
        subprocess.run(["git", "checkout", tag], cwd=repo_path, capture_output=True)
        return tag
    return "HEAD"


def ingest_into_workspace(repo_path: str, workspace_id: str) -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "src.cli",
         "--repo", repo_path,
         "--populate-memory",
         "--workspace-id", workspace_id,
         "--no-limit"],
        cwd=ROOT,
        check=False,
    )
    return result.returncode == 0


def clone_latest(repo: str, dest: str) -> str:
    subprocess.run(
        ["git", "init", dest], check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "remote", "add", "origin", f"https://github.com/{repo}"],
        cwd=dest, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "fetch", "--depth", "1", "origin", "HEAD"],
        cwd=dest, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "FETCH_HEAD"],
        cwd=dest, check=True, capture_output=True,
    )
    return checkout_latest_tag(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Python packages into MayringCoder memory")
    parser.add_argument("--packages", type=str, default=None,
                        help="Komma-getrennte GitHub-Repos, z.B. psf/requests,pallets/flask")
    parser.add_argument("--all-swebench", action="store_true",
                        help="Alle SWEbench-Lite Repos ingesten")
    parser.add_argument("--workspace", default="python_ecosystem",
                        help="Ziel-Workspace (default: python_ecosystem)")
    args = parser.parse_args()

    if args.all_swebench:
        repos = SWEBENCH_REPOS
    elif args.packages:
        repos = [r.strip() for r in args.packages.split(",") if r.strip()]
    else:
        parser.error("--packages oder --all-swebench erforderlich")

    print(f"Workspace: {args.workspace}")
    print(f"Repos ({len(repos)}): {', '.join(repos)}\n")

    rows = []
    total_start = time.time()

    for i, repo in enumerate(repos, 1):
        print(f"[{i}/{len(repos)}] {repo}")
        t0 = time.time()

        with tempfile.TemporaryDirectory() as tmp:
            try:
                version = clone_latest(repo, tmp)
            except Exception as e:
                print(f"  SKIP — clone failed: {e}")
                rows.append({"repo": repo, "version": "ERROR", "runtime_s": 0.0, "ok": False})
                continue

            print(f"  version: {version}")
            ok = ingest_into_workspace(tmp, args.workspace)

        runtime = time.time() - t0
        status = "OK" if ok else "FAIL"
        print(f"  {status}  ({runtime:.0f}s)")
        rows.append({"repo": repo, "version": version, "runtime_s": round(runtime, 1), "ok": ok})

    total_runtime = time.time() - total_start
    minutes, seconds = divmod(int(total_runtime), 60)

    print()
    print(f"{'repo':<35} {'version':<15} {'runtime_s':<12} {'status'}")
    print("-" * 75)
    for r in rows:
        status = "OK" if r["ok"] else "FAIL/SKIP"
        print(f"{r['repo']:<35} {r['version']:<15} {r['runtime_s']:<12} {status}")
    print("-" * 75)
    ok_count = sum(1 for r in rows if r["ok"])
    print(f"Ingested: {ok_count}/{len(rows)}  |  Total runtime: {minutes}m {seconds:02d}s")
    print(f"Workspace: {args.workspace}")


if __name__ == "__main__":
    main()
