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
from benchmarks.swebench_runner import ingest_repo, run_mayringcoder
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
    parser.add_argument("--repos", type=str, default=None, help="Komma-getrennte Repo-Filter, z.B. pallets/flask,psf/requests")
    parser.add_argument("--ingest-memory", action="store_true", help="Repos in Ecosystem-Workspace ingesten (side-effect)")
    parser.add_argument("--ecosystem-workspace", default="python_ecosystem", help="Workspace für Ecosystem-Ingestion (default: python_ecosystem)")
    args = parser.parse_args()

    check_mayringcoder_available()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    csv_path = output_dir / f"run_{run_ts}.csv"
    json_path = output_dir / f"run_{run_ts}_details.json"
    workspace_id = f"swebench_{run_ts}"

    repo_filter = set(args.repos.split(",")) if args.repos else None

    print(f"Loading SWEbench-Lite instances (n={args.n}, seed={args.seed}" + (f", repos={args.repos}" if repo_filter else "") + ")...")
    instances = load_instances(n=args.n, seed=args.seed, repo_filter=repo_filter)
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

            if args.ingest_memory:
                print(f"  → Ingesting into '{args.ecosystem_workspace}'...")
                ingest_repo(tmp, args.ecosystem_workspace)

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
