#!/usr/bin/env python3
"""Prepares context JSON files for version-upgrade evaluation.

Usage:
    python benchmarks/version_upgrade_prep.py
    python benchmarks/version_upgrade_prep.py --packages "psf/requests"
    python benchmarks/version_upgrade_prep.py --budget 150 --time-budget 900
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.version_upgrade_runner import run_mayringcoder_on_version
from benchmarks.version_upgrade_utils import clone_at_tag, get_gt_diff_files, summarize_run_for_prompt

UPGRADE_PAIRS = [
    {"repo": "psf/requests",      "old_tag": "v1.2.3",  "new_tag": "v2.0.0"},
    {"repo": "pallets/flask",     "old_tag": "0.12.1",  "new_tag": "1.0.0"},
    {"repo": "pytest-dev/pytest", "old_tag": "3.10.1",  "new_tag": "4.0.0"},
]

OUTPUT_DIR = ROOT / "benchmarks" / "version_upgrade_results"


def prepare_one(pair: dict, workspace_id: str, budget: int, time_budget: int, model: str | None) -> dict:
    repo, old_tag, new_tag = pair["repo"], pair["old_tag"], pair["new_tag"]
    print(f"\n{'='*60}\nPackage: {repo}  ({old_tag} → {new_tag})")

    print(f"  [1/3] Computing GT diff ({old_tag}→{new_tag})...")
    with tempfile.TemporaryDirectory() as gt_tmp:
        gt_files = get_gt_diff_files(repo, old_tag, new_tag, gt_tmp)
    print(f"  GT files ({len(gt_files)}): {sorted(gt_files)}")

    print(f"  [2/3] Cloning {old_tag} + running MayringCoder...")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as old_tmp:
        clone_at_tag(repo, old_tag, old_tmp)
        _, run_data = run_mayringcoder_on_version(
            repo_path=old_tmp,
            repo=repo,
            tag=old_tag,
            workspace_id=workspace_id,
            budget=budget,
            time_budget=time_budget,
            model=model,
        )
    runtime = time.time() - t0
    print(f"  Analysis done ({runtime:.0f}s), {len(run_data.get('results', []))} files analyzed")

    print(f"  [3/3] Building prompt summary...")
    prompt_summary = summarize_run_for_prompt(run_data, max_files=40)

    return {
        "repo": repo,
        "old_tag": old_tag,
        "new_tag": new_tag,
        "gt_files": sorted(gt_files),
        "prompt_summary": prompt_summary,
        "files_analyzed": len(run_data.get("results", [])),
        "runtime_s": round(runtime, 1),
        "workspace_id": workspace_id,
        "timestamp": datetime.now().isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Version-Upgrade Benchmark: Context Preparation")
    parser.add_argument("--packages", default=None, help="Komma-Filter, z.B. psf/requests")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--time-budget", type=int, default=600)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    pairs = UPGRADE_PAIRS
    if args.packages:
        filt = set(args.packages.split(","))
        pairs = [p for p in pairs if p["repo"] in filt]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    workspace_id = f"version_upgrade_{run_ts}"

    contexts = []
    for pair in pairs:
        try:
            ctx = prepare_one(pair, workspace_id, args.budget, args.time_budget, args.model)
            contexts.append(ctx)
            safe_name = ctx["repo"].replace("/", "__")
            out_path = OUTPUT_DIR / f"context_{safe_name}_{ctx['old_tag']}_{run_ts}.json"
            out_path.write_text(json.dumps(ctx, indent=2))
            print(f"  Saved: {out_path}")
        except Exception as e:
            print(f"  ERROR {pair['repo']}: {e}")

    summary_path = OUTPUT_DIR / f"prep_summary_{run_ts}.json"
    summary_path.write_text(json.dumps(contexts, indent=2))
    print(f"\nSummary: {summary_path}")
    print("Next: Claude Code reads context JSONs and dispatches Haiku subagents.")


if __name__ == "__main__":
    main()
