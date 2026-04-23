#!/usr/bin/env python3
"""Computes recall/precision/F1 from Haiku output + GT context files.

Haiku output JSON format: [{"repo": "psf/requests", "suggested_files": ["requests/models.py", ...]}, ...]

Usage:
    python benchmarks/version_upgrade_metrics.py \
        --haiku-output benchmarks/version_upgrade_results/haiku_output_2026-04-23.json \
        --context-dir benchmarks/version_upgrade_results/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.version_upgrade_utils import compute_metrics, filter_python_files


def load_contexts(context_dir: str) -> dict[str, dict]:
    contexts = {}
    for f in Path(context_dir).glob("context_*.json"):
        try:
            ctx = json.loads(f.read_text())
            contexts[ctx["repo"]] = ctx
        except (KeyError, json.JSONDecodeError):
            pass
    return contexts


def main() -> None:
    parser = argparse.ArgumentParser(description="Version-Upgrade Benchmark: Metrics")
    parser.add_argument("--haiku-output", required=True)
    parser.add_argument("--context-dir", default="benchmarks/version_upgrade_results/")
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    haiku_results = json.loads(Path(args.haiku_output).read_text())
    contexts = load_contexts(args.context_dir)

    rows = []
    has_findable = False
    for entry in haiku_results:
        repo = entry["repo"]
        suggested = filter_python_files(entry.get("suggested_files", []))
        ctx = contexts.get(repo)
        if not ctx:
            print(f"WARNING: no context for {repo}")
            continue
        gt = set(ctx["gt_files"])
        shown_files = ctx.get("shown_files")
        if shown_files:
            has_findable = True
        m = compute_metrics(gt, suggested, shown_files=shown_files)
        row = {
            "repo": repo,
            "old_tag": ctx["old_tag"],
            "new_tag": ctx["new_tag"],
            "gt_count": m["gt_count"],
            "suggested_count": m["suggested_count"],
            "tp": m["tp"],
            "recall": m["recall"],
            "precision": m["precision"],
            "f1": m["f1"],
            "gt_files": ",".join(sorted(gt)),
            "suggested_files": ",".join(sorted(suggested)),
        }
        if shown_files is not None:
            row.update({
                "findable_gt_count": m.get("findable_gt_count", 0),
                "findable_tp": m.get("findable_tp", 0),
                "findable_recall": m.get("findable_recall", 0.0),
                "findable_f1": m.get("findable_f1", 0.0),
            })
        rows.append(row)

    if has_findable:
        print(f"\n{'repo':<30} {'old→new':<20} {'recall':>7} {'prec':>7} {'f1':>7} {'find_rec':>9} {'find_f1':>8} {'tp/gt':>7}")
        print("-" * 97)
    else:
        print(f"\n{'repo':<30} {'old→new':<20} {'recall':>7} {'prec':>7} {'f1':>7} {'tp/gt':>7}")
        print("-" * 80)
    for r in rows:
        version = f"{r['old_tag']}→{r['new_tag']}"
        if has_findable:
            fr = r.get("findable_recall", float("nan"))
            ff = r.get("findable_f1", float("nan"))
            print(f"{r['repo']:<30} {version:<20} {r['recall']:>7.3f} {r['precision']:>7.3f} {r['f1']:>7.3f} {fr:>9.3f} {ff:>8.3f} {r['tp']}/{r['gt_count']}")
        else:
            print(f"{r['repo']:<30} {version:<20} {r['recall']:>7.3f} {r['precision']:>7.3f} {r['f1']:>7.3f} {r['tp']}/{r['gt_count']}")
    print("-" * (97 if has_findable else 80))
    if rows:
        avg_recall = sum(r["recall"] for r in rows) / len(rows)
        avg_f1 = sum(r["f1"] for r in rows) / len(rows)
        if has_findable:
            avg_fr = sum(r.get("findable_recall", 0.0) for r in rows) / len(rows)
            avg_ff = sum(r.get("findable_f1", 0.0) for r in rows) / len(rows)
            print(f"{'AVERAGE':<30} {'':<20} {avg_recall:>7.3f} {'':>7} {avg_f1:>7.3f} {avg_fr:>9.3f} {avg_ff:>8.3f}")
        else:
            print(f"{'AVERAGE':<30} {'':<20} {avg_recall:>7.3f} {'':>7} {avg_f1:>7.3f}")

    csv_path = args.output_csv or str(
        Path(args.context_dir) / f"metrics_{Path(args.haiku_output).stem}.csv"
    )
    if rows:
        # Union of all keys across rows (findable_* may be absent in legacy rows)
        all_keys: list[str] = list(dict.fromkeys(k for r in rows for k in r))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
