from __future__ import annotations

import csv
import json


FIELDNAMES = [
    "instance_id", "repo", "base_commit",
    "gt_files", "mc_files", "match",
    "findings_count", "runtime_s",
]


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_json(details: list[dict], path: str) -> None:
    with open(path, "w") as f:
        json.dump(details, f, indent=2)


def print_summary(rows: list[dict], total_runtime_s: float) -> None:
    tp = sum(1 for r in rows if r["match"] == "TP")
    total = len(rows)
    avg_findings = sum(r["findings_count"] for r in rows) / total if total else 0
    minutes, seconds = divmod(int(total_runtime_s), 60)

    print()
    print(f"{'instance_id':<35} {'repo':<20} {'gt_files':<30} {'hit':<5} {'match'}")
    print("-" * 100)
    for r in rows:
        hit = "YES" if r["match"] == "TP" else "NO"
        gt = r["gt_files"][:28] if r["gt_files"] else "-"
        print(f"{r['instance_id']:<35} {r['repo']:<20} {gt:<30} {hit:<5} {r['match']}")
    print("-" * 100)
    print(f"Recall:  {tp}/{total} ({tp/total*100:.1f}%)" if total else "Recall: n/a")
    print(f"Avg findings/instance: {avg_findings:.1f}")
    print(f"Runtime: {minutes}m {seconds:02d}s")
