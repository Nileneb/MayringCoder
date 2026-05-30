"""Honest A/B: does cloud-memory help the agent? Same model, memory ON vs OFF.

For each task in task_suite.yaml we run run_task_with_memory twice — once with
memory (ambient context + search_memory tool, cloud-first) and once with
disable_memory=True (pure model) — and score by OBJECTIVE keyword hits (no
LLM-judge, which could bias). We report per-task + per-category + overall, and
we report it faithfully whether memory wins, loses, or ties.

Run:  MAYRING_API_URL=https://mcp.linn.games OLLAMA_URL=http://localhost:11434 \
      python benchmarks/memory_ab_eval.py [--model qwen3.5:9b] [--repeats 1]
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import yaml

from mayring_pi_agent.pi import run_task_with_memory

SUITE = Path(__file__).parent / "task_suite.yaml"


def score(answer: str, keywords: list[str]) -> tuple[int, int, list[str]]:
    a = (answer or "").lower()
    hits = [k for k in keywords if k.lower() in a]
    return len(hits), len(keywords), hits


def run_arm(task: str, model: str, ollama: str, disable: bool) -> tuple[str, float]:
    t0 = time.time()
    try:
        out = run_task_with_memory(
            task=task, ollama_url=ollama, model=model,
            disable_memory=disable, num_predict=2048, timeout=180,
        )
    except Exception as exc:  # report, never hide
        out = f"<ERROR: {exc!r}>"
    return out, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("AB_MODEL", "qwen3.5:9b"))
    ap.add_argument("--repeats", type=int, default=int(os.getenv("AB_REPEATS", "1")))
    args = ap.parse_args()
    ollama = os.getenv("OLLAMA_URL", "http://localhost:11434")

    tasks = yaml.safe_load(SUITE.read_text())["tasks"]
    print(f"# Memory A/B — model={args.model} repeats={args.repeats} tasks={len(tasks)}\n")

    rows = []
    for t in tasks:
        for r in range(args.repeats):
            on, on_s = run_arm(t["task"], args.model, ollama, disable=False)
            off, off_s = run_arm(t["task"], args.model, ollama, disable=True)
            on_h, n, _ = score(on, t["expected_keywords"])
            off_h, _, _ = score(off, t["expected_keywords"])
            rows.append({
                "id": t["id"], "cat": t["category"], "req_mem": t["requires_memory"],
                "on": on_h / n, "off": off_h / n, "on_t": on_s, "off_t": off_s,
            })
            verdict = "ON>" if on_h > off_h else ("OFF>" if off_h > on_h else "tie")
            print(f"{t['id']:<22} req_mem={str(t['requires_memory']):<5} "
                  f"ON={on_h}/{n} OFF={off_h}/{n}  {verdict}  ({on_s:.0f}s/{off_s:.0f}s)")

    print("\n## Per-category (mean keyword-hit fraction)")
    cats = {}
    for row in rows:
        cats.setdefault(row["cat"], []).append(row)
    for cat, rs in sorted(cats.items()):
        on = sum(r["on"] for r in rs) / len(rs)
        off = sum(r["off"] for r in rs) / len(rs)
        print(f"  {cat:<22} ON={on:.2f} OFF={off:.2f}  Δ={on-off:+.2f}  (n={len(rs)})")

    print("\n## Split by requires_memory")
    for flag in (True, False):
        rs = [r for r in rows if r["req_mem"] is flag]
        if not rs:
            continue
        on = sum(r["on"] for r in rs) / len(rs)
        off = sum(r["off"] for r in rs) / len(rs)
        print(f"  requires_memory={str(flag):<5} ON={on:.2f} OFF={off:.2f}  Δ={on-off:+.2f}  (n={len(rs)})")

    on_all = sum(r["on"] for r in rows) / len(rows)
    off_all = sum(r["off"] for r in rows) / len(rows)
    wins_on = sum(1 for r in rows if r["on"] > r["off"])
    wins_off = sum(1 for r in rows if r["off"] > r["on"])
    ties = len(rows) - wins_on - wins_off
    print(f"\n## OVERALL  ON={on_all:.2f} OFF={off_all:.2f}  Δ={on_all-off_all:+.2f}")
    print(f"   task-wins: ON={wins_on} OFF={wins_off} tie={ties} (of {len(rows)})")


if __name__ == "__main__":
    main()
