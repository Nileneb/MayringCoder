"""Export a judge-relevance fine-tuning dataset (#260).

One JSONL row per chunk that has explicit rating feedback:

    {
      "input":  {"query": "...", "chunk_text": "..."},
      "output": 0.75,                 # relevance score in [0, 1]
      "workspace_id": "...",
      "chunk_id": "chk_...",
      "n_ratings": 2
    }

Source:
  * ``chunk_feedback`` (memory.db) — ``signal`` on the 1..5 rating scale,
    mapped to ``(avg_rating - 1) / 4`` → [0, 1]. Legacy ``positive`` / ``negative``
    map to 4 / 2 (same convention as tools/export_retrieval_dataset.py).
  * ``chunks`` — the chunk text (the second half of the input pair).
  * query: taken from ``chunk_feedback.metadata.query_context`` when present.

⚠ Data prerequisite: locally most rating rows have NO ``query_context`` (they
were written by the auto memory-context generator, not by a real search). Those
rows emit ``query: ""`` — usable only after a backfill that records the query
alongside the rating (see docs/specialist-models.md → judge-relevance). The
exporter is the pipeline; the query backfill is the gating data task.

Usage:
    python tools/export_judge_relevance_dataset.py \
        --db cache/memory.db \
        --out cache/finetuning/judge_relevance_dataset.jsonl \
        --require-query        # skip rows without a query_context
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from mayring_core.config import CACHE_DIR

DEFAULT_OUT = CACHE_DIR / "finetuning" / "judge_relevance_dataset.jsonl"


def _rating_value(signal: str) -> float | None:
    """Map a feedback signal to a 1..5 rating, or None if it carries no signal."""
    if signal in ("1", "2", "3", "4", "5"):
        return float(signal)
    if signal == "positive":
        return 4.0
    if signal == "negative":
        return 2.0
    return None  # neutral / other → no training signal


def export(db_path: Path, out: Path, days: int, require_query: bool) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT cf.chunk_id, cf.signal, cf.metadata, cf.workspace_id, "
            "       c.text AS chunk_text "
            "FROM chunk_feedback cf JOIN chunks c ON c.chunk_id = cf.chunk_id "
            "WHERE cf.created_at > datetime('now', ?)",
            (f"-{days} days",),
        ).fetchall()

        # Aggregate ratings per chunk; remember a query_context if any row has one.
        agg: dict[str, dict] = {}
        for r in rows:
            val = _rating_value(r["signal"])
            if val is None:
                continue
            try:
                meta = json.loads(r["metadata"] or "{}")
            except (TypeError, ValueError):
                meta = {}
            entry = agg.setdefault(r["chunk_id"], {
                "ratings": [], "query": "",
                "text": r["chunk_text"] or "",
                "workspace_id": r["workspace_id"] or "",
            })
            entry["ratings"].append(val)
            if not entry["query"]:
                entry["query"] = (meta.get("query_context") or "").strip()

        written = 0
        with out.open("w", encoding="utf-8") as f:
            for cid, e in agg.items():
                if require_query and not e["query"]:
                    continue
                avg = sum(e["ratings"]) / len(e["ratings"])
                score = round((avg - 1.0) / 4.0, 4)
                f.write(json.dumps({
                    "input": {"query": e["query"], "chunk_text": e["text"]},
                    "output": score,
                    "workspace_id": e["workspace_id"],
                    "chunk_id": cid,
                    "n_ratings": len(e["ratings"]),
                }, ensure_ascii=False) + "\n")
                written += 1
        return written
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(CACHE_DIR / "memory.db"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--days", type=int, default=3650)
    ap.add_argument(
        "--require-query", action="store_true",
        help="skip rows without a metadata.query_context (recommended for training)",
    )
    args = ap.parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"db not found: {db_path}")
        return 2
    n = export(db_path, Path(args.out), args.days, args.require_query)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] wrote {n} judge-relevance rows → {args.out} "
          f"(require_query={args.require_query})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
