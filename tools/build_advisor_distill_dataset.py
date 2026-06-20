"""Build the advisor-distillation dataset from Claude's span-judge labels (Pfad B).

A (claude-as-teacher) writes Claude relevance scores into ``span_judge_cache``
tagged ``claude-prewarm``. Those (query, chunk, score) tuples are EXACTLY the
input/output pairs needed to fine-tune the local judge model so it replaces the
runtime advisor (``pi_judge_relevance`` / inject-advisor) — same RELEVANCE_RUBRIC,
same task. So B reuses A's output: every pair Claude judges for the reranker also
becomes a distillation example, no extra labelling.

``span_judge_cache`` stores only ``query_hash`` (not the query text), so we
rebuild the hash→query map from ``context_feedback_log`` (the source of the
prewarm queries). One row per (query, chunk):

    {"query": str, "chunk_id": str, "text": str, "score": float}

The judge fine-tune variant (deferred) wraps these in RELEVANCE_RUBRIC chat
messages; kept rubric-agnostic here so the same dataset can train a per-pair
regressor or a batched-JSON judge.

Usage:
    python tools/build_advisor_distill_dataset.py \
        --db cache/memory.db --out cache/finetuning/advisor_distill.jsonl
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

try:
    import span_judge as _sj
except ImportError:
    from tools import span_judge as _sj

PREWARM_MODEL = "claude-prewarm"


def _hash_to_query(conn: sqlite3.Connection) -> dict[str, str]:
    """Map span_judge query_hash → query text via context_feedback_log."""
    out: dict[str, str] = {}
    for (query,) in conn.execute(
        "SELECT DISTINCT query FROM context_feedback_log WHERE query != ''"
    ):
        out.setdefault(_sj.query_hash(query), query)
    return out


def build_records(conn: sqlite3.Connection, model: str = PREWARM_MODEL) -> list[dict]:
    """One {query, chunk_id, text, score} record per Claude-judged pair."""
    h2q = _hash_to_query(conn)
    rows = conn.execute(
        "SELECT s.query_hash, s.chunk_id, s.score, c.text "
        "FROM span_judge_cache s JOIN chunks c ON c.chunk_id = s.chunk_id "
        "WHERE s.model = ?",
        (model,),
    ).fetchall()
    records: list[dict] = []
    for qhash, chunk_id, score, text in rows:
        query = h2q.get(qhash)
        if not query or not text:
            continue
        records.append({
            "query": query,
            "chunk_id": chunk_id,
            "text": text,
            "score": float(score),
        })
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="cache/memory.db")
    ap.add_argument("--out", default="cache/finetuning/advisor_distill.jsonl")
    ap.add_argument("--model", default=PREWARM_MODEL)
    args = ap.parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"db not found: {db_path}")
        return 2
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        records = build_records(conn, args.model)
    finally:
        conn.close()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} advisor-distill records → {out} (model={args.model})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
