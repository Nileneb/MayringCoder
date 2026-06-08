"""Claude cache pre-warm for the span-judge reranker pipeline.

The span-judge export (tools/export_retrieval_dataset.py --span-judge) is
cache-first: any (query, chunk) relevance score already in ``span_judge_cache``
skips the Ollama/GPU call entirely. This tool lets *Claude* contribute a share
of those scores instead of the local GPU — when Claude triggers a retrain it can
judge a batch of pairs itself (it is a strong relevance judge) and pre-warm the
cache, so the subsequent export only burns GPU/Cloud on the remainder.

Two modes:

  python tools/span_judge_prewarm.py --dump --days 30 --out pairs.json [--limit 300]
      Emit the UNCACHED (query, chunk) pairs for the window as JSON batches
      (one batch per query, ≤MAX_CHUNKS_PER_CALL chunks each) with chunk texts.

  python tools/span_judge_prewarm.py --ingest scores.json
      Read [{"query": "...", "scores": {"<chunk_id>": 0..1, ...}}, ...] and
      write them to span_judge_cache tagged model="claude-prewarm" (so they are
      distinguishable from Ollama scores in the cache for auditing).

Claude's judging rubric is span_judge.RELEVANCE_RUBRIC — identical to the Ollama
judge so the teacher labels stay consistent regardless of who produced them.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from mayring_core.config import CACHE_DIR

try:
    import span_judge as _sj
    from export_retrieval_dataset import fetch_feedback_events
except ImportError:
    from tools import span_judge as _sj
    from tools.export_retrieval_dataset import fetch_feedback_events

CLAUDE_PREWARM_MODEL = "claude-prewarm"


def _candidate_pairs(conn: sqlite3.Connection, days: int) -> dict[str, list[str]]:
    """{query → ordered unique chunk_ids} over the training window, dedup
    across repeated events of the same query."""
    by_query: dict[str, list[str]] = {}
    for row in fetch_feedback_events(conn, days):
        query = row["query"]
        try:
            chunks = json.loads(row["trigger_ids"])
        except (TypeError, ValueError):
            continue
        seen = by_query.setdefault(query, [])
        for cid in chunks:
            if cid and cid not in seen:
                seen.append(cid)
    return by_query


def dump(conn: sqlite3.Connection, days: int, limit: int) -> list[dict]:
    """Uncached (query, chunk) pairs as ≤MAX_CHUNKS_PER_CALL batches, capped at
    ``limit`` total pairs (0 = no cap)."""
    _sj.ensure_cache_table(conn)
    batches: list[dict] = []
    emitted = 0
    for query, chunk_ids in _candidate_pairs(conn, days).items():
        if limit and emitted >= limit:
            break
        already = _sj._read_cache(conn, _sj.query_hash(query), chunk_ids)
        missing = [c for c in chunk_ids if c not in already]
        if not missing:
            continue
        texts = _sj._chunk_texts(conn, missing)
        items = [(c, (texts.get(c) or "")[: _sj.CHUNK_TEXT_LIMIT])
                 for c in missing if texts.get(c)]
        for i in range(0, len(items), _sj.MAX_CHUNKS_PER_CALL):
            if limit and emitted >= limit:
                break
            window = items[i:i + _sj.MAX_CHUNKS_PER_CALL]
            if limit:
                window = window[: max(0, limit - emitted)]
            if not window:
                continue
            batches.append({
                "query": query,
                "query_hash": _sj.query_hash(query),
                "chunks": [{"chunk_id": c, "text": t} for c, t in window],
            })
            emitted += len(window)
    return batches


def ingest(conn: sqlite3.Connection, scores_doc: list[dict]) -> int:
    """Write Claude's scores to span_judge_cache. Returns pairs written."""
    _sj.ensure_cache_table(conn)
    written = 0
    for entry in scores_doc:
        query = entry.get("query")
        scores = entry.get("scores") or {}
        if not query or not scores:
            continue
        clean = {str(cid): max(0.0, min(1.0, float(v)))
                 for cid, v in scores.items()}
        _sj._write_cache(conn, _sj.query_hash(query), clean, CLAUDE_PREWARM_MODEL)
        written += len(clean)
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(CACHE_DIR / "memory.db"))
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dump", action="store_true",
                      help="emit uncached (query,chunk) pairs to --out")
    mode.add_argument("--ingest", metavar="SCORES_JSON",
                      help="write Claude's scores from this file into the cache")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--out", default="span_judge_pairs.json")
    ap.add_argument("--limit", type=int, default=300,
                    help="max pairs to dump (0 = no cap)")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"db not found: {db_path}")
        return 2
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        if args.dump:
            batches = dump(conn, args.days, args.limit)
            n_pairs = sum(len(b["chunks"]) for b in batches)
            Path(args.out).write_text(
                json.dumps(batches, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"dumped {n_pairs} uncached pairs in {len(batches)} batches "
                  f"→ {args.out} (window={args.days}d, limit={args.limit})")
            print("Judge each batch per span_judge.RELEVANCE_RUBRIC, write "
                  '[{"query":..., "scores":{chunk_id:0..1}}] and run --ingest.')
        else:
            doc = json.loads(Path(args.ingest).read_text(encoding="utf-8"))
            n = ingest(conn, doc)
            print(f"ingested {n} scores → span_judge_cache "
                  f"(model={CLAUDE_PREWARM_MODEL})")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
