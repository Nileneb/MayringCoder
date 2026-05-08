"""Export retrieval dataset for reranker training.

Joins ``context_feedback_log`` (one row per /memory/search call with the
top-K chunk_ids and the per-stage scores) with ``chunk_feedback`` (signal
posted by the Stop hook after the assistant's reply) to produce one
training row per (query, chunk) pair:

    {
      "query":     "...",
      "chunk_id":  "chk_...",
      "features":  {"v": .., "s": .., "r": .., "a": .., "f": ..},
      "label":     1 | 0,
      "captured_at": "...",
      "workspace_id": "...",
    }

This is the dataset Pipeline 2 of Issue #87 needs. Initially the data
volume is small (~hundreds of feedback rows) so a logistic-regression /
small-tree reranker is the realistic target — not a transformer.

Usage:
    python tools/export_retrieval_dataset.py
        --out cache/finetuning/retrieval_dataset.jsonl
        --days 30
        --negative-mode unlabeled

Negative-mode determines how unlabeled chunks become training negatives:
  * ``unlabeled``  — all top-K chunks that did NOT get positive feedback
                     are negatives (faster to label, noisier).
  * ``explicit``   — only chunks with explicit ``negative`` feedback.
                     Higher-precision negatives, far fewer rows.

Default ``unlabeled`` because we have far more positives than explicit
negatives and we need the volume to learn weights.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.config import CACHE_DIR

DEFAULT_OUT = CACHE_DIR / "finetuning" / "retrieval_dataset.jsonl"


def _label_map(conn: sqlite3.Connection, days: int) -> dict[str, int]:
    """{chunk_id: 1 if positive feedback dominates else 0/-1}."""
    rows = conn.execute(
        "SELECT chunk_id, signal FROM chunk_feedback "
        "WHERE created_at > datetime('now', ?)",
        (f"-{days} days",),
    ).fetchall()
    counts: dict[str, dict[str, int]] = {}
    for cid, sig in rows:
        bucket = counts.setdefault(cid, {"pos": 0, "neg": 0})
        if sig in ("positive", "1", "2", "3", "4", "5"):
            bucket["pos"] += 1
        elif sig == "negative":
            bucket["neg"] += 1
    out: dict[str, int] = {}
    for cid, c in counts.items():
        if c["pos"] > c["neg"]:
            out[cid] = 1
        elif c["neg"] > c["pos"]:
            out[cid] = -1
        # else unknown → omit
    return out


FEATURES_OUT = ("v", "s", "r", "a", "sf", "sl")


def _normalize_features(feats: dict) -> dict | None:
    """Return a 6-key dict {v, s, r, a, sf, sl}.

    Old logs (before the v2-feature-set commit) only have {v, s, r, a, f}.
    For backward compat we synthesize sf=0.5 + sl=0.5 (= 'no signal yet'
    sentinel value, NOT a 'neutral verdict' — user feedback itself stays
    binary in chunk_feedback). Old rows then contribute to the model fit
    via the other 4 features without polluting sf/sl with a fake signal.
    New logs have all 6.
    """
    if not isinstance(feats, dict):
        return None
    out = {
        "v":  float(feats.get("v",  0.0) or 0.0),
        "s":  float(feats.get("s",  0.0) or 0.0),
        "r":  float(feats.get("r",  0.0) or 0.0),
        "a":  float(feats.get("a",  0.0) or 0.0),
        "sf": float(feats.get("sf", 0.5) if "sf" in feats else 0.5),
        "sl": float(feats.get("sl", 0.5) if "sl" in feats else 0.5),
    }
    return out


def export(
    db_path: Path, out: Path, days: int, negative_mode: str,
) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        labels = _label_map(conn, days)
        rows = conn.execute(
            "SELECT id, query, trigger_ids, stage_scores, captured_at, "
            "       workspace_id "
            "FROM context_feedback_log "
            "WHERE captured_at > datetime('now', ?) "
            "AND query != '' AND stage_scores != '{}' "
            "ORDER BY captured_at DESC",
            (f"-{days} days",),
        ).fetchall()
        written = 0
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                try:
                    chunks = json.loads(row["trigger_ids"])
                    stage = json.loads(row["stage_scores"])
                except (TypeError, ValueError):
                    continue
                for cid in chunks:
                    feats = _normalize_features(stage.get(cid))
                    if feats is None:
                        continue
                    lbl = labels.get(cid, 0)
                    if lbl == -1:
                        label = 0
                    elif lbl == 1:
                        label = 1
                    elif negative_mode == "explicit":
                        continue
                    else:
                        label = 0
                    f.write(json.dumps({
                        "query":        row["query"],
                        "chunk_id":     cid,
                        "features":     feats,
                        "label":        label,
                        "captured_at":  row["captured_at"],
                        "workspace_id": row["workspace_id"] or "",
                    }) + "\n")
                    written += 1
        return written
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(CACHE_DIR / "memory.db"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument(
        "--negative-mode", choices=["unlabeled", "explicit"],
        default="unlabeled",
    )
    args = ap.parse_args()
    db_path = Path(args.db)
    out_path = Path(args.out)
    if not db_path.exists():
        print(f"db not found: {db_path}")
        return 2
    n = export(db_path, out_path, args.days, args.negative_mode)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] wrote {n} rows → {out_path} "
          f"(window={args.days}d, neg_mode={args.negative_mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
