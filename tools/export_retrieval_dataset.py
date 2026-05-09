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

# 32% der Trainings-Events kommen aus Smoke-Tests / Task-Notifications /
# Marker-Tokens — die haben keinen User-Retrieval-Signal-Wert und ziehen
# das Modell in degenerierte Pattern (z.B. negative Vector-Weights weil
# die Smoke-Test-Patterns nicht-vector-typisch sind). Filter NUR diese
# offensichtlich synthetischen Events; alles andere bleibt drin.
NOISE_QUERY_PATTERNS = (
    "smoke %", "marker token%", "<task-notification>%",
    "reranker rollback%", "_rerank candidates%", "fix bug",
    "reasons probe%", "smoke watcher%", "smoke reasons probe%",
    "smoke check %",
)


def _igio_axis_map(conn: sqlite3.Connection) -> dict[str, str]:
    """Return {chunk_id → igio_axis} for chunks with a classified axis.

    IGIO axis (issue/goal/intervention/outcome) is a strong retrieval
    signal that vector similarity misses — outcome chunks get
    referenced ~6× more often than intervention chunks in real
    user-feedback data. Adding it as a feature lifted v4 model AUC
    from 0.73 to 0.76 in offline eval.
    """
    out = {}
    for cid, axis in conn.execute(
        "SELECT chunk_id, igio_axis FROM chunks "
        "WHERE igio_axis IS NOT NULL AND igio_axis != ''"
    ):
        if axis:
            out[cid] = axis
    return out


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


# IGIO axes — one-hot encoded into features. 'unknown' covers chunks
# without a classified axis (~92% today; backfill cron drives this down).
IGIO_AXES = ("issue", "goal", "intervention", "outcome", "unknown")
# WHY(#187): pt (predicted-topic-boost) und re (rationale-presence) sind seit
# commit 46e9c2e/c9db1bf live im API-Response, aber bislang nicht im Trainer
# — Phantom-Features. Hier zur FEATURES_OUT hinzu damit der nächste
# train_reranker.py-Run sie als Eingabe sieht und ein Gewicht lernt.
FEATURES_OUT = ("v", "s", "r", "a", "pt", "re") + tuple(f"igio_{a}" for a in IGIO_AXES)


def _normalize_features(
    feats: dict,
    chunk_id: str,
    igio_map: dict[str, str],
) -> dict | None:
    """Return retrieval features + IGIO one-hot for one (event, chunk) row.

    Dropped vs the legacy 6-feature set:
      * `sf` and `sl`: target leakage. `sf` is computed from
        chunk_feedback which is the same source as the label — model
        was learning sf as proxy for the label (sf weight 8.77 in v2).
      * `f` (score_final): linear combination of others, multikollin.

    Added:
      * `igio_<axis>` one-hot: outcome-axis chunks get referenced ~6×
        more often than intervention-axis chunks; a real retrieval
        signal that pure vector similarity does not capture.
    """
    if not isinstance(feats, dict):
        return None
    axis = igio_map.get(chunk_id, "unknown")
    out = {
        "v": float(feats.get("v", 0.0) or 0.0),
        "s": float(feats.get("s", 0.0) or 0.0),
        "r": float(feats.get("r", 0.0) or 0.0),
        "a": float(feats.get("a", 0.0) or 0.0),
        # Issue #187: pt = score_predicted_topic, re = 1.0 wenn der chunk
        # eine rationale-edge hatte. Defaults 0.0 für Backward-Compat mit
        # alten context_feedback_log-rows (die sf/sl/pt/re nicht hatten).
        "pt": float(feats.get("pt", 0.0) or 0.0),
        "re": float(feats.get("re", 0.0) or 0.0),
    }
    for a in IGIO_AXES:
        out[f"igio_{a}"] = 1.0 if axis == a else 0.0
    return out


def export(
    db_path: Path, out: Path, days: int, negative_mode: str,
) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        igio_map = _igio_axis_map(conn)
        # Label aus dem PER-EVENT-Signal `was_referenced` statt aus dem
        # GLOBALEN chunk_feedback-Aggregat. Das alte _label_map(conn,days)
        # gab pro chunk_id ein binary "irgendwo schonmal positives
        # Feedback gehabt" — das machte 9 chunks für 42% aller Positives
        # verantwortlich (Modell lernte chunk-id-leaderboard) UND
        # produzierte sf↔label-Leakage (sf wird aus selber chunk_feedback
        # Tabelle berechnet wie das Label).
        # was_referenced ist event-level: hat der Assistant nach
        # Injection eines chunks dessen source_id in der Antwort
        # referenziert. Nicht perfekt (event-level statt chunk-level),
        # aber kein Target-Leakage und keine chunk-id-Konzentration.
        noise_filter = " AND ".join(
            "query NOT LIKE ?" for _ in NOISE_QUERY_PATTERNS
        )
        rows = conn.execute(
            "SELECT id, query, trigger_ids, stage_scores, was_referenced, "
            "       captured_at, workspace_id "
            "FROM context_feedback_log "
            "WHERE captured_at > datetime('now', ?) "
            f"AND query != '' AND stage_scores != '{{}}' "
            f"AND {noise_filter} "
            "ORDER BY captured_at DESC",
            (f"-{days} days", *NOISE_QUERY_PATTERNS),
        ).fetchall()
        written = 0
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                try:
                    chunks = json.loads(row["trigger_ids"])
                    stage = json.loads(row["stage_scores"])
                except (TypeError, ValueError):
                    continue
                ev_label = int(row["was_referenced"] or 0)
                if ev_label == 0 and negative_mode == "explicit":
                    continue
                for cid in chunks:
                    feats = _normalize_features(stage.get(cid), cid, igio_map)
                    if feats is None:
                        continue
                    f.write(json.dumps({
                        "query":        row["query"],
                        "chunk_id":     cid,
                        "features":     feats,
                        "label":        ev_label,
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
