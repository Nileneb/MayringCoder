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
        --negative-mode explicit

Negative-mode determines how unlabeled chunks become training negatives:
  * ``explicit``   — only chunks with explicit negativ-Rating (1-2★) ODER
                     was_referenced=1 als Positive. Hochpräzise, menschen-gelabelt.
  * ``unlabeled``  — ALLE top-K-Chunks ohne positives Feedback werden Negative.

WHY(2026-06-05) Default ``explicit`` (vorher ``unlabeled``): top-K-Chunks haben PER
DEFINITION hohe Vektor-Ähnlichkeit (sie wurden ja retrieved). Sie pauschal als
Negative zu labeln nur weil sie nicht in der Antwort zitiert wurden (was_referenced=0
heißt NICHT irrelevant — Kontext kann die Antwort speisen ohne zitiert zu werden)
lehrt das Modell „hohe v → negativ" → das v-Gewicht wird NEGATIV → der Loader rejected
das ganze Modell (v/s-hard-gate, #180) → v2 ging nie live. Mensch-gelabelte
Negative (Rating 1-2★) haben diesen Bias nicht. ``unlabeled`` nur noch für
Volume-Experimente, idealerweise kombiniert mit ``--span-judge`` (LLM-Relabeling).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from mayring_core.config import CACHE_DIR

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


def _label_map(
    conn: sqlite3.Connection, days: int
) -> dict[str, tuple[int, float]]:
    """{chunk_id: (label, sample_weight)} from rating 1..5 feedback.

    WHY(#209, rating-weighted): vor diesem fix wurde rating 1..5 binär
    kollabiert — "5★ primary source" und "1★ irrelevant" wurden gleich
    behandelt (beide → pos), und neutrale 3★ leakten als pos-signal rein.
    Resultat: model lernt nichts aus der Skala.

    Neuer mapping:
      rating 5  → label 1, weight 1.0  (starkes positiv)
      rating 4  → label 1, weight 0.5
      rating 3  → omit (neutral)
      rating 2  → label 0, weight 0.5
      rating 1  → label 0, weight 1.0  (starkes negativ)

    Bei mehreren ratings: avg(rating) entscheidet, weight wird mit
    n-bonus (1 + 0.1*(n-1)) skaliert sodass mehr feedback = höheres
    confidence. Legacy-signals werden auf rating-äquivalente gemappt:
    positive → 4, negative → 2 (default weight 0.5 weil low confidence).
    """
    rows = conn.execute(
        "SELECT chunk_id, signal FROM chunk_feedback "
        "WHERE created_at > datetime('now', ?)",
        (f"-{days} days",),
    ).fetchall()

    chunk_ratings: dict[str, list[float]] = {}
    for cid, sig in rows:
        if sig in ("1", "2", "3", "4", "5"):
            chunk_ratings.setdefault(cid, []).append(float(sig))
        elif sig == "positive":
            chunk_ratings.setdefault(cid, []).append(4.0)
        elif sig == "negative":
            chunk_ratings.setdefault(cid, []).append(2.0)

    out: dict[str, tuple[int, float]] = {}
    for cid, ratings in chunk_ratings.items():
        avg = sum(ratings) / len(ratings)
        n_bonus = 1 + 0.1 * (len(ratings) - 1)

        if avg >= 4.5:
            out[cid] = (1, 1.0 * n_bonus)
        elif avg >= 3.5:
            out[cid] = (1, 0.5 * n_bonus)
        elif avg <= 1.5:
            out[cid] = (0, 1.0 * n_bonus)
        elif avg <= 2.5:
            out[cid] = (0, 0.5 * n_bonus)
        # avg in (2.5, 3.5) → omit (neutral, no training signal)
    return out


# IGIO axes — one-hot encoded into features. 'unknown' covers chunks
# without a classified axis (~92% today; backfill cron drives this down).
IGIO_AXES = ("issue", "goal", "intervention", "outcome", "unknown")
# WHY(#187/2026-06-05): pt (predicted-topic) ist live im API-Response UND seit
# 2026-06-05 im stage-Dict am Inferenz-Pfad → echtes lernbares Feature. re
# (rationale-presence) RAUS: erst nach dem Scoring bekannt (wiki_edges-Query pro
# Chunk), nie im stage-Dict → trainiert-aber-nie-genutzt + sein Negativ-Gewicht
# (-0.67) rejectete das ganze v2-Modell. Nicht mehr exportieren.
FEATURES_OUT = ("v", "s", "r", "a", "pt") + tuple(f"igio_{a}" for a in IGIO_AXES)

# Span-Judge-Schwellen (Offline-Teacher, #SSA): nur Rows OHNE explizites
# Human-Rating werden anhand des LLM-Relevanz-Scores korrigiert. Der
# Score ist KEIN Modell-Feature (Trainer liest nur row["features"]) —
# er verfeinert Label/Sample-Weight, damit der günstige Linear-Reranker
# auf saubererem Signal lernt. Siehe tools/span_judge.py.
SPAN_LOW = 0.25
SPAN_HIGH = 0.75

# span_judge_fn: (conn, query, chunk_ids) -> {chunk_id: score 0..1}.
# Injizierbar für Tests; in der CLI gebunden an span_judge.scores_for_query.
SpanJudgeFn = Callable[[sqlite3.Connection, str, "list[str]"], "dict[str, float]"]


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
    span_judge_fn: "SpanJudgeFn | None" = None,
) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        igio_map = _igio_axis_map(conn)
        # WHY(#209, rating-weighted): chunks die EXPLIZITES rating-feedback
        # haben bekommen einen sample_weight basierend auf der rating-skala
        # (5★ = strong positive, 1★ = strong negative). Chunks ohne explicit
        # rating bleiben bei weight=1.0 (default).
        # Das gibt dem trainer mehr signal aus der 5-stufen-skala ohne sie
        # ins label zu mappen (das bleibt was_referenced, kein leakage).
        rating_weights = _label_map(conn, days)
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
        total = len(rows)
        # Emit ~25 PROGRESS markers — the API runner parses them into the live
        # frontend bar. Per-event granularity matters most with --span-judge
        # (one Ollama call/event = the slow part).
        progress_every = max(1, total // 25)
        with out.open("w", encoding="utf-8") as f:
            for ei, row in enumerate(rows):
                if ei % progress_every == 0 or ei == total - 1:
                    print(f"PROGRESS {ei + 1}/{total}", flush=True)
                try:
                    chunks = json.loads(row["trigger_ids"])
                    stage = json.loads(row["stage_scores"])
                except (TypeError, ValueError):
                    continue
                ev_label = int(row["was_referenced"] or 0)
                # span_judge darf False-Negatives (was_referenced=0)
                # promoten, daher im Judge-Modus nicht vorab skippen.
                if (ev_label == 0 and negative_mode == "explicit"
                        and span_judge_fn is None):
                    continue
                span_scores: dict[str, float] = {}
                if span_judge_fn is not None:
                    span_scores = span_judge_fn(
                        conn, row["query"], list(chunks)
                    ) or {}
                for cid in chunks:
                    feats = _normalize_features(stage.get(cid), cid, igio_map)
                    if feats is None:
                        continue
                    # WHY(#209): rating-weight (1..5★) als sample_weight.
                    # explicit rating → weight 0.5..1.1; ohne rating → 1.0.
                    # Plus: bei Konflikt zwischen was_referenced=0 und
                    # avg_rating>=4 (chunk wurde NICHT referenziert in der
                    # antwort, aber user hat ihn später als 5★ gerated):
                    # rating überstimmt label, weil der user die ground
                    # truth ist (was_referenced ist nur ein proxy).
                    rating_info = rating_weights.get(cid)
                    span_score = span_scores.get(cid)
                    final_label = ev_label
                    final_weight = 1.0
                    if rating_info is not None:
                        rating_label, rating_weight = rating_info
                        final_label = rating_label
                        final_weight = rating_weight
                    elif span_score is not None:
                        # Span-Judge verfeinert NUR Rows ohne explizites
                        # Rating (Human-Rating bleibt Ground-Truth).
                        # False-Positive (referenziert, Judge sagt
                        # irrelevant) → down-weight. False-Negative (nicht
                        # referenziert, Judge sagt hoch-relevant) → als
                        # positiv mit moderatem Gewicht aufnehmen.
                        if ev_label == 1 and span_score < SPAN_LOW:
                            final_weight = 0.3
                        elif ev_label == 0 and span_score > SPAN_HIGH:
                            final_label = 1
                            final_weight = 0.5
                    record = {
                        "query":         row["query"],
                        "chunk_id":      cid,
                        "features":      feats,
                        "label":         final_label,
                        "sample_weight": final_weight,
                        "captured_at":   row["captured_at"],
                        "workspace_id":  row["workspace_id"] or "",
                    }
                    if span_judge_fn is not None:
                        # Top-Level-Key, NICHT in features: der Trainer
                        # liest row["features"] und ignoriert das hier.
                        # Nur für Analyse/Debug der Refinement-Wirkung.
                        record["span_score"] = span_score
                    f.write(json.dumps(record) + "\n")
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
        default="explicit",  # WHY(2026-06-05): unlabeled labelte high-v top-K als Negative → v<0 → Modell-Reject
    )
    ap.add_argument(
        "--span-judge", action="store_true",
        help="use the offline LLM relevance judge (Ollama) to refine "
             "noisy was_referenced labels — see tools/span_judge.py",
    )
    args = ap.parse_args()
    db_path = Path(args.db)
    out_path = Path(args.out)
    if not db_path.exists():
        print(f"db not found: {db_path}")
        return 2
    span_judge_fn = None
    if args.span_judge:
        try:
            import span_judge as _span_judge
        except ImportError:
            from tools import span_judge as _span_judge
        span_judge_fn = _span_judge.scores_for_query
    n = export(db_path, out_path, args.days, args.negative_mode,
               span_judge_fn=span_judge_fn)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] wrote {n} rows → {out_path} "
          f"(window={args.days}d, neg_mode={args.negative_mode}, "
          f"span_judge={'on' if args.span_judge else 'off'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
