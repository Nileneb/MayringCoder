"""Train a learned reranker on retrieval feedback data.

Reads JSONL produced by ``tools/export_retrieval_dataset.py`` (rows with
features {v,s,r,a,f} + binary label), trains logistic regression, writes
the calibrated weights to ``cache/rerank_v2.json``. The runtime then
loads those weights via env flag ``RERANKER_VERSION=v2`` to replace the
hand-tuned ``_WEIGHTS`` dict in ``src/memory/retrieval.py``.

The model is intentionally tiny — five features, linear, no embeddings.
With ~hundreds of feedback rows this is the right capacity. Replace the
estimator later when data volume justifies it.

Output schema (cache/rerank_v2.json):

    {
      "version": "v2",
      "estimator": "logistic_regression",
      "trained_at": "...",
      "n_train": 412,
      "n_test": 104,
      "metrics": {"auc": 0.71, "p@1": 0.52, "ndcg@5": 0.58},
      "weights": {"v": 1.27, "s": 0.41, "r": 0.05, "a": 0.18, "f": 0.62},
      "intercept": -0.31
    }

Usage:
    python tools/train_reranker.py
        --in cache/finetuning/retrieval_dataset.jsonl
        --out cache/rerank_v2.json

Requires sklearn — install via ``pip install scikit-learn`` if missing.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from mayring_core.config import CACHE_DIR

DEFAULT_IN = CACHE_DIR / "finetuning" / "retrieval_dataset.jsonl"
DEFAULT_OUT = CACHE_DIR / "rerank_v2.json"
# Memory-Injection v3 features (Issue #180 fix).
# Dropped from old (v2) feature set:
#   * `f`: linear combination of others → multikollin → degenerate weights
#   * `sf`: target leakage. score_feedback is computed FROM the same
#     chunk_feedback table that v2's chunk-global label was derived from.
#     v2 learned sf=8.77, v=-0.51 — i.e. "use feedback as proxy for
#     label, then negative-correct everything else". Removing sf forces
#     the model to learn from actual retrieval signals.
#   * `sl`: LLM-advisor score. ~70% of rows have sl=0.5 (default 'no
#     signal yet') — almost constant, low information.
# Added: igio_<axis> one-hot. IGIO axis (issue/goal/intervention/
# outcome) is a real semantic discriminator missed by vector similarity
# — empirically lifts AUC by +0.03 in offline eval.
IGIO_AXES = ("issue", "goal", "intervention", "outcome", "unknown")
# WHY(#187): pt + re sind seit commit 46e9c2e/c9db1bf live im API-Response
# (score_predicted_topic, rationale_edges). Vorher: phantom-features ohne
# Trainings-Pfad. Jetzt im Trainings-Set, sodass v2-Modell sie als Gewicht
# lernen kann.
FEATURES = ("v", "s", "r", "a", "pt", "re") + tuple(f"igio_{a}" for a in IGIO_AXES)
MIN_ROWS = 50
MIN_POSITIVES = 10


def _load(
    path: Path,
) -> tuple[list[list[float]], list[int], list[float], list[str]]:
    """Return (X, y, sample_weight, chunk_ids).

    sample_weight per row (#209): rating 5★ = 1.0, 4★ = 0.5, 2★ = 0.5,
    1★ = 1.0 (negativ). Rows ohne explicit rating bekommen 1.0.
    Multi-rating chunks bekommen n-bonus (1 + 0.1*(n-1), clipped).
    """
    X: list[list[float]] = []
    y: list[int] = []
    w: list[float] = []
    cids: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            feats = row.get("features") or {}
            X.append([float(feats.get(k, 0.0)) for k in FEATURES])
            y.append(int(row.get("label", 0)))
            w.append(float(row.get("sample_weight", 1.0)))
            cids.append(row.get("chunk_id", ""))
    return X, y, w, cids


def _ndcg_at_k(labels: list[int], k: int) -> float:
    if not labels:
        return 0.0
    gains = labels[:k]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(labels, reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


def train(in_path: Path, out_path: Path) -> int:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("Fehler: sklearn nicht installiert (pip install scikit-learn)")
        return 2
    X, y, w, _cids = _load(in_path)
    if len(X) < MIN_ROWS:
        print(f"zu wenig Daten: {len(X)} rows < {MIN_ROWS} (warte auf mehr Feedback)")
        return 1
    pos = sum(y)
    if pos < MIN_POSITIVES:
        print(f"zu wenig Positives: {pos} < {MIN_POSITIVES}")
        return 1
    # WHY(#209): split inkl. sample_weight, sodass die ratings auch in
    # train/test korrekt mitgeführt werden.
    Xtr, Xte, ytr, yte, wtr, _wte = train_test_split(
        X, y, w, test_size=0.2, random_state=42,
        stratify=y if pos < len(y) else None,
    )
    # L2 (Ridge) mit C=0.1 = stark regularisiert. Multikollinearität
    # zwischen v↔s (corr=+0.51), v↔r (+0.43), s↔r (+0.66) führt unter
    # default-C zu willkürlichen Vorzeichen-Flips zwischen den
    # korrelierten Features (so wurde v=-0.51 in v2 gelernt). Mit
    # C=0.1 schmiert ridge das Gewicht über die korrelierten Features
    # → keine 'cancel-out' Vorzeichen mehr, IGIO-Features bekommen
    # ihren echten Effekt-Anteil. AUC nahezu identisch zu unregulierter
    # version; Weights sind interpretierbarer + multi-tenant-stabiler.
    clf = LogisticRegression(max_iter=500, class_weight="balanced", C=0.1)
    # WHY(#209): sample_weight führt rating-skala 1..5 als training-signal
    # ein. 5★-feedback bekommt doppelt so viel pull wie 4★. Zusammen mit
    # class_weight='balanced' (klassen-imbalance) ergibt sich effektives
    # gewicht = class_weight * sample_weight.
    clf.fit(Xtr, ytr, sample_weight=wtr)
    proba = clf.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, proba)) if len(set(yte)) > 1 else 0.0
    sorted_pairs = sorted(zip(proba, yte), key=lambda p: p[0], reverse=True)
    sorted_labels = [int(lbl) for _, lbl in sorted_pairs]
    p_at_1 = float(sum(sorted_labels[:1]) / max(1, len(sorted_labels[:1])))
    ndcg_5 = _ndcg_at_k(sorted_labels, 5)

    weights = {f: round(float(w), 4) for f, w in zip(FEATURES, clf.coef_[0])}

    # Trainer-side Sanity-Gate: lehne degenerierte Modelle SCHON HIER ab,
    # nicht erst beim Loader. Vector + Symbolic sind by-design retrieval-
    # positive Signale; ein Modell mit negativen Weights auf v oder s
    # hat aus den Daten Confounder gelernt (siehe Issue #180). Ohne
    # diese Prüfung schreibt der Cron stillschweigend ein Modell file,
    # das der runtime _load_model dann silent abweist → "Cron success,
    # Modell aber nicht aktiv" wäre täglich passiert ohne dass es jemand
    # merkt. Hier hart abbrechen + alte rerank_v2.json unangetastet.
    v_w = float(weights.get("v", 0.0))
    s_w = float(weights.get("s", 0.0))
    if v_w < 0 or s_w < 0:
        print(
            f"REJECTED: Modell hat negative retrieval-positive weights "
            f"(v={v_w:+.3f}, s={s_w:+.3f}). Schreibe NICHT auf {out_path}. "
            f"Bug in features oder labels — siehe Issue #180. "
            f"Volle Weights: {weights}"
        )
        return 3
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "version":     "v2",
            "estimator":   "logistic_regression",
            "trained_at":  datetime.now(timezone.utc).isoformat(),
            "n_train":     len(Xtr),
            "n_test":      len(Xte),
            "metrics":     {"auc": round(auc, 4), "p_at_1": round(p_at_1, 4),
                            "ndcg_at_5": ndcg_5},
            "weights":     weights,
            "intercept":   round(float(clf.intercept_[0]), 4),
        }, f, indent=2, sort_keys=True)
    print(f"v2 model written to {out_path}: weights={weights}, "
          f"auc={auc:.3f}, p@1={p_at_1:.3f}, ndcg@5={ndcg_5:.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    in_path = Path(args.inp)
    if not in_path.exists():
        print(f"input not found: {in_path} — run export_retrieval_dataset.py first")
        return 2
    return train(in_path, Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
