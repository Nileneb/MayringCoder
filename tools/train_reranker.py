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

from src.config import CACHE_DIR

DEFAULT_IN = CACHE_DIR / "finetuning" / "retrieval_dataset.jsonl"
DEFAULT_OUT = CACHE_DIR / "rerank_v2.json"
# 6 features explicitly logged in stage_scores. The earlier 5-feature
# set used ``f`` (score_final) which is a linear combination of the
# others — LogReg learned a negative weight on ``v`` to cancel f's
# vector contribution, making the weights uninterpretable. Replacing
# ``f`` with the two missing direct signals (sf, sl) breaks that.
FEATURES = ("v", "s", "r", "a", "sf", "sl")
MIN_ROWS = 50
MIN_POSITIVES = 10


def _load(path: Path) -> tuple[list[list[float]], list[int], list[str]]:
    X: list[list[float]] = []
    y: list[int] = []
    cids: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            feats = row.get("features") or {}
            X.append([float(feats.get(k, 0.0)) for k in FEATURES])
            y.append(int(row.get("label", 0)))
            cids.append(row.get("chunk_id", ""))
    return X, y, cids


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
    X, y, _cids = _load(in_path)
    if len(X) < MIN_ROWS:
        print(f"zu wenig Daten: {len(X)} rows < {MIN_ROWS} (warte auf mehr Feedback)")
        return 1
    pos = sum(y)
    if pos < MIN_POSITIVES:
        print(f"zu wenig Positives: {pos} < {MIN_POSITIVES}")
        return 1
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42,
                                          stratify=y if pos < len(y) else None)
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, proba)) if len(set(yte)) > 1 else 0.0
    sorted_pairs = sorted(zip(proba, yte), key=lambda p: p[0], reverse=True)
    sorted_labels = [int(lbl) for _, lbl in sorted_pairs]
    p_at_1 = float(sum(sorted_labels[:1]) / max(1, len(sorted_labels[:1])))
    ndcg_5 = _ndcg_at_k(sorted_labels, 5)

    weights = {f: round(float(w), 4) for f, w in zip(FEATURES, clf.coef_[0])}
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
