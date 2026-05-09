"""Memory-Injection v2.0 reranker — runtime model loader + scorer.

Loads ``cache/rerank_v2.json`` (produced by ``tools/train_reranker.py``)
and exposes:

* ``get_active_reranker(query_hint)`` — decides which version to use
  for a given request.
* ``score_v2(stage_scores, model)`` — applies the learned weights to
  the 5 stage-score features (v, s, r, a, f) and returns a sigmoid
  probability in [0, 1] for ranking.

Selection precedence:

    RERANKER_VERSION=v1  →   always v1
    RERANKER_VERSION=v2  →   v2 if model file present, else v1 (silent fallback)
    RERANKER_VERSION=auto →  hash(query_hint) % 2 split (50/50 A/B)
    unset                 →  v1 (default — no surprise behaviour change)

The model JSON is loaded once and cached in process; ``invalidate_v2_cache()``
forces a reload after a fresh training run.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# Memory-Injection v2.0 — explicit per-stage features. ``f`` (score_final)
# was the original 5th feature but caused multi-collinearity (negative
# weight on ``v`` because ``f`` already contained 0.30*v). Replaced with
# ``sf`` (feedback signal) + ``sl`` (LLM advisor) so the model gets the
# full v1 signal set without any single feature being a sum of the others.
_FEATURES = ("v", "s", "r", "a", "sf", "sl")
_LOCK = threading.Lock()
_CACHED_MODEL: dict[str, Any] | None = None
_CACHED_MTIME: float = 0.0


def _model_path() -> Path:
    from src.config import CACHE_DIR
    return CACHE_DIR / "rerank_v2.json"


def _load_model() -> dict[str, Any] | None:
    """Return the v2 model dict if the file exists and parses, else None.
    Result is cached based on file mtime so retraining picks up cleanly
    on the next call without restart."""
    global _CACHED_MODEL, _CACHED_MTIME
    path = _model_path()
    try:
        st = path.stat()
    except (FileNotFoundError, OSError):
        with _LOCK:
            _CACHED_MODEL = None
            _CACHED_MTIME = 0.0
        return None
    if _CACHED_MODEL is not None and st.st_mtime == _CACHED_MTIME:
        return _CACHED_MODEL
    with _LOCK:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            _log.warning("rerank_v2.json unreadable: %s — falling back to v1", e)
            _CACHED_MODEL = None
            _CACHED_MTIME = st.st_mtime
            return None
        if not isinstance(data, dict) or "weights" not in data:
            _log.warning("rerank_v2.json malformed — falling back to v1")
            _CACHED_MODEL = None
            _CACHED_MTIME = st.st_mtime
            return None
        # Sanity-check: ein gelerntes Modell mit NEGATIVEM Vector- ODER
        # Symbolic-Gewicht ist degeneriert. Vector-Similarity und
        # Symbolic-Token-Overlap sind beides retrieval-positive Signale —
        # negative Gewichte heißen das Modell hat etwas Unsinniges aus
        # collinearen Features oder feedback-Lecks gelernt (gesehen bei
        # n_train=1732, sf_weight=8.77, v_weight=-0.51 → 'symbolic-only'-
        # ranking, vector-Treffer wurden RUNTER sortiert). Lieber sauber
        # auf v1 fallen als degenerierte rankings ausliefern. Trainings-
        # bug muss am train_reranker.py-loop behoben werden, nicht hier.
        weights = data.get("weights") or {}
        v_w = float(weights.get("v") or 0.0)
        s_w = float(weights.get("s") or 0.0)
        if v_w < 0 or s_w < 0:
            _log.error(
                "rerank_v2.json degenerate (v=%.3f, s=%.3f); refusing to load. "
                "Reject training runs with negative vector/symbolic weights — "
                "those are retrieval-positive features.", v_w, s_w,
            )
            _CACHED_MODEL = None
            _CACHED_MTIME = st.st_mtime
            return None
        _CACHED_MODEL = data
        _CACHED_MTIME = st.st_mtime
        return data


def invalidate_v2_cache() -> None:
    """Force a reload on the next call. Call after a training run."""
    global _CACHED_MODEL, _CACHED_MTIME
    with _LOCK:
        _CACHED_MODEL = None
        _CACHED_MTIME = 0.0


def _ab_pick(query_hint: str | None) -> str:
    """50/50 deterministic split. Same query routes to the same version
    so A/B comparisons stay stable across paginated calls."""
    h = hashlib.sha256((query_hint or "").encode("utf-8")).hexdigest()
    return "v2" if int(h[:8], 16) % 2 == 0 else "v1"


def _default_state_path() -> Path:
    from src.config import CACHE_DIR
    return CACHE_DIR / "rerank_default.txt"


def _read_runtime_default() -> str:
    """Persisted default decided by the auto-rollout cron.

    Lives in cache/rerank_default.txt because env vars need a restart
    to take effect; a tiny file lets the cron flip the default
    runtime without touching docker-compose. Values: 'auto' / 'v1' /
    'v2'. Default 'auto' so the system runs a 50/50 A/B from day one
    rather than silently sitting on v1 forever.
    """
    p = _default_state_path()
    try:
        v = p.read_text(encoding="utf-8").strip().lower()
        if v in ("v1", "v2", "auto"):
            return v
    except (OSError, FileNotFoundError):
        pass
    return "auto"


def write_runtime_default(version: str) -> str:
    """Set the persisted default. Used by the auto-rollout cron after
    it sees a 25%+ uplift. Returns the value actually written."""
    if version not in ("v1", "v2", "auto"):
        raise ValueError(f"invalid default version: {version!r}")
    p = _default_state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(version, encoding="utf-8")
    return version


def get_active_reranker(
    query_hint: str | None = None,
    explicit_override: str | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Resolve which reranker version to use for THIS call.

    Precedence (highest first):
      1. ``explicit_override`` from request body (?reranker=v2)
      2. ``RERANKER_VERSION`` env var
      3. ``cache/rerank_default.txt`` runtime state file
         (the auto-rollout cron flips this on 25%+ uplift)
      4. fallback: ``auto`` (50/50 A/B)

    Args:
        query_hint: Stable string (query text or session id) that A/B
            mode hashes for the deterministic split.
        explicit_override: Per-request override ('v1' / 'v2' / 'auto').

    Returns:
        (version, model_dict | None)
    """
    raw = (
        explicit_override
        or os.getenv("RERANKER_VERSION")
        or _read_runtime_default()
        or "auto"
    ).lower()
    if raw not in ("v1", "v2", "auto"):
        raw = "auto"
    if raw == "auto":
        raw = _ab_pick(query_hint)
    if raw == "v2":
        model = _load_model()
        if model is None:
            return "v1", None  # silent fallback — never crash on missing model
        return "v2", model
    return "v1", None


def score_v2(stage_scores: dict[str, float], model: dict[str, Any]) -> float:
    """Apply learned weights to {v,s,r,a,f}. Returns probability in [0,1]
    via sigmoid so the score is comparable to v1's [0,1]-ish range."""
    weights = model.get("weights", {})
    intercept = float(model.get("intercept", 0.0))
    z = intercept
    for f in _FEATURES:
        z += float(weights.get(f, 0.0)) * float(stage_scores.get(f, 0.0))
    if z > 30:
        return 1.0
    if z < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def model_summary() -> dict[str, Any]:
    """For /stats/admin endpoints — quick view of the active model."""
    model = _load_model()
    if model is None:
        return {"active": False, "reason": "no rerank_v2.json"}
    return {
        "active": True,
        "version": model.get("version"),
        "trained_at": model.get("trained_at"),
        "n_train": model.get("n_train"),
        "n_test": model.get("n_test"),
        "metrics": model.get("metrics"),
        "weights": model.get("weights"),
        "intercept": model.get("intercept"),
    }
