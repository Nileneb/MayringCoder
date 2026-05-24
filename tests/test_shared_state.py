"""shared_state: fail-soft Redis L2 for cross-worker dashboard consistency (§5.3).

Covers both modes:
  * No Redis (MAYRING_REDIS_URL unset) → cache is a no-op, activations_redis None
    (so call sites transparently fall back to their L1 dict/deque). This is the
    mode unit tests + a Redis outage run in.
  * Redis present (injected fake client) → reads/writes go to the shared store.
"""
from __future__ import annotations

import importlib
import json

import pytest


@pytest.fixture
def ss(monkeypatch):
    import src.api.shared_state as ss
    # reset module state for isolation
    monkeypatch.setattr(ss, "_client", None, raising=False)
    monkeypatch.setattr(ss, "_next_retry", 0.0, raising=False)
    monkeypatch.setattr(ss, "_degraded_logged", False, raising=False)
    return ss


def test_no_redis_is_noop(ss, monkeypatch):
    monkeypatch.delenv("MAYRING_REDIS_URL", raising=False)
    assert ss.cache_get("k") is None
    ss.cache_set("k", {"v": 1}, 30)        # no raise
    assert ss.cache_get("k") is None        # nothing stored (no-op)
    ss.cache_del("k")                        # no raise
    ss.activation_push({"q": "x"})          # no raise
    assert ss.activations_redis() is None    # signal: caller uses local deque


class _FakeRedis:
    def __init__(self):
        self.kv: dict[str, str] = {}
        self.lists: dict[str, list] = {}

    def get(self, k):
        return self.kv.get(k)

    def set(self, k, v, ex=None):
        self.kv[k] = v

    def delete(self, k):
        self.kv.pop(k, None)

    def lpush(self, k, v):
        self.lists.setdefault(k, []).insert(0, v)

    def ltrim(self, k, start, end):
        self.lists[k] = self.lists.get(k, [])[start:end + 1]

    def lrange(self, k, start, end):
        return self.lists.get(k, [])[start:end + 1]


def test_redis_cache_roundtrip(ss, monkeypatch):
    monkeypatch.setattr(ss, "_client", _FakeRedis(), raising=False)
    ss.cache_set("stats:summary", {"chunks": 5}, 30)
    assert ss.cache_get("stats:summary") == {"chunks": 5}
    ss.cache_del("stats:summary")
    assert ss.cache_get("stats:summary") is None


def test_redis_activations_shared(ss, monkeypatch):
    monkeypatch.setattr(ss, "_client", _FakeRedis(), raising=False)
    ss.activation_push({"query": "a", "ts": 1})
    ss.activation_push({"query": "b", "ts": 2})
    acts = ss.activations_redis()
    assert acts is not None and len(acts) == 2
    assert acts[0]["query"] == "b"  # newest-first (lpush)


def test_redis_error_degrades_to_none(ss, monkeypatch):
    class _Boom:
        def get(self, k): raise RuntimeError("conn lost")
        def set(self, *a, **k): raise RuntimeError("conn lost")
    monkeypatch.setattr(ss, "_client", _Boom(), raising=False)
    assert ss.cache_get("k") is None      # swallowed → None (logged once, not raised)
    ss.cache_set("k", 1, 30)               # no raise
