"""Cross-worker shared state via Redis (api-concurrency-capacity §5.3 / Phase 3).

WHY: with ``uvicorn --workers N`` each worker has its OWN in-process caches
(``_STATS_CACHE``, ``_DASH_CACHE``) and activation ring buffer
(``_RECENT_ACTIVATIONS``). A dashboard poll is round-robined across workers, so
it sees a different worker's cache/ring each time → numbers flicker and the
activations widget only shows ~1/N of activity. Redis gives all workers ONE
shared view.

Design — additive L2, fail-soft:
  * The call sites KEEP their per-process L1 caches (so a Redis outage degrades
    to today's per-worker behaviour, not a SQLite flood, and so unit tests with
    no Redis are unchanged).
  * Redis is the shared L2: authoritative when reachable.
  * cache_*  : redis up → read/write Redis; redis down → no-op (caller's L1 +
               recompute handle it). cache_get returns None on miss OR no-redis.
  * activation_push / activations_redis: redis up → Redis list; redis down →
    push is a no-op and activations_redis returns None so the caller falls back
    to its local deque.

Connection is lazy with a short retry window, so a deploy where Redis boots a
few seconds after the API self-heals. Socket timeouts are 0.5s so a Redis hiccup
never stalls the request path.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

_log = logging.getLogger(__name__)

_NS = "mayring:"
_lock = threading.Lock()
_client: Any = None
_next_retry = 0.0
_degraded_logged = False


def _get_client() -> Any:
    global _client, _next_retry
    if _client is not None:
        return _client
    now = time.monotonic()
    if now < _next_retry:
        return None
    with _lock:
        if _client is not None:
            return _client
        if time.monotonic() < _next_retry:
            return None
        url = os.getenv("MAYRING_REDIS_URL", "")
        if not url:
            _next_retry = time.monotonic() + 60.0
            return None
        try:
            import redis
            c = redis.Redis.from_url(
                url, decode_responses=True,
                socket_connect_timeout=0.5, socket_timeout=0.5,
            )
            c.ping()
            _client = c
            return c
        except Exception as e:  # noqa: BLE001 — any redis/connection error → degrade
            _next_retry = time.monotonic() + 5.0
            _degrade(e)
            return None


def _degrade(exc: Exception) -> None:
    global _degraded_logged
    if not _degraded_logged:
        _log.warning(
            "shared_state: Redis unavailable (%s) — per-process fallback "
            "(dashboards may differ across workers until Redis is back)", exc,
        )
        _degraded_logged = True


# ---------------------------------------------------------------------------
# Shared TTL cache (L2). Call sites keep their own L1 dict.
# ---------------------------------------------------------------------------

def cache_get(key: str) -> Any | None:
    """Shared cache read. None = miss OR Redis unavailable (caller checks L1)."""
    c = _get_client()
    if c is None:
        return None
    try:
        raw = c.get(_NS + "cache:" + key)
        return json.loads(raw) if raw is not None else None
    except Exception as e:  # noqa: BLE001
        _degrade(e)
        return None


def cache_set(key: str, value: Any, ttl_s: float) -> None:
    c = _get_client()
    if c is None:
        return
    try:
        c.set(_NS + "cache:" + key, json.dumps(value, default=str), ex=max(1, int(ttl_s)))
    except Exception as e:  # noqa: BLE001
        _degrade(e)


def cache_del(key: str) -> None:
    c = _get_client()
    if c is None:
        return
    try:
        c.delete(_NS + "cache:" + key)
    except Exception as e:  # noqa: BLE001
        _degrade(e)


# ---------------------------------------------------------------------------
# Shared activation ring (L2). Caller keeps a local deque as the L1 fallback.
# ---------------------------------------------------------------------------

_ACT_KEY = _NS + "activations"
_ACT_MAX = 200


def activation_push(ev: dict) -> None:
    """Write-through a search activation to the shared ring (no-op if no Redis)."""
    c = _get_client()
    if c is None:
        return
    try:
        c.lpush(_ACT_KEY, json.dumps(ev, default=str))
        c.ltrim(_ACT_KEY, 0, _ACT_MAX - 1)
    except Exception as e:  # noqa: BLE001
        _degrade(e)


def activations_redis() -> list[dict] | None:
    """Shared activations, newest-first. None = Redis unavailable → use local deque."""
    c = _get_client()
    if c is None:
        return None
    try:
        return [json.loads(x) for x in c.lrange(_ACT_KEY, 0, _ACT_MAX - 1)]
    except Exception as e:  # noqa: BLE001
        _degrade(e)
        return None
