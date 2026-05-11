"""Tests for /stats/summary TTL-cache + stale-fallback (2026-05-11 fix)."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest
from fastapi import HTTPException


@pytest.fixture(autouse=True)
def reset_cache():
    from src.api.server import _STATS_CACHE
    _STATS_CACHE["fresh"] = None
    _STATS_CACHE["stale"] = None
    _STATS_CACHE["expires_at"] = 0.0
    yield
    _STATS_CACHE["fresh"] = None
    _STATS_CACHE["stale"] = None
    _STATS_CACHE["expires_at"] = 0.0


def _fake_uncached_response():
    return {
        "chunks": {"active": 100, "total": 120},
        "sources": {"count": 30},
        "feedback": {"stars": {}, "total": 0, "avg": 0},
        "ingestion": {"last_hour": 0, "last_24h": 0},
        "recent_ops": [],
        "recent_jobs": [],
        "llm_calls": {"last_24h": 0, "recent": []},
    }


def test_first_call_hits_uncached_and_marks_fresh():
    from src.api.server import stats_summary
    with patch("src.api.server._stats_summary_uncached", return_value=_fake_uncached_response()) as mock_func:
        result = stats_summary(workspace_id="default")
    assert mock_func.call_count == 1
    assert result["_cache_status"] == "fresh"
    assert result["chunks"]["active"] == 100


def test_second_call_within_ttl_uses_cache():
    from src.api.server import stats_summary
    with patch("src.api.server._stats_summary_uncached", return_value=_fake_uncached_response()) as mock_func:
        result1 = stats_summary(workspace_id="default")
        result2 = stats_summary(workspace_id="default")
    # Only one heavy call despite two requests
    assert mock_func.call_count == 1
    assert result1["_cache_status"] == "fresh"
    assert result2["_cache_status"] == "hit"


def test_call_after_ttl_expiry_refreshes():
    from src.api.server import stats_summary, _STATS_CACHE
    with patch("src.api.server._stats_summary_uncached", return_value=_fake_uncached_response()) as mock_func:
        stats_summary(workspace_id="default")
        # Manually expire
        _STATS_CACHE["expires_at"] = time.time() - 1
        stats_summary(workspace_id="default")
    assert mock_func.call_count == 2


def test_db_crash_returns_stale_cache():
    """Wenn DB crash + es gibt stale cache → return stale mit marker."""
    from src.api.server import stats_summary, _STATS_CACHE

    # Erste call: populate cache
    with patch("src.api.server._stats_summary_uncached", return_value=_fake_uncached_response()):
        stats_summary(workspace_id="default")

    # Expire fresh slot, crash next live-call → should return stale
    _STATS_CACHE["expires_at"] = time.time() - 1
    with patch("src.api.server._stats_summary_uncached", side_effect=RuntimeError("DB timeout")):
        result = stats_summary(workspace_id="default")
    assert result["_cache_status"] == "stale"
    assert "DB timeout" in result["_stale_reason"]
    assert result["chunks"]["active"] == 100  # cached data preserved


def test_db_crash_without_cache_returns_503():
    """Erste call jemals + DB down → 503, kein stale verfügbar."""
    from src.api.server import stats_summary
    with patch("src.api.server._stats_summary_uncached", side_effect=RuntimeError("DB locked")):
        with pytest.raises(HTTPException) as exc_info:
            stats_summary(workspace_id="default")
    assert exc_info.value.status_code == 503
    assert "DB locked" in exc_info.value.detail


def test_consecutive_crashes_keep_serving_stale_forever():
    """Wenn DB lange down ist und cache stale, soll trotzdem serviced werden."""
    from src.api.server import stats_summary, _STATS_CACHE

    # populate
    with patch("src.api.server._stats_summary_uncached", return_value=_fake_uncached_response()):
        stats_summary(workspace_id="default")

    # 5x crash in a row — alle sollten stale zurückgeben
    _STATS_CACHE["expires_at"] = 0  # immer abgelaufen, immer DB versuchen
    with patch("src.api.server._stats_summary_uncached", side_effect=RuntimeError("still down")):
        for _ in range(5):
            r = stats_summary(workspace_id="default")
            assert r["_cache_status"] == "stale"
            assert r["chunks"]["active"] == 100


def test_recovery_after_crash_updates_cache():
    """DB war crash, kommt zurück → next call refresht cache mit neuen daten."""
    from src.api.server import stats_summary, _STATS_CACHE

    # Initialer fresh-cache
    with patch("src.api.server._stats_summary_uncached", return_value=_fake_uncached_response()):
        stats_summary(workspace_id="default")

    _STATS_CACHE["expires_at"] = 0  # expire

    # DB crash
    with patch("src.api.server._stats_summary_uncached", side_effect=RuntimeError("transient")):
        stale = stats_summary(workspace_id="default")
    assert stale["_cache_status"] == "stale"

    # DB recovers with new numbers
    new_response = dict(_fake_uncached_response(), chunks={"active": 999, "total": 1000})
    with patch("src.api.server._stats_summary_uncached", return_value=new_response):
        result = stats_summary(workspace_id="default")
    assert result["_cache_status"] == "fresh"
    assert result["chunks"]["active"] == 999  # cache updated to new numbers


def test_bust_stats_cache_clears_fresh_keeps_stale():
    from src.api.server import bust_stats_cache, _STATS_CACHE
    _STATS_CACHE["fresh"] = {"x": 1}
    _STATS_CACHE["stale"] = {"x": 1}
    _STATS_CACHE["expires_at"] = 9e9  # far future
    bust_stats_cache()
    assert _STATS_CACHE["fresh"] is None
    assert _STATS_CACHE["expires_at"] == 0.0
    # stale slot preserved (disaster-fallback)
    assert _STATS_CACHE["stale"] == {"x": 1}
