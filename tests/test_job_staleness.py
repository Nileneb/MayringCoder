"""Zombie-debounce guard (2026-06-21): a populate frozen at 'started' for 8 days
blocked all app.linn.games re-ingest because enqueue_populate's debounce reused it.
is_job_alive() rejects stale jobs; reconcile_stale_jobs() reaps them on startup."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.api import job_queue as Q


def _iso(age_seconds: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=age_seconds)).isoformat()


def test_is_job_alive_fresh_started():
    assert Q.is_job_alive({"status": "started", "started_at": _iso(10)}) is True


def test_is_job_alive_rejects_stale_started():
    old = Q.STALE_JOB_SECONDS + 60
    assert Q.is_job_alive({"status": "started", "started_at": _iso(old)}) is False


def test_is_job_alive_done_is_not_alive():
    assert Q.is_job_alive({"status": "done", "started_at": _iso(10)}) is False


def test_is_job_alive_missing_started_at_is_alive():
    # unparseable/missing timestamp → don't reap on bad data
    assert Q.is_job_alive({"status": "running"}) is True


def test_reconcile_reaps_only_stale(monkeypatch, tmp_path):
    fresh = Q.make_job("ws", repo="r-fresh")
    Q._JOBS[fresh]["started_at"] = _iso(10)
    stale = Q.make_job("ws", repo="r-stale")
    Q._JOBS[stale]["started_at"] = _iso(Q.STALE_JOB_SECONDS + 600)

    reaped = Q.reconcile_stale_jobs()

    assert reaped >= 1
    assert Q._JOBS[stale]["status"] == "error"
    assert "stale" in Q._JOBS[stale]["output"]
    assert Q._JOBS[fresh]["status"] == "started"  # fresh untouched
