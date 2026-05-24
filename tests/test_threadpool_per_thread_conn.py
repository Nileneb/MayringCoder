"""§5.2: get_conn gives each worker thread its OWN SQLite connection.

WHY(api-concurrency-capacity §5.2): the /memory/search body runs in
run_in_threadpool so the event loop stays free under the hook's 3 concurrent
searches (→ /health stays fast). Sharing ONE sqlite connection across those
threads deadlocked (8b0ff34, reverted bbabe22). Per-thread connections let
concurrent searches' DB work proceed (WAL reads parallel; writes serialise via
the file lock, no corruption). An explicitly injected connection
(dependencies._conn, used by tests) still wins on every thread so test overrides
and sequential single-connection callers keep working.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import src.api.dependencies as deps


def _reset(monkeypatch, tmp_path):
    monkeypatch.setattr(deps, "_conn", None, raising=False)
    monkeypatch.setattr(deps, "_conn_local", threading.local(), raising=False)
    monkeypatch.setenv("MAYRING_LOCAL_DB", str(tmp_path / "tl.db"))


def test_distinct_connection_per_thread(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path)
    main_conn = deps.get_conn()
    assert deps.get_conn() is main_conn  # stable within a thread

    with ThreadPoolExecutor(max_workers=1) as ex:
        worker_conn = ex.submit(deps.get_conn).result()
        worker_conn_again = ex.submit(deps.get_conn).result()

    assert worker_conn is not main_conn          # worker thread gets its own
    assert worker_conn is worker_conn_again       # ...and reuses it within that thread


def test_explicit_override_wins_on_all_threads(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path)
    sentinel = object()
    monkeypatch.setattr(deps, "_conn", sentinel, raising=False)

    assert deps.get_conn() is sentinel
    with ThreadPoolExecutor(max_workers=1) as ex:
        assert ex.submit(deps.get_conn).result() is sentinel
