"""Hermetic tests for pi_jobs lifecycle + worker-loop.

The worker is exercised end-to-end with a mocked `run_task_with_memory` so
the test runs without Ollama. Schema migration is verified in isolation.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.agents import pi_jobs, pi_worker
from src.memory.store import init_memory_db


@pytest.fixture
def db(tmp_path: Path):
    p = tmp_path / "memory.db"
    init_memory_db(p).close()
    yield p


def _wait_for_status(db_path: Path, job_id: str, target: str, timeout: float = 5.0):
    start = time.time()
    while time.time() - start < timeout:
        row = sqlite3.connect(db_path).execute(
            "SELECT status FROM pi_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if row and row[0] == target:
            return
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} stuck at {row[0] if row else 'missing'}")


# ----- Schema migration --------------------------------------------------


def test_pi_jobs_table_created(db: Path) -> None:
    cols = {r[1] for r in sqlite3.connect(db).execute(
        "PRAGMA table_info(pi_jobs)"
    ).fetchall()}
    assert {
        "job_id", "task_text", "repo_slug", "workspace_id", "status",
        "prefer", "ollama_url", "model", "result_json", "error",
        "timeout_s", "created_at", "started_at", "finished_at",
    } <= cols


def test_init_memory_db_idempotent(db: Path) -> None:
    init_memory_db(db).close()  # second call must not crash on existing index


# ----- Lifecycle ---------------------------------------------------------


def test_insert_job_starts_queued(db: Path) -> None:
    job = pi_jobs.insert_job("hi", db_path=db)
    fetched = pi_jobs.get_job(job.job_id, db_path=db)
    assert fetched is not None
    assert fetched.status == "queued"
    assert fetched.task_text == "hi"


def test_insert_job_rejects_invalid_prefer(db: Path) -> None:
    with pytest.raises(ValueError):
        pi_jobs.insert_job("x", prefer="never", db_path=db)


def test_claim_next_returns_oldest_first(db: Path) -> None:
    a = pi_jobs.insert_job("first", db_path=db)
    time.sleep(0.01)
    b = pi_jobs.insert_job("second", db_path=db)
    claimed = pi_jobs.claim_next(db_path=db)
    assert claimed is not None
    assert claimed.job_id == a.job_id
    assert claimed.status == "running"
    # The other job is still queued
    assert pi_jobs.get_job(b.job_id, db_path=db).status == "queued"


def test_claim_next_returns_none_when_empty(db: Path) -> None:
    assert pi_jobs.claim_next(db_path=db) is None


def test_claim_next_atomic_under_concurrency(db: Path) -> None:
    """Two concurrent claim_next() must not both grab the same row."""
    pi_jobs.insert_job("only-one", db_path=db)

    import threading
    results: list = []
    barrier = threading.Barrier(2)

    def worker() -> None:
        barrier.wait()
        results.append(pi_jobs.claim_next(db_path=db))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    claimed = [r for r in results if r is not None]
    assert len(claimed) == 1, f"both threads claimed: {results}"


def test_complete_job_sets_result(db: Path) -> None:
    job = pi_jobs.insert_job("hi", db_path=db)
    pi_jobs.claim_next(db_path=db)
    pi_jobs.complete_job(job.job_id, {"text": "done"}, db_path=db)
    fetched = pi_jobs.get_job(job.job_id, db_path=db)
    assert fetched.status == "completed"
    assert fetched.to_dict()["result"] == {"text": "done"}
    assert fetched.finished_at != ""


def test_fail_job_records_error(db: Path) -> None:
    job = pi_jobs.insert_job("hi", db_path=db)
    pi_jobs.claim_next(db_path=db)
    pi_jobs.fail_job(job.job_id, "ollama down", db_path=db)
    fetched = pi_jobs.get_job(job.job_id, db_path=db)
    assert fetched.status == "failed"
    assert "ollama down" in fetched.error


def test_list_recent_filters_active(db: Path) -> None:
    a = pi_jobs.insert_job("a", db_path=db)
    b = pi_jobs.insert_job("b", db_path=db)
    pi_jobs.claim_next(db_path=db)
    pi_jobs.complete_job(a.job_id, {"text": "x"}, db_path=db)
    active = pi_jobs.list_recent(only_active=True, db_path=db)
    assert {j.job_id for j in active} == {b.job_id}
    all_jobs = pi_jobs.list_recent(only_active=False, db_path=db)
    assert {j.job_id for j in all_jobs} == {a.job_id, b.job_id}


# ----- Worker loop end-to-end --------------------------------------------


def test_worker_drains_queued_job(db: Path, monkeypatch) -> None:
    """A queued job is claimed by the worker, executed (mocked LLM), and
    persisted as 'completed'."""
    monkeypatch.setattr(
        "src.agents.pi_jobs.CACHE_DIR",
        db.parent,
        raising=True,
    )

    pi_worker.stop()  # ensure clean state if a previous test left state
    with patch("src.agents.pi.run_task_with_memory", return_value="hello world"):
        pi_worker.start(poll_interval=0.05)
        try:
            job = pi_jobs.insert_job("hi", db_path=db)
            _wait_for_status(db, job.job_id, "completed", timeout=5.0)
        finally:
            pi_worker.stop()

    fetched = pi_jobs.get_job(job.job_id, db_path=db)
    assert fetched.status == "completed"
    assert fetched.to_dict()["result"] == {"text": "hello world"}


def test_worker_marks_failure_on_exception(db: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.agents.pi_jobs.CACHE_DIR", db.parent, raising=True)
    pi_worker.stop()
    with patch(
        "src.agents.pi.run_task_with_memory",
        side_effect=RuntimeError("ollama down"),
    ):
        pi_worker.start(poll_interval=0.05)
        try:
            job = pi_jobs.insert_job("boom", db_path=db)
            _wait_for_status(db, job.job_id, "failed", timeout=5.0)
        finally:
            pi_worker.stop()

    fetched = pi_jobs.get_job(job.job_id, db_path=db)
    assert fetched.status == "failed"
    assert "ollama down" in fetched.error


def test_worker_start_is_idempotent() -> None:
    pi_worker.stop()
    assert pi_worker.start() is True
    assert pi_worker.start() is False
    pi_worker.stop()


def test_worker_disabled_via_env(monkeypatch) -> None:
    monkeypatch.setenv("PI_ASYNC_DISABLED", "1")
    pi_worker.stop()
    assert pi_worker.start() is False
    assert pi_worker._is_running() is True  # marked started, but no thread
    pi_worker.stop()


# ----- Ollama URL resolver (per-job / config / env / default) ---------------


def test_resolve_ollama_url_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MAYRING_OLLAMA_CONFIG", str(tmp_path / "missing.conf"))
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    assert pi_worker._resolve_ollama_url("") == "http://localhost:11434"


def test_resolve_ollama_url_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MAYRING_OLLAMA_CONFIG", str(tmp_path / "missing.conf"))
    monkeypatch.setenv("OLLAMA_URL", "http://laptop.lan:11434")
    assert pi_worker._resolve_ollama_url("") == "http://laptop.lan:11434"


def test_resolve_ollama_url_config_file_overrides_env(
    monkeypatch, tmp_path: Path,
) -> None:
    """Config file overrides ENV — runtime switch without restart."""
    cfg = tmp_path / "ollama.conf"
    cfg.write_text("https://three.linn.games\n")
    monkeypatch.setenv("MAYRING_OLLAMA_CONFIG", str(cfg))
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:11434")
    assert pi_worker._resolve_ollama_url("") == "https://three.linn.games"


def test_resolve_ollama_url_config_file_skips_comments(
    monkeypatch, tmp_path: Path,
) -> None:
    cfg = tmp_path / "ollama.conf"
    cfg.write_text("# pinned to remote GPU\nhttps://three.linn.games\n")
    monkeypatch.setenv("MAYRING_OLLAMA_CONFIG", str(cfg))
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:11434")
    assert pi_worker._resolve_ollama_url("") == "https://three.linn.games"


def test_resolve_ollama_url_per_job_wins_over_everything(
    monkeypatch, tmp_path: Path,
) -> None:
    cfg = tmp_path / "ollama.conf"
    cfg.write_text("https://three.linn.games\n")
    monkeypatch.setenv("MAYRING_OLLAMA_CONFIG", str(cfg))
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:11434")
    assert (
        pi_worker._resolve_ollama_url("https://ollama.cloud")
        == "https://ollama.cloud"
    )


def test_resolve_ollama_url_runtime_switch(
    monkeypatch, tmp_path: Path,
) -> None:
    """Rewriting the config file flips backends — no restart needed."""
    cfg = tmp_path / "ollama.conf"
    monkeypatch.setenv("MAYRING_OLLAMA_CONFIG", str(cfg))
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:11434")

    cfg.write_text("https://three.linn.games\n")
    assert pi_worker._resolve_ollama_url("") == "https://three.linn.games"

    cfg.write_text("https://ollama.cloud\n")
    assert pi_worker._resolve_ollama_url("") == "https://ollama.cloud"

    cfg.unlink()
    assert pi_worker._resolve_ollama_url("") == "http://localhost:11434"


# ----- Phase 2: cloud-scope jobs -------------------------------------------


def test_local_claim_ignores_cloud_jobs(db: Path) -> None:
    """A cloud-scope job must not be picked up by the local worker."""
    pi_jobs.insert_cloud_job("cloud-only", capability_required="local-gpu", db_path=db)
    assert pi_jobs.claim_next(db_path=db) is None


def test_claim_cloud_next_requires_capability(db: Path) -> None:
    pi_jobs.insert_cloud_job(
        "needs-gpu", capability_required="local-gpu", db_path=db,
    )
    no_gpu = pi_jobs.claim_cloud_next("wkr_no", capabilities=["cpu"], db_path=db)
    assert no_gpu is None
    gpu = pi_jobs.claim_cloud_next("wkr_yes", capabilities=["local-gpu"], db_path=db)
    assert gpu is not None
    assert gpu.claimed_by == "wkr_yes"
    assert gpu.scope == "cloud"
    assert gpu.status == "running"


def test_claim_cloud_next_no_required_matches_any_worker(db: Path) -> None:
    pi_jobs.insert_cloud_job("any", db_path=db)
    j = pi_jobs.claim_cloud_next("wkr_any", capabilities=[], db_path=db)
    assert j is not None and j.claimed_by == "wkr_any"


def test_claim_cloud_next_returns_none_when_empty(db: Path) -> None:
    assert pi_jobs.claim_cloud_next("wkr", db_path=db) is None


def test_claim_cloud_next_atomic_under_concurrency(db: Path) -> None:
    pi_jobs.insert_cloud_job(
        "only-one", capability_required="local-gpu", db_path=db,
    )

    import threading
    results: list = []
    barrier = threading.Barrier(2)

    def worker(wid: str) -> None:
        barrier.wait()
        results.append(
            pi_jobs.claim_cloud_next(wid, capabilities=["local-gpu"], db_path=db)
        )

    threads = [
        threading.Thread(target=worker, args=(f"wkr_{i}",)) for i in range(2)
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]

    claimed = [r for r in results if r is not None]
    assert len(claimed) == 1


def test_claim_cloud_next_filters_by_workspace(db: Path) -> None:
    pi_jobs.insert_cloud_job("ws-a", workspace_id="alpha", db_path=db)
    pi_jobs.insert_cloud_job("ws-b", workspace_id="beta", db_path=db)
    j = pi_jobs.claim_cloud_next("wkr", workspace_id="beta", db_path=db)
    assert j is not None and j.task_text == "ws-b"


def test_list_recent_filter_by_scope(db: Path) -> None:
    pi_jobs.insert_job("local-1", db_path=db)
    pi_jobs.insert_cloud_job("cloud-1", db_path=db)
    cloud_only = pi_jobs.list_recent(scope="cloud", db_path=db)
    assert {j.task_text for j in cloud_only} == {"cloud-1"}
    local_only = pi_jobs.list_recent(scope="local", db_path=db)
    assert {j.task_text for j in local_only} == {"local-1"}


def test_insert_cloud_job_rejects_invalid_prefer(db: Path) -> None:
    with pytest.raises(ValueError):
        pi_jobs.insert_cloud_job("x", prefer="nope", db_path=db)


def test_cloud_tools_register_without_error() -> None:
    """`register_pi_queue_tools` is callable and adds tools to the MCP."""
    from mcp.server.fastmcp import FastMCP
    from src.api.mcp_pi_tools import register_pi_queue_tools
    mcp = FastMCP("test")
    register_pi_queue_tools(mcp)  # must not raise


# ----- Defence-in-depth: pi_jobs phase-2 column lazy migration -----------


def test_ensure_pi_jobs_phase2_columns_adds_missing(tmp_path: Path) -> None:
    """Legacy DB without scope/capability_required/claimed_by/claimed_at —
    helper applies the missing columns idempotently."""
    import sqlite3
    from src.api.mcp_pi_tools import _ensure_pi_jobs_phase2_columns

    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE pi_jobs (job_id TEXT PRIMARY KEY, task_text TEXT, "
        "status TEXT DEFAULT 'queued', created_at TEXT NOT NULL)"
    )
    cols_before = {r[1] for r in conn.execute("PRAGMA table_info(pi_jobs)").fetchall()}
    assert "scope" not in cols_before

    _ensure_pi_jobs_phase2_columns(conn)
    cols_after = {r[1] for r in conn.execute("PRAGMA table_info(pi_jobs)").fetchall()}
    assert {"scope", "capability_required", "claimed_by", "claimed_at"} <= cols_after

    # Idempotent — second call is a no-op.
    _ensure_pi_jobs_phase2_columns(conn)


def test_ensure_pi_jobs_phase2_columns_handles_missing_table(
    tmp_path: Path, caplog,
) -> None:
    """If pi_jobs table is gone entirely, the helper logs and returns —
    must NOT crash, must NOT silently mask the deployment problem."""
    import logging
    import sqlite3
    from src.api.mcp_pi_tools import _ensure_pi_jobs_phase2_columns

    conn = sqlite3.connect(tmp_path / "no-table.db")
    with caplog.at_level(logging.ERROR, logger="src.api.mcp_pi_tools"):
        _ensure_pi_jobs_phase2_columns(conn)
    assert any("pi_jobs table is missing" in r.message for r in caplog.records)


def test_server_startup_runs_schema_migration(monkeypatch, tmp_path: Path) -> None:
    """The FastAPI startup hook in server.py calls init_memory_db so
    legacy DBs get migrated on every boot."""
    monkeypatch.setenv("MAYRING_LOCAL_DB", str(tmp_path / "fresh.db"))

    # Reset cached singleton so the test gets a clean import path.
    import src.api.dependencies as deps
    deps._conn = None

    from fastapi.testclient import TestClient
    from src.api.server import app

    with TestClient(app) as client:
        # Hitting any endpoint forces lifespan startup.
        resp = client.get("/health")
        assert resp.status_code == 200

    # The schema migration must have created the pi_jobs table with phase-2 cols.
    import sqlite3
    cols = {r[1] for r in sqlite3.connect(tmp_path / "fresh.db").execute(
        "PRAGMA table_info(pi_jobs)"
    ).fetchall()}
    assert {"scope", "capability_required", "claimed_by", "claimed_at"} <= cols


# ----- Tenant scoping (cross-workspace isolation) -------------------------


def test_get_job_scoped_by_workspace_returns_none_for_other_tenant(db: Path) -> None:
    """A caller with workspace=alpha must NOT be able to read a job in beta."""
    job = pi_jobs.insert_job("secret", workspace_id="beta", db_path=db)
    assert pi_jobs.get_job(job.job_id, workspace_id="alpha", db_path=db) is None
    assert pi_jobs.get_job(job.job_id, workspace_id="beta", db_path=db) is not None


def test_get_job_unscoped_still_reads_any_row(db: Path) -> None:
    """Internal callers (worker loops) may omit workspace_id."""
    job = pi_jobs.insert_job("internal", workspace_id="beta", db_path=db)
    assert pi_jobs.get_job(job.job_id, db_path=db) is not None


def test_list_recent_scoped_by_workspace(db: Path) -> None:
    pi_jobs.insert_job("alpha-1", workspace_id="alpha", db_path=db)
    pi_jobs.insert_job("alpha-2", workspace_id="alpha", db_path=db)
    pi_jobs.insert_job("beta-1", workspace_id="beta", db_path=db)
    alpha = pi_jobs.list_recent(workspace_id="alpha", db_path=db)
    assert {j.task_text for j in alpha} == {"alpha-1", "alpha-2"}
    beta = pi_jobs.list_recent(workspace_id="beta", db_path=db)
    assert {j.task_text for j in beta} == {"beta-1"}


def test_list_recent_combines_workspace_and_scope_filters(db: Path) -> None:
    pi_jobs.insert_job("alpha-local", workspace_id="alpha", db_path=db)
    pi_jobs.insert_cloud_job("alpha-cloud", workspace_id="alpha", db_path=db)
    pi_jobs.insert_cloud_job("beta-cloud", workspace_id="beta", db_path=db)
    out = pi_jobs.list_recent(
        scope="cloud", workspace_id="alpha", db_path=db,
    )
    assert {j.task_text for j in out} == {"alpha-cloud"}
