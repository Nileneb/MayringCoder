import asyncio
import sqlite3
from pathlib import Path

from mayring_core.memory.store import init_memory_db


def test_record_pi_task_upserts_and_get_projection_has_no_updated_at(tmp_path: Path, monkeypatch):
    """Pi-Agent observability (2026-05-27): locally-run pi-tasks never reached the
    cloud, so /stats/pi-tasks was empty. Two bugs fixed together:
      1. POST /stats/pi-tasks/record now mirrors a task into cloud pi_jobs
         (upsert by job_id, workspace-scoped).
      2. GET /stats/pi-tasks selected a non-existent `updated_at` column → the
         SELECT raised → fail-soft returned []  → the page showed nothing even
         WITH rows. The projection now uses `finished_at` (a real column).
    """
    import src.api.routes.dashboard as dash

    db = tmp_path / "m.db"
    init_memory_db(db).close()
    conn = sqlite3.connect(str(db))
    monkeypatch.setattr(dash, "_conn", lambda: conn)

    ws = "ws-1"
    rec = dash._PiTaskRecord(job_id="pij_x", task_text="do X", status="queued")
    asyncio.run(dash.record_pi_task(rec, workspace_id=ws))
    done = dash._PiTaskRecord(job_id="pij_x", task_text="do X", status="completed", result="OK")
    asyncio.run(dash.record_pi_task(done, workspace_id=ws))

    # The exact (fixed) GET projection must run and return the upserted row.
    rows = conn.execute(
        "SELECT job_id, task_text, status, prefer, scope, model, error, "
        "       created_at, finished_at FROM pi_jobs WHERE workspace_id=?",
        (ws,),
    ).fetchall()
    assert len(rows) == 1, "same job_id must upsert, not duplicate"
    assert rows[0][2] == "completed", "second record must update status"

    # Regression guard: pi_jobs has no updated_at; finished_at is the real column.
    cols = [r[1] for r in conn.execute("PRAGMA table_info(pi_jobs)").fetchall()]
    assert "updated_at" not in cols
    assert "finished_at" in cols


def test_record_pi_task_scopes_by_workspace(tmp_path: Path, monkeypatch):
    import src.api.routes.dashboard as dash

    db = tmp_path / "m.db"
    init_memory_db(db).close()
    conn = sqlite3.connect(str(db))
    monkeypatch.setattr(dash, "_conn", lambda: conn)

    asyncio.run(dash.record_pi_task(dash._PiTaskRecord(job_id="a", status="completed"), workspace_id="ws-a"))
    asyncio.run(dash.record_pi_task(dash._PiTaskRecord(job_id="b", status="completed"), workspace_id="ws-b"))
    n_a = conn.execute("SELECT COUNT(*) FROM pi_jobs WHERE workspace_id='ws-a'").fetchone()[0]
    assert n_a == 1, "record must be written under the caller's workspace"
