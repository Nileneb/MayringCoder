"""get_job finds jobs created by ANOTHER uvicorn worker (via the shared file).

WHY(smoke-fix 2026-05-24): _JOBS is per-process. Under --workers 4 a job created
on worker A (POST /populate) was invisible to worker B (GET /jobs/{id}) → 404 →
pipeline_stage_observability smoke red. _save_jobs writes the whole registry to
the shared-volume file atomically, so get_job re-reads it on a local miss.
"""
from __future__ import annotations


def test_get_job_falls_back_to_shared_file(tmp_path, monkeypatch):
    import src.api.job_queue as jq
    # Isolate the shared state file WITHOUT importlib.reload. WHY(flake-fix
    # 2026-05-25): reload rebinds jq._JOBS to a NEW dict, but server.py/jobs.py
    # keep the OLD dict they imported via `from job_queue import _JOBS`. After
    # this test ran, every later test's make_job wrote to the new dict while the
    # route's _fake_checker read srv._JOBS (old) → flaky KeyError in
    # test_populate_fires_full_v2_chain_on_success (green isolated, red in-suite).
    # setattr rebinds only this module's attr; no divergence.
    monkeypatch.setattr(jq, "_JOBS_STATE_FILE", tmp_path / "jobs_state.json")
    jq._JOBS.clear()

    job_id = jq.make_job("ws-1")
    assert jq.get_job(job_id) is not None  # present in this process

    # Simulate a DIFFERENT worker process: its in-memory registry never saw
    # this job. The shared file (written by make_job → _save_jobs) still has it.
    jq._JOBS.clear()
    found = jq.get_job(job_id)
    assert found is not None, "get_job must re-read the shared file on a local miss"
    assert found["job_id"] == job_id
    assert found["status"] == "started"

    # A truly unknown id is still None (not a silent empty dict).
    assert jq.get_job("deadbeef") is None
