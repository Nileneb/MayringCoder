from __future__ import annotations
from unittest.mock import patch, AsyncMock


def test_enqueue_populate_starts_a_job(monkeypatch, tmp_path):
    import src.api.job_queue as jq
    monkeypatch.setattr(jq, "_JOBS_STATE_FILE", tmp_path / "jobs.json")
    jq._JOBS.clear()
    from src.api.routes import jobs
    with patch("src.api.routes.jobs._run_with_v2_postingest", new_callable=AsyncMock) as m, \
         patch("src.api.routes.jobs.asyncio.create_task", new_callable=AsyncMock):
        jid = jobs.enqueue_populate("https://github.com/a/b", "ws-1")
    assert jid and jq.get_job(jid) is not None
    assert m.called  # the v2-chain background task was scheduled


def test_enqueue_populate_debounces_running_repo(monkeypatch, tmp_path):
    import src.api.job_queue as jq
    monkeypatch.setattr(jq, "_JOBS_STATE_FILE", tmp_path / "jobs.json")
    jq._JOBS.clear()
    from src.api.routes import jobs
    with patch("src.api.routes.jobs._run_with_v2_postingest", new_callable=AsyncMock), \
         patch("src.api.routes.jobs.asyncio.create_task", new_callable=AsyncMock):
        jid1 = jobs.enqueue_populate("https://github.com/a/b", "ws-1")
        jid2 = jobs.enqueue_populate("https://github.com/a/b", "ws-1")  # same repo, still running
    assert jid2 == jid1, "a populate already running for this repo must be reused, not duplicated"


def test_enqueue_populate_persists_repo_for_cross_worker_debounce(monkeypatch, tmp_path):
    import src.api.job_queue as jq
    monkeypatch.setattr(jq, "_JOBS_STATE_FILE", tmp_path / "jobs.json")
    jq._JOBS.clear()
    from src.api.routes import jobs
    with patch("src.api.routes.jobs._run_with_v2_postingest", new_callable=AsyncMock), \
         patch("src.api.routes.jobs.asyncio.create_task", new_callable=AsyncMock):
        jid = jobs.enqueue_populate("https://github.com/a/b", "ws-1")
    # simulate another worker: read ONLY from the shared file, not local _JOBS
    persisted = jq._load_jobs()
    assert persisted[jid].get("repo") == "https://github.com/a/b", \
        "repo tag must be persisted so a different worker's debounce can match it"


def test_enqueue_populate_does_not_debounce_across_workspaces(monkeypatch, tmp_path):
    import src.api.job_queue as jq
    monkeypatch.setattr(jq, "_JOBS_STATE_FILE", tmp_path / "jobs.json")
    jq._JOBS.clear()
    from src.api.routes import jobs
    with patch("src.api.routes.jobs._run_with_v2_postingest", new_callable=AsyncMock), \
         patch("src.api.routes.jobs.asyncio.create_task", new_callable=AsyncMock):
        jid_a = jobs.enqueue_populate("https://github.com/a/b", "ws-1")
        jid_b = jobs.enqueue_populate("https://github.com/a/b", "ws-2")  # same repo, other workspace
    assert jid_a != jid_b, "debounce must be workspace-scoped — different workspace gets its own job"
