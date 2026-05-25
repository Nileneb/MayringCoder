from __future__ import annotations
import asyncio
from unittest.mock import patch, MagicMock


def test_enqueue_populate_starts_a_job(monkeypatch, tmp_path):
    import src.api.job_queue as jq
    monkeypatch.setattr(jq, "_JOBS_STATE_FILE", tmp_path / "jobs.json")
    jq._JOBS.clear()
    from src.api.routes import jobs
    with patch("src.api.routes.jobs._run_with_v2_postingest") as m, \
         patch("src.api.routes.jobs.asyncio.create_task"):
        jid = jobs.enqueue_populate("https://github.com/a/b", "ws-1")
    assert jid and jq.get_job(jid) is not None
    assert m.called  # the v2-chain background task was scheduled


def test_enqueue_populate_debounces_running_repo(monkeypatch, tmp_path):
    import src.api.job_queue as jq
    monkeypatch.setattr(jq, "_JOBS_STATE_FILE", tmp_path / "jobs.json")
    jq._JOBS.clear()
    from src.api.routes import jobs
    with patch("src.api.routes.jobs._run_with_v2_postingest"), \
         patch("src.api.routes.jobs.asyncio.create_task"):
        jid1 = jobs.enqueue_populate("https://github.com/a/b", "ws-1")
        jid2 = jobs.enqueue_populate("https://github.com/a/b", "ws-1")  # same repo, still running
    assert jid2 == jid1, "a populate already running for this repo must be reused, not duplicated"
