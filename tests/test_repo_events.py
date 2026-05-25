from __future__ import annotations
import inspect
from unittest.mock import patch, AsyncMock, MagicMock, call

import src.api.dependencies as _deps


def _capture_task(coro):
    """Stand-in for asyncio.create_task in sync tests: dispose the coroutine
    so it is not reported as 'never awaited', and return a placeholder."""
    if inspect.iscoroutine(coro):
        coro.close()
    return MagicMock()


def test_enqueue_populate_starts_a_job(monkeypatch, tmp_path):
    import src.api.job_queue as jq
    monkeypatch.setattr(jq, "_JOBS_STATE_FILE", tmp_path / "jobs.json")
    jq._JOBS.clear()
    from src.api.routes import jobs
    with patch("src.api.routes.jobs._run_with_v2_postingest", new_callable=AsyncMock) as m, \
         patch("src.api.routes.jobs.asyncio.create_task", side_effect=_capture_task):
        jid = jobs.enqueue_populate("https://github.com/a/b", "ws-1")
    assert jid and jq.get_job(jid) is not None
    assert m.called  # the v2-chain background task was scheduled


def test_enqueue_populate_debounces_running_repo(monkeypatch, tmp_path):
    import src.api.job_queue as jq
    monkeypatch.setattr(jq, "_JOBS_STATE_FILE", tmp_path / "jobs.json")
    jq._JOBS.clear()
    from src.api.routes import jobs
    with patch("src.api.routes.jobs._run_with_v2_postingest", new_callable=AsyncMock), \
         patch("src.api.routes.jobs.asyncio.create_task", side_effect=_capture_task):
        jid1 = jobs.enqueue_populate("https://github.com/a/b", "ws-1")
        jid2 = jobs.enqueue_populate("https://github.com/a/b", "ws-1")  # same repo, still running
    assert jid2 == jid1, "a populate already running for this repo must be reused, not duplicated"


def test_enqueue_populate_persists_repo_for_cross_worker_debounce(monkeypatch, tmp_path):
    import src.api.job_queue as jq
    monkeypatch.setattr(jq, "_JOBS_STATE_FILE", tmp_path / "jobs.json")
    jq._JOBS.clear()
    from src.api.routes import jobs
    with patch("src.api.routes.jobs._run_with_v2_postingest", new_callable=AsyncMock), \
         patch("src.api.routes.jobs.asyncio.create_task", side_effect=_capture_task):
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
         patch("src.api.routes.jobs.asyncio.create_task", side_effect=_capture_task):
        jid_a = jobs.enqueue_populate("https://github.com/a/b", "ws-1")
        jid_b = jobs.enqueue_populate("https://github.com/a/b", "ws-2")  # same repo, other workspace
    assert jid_a != jid_b, "debounce must be workspace-scoped — different workspace gets its own job"


# ---------------------------------------------------------------------------
# POST /repo-events tests (Task 2)
# ---------------------------------------------------------------------------

def _make_repo_events_client(monkeypatch, tmp_path):
    from mayring_core.memory.db_adapter import DBAdapter
    from mayring_core.memory.store import _init_schema
    adapter = DBAdapter.create(tmp_path / "test.db", check_same_thread=False)
    _init_schema(adapter)
    monkeypatch.setattr(_deps, "_conn", adapter)

    import src.api.job_queue as jq
    monkeypatch.setattr(jq, "_JOBS_STATE_FILE", tmp_path / "jobs.json")
    jq._JOBS.clear()
    from fastapi.testclient import TestClient
    from src.api import server as srv
    from src.api import auth as auth_module
    from src.api.jwt_auth import TokenInfo
    async def _svc():
        return TokenInfo(workspace_id="system", scopes=("*",))
    srv.app.dependency_overrides[auth_module.get_token_info] = _svc
    return TestClient(srv.app)


def test_repo_events_push_enqueues_populate(monkeypatch, tmp_path):
    client = _make_repo_events_client(monkeypatch, tmp_path)
    try:
        with patch("src.api.routes.repo_events.enqueue_populate", return_value="job-1") as m:
            r = client.post("/repo-events",
                            json={"event_type": "push", "repo": "https://github.com/a/b",
                                  "sha": "abc"},
                            headers={"Authorization": "Bearer t"})
        assert r.status_code == 200
        assert m.called
        assert m.call_args.args == ("https://github.com/a/b", "system"), \
            "enqueue_populate must receive (repo, resolved_workspace_id)"
        body = r.json()
        assert body["action"] == "populate"
        assert body["job_id"] == "job-1"
        assert body["workspace_id"] == "system"
        # verify match-or-create persisted a projects row in the isolated DB
        row = _deps._conn.execute(
            "SELECT workspace_id FROM projects WHERE source_type='github' AND source_ref=?",
            ("https://github.com/a/b",),
        ).fetchone()
        assert row is not None and row[0] == "system", \
            "unknown repo must be match-or-created under 'system'"
    finally:
        from src.api import server as srv
        srv.app.dependency_overrides.clear()


def test_workflow_run_records_hook_event(monkeypatch, tmp_path):
    client = _make_repo_events_client(monkeypatch, tmp_path)
    from src.api.dependencies import get_conn
    try:
        r = client.post("/repo-events", headers={"Authorization": "Bearer t"},
            json={"event_type": "workflow_run", "repo": "https://github.com/a/b",
                  "sha": "deadbeef", "conclusion": "failure", "workflow": "ci"})
        assert r.status_code == 200 and r.json()["action"] == "repo_ci"
        rows = get_conn().execute(
            "SELECT hook_type, payload FROM hook_events WHERE hook_type='repo_ci'").fetchall()
        assert any('"sha": "deadbeef"' in row[1] for row in rows)
        # idempotent: same event again → still one row
        client.post("/repo-events", headers={"Authorization": "Bearer t"},
            json={"event_type": "workflow_run", "repo": "https://github.com/a/b",
                  "sha": "deadbeef", "conclusion": "failure", "workflow": "ci"})
        rows2 = get_conn().execute(
            "SELECT id FROM hook_events WHERE hook_type='repo_ci' AND payload LIKE '%deadbeef%'").fetchall()
        assert len(rows2) == 1, "re-delivered event must not duplicate"
    finally:
        from src.api import server as srv; srv.app.dependency_overrides.clear()


def test_security_event_records_and_dedups_with_none_fields(monkeypatch, tmp_path):
    client = _make_repo_events_client(monkeypatch, tmp_path)
    from src.api.dependencies import get_conn
    try:
        payload = {"event_type": "security", "repo": "https://github.com/a/b",
                   "severity": "high", "summary": "CVE-1 in dep"}
        assert client.post("/repo-events", headers={"Authorization": "Bearer t"}, json=payload).status_code == 200
        client.post("/repo-events", headers={"Authorization": "Bearer t"}, json=payload)  # re-deliver
        rows = get_conn().execute(
            "SELECT id FROM hook_events WHERE hook_type='repo_security'").fetchall()
        assert len(rows) == 1, "security re-delivery (None sha/workflow) must dedup"
    finally:
        from src.api import server as srv; srv.app.dependency_overrides.clear()


def test_repo_events_rejects_non_privileged(monkeypatch, tmp_path):
    import src.api.job_queue as jq
    monkeypatch.setattr(jq, "_JOBS_STATE_FILE", tmp_path / "jobs.json")
    jq._JOBS.clear()
    from fastapi.testclient import TestClient
    from src.api import server as srv
    from src.api import auth as auth_module
    from src.api.jwt_auth import TokenInfo
    async def _unprivileged():
        return TokenInfo(workspace_id="bene", scopes=("mcp:memory",))
    srv.app.dependency_overrides[auth_module.get_token_info] = _unprivileged
    try:
        client = TestClient(srv.app)
        r = client.post("/repo-events",
                        json={"event_type": "push", "repo": "https://github.com/a/b"},
                        headers={"Authorization": "Bearer t"})
        assert r.status_code == 403
    finally:
        srv.app.dependency_overrides.clear()
