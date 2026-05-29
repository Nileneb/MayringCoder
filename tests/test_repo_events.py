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
        # match-or-create persistiert die KANONISCHE source_ref (slug 'a/b'), nicht die
        # rohe URL — so matcht eine spätere URL/SSH-Variante dasselbe Projekt (Dedup-Fix).
        row = _deps._conn.execute(
            "SELECT workspace_id FROM projects WHERE source_type='github' AND source_ref=?",
            ("a/b",),
        ).fetchone()
        assert row is not None and row[0] == "system", \
            "unknown repo → canonical-slug project under 'system' (no user ws in test)"
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


def test_workflow_run_failure_creates_issue_chunk(monkeypatch, tmp_path):
    client = _make_repo_events_client(monkeypatch, tmp_path)
    from src.api.dependencies import get_conn
    try:
        client.post("/repo-events", headers={"Authorization": "Bearer t"},
            json={"event_type": "workflow_run", "repo": "https://github.com/a/b",
                  "sha": "c0ffee", "conclusion": "failure", "workflow": "tests"})
        rows = get_conn().execute(
            "SELECT text, igio_axis FROM chunks c JOIN sources s ON c.source_id=s.source_id "
            "WHERE s.source_type='repo_event'").fetchall()
        assert rows, "a repo_event chunk must be created"
        assert any(r[1] == "issue" and "tests" in r[0] for r in rows)
    finally:
        from src.api import server as srv; srv.app.dependency_overrides.clear()


def test_workflow_run_success_creates_outcome_chunk(monkeypatch, tmp_path):
    client = _make_repo_events_client(monkeypatch, tmp_path)
    from src.api.dependencies import get_conn
    try:
        client.post("/repo-events", headers={"Authorization": "Bearer t"},
            json={"event_type": "workflow_run", "repo": "https://github.com/a/b",
                  "sha": "beef", "conclusion": "success", "workflow": "deploy"})
        rows = get_conn().execute(
            "SELECT igio_axis FROM chunks c JOIN sources s ON c.source_id=s.source_id "
            "WHERE s.source_type='repo_event'").fetchall()
        assert any(r[0] == "outcome" for r in rows)
    finally:
        from src.api import server as srv; srv.app.dependency_overrides.clear()


def test_security_event_creates_issue_chunk(monkeypatch, tmp_path):
    client = _make_repo_events_client(monkeypatch, tmp_path)
    from src.api.dependencies import get_conn
    try:
        client.post("/repo-events", headers={"Authorization": "Bearer t"},
            json={"event_type": "security", "repo": "https://github.com/a/b",
                  "severity": "high", "summary": "CVE-1"})
        rows = get_conn().execute(
            "SELECT text, igio_axis FROM chunks c JOIN sources s ON c.source_id=s.source_id "
            "WHERE s.source_type='repo_event'").fetchall()
        assert any(r[1] == "issue" and "CVE-1" in r[0] for r in rows)
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


def test_resolve_workspace_defaults_to_sole_user_ws(tmp_path):
    """Unbekanntes Repo → einziger kind='user'-Workspace (NICHT 'system'), damit Repo-CI/
    Security-Events im user-gescopten Dashboard sichtbar werden (Blackbox-Fix #4 2026-05-29)."""
    from mayring_core.memory.store import init_memory_db
    from src.api.routes.repo_events import _resolve_workspace
    conn = init_memory_db(tmp_path / "m.db")
    now = "2026-05-29T00:00:00Z"
    conn.execute("INSERT INTO workspaces(id,kind,display_name,created_at,updated_at) "
                 "VALUES ('ws-bene','user','Bene',?,?)", (now, now))
    conn.commit()
    assert _resolve_workspace(conn, "https://github.com/x/new-repo") == "ws-bene"
    assert _resolve_workspace(conn, "https://github.com/x/new-repo") == "ws-bene"  # idempotent
    conn.close()


def test_resolve_workspace_multi_user_falls_back_to_system(tmp_path):
    """Mehrere User → kein Raten, Fallback 'system' (dann braucht es repo-owner→ws-Mapping)."""
    from mayring_core.memory.store import init_memory_db
    from src.api.routes.repo_events import _resolve_workspace
    conn = init_memory_db(tmp_path / "m.db")
    now = "2026-05-29T00:00:00Z"
    for wid in ("ws-a", "ws-b"):
        conn.execute("INSERT INTO workspaces(id,kind,display_name,created_at,updated_at) "
                     "VALUES (?,'user','x',?,?)", (wid, now, now))
    conn.commit()
    assert _resolve_workspace(conn, "https://github.com/x/repo") == "system"
    conn.close()


def test_resolve_workspace_smoke_repo_stays_system(tmp_path):
    """Smoke-Suite-Wegwerf-Repos → IMMER 'system', auch bei single-user, sonst pollutet
    jeder Smoke-Lauf die Projekt-Sicht des Users (Über-Claim-Fix 2026-05-29)."""
    from mayring_core.memory.store import init_memory_db
    from src.api.routes.repo_events import _resolve_workspace
    conn = init_memory_db(tmp_path / "m.db")
    now = "2026-05-29T00:00:00Z"
    conn.execute("INSERT INTO workspaces(id,kind,display_name,created_at,updated_at) "
                 "VALUES ('ws-bene','user','Bene',?,?)", (now, now))
    conn.commit()
    assert _resolve_workspace(conn, "https://github.com/smoke/repo-1780000000") == "system"
    assert _resolve_workspace(conn, "https://github.com/Nileneb/app.linn.games") == "ws-bene"
    conn.close()


def test_canonical_repo_ref_collapses_url_slug_ssh():
    """URL, Slug und SSH-Form desselben Repos → EINE kanonische source_ref, sonst legen
    /projects/route + /repo-events Dubletten an (Datenmüll-Fix 2026-05-29)."""
    from src.api.routes.projects import canonical_repo_ref
    a = canonical_repo_ref("https://github.com/Nileneb/app.linn.games")
    b = canonical_repo_ref("nileneb/app.linn.games")
    c = canonical_repo_ref("git@github.com:Nileneb/app.linn.games.git")
    assert a == b == c == "nileneb/app.linn.games"


def test_resolve_workspace_stores_canonical_ref(tmp_path):
    """/repo-events legt das Projekt unter der kanonischen Slug-Form an — eine spätere
    URL-Variante desselben Repos findet dasselbe Projekt (keine zweite Dublette)."""
    from mayring_core.memory.store import init_memory_db
    from src.api.routes.repo_events import _resolve_workspace
    conn = init_memory_db(tmp_path / "m.db")
    now = "2026-05-29T00:00:00Z"
    conn.execute("INSERT INTO workspaces(id,kind,display_name,created_at,updated_at) "
                 "VALUES ('ws-bene','user','Bene',?,?)", (now, now))
    conn.commit()
    _resolve_workspace(conn, "https://github.com/Nileneb/app.linn.games")
    # zweite Variante (slug) desselben Repos → KEIN zweites Projekt
    _resolve_workspace(conn, "nileneb/app.linn.games")
    n = conn.execute("SELECT COUNT(*) FROM projects WHERE source_type='github'").fetchone()[0]
    assert n == 1, "URL + Slug desselben Repos dürfen nur EIN Projekt sein"
    conn.close()
