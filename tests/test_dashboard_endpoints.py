"""Dashboard endpoints — read-only aggregations over existing tables.

Each test exercises one endpoint with a tiny in-memory DB seeded with the
exact rows needed to hit every code path. No HTTP layer involved; the
endpoints are async functions called directly with a stub workspace_id.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.api.routes import dashboard
from mayring_core.memory.store import init_memory_db


@pytest.fixture(autouse=True)
def _isolate_job_queue():
    """jobs_history reads the process-global job_queue._JOBS and returns only
    the most-recent `limit` jobs. Other test modules (test_job_progress,
    test_duel, test_v2_ops_endpoints) leave jobs there with current
    timestamps; without isolation they crowd out this file's older seeded
    jobs and the assertions flake depending on collection order. Snapshot →
    clear → restore makes these tests deterministic in any order."""
    from src.api import job_queue
    saved = dict(job_queue._JOBS)
    job_queue._JOBS.clear()
    try:
        yield
    finally:
        job_queue._JOBS.clear()
        job_queue._JOBS.update(saved)


@pytest.fixture
def seeded_db(tmp_path, monkeypatch) -> sqlite3.Connection:
    """Memory DB pre-loaded with one row per dashboard-relevant table."""
    monkeypatch.setattr(dashboard, "_conn", lambda: conn)
    db_path = tmp_path / "memory.db"
    conn = init_memory_db(db_path)
    now = datetime.now(timezone.utc).isoformat()

    conn.execute(
        "INSERT INTO ingestion_log (source_id, event_type, payload, created_at) "
        "VALUES (?,?,?,?)",
        ("repo:x:foo.py", "ingest_done", '{"chunks":3}', now),
    )
    conn.execute(
        "INSERT INTO sources (source_id, source_type, repo, path, captured_at, "
        "                     workspace_id, visibility, user_id) "
        "VALUES (?,?,?,?,?,?,?,?)",
        ("repo:x:foo.py", "repo_file", "x", "foo.py", now, "user-2", "private", "user-2"),
    )
    conn.execute(
        "INSERT INTO chunks (chunk_id, source_id, text, text_hash, created_at, "
        "                    workspace_id) "
        "VALUES (?,?,?,?,?,?)",
        ("chk_a", "repo:x:foo.py", "hello", "sha256:abc", now, "user-2"),
    )
    conn.execute(
        "INSERT INTO chunk_feedback (chunk_id, signal, created_at) "
        "VALUES (?,?,?)",
        ("chk_a", "positive", now),
    )
    conn.execute(
        "INSERT INTO chunk_source_refs (canonical_chunk_id, source_id, "
        "                                workspace_id, created_at) "
        "VALUES (?,?,?,?)",
        ("chk_a", "repo:x:foo.py", "user-2", now),
    )
    conn.execute(
        "INSERT INTO chunk_source_refs (canonical_chunk_id, source_id, "
        "                                workspace_id, created_at) "
        "VALUES (?,?,?,?)",
        ("chk_a", "repo:x:bar.py", "user-2", now),
    )
    conn.execute(
        "INSERT INTO context_feedback_log (trigger_ids, context_text, "
        "                                   was_referenced, led_to_retrieval, "
        "                                   relevance_score, captured_at) "
        "VALUES (?,?,?,?,?,?)",
        ('["t1"]', "ctx", 1, 1, 0.8, now),
    )
    conn.execute(
        "INSERT INTO trigger_stats (trigger_id, fire_count, ref_count, "
        "                            is_active, last_fired) "
        "VALUES (?,?,?,?,?)",
        ("t1", 10, 7, 1, now),
    )
    conn.execute(
        "INSERT INTO topic_transitions (from_topic, to_topic, count, last_seen) "
        "VALUES (?,?,?,?)",
        ("auth", "session", 5, now),
    )
    conn.execute(
        "INSERT INTO llm_calls_log (call_type, model, prompt, response, "
        "                            tool_calls, duration_ms, workspace_id, created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        ("vector_search", "nomic-embed-text", "stop hook",
         '{"vector_stage":"ok(max_score=0.5,matches=3,mean_dist=0.6)"}',
         0, 12, "user-2", now),
    )
    conn.commit()
    yield conn
    conn.close()


def _run(coro):
    # asyncio.run() each call: simpler than juggling a shared loop, and
    # Python 3.13 deprecates get_event_loop() outside an existing loop.
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 1 recent_ops
# ---------------------------------------------------------------------------

def test_recent_ops_returns_ingestion_events(seeded_db):
    res = _run(dashboard.recent_ops(workspace_id="user-2"))
    assert res["workspace_id"] == "user-2"
    assert len(res["ops"]) == 1
    assert res["ops"][0]["event_type"] == "ingest_done"
    assert res["ops"][0]["payload"] == {"chunks": 3}


# ---------------------------------------------------------------------------
# Hook-A: /stats/notifications/ingest (plugin watch-findings → hook_events)
# ---------------------------------------------------------------------------

def _admin_info():
    from src.api.jwt_auth import TokenInfo
    return TokenInfo(workspace_id="system", sub="admin", scopes=("*",))


def test_notifications_ingest_inserts_idempotent_and_enforces_allowlist(seeded_db):
    from src.api.routes.dashboard import NotificationIngest, NotificationEvent, ingest_notifications
    repo = "github.com/smoke/repo-notif-unit"  # → 'system' workspace (no user pollution)

    body = NotificationIngest(events=[
        NotificationEvent(hook_type="repo_pull", repo=repo, number=7,
                          summary="new PR", url="https://gh/pr/7"),
    ])
    res = _run(ingest_notifications(body=body, info=_admin_info()))
    assert res["ok"] is True
    assert res["inserted"] == 1

    # The row landed in hook_events under the resolved (system) workspace.
    row = seeded_db.execute(
        "SELECT hook_type, payload FROM hook_events WHERE hook_type='repo_pull'"
    ).fetchone()
    assert row is not None and '"number": 7' in row[1]

    # Idempotent: re-POST of the exact same event is skipped, not duplicated.
    res2 = _run(ingest_notifications(body=body, info=_admin_info()))
    assert res2["inserted"] == 0 and res2["skipped"] == 1

    # Allow-list: ci/security come via /repo-events → rejected here.
    ci = NotificationIngest(events=[NotificationEvent(hook_type="repo_ci", repo=repo)])
    res3 = _run(ingest_notifications(body=ci, info=_admin_info()))
    assert res3["inserted"] == 0 and res3["skipped"] == 1


def test_notifications_ingest_requires_privileged(seeded_db):
    from fastapi import HTTPException
    from src.api.jwt_auth import TokenInfo
    from src.api.routes.dashboard import NotificationIngest, NotificationEvent, ingest_notifications
    body = NotificationIngest(events=[
        NotificationEvent(hook_type="repo_pull", repo="github.com/smoke/repo-x"),
    ])
    with pytest.raises(HTTPException) as exc:
        _run(ingest_notifications(body=body, info=TokenInfo(workspace_id="u", sub="u", scopes=())))
    assert exc.value.status_code == 403


def test_recent_ops_filters_by_source_id(seeded_db):
    res = _run(dashboard.recent_ops(source_id="nope", workspace_id="user-2"))
    assert res["ops"] == []


# ---------------------------------------------------------------------------
# 3 feedback_log
# ---------------------------------------------------------------------------

def test_feedback_log_computes_referenced_rate(seeded_db):
    res = _run(dashboard.feedback_log(workspace_id="user-2"))
    assert res["injections_24h"] == 1
    assert res["referenced_24h"] == 1
    assert res["referenced_rate"] == 1.0
    assert res["recent"][0]["was_referenced"] is True


# ---------------------------------------------------------------------------
# 4 source_refs
# ---------------------------------------------------------------------------

def test_source_refs_groups_by_chunk(seeded_db):
    res = _run(dashboard.source_refs(min_sources=2, workspace_id="user-2"))
    assert len(res["refs"]) == 1
    assert res["refs"][0]["canonical_chunk_id"] == "chk_a"
    assert res["refs"][0]["source_count"] == 2


def test_source_refs_min_sources_filters_singletons(seeded_db):
    res = _run(dashboard.source_refs(min_sources=99, workspace_id="user-2"))
    assert res["refs"] == []


# ---------------------------------------------------------------------------
# 5 triggers
# ---------------------------------------------------------------------------

def test_triggers_computes_ratio(seeded_db):
    res = _run(dashboard.triggers(workspace_id="user-2"))
    assert len(res["triggers"]) == 1
    assert res["triggers"][0]["fire_count"] == 10
    assert res["triggers"][0]["ratio"] == 0.7


# ---------------------------------------------------------------------------
# 6 topic_flow
# ---------------------------------------------------------------------------

def test_topic_flow_returns_transitions(seeded_db):
    res = _run(dashboard.topic_flow(workspace_id="user-2"))
    assert len(res["flows"]) == 1
    assert res["flows"][0]["from_topic"] == "auth"
    assert res["flows"][0]["to_topic"] == "session"


def test_topic_flow_filter_from(seeded_db):
    res = _run(dashboard.topic_flow(from_topic="nope", workspace_id="user-2"))
    assert res["flows"] == []


# ---------------------------------------------------------------------------
# 7 pi_tasks (returns empty without pi_jobs schema, must not crash)
# ---------------------------------------------------------------------------

def test_pi_tasks_no_pi_jobs_table_returns_empty(seeded_db):
    """Older DBs lack pi_jobs; endpoint must fail soft."""
    seeded_db.execute("DROP TABLE IF EXISTS pi_jobs")
    seeded_db.commit()
    res = _run(dashboard.pi_tasks(workspace_id="user-2"))
    assert res["tasks"] == []


# ---------------------------------------------------------------------------
# 9 workspaces
# ---------------------------------------------------------------------------

def test_workspaces_tenant_sees_only_own(seeded_db, monkeypatch):
    """Non-admin token without memberships: scoped to active workspace_id only.

    WHY(tenancy-audit 2026-05-31): info is injected via Depends(get_token_info),
    NOT mcp_auth._TOKEN_CTX — the old test set _TOKEN_CTX (a ContextVar only the
    MCP sub-app populates), so it green-lit the broken REST path where info=None."""
    from src.api.jwt_auth import TokenInfo
    res = _run(dashboard.workspaces(
        workspace_id="user-2", info=TokenInfo(workspace_id="user-2", scopes=())))
    assert len(res["workspaces"]) == 1
    assert res["workspaces"][0]["workspace_id"] == "user-2"
    assert res["workspaces"][0]["chunks"] == 1


def test_workspaces_tenant_with_memberships_sees_all(seeded_db, monkeypatch):
    """V2: token with memberships[] lists every ws the user is a member of.
    Regression for #195: pre-V2 the non-admin branch returned exactly 1 row,
    hiding org-workspaces from the dashboard even when chunks were ingested."""
    from src.api.jwt_auth import TokenInfo, Membership
    # Seed: insert source FIRST (FK from chunks.source_id), then chunk.
    seeded_db.execute(
        "INSERT INTO sources (source_id, workspace_id, source_type, repo, path, captured_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("src-org", "ws-acme", "memory", "", "manual", "2026-05-10T10:00:00Z"),
    )
    seeded_db.execute(
        "INSERT INTO chunks (chunk_id, source_id, workspace_id, text, "
        "created_at, is_active) VALUES (?, ?, ?, ?, ?, 1)",
        ("c-org-1", "src-org", "ws-acme", "org-content", "2026-05-10T10:00:00Z"),
    )
    seeded_db.commit()

    info = TokenInfo(
        workspace_id="user-2",
        scopes=(),
        memberships=(
            Membership(id="user-2", type="personal", role="owner"),
            Membership(id="ws-acme", type="organization", role="editor"),
        ),
    )
    res = _run(dashboard.workspaces(workspace_id="user-2", info=info))
    ws_ids = {w["workspace_id"] for w in res["workspaces"]}
    assert ws_ids == {"user-2", "ws-acme"}
    types = {w["workspace_id"]: w["type"] for w in res["workspaces"]}
    assert types["user-2"] == "personal"
    assert types["ws-acme"] == "organization"


# ---------------------------------------------------------------------------
# 10 vector_trend
# ---------------------------------------------------------------------------

def test_vector_trend_parses_ok_diag(seeded_db):
    res = _run(dashboard.vector_trend(workspace_id="user-2"))
    assert res["logged_24h"] == 1
    assert res["success_rate"] == 1.0
    assert res["mean_max_score"] == pytest.approx(0.5, abs=0.01)


def test_vector_trend_no_logs_returns_zeroes(tmp_path, monkeypatch):
    """Empty llm_calls_log → all zeros, no division-by-zero."""
    db = init_memory_db(tmp_path / "empty.db")
    monkeypatch.setattr(dashboard, "_conn", lambda: db)
    res = _run(dashboard.vector_trend(workspace_id="user-2"))
    assert res["logged_24h"] == 0
    assert res["success_rate"] == 0.0
    assert res["mean_max_score"] == 0.0


# ---------------------------------------------------------------------------
# 8 activations + 2 jobs_history rely on in-process state — quick smoke tests
# ---------------------------------------------------------------------------

def test_activations_returns_recent_searches(monkeypatch):
    """RECENT_ACTIVATIONS deque populated by run_search; admin sees all."""
    from src.api import memory_service
    from src.api.jwt_auth import TokenInfo
    memory_service._RECENT_ACTIVATIONS.clear()
    memory_service._RECENT_ACTIVATIONS.append(
        {"workspace_id": "user-2", "query": "q1", "source_ids": ["s1"], "ts": 1}
    )
    memory_service._RECENT_ACTIVATIONS.append(
        {"workspace_id": "user-99", "query": "q2", "source_ids": ["s2"], "ts": 2}
    )
    res = asyncio.run(dashboard.activations(
        workspace_id="user-2", info=TokenInfo(workspace_id="user-2", scopes=("admin",))))
    assert len(res["activations"]) == 2


def test_activations_tenant_filters_to_own_workspace(monkeypatch):
    from src.api import memory_service
    from src.api.jwt_auth import TokenInfo
    memory_service._RECENT_ACTIVATIONS.clear()
    memory_service._RECENT_ACTIVATIONS.append(
        {"workspace_id": "user-2", "query": "q1", "source_ids": [], "ts": 1}
    )
    memory_service._RECENT_ACTIVATIONS.append(
        {"workspace_id": "user-99", "query": "q2", "source_ids": [], "ts": 2}
    )
    res = asyncio.run(dashboard.activations(
        workspace_id="user-2", info=TokenInfo(workspace_id="user-2", scopes=())))
    assert len(res["activations"]) == 1
    assert res["activations"][0]["query"] == "q1"


def test_jobs_history_reads_in_memory_dict():
    from src.api import job_queue
    job_queue._JOBS["jobX"] = {
        "job_id": "jobX",
        "status": "done",
        "started_at": "2026-05-08T00:00:00Z",
        "workspace_id": "user-2",
    }
    try:
        res = _run(dashboard.jobs_history(workspace_id="user-2"))
        assert any(j["job_id"] == "jobX" for j in res["jobs"])
    finally:
        del job_queue._JOBS["jobX"]


def test_jobs_history_reads_cross_worker_shared_file(tmp_path, monkeypatch):
    """A job created by ANOTHER uvicorn worker lives only in the shared state
    file, not this process's _JOBS. jobs_history must still surface it — under
    --workers 4 the Pi-Agent dashboard job list was empty before the _load_jobs
    merge (only saw the serving worker's per-process _JOBS)."""
    from src.api import job_queue
    monkeypatch.setattr(job_queue, "_JOBS_STATE_FILE", tmp_path / "jobs.json")
    job_queue._JOBS["fromA"] = {
        "job_id": "fromA", "status": "done",
        "started_at": "2026-05-08T00:00:00Z", "workspace_id": "user-2",
    }
    job_queue._save_jobs()        # worker A persists to the shared file
    job_queue._JOBS.clear()       # simulate worker B: never saw it in-process
    res = _run(dashboard.jobs_history(workspace_id="user-2"))
    assert any(j["job_id"] == "fromA" for j in res["jobs"]), \
        "jobs_history must read the cross-worker shared file, not just local _JOBS"


def test_jobs_history_filters_other_workspaces(tmp_path, monkeypatch):
    """Bene's dashboard must not show jobs that belong to another tenant
    or to the system maintenance bucket — pre-fix, smoke-runs (ws=system)
    leaked into the user-facing memory-dashboard job history.

    Isolated state-file: workspace='system' sees ALL jobs, so a populated
    local cache/jobs_state.json would crowd the seeds past limit=50 and flake.
    """
    from src.api import job_queue
    monkeypatch.setattr(job_queue, "_JOBS_STATE_FILE", tmp_path / "jobs.json")
    job_queue._JOBS.update({
        "ws_bene":   {"job_id": "ws_bene",   "status": "done",
                      "started_at": "2026-05-08", "workspace_id": "bene"},
        "ws_system": {"job_id": "ws_system", "status": "done",
                      "started_at": "2026-05-08", "workspace_id": "system"},
        "ws_other":  {"job_id": "ws_other",  "status": "done",
                      "started_at": "2026-05-08", "workspace_id": "alice"},
    })
    try:
        # Bene-tenant: nur eigene jobs
        res = _run(dashboard.jobs_history(workspace_id="bene"))
        ids = {j["job_id"] for j in res["jobs"]}
        assert "ws_bene" in ids
        assert "ws_system" not in ids
        assert "ws_other" not in ids

        # system (Service-Token / Admin): sieht alles
        res = _run(dashboard.jobs_history(workspace_id="system"))
        ids = {j["job_id"] for j in res["jobs"]}
        assert {"ws_bene", "ws_system", "ws_other"} <= ids
    finally:
        for k in ("ws_bene", "ws_system", "ws_other"):
            job_queue._JOBS.pop(k, None)


def test_jobs_history_hides_smoke_jobs_by_default(tmp_path, monkeypatch):
    """#253: smoke-getriggerte populate-jobs (source='smoke') failen bewusst
    und landen in workspace:system — sie rauschen die job-history zu. Default
    blendet sie aus. Der smoke-CHECK bleibt grün (er prüft /jobs/{id}, nicht
    diese history). Isoliertes state-file, damit echte cache-jobs die Seeds
    nicht über das limit verdrängen."""
    from src.api import job_queue
    monkeypatch.setattr(job_queue, "_JOBS_STATE_FILE", tmp_path / "jobs.json")
    job_queue._JOBS.update({
        "real1":  {"job_id": "real1", "status": "done", "started_at": "2026-05-08",
                   "workspace_id": "system"},
        "smoke1": {"job_id": "smoke1", "status": "error", "started_at": "2026-05-08",
                   "workspace_id": "system", "source": "smoke"},
    })
    try:
        res = _run(dashboard.jobs_history(workspace_id="system"))
        ids = {j["job_id"] for j in res["jobs"]}
        assert "real1" in ids
        assert "smoke1" not in ids, "smoke-jobs müssen per default ausgeblendet sein"
    finally:
        for k in ("real1", "smoke1"):
            job_queue._JOBS.pop(k, None)


def test_jobs_history_includes_smoke_when_requested(tmp_path, monkeypatch):
    """include_smoke=True macht die gefilterten jobs sichtbar; der source-tag
    wird mit ausgegeben, damit die UI sie als smoke labeln kann."""
    from src.api import job_queue
    monkeypatch.setattr(job_queue, "_JOBS_STATE_FILE", tmp_path / "jobs.json")
    job_queue._JOBS["smoke2"] = {
        "job_id": "smoke2", "status": "error", "started_at": "2026-05-08",
        "workspace_id": "system", "source": "smoke",
    }
    try:
        res = _run(dashboard.jobs_history(workspace_id="system", include_smoke=True))
        smoke = next((j for j in res["jobs"] if j["job_id"] == "smoke2"), None)
        assert smoke is not None
        assert smoke["source"] == "smoke"
    finally:
        job_queue._JOBS.pop("smoke2", None)


def test_jobs_history_status_filter():
    from src.api import job_queue
    job_queue._JOBS.update({
        "j1": {"job_id": "j1", "status": "done", "started_at": "2026-05-08", "workspace_id": "user-2"},
        "j2": {"job_id": "j2", "status": "error", "started_at": "2026-05-08", "workspace_id": "user-2"},
    })
    try:
        res = _run(dashboard.jobs_history(status="error", workspace_id="user-2"))
        statuses = {j["status"] for j in res["jobs"]}
        assert statuses == {"error"}
    finally:
        del job_queue._JOBS["j1"]
        del job_queue._JOBS["j2"]


def test_jobs_history_includes_error_tail_for_failed_jobs():
    """Bug: app.linn.games-Dashboard zeigte status='error' ohne jegliche
    Detail-Info — das output-Field wurde server-side gestrippt. Fix: bei
    status='error' liefert die Response jetzt error_tail (letzte 1200
    chars vom output, typischerweise Traceback)."""
    from src.api import job_queue
    long_traceback = (
        "INFO: starting ingestion ...\n"
        "WARNING: connection slow\n"
        + ("filler-line\n" * 200)
        + "Traceback (most recent call last):\n"
        + '  File "/app/src/foo.py", line 42, in handle\n'
        + "    raise ValueError(\"unexpected token\")\n"
        + "ValueError: unexpected token\n"
    )
    job_queue._JOBS.update({
        "err1": {
            "job_id": "err1", "status": "error", "started_at": "2026-05-08",
            "workspace_id": "user-2", "output": long_traceback,
        },
        "ok1": {
            "job_id": "ok1", "status": "done", "started_at": "2026-05-08",
            "workspace_id": "user-2", "output": "all good",
        },
        "err_short": {
            "job_id": "err_short", "status": "error", "started_at": "2026-05-08",
            "workspace_id": "user-2", "output": "short fail message",
        },
    })
    try:
        res = _run(dashboard.jobs_history(workspace_id="user-2"))
        by_id = {j["job_id"]: j for j in res["jobs"]}

        # done-jobs: error_tail muss None sein (kein leak von output bei success)
        assert by_id["ok1"]["error_tail"] is None

        # error-jobs: error_tail enthält den Traceback
        tail = by_id["err1"]["error_tail"]
        assert tail is not None
        assert "ValueError: unexpected token" in tail
        assert tail.startswith("…\n"), "lange outputs müssen mit Ellipsis prefix kommen"
        assert len(tail) <= 1300, "error_tail darf nicht das gesamte output zurückspielen"

        # kurze error-outputs: kein Truncate, kein Ellipsis-prefix
        short_tail = by_id["err_short"]["error_tail"]
        assert short_tail == "short fail message"
    finally:
        for k in ("err1", "ok1", "err_short"):
            job_queue._JOBS.pop(k, None)


def test_jobs_history_error_tail_none_when_output_empty():
    """Edge: wenn ein error-Job ohne output-string gespeichert wurde
    (z.B. crash vor erstem stdout), darf error_tail nicht leerer String
    sein — UI testet truthiness, leerer String würde 'Details'-Button
    fälschlich aktivieren ohne Inhalt."""
    from src.api import job_queue
    job_queue._JOBS["empty_err"] = {
        "job_id": "empty_err", "status": "error", "started_at": "2026-05-08",
        "workspace_id": "user-2",
    }
    try:
        res = _run(dashboard.jobs_history(workspace_id="user-2"))
        j = next(x for x in res["jobs"] if x["job_id"] == "empty_err")
        assert j["error_tail"] is None
    finally:
        del job_queue._JOBS["empty_err"]


# ---------------------------------------------------------------------------
# job_queue persistence: round-trip via JSON file
# ---------------------------------------------------------------------------

def test_save_and_load_jobs_round_trip(tmp_path, monkeypatch):
    """_save_jobs writes the dict, _load_jobs reads it back identically.

    Avoids `importlib.reload` because other tests in the suite hold imported
    references to ``_JOBS`` and would see a stale dict after the reload.
    """
    from src.api import job_queue

    state = tmp_path / "jobs.json"
    monkeypatch.setattr(job_queue, "_JOBS_STATE_FILE", state)

    job_queue._JOBS["jPersist"] = {
        "job_id": "jPersist",
        "status": "done",
        "started_at": "2026-05-08T00:00:00Z",
        "workspace_id": "user-2",
    }
    try:
        job_queue._save_jobs()
        assert state.exists()
        loaded = job_queue._load_jobs()
        assert "jPersist" in loaded
        assert loaded["jPersist"]["status"] == "done"
    finally:
        del job_queue._JOBS["jPersist"]


def test_make_job_records_source(tmp_path, monkeypatch):
    """#253: make_job tags the record with an optional source so the dashboard
    can default-filter smoke-triggered jobs. Default is empty (visible)."""
    from src.api import job_queue
    monkeypatch.setattr(job_queue, "_JOBS_STATE_FILE", tmp_path / "jobs.json")
    jid_smoke = job_queue.make_job("system", source="smoke")
    jid_plain = job_queue.make_job("system")
    try:
        assert job_queue._JOBS[jid_smoke]["source"] == "smoke"
        assert job_queue._JOBS[jid_plain]["source"] == ""
    finally:
        del job_queue._JOBS[jid_smoke]
        del job_queue._JOBS[jid_plain]


def test_load_jobs_handles_missing_file(tmp_path, monkeypatch):
    from src.api import job_queue
    monkeypatch.setattr(job_queue, "_JOBS_STATE_FILE", tmp_path / "nope.json")
    assert job_queue._load_jobs() == {}


def test_load_jobs_handles_corrupt_file(tmp_path, monkeypatch):
    """A truncated/garbage state file must not take the API down."""
    from src.api import job_queue
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    monkeypatch.setattr(job_queue, "_JOBS_STATE_FILE", bad)
    assert job_queue._load_jobs() == {}


def test_make_job_writes_state_file(tmp_path, monkeypatch):
    """make_job() triggers a save so the dashboard sees the row immediately."""
    from src.api import job_queue
    state = tmp_path / "jobs.json"
    monkeypatch.setattr(job_queue, "_JOBS_STATE_FILE", state)
    jid = job_queue.make_job("user-2")
    try:
        assert state.exists()
        payload = json.loads(state.read_text())
        assert jid in payload
        assert payload[jid]["workspace_id"] == "user-2"
    finally:
        del job_queue._JOBS[jid]
