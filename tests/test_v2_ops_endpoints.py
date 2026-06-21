"""Issue #52-55 — v2.0 operational endpoints + post-ingest chain.

Tests ensure:
  - /wiki/generate, /ambient/snapshot, /predictive/rebuild-transitions
    all spawn a background job via _run_checker_job with the right flags,
    without requiring a server restart (KISS: no CLI-only path).
  - /populate and /issues/ingest trigger the v2.0 chain after success.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    # Bypass Laravel-Sanctum auth; every request gets workspace="default"
    from src.api import server as srv
    from src.api import auth as auth_module

    async def _fake_get_workspace():
        return "default"

    srv.app.dependency_overrides[auth_module.get_workspace] = _fake_get_workspace
    yield TestClient(srv.app)
    srv.app.dependency_overrides.clear()


def _call(client, endpoint, payload):
    return client.post(endpoint, json=payload, headers={"Authorization": "Bearer tst"})


def _cmd_for(mock_run_job, call_index: int) -> list[str]:
    return list(mock_run_job.call_args_list[call_index].args[1])


class TestV2Endpoints:
    def test_wiki_generate_spawns_job_with_flag(self, client):
        with patch("src.api.routes.jobs._run_checker_job", new_callable=AsyncMock) as m:
            r = _call(client, "/wiki/generate",
                      {"repo": "https://github.com/a/b", "wiki_type": "code"})
        assert r.status_code == 200
        assert r.json()["status"] == "started"
        cmd = _cmd_for(m, 0)
        assert "--generate-wiki" in cmd
        assert "--wiki-type" in cmd
        assert cmd[cmd.index("--wiki-type") + 1] == "code"
        assert cmd[cmd.index("--repo") + 1] == "https://github.com/a/b"

    def test_ambient_snapshot_spawns_generate_ambient(self, client):
        with patch("src.api.routes.jobs._run_checker_job", new_callable=AsyncMock) as m:
            r = _call(client, "/ambient/snapshot", {"repo": "https://github.com/a/b"})
        assert r.status_code == 200
        assert "--generate-ambient" in _cmd_for(m, 0)

    def test_predictive_rebuild_spawns_transitions_flag(self, client):
        with patch("src.api.routes.jobs._run_checker_job", new_callable=AsyncMock) as m:
            r = _call(client, "/predictive/rebuild-transitions", {"repo": None})
        assert r.status_code == 200
        assert "--rebuild-transitions" in _cmd_for(m, 0)

    def test_predictive_rebuild_respects_optional_repo(self, client):
        with patch("src.api.routes.jobs._run_checker_job", new_callable=AsyncMock) as m:
            r = _call(client, "/predictive/rebuild-transitions",
                      {"repo": "https://github.com/a/b"})
        assert r.status_code == 200
        cmd = _cmd_for(m, 0)
        assert "--rebuild-transitions" in cmd
        assert cmd[cmd.index("--repo") + 1] == "https://github.com/a/b"


class TestPostIngestV2Chain:
    def test_populate_fires_full_v2_chain_on_success(self, client):
        from src.api import server as srv

        calls: list[list[str]] = []

        async def _fake_checker(job_id, args, workspace_id):
            calls.append(list(args))
            srv._JOBS[job_id]["status"] = "done"

        with patch("src.api.routes.jobs._run_checker_job", side_effect=_fake_checker):
            r = _call(client, "/populate", {"repo": "https://github.com/a/b"})
        assert r.status_code == 200
        job_id = r.json()["job_id"]

        # Chain: overview + wiki + ambient + predictive + images + rationale
        assert set(srv._JOBS[job_id]["v2_jobs"].keys()) == {
            "overview", "wiki", "ambient", "predictive", "images", "rationale",
        }
        for v2_id in srv._JOBS[job_id]["v2_jobs"].values():
            assert v2_id in srv._JOBS

        assert any("--generate-ambient" in a for a in calls)
        assert any("--rebuild-transitions" in a for a in calls)
        assert any("overview" in a and "--mode" in a for a in calls)
        assert any("--generate-wiki" in a for a in calls)
        # Regression: populate must carry --memory-categorize so chunks get labels
        populate_call = next(a for a in calls if "--populate-memory" in a)
        assert "--memory-categorize" in populate_call

    def test_wiki_skipped_when_overview_fails(self, client):
        from src.api import server as srv

        async def _fake_checker(job_id, args, workspace_id):
            if "overview" in args:
                srv._JOBS[job_id]["status"] = "error"
            else:
                srv._JOBS[job_id]["status"] = "done"

        with patch("src.api.routes.jobs._run_checker_job", side_effect=_fake_checker):
            r = _call(client, "/populate", {"repo": "https://github.com/a/b"})
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        wiki_id = srv._JOBS[job_id]["v2_jobs"]["wiki"]
        assert srv._JOBS[wiki_id]["status"] == "error"
        assert "overview" in srv._JOBS[wiki_id]["output"]

    def test_populate_skips_v2_chain_on_failure(self, client):
        from src.api import server as srv

        async def _fake_checker(job_id, args, workspace_id):
            srv._JOBS[job_id]["status"] = "error"

        with patch("src.api.routes.jobs._run_checker_job", side_effect=_fake_checker):
            r = _call(client, "/populate", {"repo": "https://github.com/a/b"})
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        assert "v2_jobs" not in srv._JOBS[job_id]

    def test_issues_ingest_also_fires_v2_chain(self, client):
        from src.api import server as srv

        async def _fake_checker(job_id, args, workspace_id):
            srv._JOBS[job_id]["status"] = "done"

        with patch("src.api.routes.jobs._run_checker_job", side_effect=_fake_checker):
            r = _call(client, "/issues/ingest",
                      {"repo": "https://github.com/a/b", "state": "open"})
        assert r.status_code == 200
        assert "v2_jobs" in srv._JOBS[r.json()["job_id"]]


class TestLostUpdateGuard:
    """A populate clones HEAD-at-run; rapid pushes during the (long) ingest get
    swallowed by the debounce. The guard re-ingests once when HEAD moved past the
    triggering commit. See jobs.py / repo_events.py."""

    def _run_populate_chain(self, monkeypatch, *, trigger_sha, head_now, source=""):
        import asyncio
        from src.api.routes import jobs as J
        from src.api import job_queue as Q

        async def _fake_checker(job_id, args, workspace_id):
            Q._JOBS[job_id]["status"] = "done"

        monkeypatch.setattr(J, "_run_checker_job", _fake_checker)
        monkeypatch.setattr(J, "_resolve_head_sha", lambda repo: head_now)

        followups: list[str] = []

        def _spy_enqueue(repo, ws, extra_args=None, source="", head_sha=None):
            followups.append(source)
            return "spy-followup"

        job_id = Q.make_job("default", repo="https://github.com/a/b",
                            source=source, head_sha=trigger_sha)
        monkeypatch.setattr(J, "enqueue_populate", _spy_enqueue)
        asyncio.run(J._run_with_v2_postingest(
            job_id, ["--populate-memory"], "default", "https://github.com/a/b"))
        return followups

    def test_reingest_on_sha_mismatch(self, monkeypatch):
        followups = self._run_populate_chain(
            monkeypatch, trigger_sha="oldsha", head_now="newsha")
        assert followups == ["lost-update-followup"]

    def test_no_reingest_on_sha_match(self, monkeypatch):
        followups = self._run_populate_chain(
            monkeypatch, trigger_sha="samesha", head_now="samesha")
        assert followups == []

    def test_followup_does_not_recurse(self, monkeypatch):
        # a guard-spawned follow-up must NOT spawn another, even if HEAD moved
        followups = self._run_populate_chain(
            monkeypatch, trigger_sha="oldsha", head_now="newsha",
            source="lost-update-followup")
        assert followups == []

    def test_manual_populate_without_sha_skips_guard(self, monkeypatch):
        # manual /populate carries no triggering sha → no guard, no network call
        followups = self._run_populate_chain(
            monkeypatch, trigger_sha=None, head_now="whatever")
        assert followups == []
