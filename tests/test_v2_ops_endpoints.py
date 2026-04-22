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
        with patch("src.api.server._run_checker_job", new_callable=AsyncMock) as m:
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
        with patch("src.api.server._run_checker_job", new_callable=AsyncMock) as m:
            r = _call(client, "/ambient/snapshot", {"repo": "https://github.com/a/b"})
        assert r.status_code == 200
        assert "--generate-ambient" in _cmd_for(m, 0)

    def test_predictive_rebuild_spawns_transitions_flag(self, client):
        with patch("src.api.server._run_checker_job", new_callable=AsyncMock) as m:
            r = _call(client, "/predictive/rebuild-transitions", {"repo": None})
        assert r.status_code == 200
        assert "--rebuild-transitions" in _cmd_for(m, 0)

    def test_predictive_rebuild_respects_optional_repo(self, client):
        with patch("src.api.server._run_checker_job", new_callable=AsyncMock) as m:
            r = _call(client, "/predictive/rebuild-transitions",
                      {"repo": "https://github.com/a/b"})
        assert r.status_code == 200
        cmd = _cmd_for(m, 0)
        assert "--rebuild-transitions" in cmd
        assert cmd[cmd.index("--repo") + 1] == "https://github.com/a/b"


class TestPostIngestV2Chain:
    def test_populate_fires_wiki_ambient_predictive_on_success(self, client):
        from src.api import server as srv

        async def _fake_checker(job_id, args, workspace_id):
            srv._JOBS[job_id]["status"] = "done"

        with patch("src.api.server._run_checker_job", side_effect=_fake_checker):
            r = _call(client, "/populate", {"repo": "https://github.com/a/b"})
        assert r.status_code == 200
        job_id = r.json()["job_id"]

        # The post-ingest chain sets _JOBS[job_id]['v2_jobs'] with 3 entries
        assert set(srv._JOBS[job_id]["v2_jobs"].keys()) == {"wiki", "ambient", "predictive"}
        # all three v2 job ids exist in the registry
        for v2_id in srv._JOBS[job_id]["v2_jobs"].values():
            assert v2_id in srv._JOBS

    def test_populate_skips_v2_chain_on_failure(self, client):
        from src.api import server as srv

        async def _fake_checker(job_id, args, workspace_id):
            srv._JOBS[job_id]["status"] = "error"

        with patch("src.api.server._run_checker_job", side_effect=_fake_checker):
            r = _call(client, "/populate", {"repo": "https://github.com/a/b"})
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        assert "v2_jobs" not in srv._JOBS[job_id]

    def test_issues_ingest_also_fires_v2_chain(self, client):
        from src.api import server as srv

        async def _fake_checker(job_id, args, workspace_id):
            srv._JOBS[job_id]["status"] = "done"

        with patch("src.api.server._run_checker_job", side_effect=_fake_checker):
            r = _call(client, "/issues/ingest",
                      {"repo": "https://github.com/a/b", "state": "open"})
        assert r.status_code == 200
        assert "v2_jobs" in srv._JOBS[r.json()["job_id"]]
