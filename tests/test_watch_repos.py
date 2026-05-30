import pytest

from src.api import watch_store as ws


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_store_path", lambda: tmp_path / "watched_repos.json")
    return tmp_path


def test_set_and_get_roundtrip(store):
    ws.set_watched("ws1", "nileneb/foo", active=True, alerts=["ci", "code_scanning"])
    repos = ws.get_watched("ws1")
    assert len(repos) == 1
    r = repos[0]
    assert r["repo_slug"] == "nileneb/foo"
    assert r["active"] is True
    assert r["alerts"] == ["ci", "code_scanning"]


def test_toggle_inactive(store):
    ws.set_watched("ws1", "nileneb/foo", active=True, alerts=["ci"])
    ws.set_watched("ws1", "nileneb/foo", active=False, alerts=["ci"])
    repos = ws.get_watched("ws1")
    assert repos[0]["active"] is False


def test_workspace_isolation(store):
    ws.set_watched("ws1", "nileneb/foo", active=True, alerts=["ci"])
    ws.set_watched("ws2", "nileneb/bar", active=True, alerts=["ci"])
    assert {r["repo_slug"] for r in ws.get_watched("ws1")} == {"nileneb/foo"}
    assert {r["repo_slug"] for r in ws.get_watched("ws2")} == {"nileneb/bar"}


def test_active_slugs_helper(store):
    ws.set_watched("ws1", "nileneb/foo", active=True, alerts=["ci"])
    ws.set_watched("ws1", "nileneb/bar", active=False, alerts=["ci"])
    active = ws.active_watch_map("ws1")
    assert active == {"nileneb/foo": ["ci"]}  # only active repos, slug→alerts


def test_route_activate_triggers_ingest(store, monkeypatch):
    import asyncio
    from unittest.mock import patch
    from src.api.routes import watch_repos as route

    with patch("src.api.routes.jobs.enqueue_populate", return_value="job-123") as enq:
        out = asyncio.run(route.set_watch_repo(
            route.WatchRepoRequest(repo_slug="nileneb/foo", active=True, alerts=["ci"]),
            workspace_id="ws1"))
    assert enq.called and enq.call_args[0][0] == "nileneb/foo"
    assert out["repo"]["active"] is True and out["repo"]["ingested_at"]
    listed = asyncio.run(route.list_watch_repos(workspace_id="ws1"))
    assert listed["repos"][0]["repo_slug"] == "nileneb/foo"


def test_route_deactivate_no_ingest(store):
    import asyncio
    from unittest.mock import patch
    from src.api.routes import watch_repos as route
    with patch("src.api.routes.jobs.enqueue_populate") as enq:
        asyncio.run(route.set_watch_repo(
            route.WatchRepoRequest(repo_slug="nileneb/foo", active=False, alerts=["ci"]),
            workspace_id="ws1"))
    assert not enq.called  # deactivating must not ingest
