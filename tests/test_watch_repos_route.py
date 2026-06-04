import importlib


def test_set_watch_repo_threads_hook_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    import mayring_core.config as cfg
    importlib.reload(cfg)
    from src.api import watch_store
    importlib.reload(watch_store)

    captured = {}

    def fake_set(ws, slug, **kw):
        captured.update({"ws": ws, "slug": slug, **kw})
        return {"repo_slug": slug, "active": kw["active"]}

    monkeypatch.setattr(watch_store, "set_watched", fake_set)
    monkeypatch.setattr("src.api.routes.jobs.enqueue_populate", lambda *a, **k: "job-1")

    # Mount ONLY the watch_repos router (avoids importing src.api.server, whose
    # a2a_relay import needs the a2a-sdk that is absent in the local venv).
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.routes import watch_repos as wr
    from src.api.auth import get_workspace
    app = FastAPI()
    app.include_router(wr.router)
    app.dependency_overrides[get_workspace] = lambda: "wsX"
    client = TestClient(app)
    r = client.post("/stats/watch-repos", json={
        "repo_slug": "Nileneb/Foo", "active": True, "alerts": ["ci"],
        "hook_id": 77, "secret": "shh", "source": "webhook",
    })
    app.dependency_overrides.clear()
    assert r.status_code == 200, r.text
    assert captured["hook_id"] == 77
    assert captured["secret"] == "shh"
    assert captured["source"] == "webhook"
