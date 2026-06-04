import importlib


def _fresh_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    import mayring_core.config as cfg
    importlib.reload(cfg)
    from src.api import watch_store
    importlib.reload(watch_store)
    return watch_store


def test_set_and_find_active_by_repo_with_secret(tmp_path, monkeypatch):
    ws = _fresh_store(tmp_path, monkeypatch)
    ws.set_watched("wsA", "Nileneb/Foo", active=True, alerts=["ci"],
                   ingested_at=None, hook_id=123, secret="s3cr3t", source="webhook")
    rec = ws.find_active_by_repo("nileneb/foo")
    assert rec is not None
    assert rec["workspace_id"] == "wsA"
    assert rec["secret"] == "s3cr3t"
    assert rec["hook_id"] == 123
    listed = ws.get_watched("wsA")
    assert listed[0]["hook_id"] == 123
    assert "secret" not in listed[0]


def test_find_active_by_repo_ignores_inactive(tmp_path, monkeypatch):
    ws = _fresh_store(tmp_path, monkeypatch)
    ws.set_watched("wsA", "Nileneb/Foo", active=False, alerts=["ci"],
                   hook_id=9, secret="x", source="webhook")
    assert ws.find_active_by_repo("Nileneb/Foo") is None
