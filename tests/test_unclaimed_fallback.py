import importlib


def test_effective_workspace_unclaimed(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    import src.api.mcp_auth as m
    importlib.reload(m)
    monkeypatch.setattr(m, "_current_token_info", lambda: None)
    ws1 = m._effective_workspace_id()
    ws2 = m._effective_workspace_id()
    assert ws1.startswith("unclaimed:")
    assert ws1 == ws2          # stable device id
    assert ws1 != "default"


def test_local_device_id_persists(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    import src.api.mcp_auth as m
    importlib.reload(m)
    a = m._local_device_id()
    b = m._local_device_id()
    assert a == b and len(a) >= 8
