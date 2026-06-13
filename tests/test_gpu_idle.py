"""Spec B1: qwen-activity stamp + GET /stats/gpu-idle.

Time is controlled via monkeypatch (no sleep). The stamp is forced to the
cache-file backend by leaving MAYRING_REDIS_URL unset and pointing
MAYRING_CACHE_DIR at a tmp dir, so the test never touches a real Redis.
"""
from __future__ import annotations

import pytest

from src.api.jwt_auth import TokenInfo
from src.api.routes.stats_admin import gpu_idle


@pytest.fixture(autouse=True)
def file_backed_stamp(tmp_path, monkeypatch):
    monkeypatch.delenv("MAYRING_REDIS_URL", raising=False)
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    import mayring_core.qwen_activity as qa
    # reset the lazy redis cache so the env change takes effect per test
    qa._redis_client = None
    qa._redis_tried = False
    yield


def _admin() -> TokenInfo:
    return TokenInfo(workspace_id="ws-test", scopes=("*",))


def test_fresh_stamp_not_idle(monkeypatch):
    import mayring_core.qwen_activity as qa
    qa.stamp(now=1000.0)
    monkeypatch.setattr("src.api.routes.stats_admin.time.time", lambda: 1005.0)
    res = gpu_idle(idle_for=20, info=_admin())
    assert res["idle"] is False
    assert res["idle_seconds"] == pytest.approx(5.0)
    assert res["last_activity"] is not None


def test_old_stamp_is_idle(monkeypatch):
    import mayring_core.qwen_activity as qa
    qa.stamp(now=1000.0)
    monkeypatch.setattr("src.api.routes.stats_admin.time.time", lambda: 1100.0)
    res = gpu_idle(idle_for=20, info=_admin())
    assert res["idle"] is True
    assert res["idle_seconds"] == pytest.approx(100.0)


def test_missing_stamp_is_idle():
    res = gpu_idle(idle_for=20, info=_admin())
    assert res["idle"] is True
    assert res["idle_seconds"] == float("inf")
    assert res["last_activity"] is None


def test_idle_for_param_threshold(monkeypatch):
    import mayring_core.qwen_activity as qa
    qa.stamp(now=1000.0)
    monkeypatch.setattr("src.api.routes.stats_admin.time.time", lambda: 1030.0)
    # 30s idle: below 60 → busy, at/above 20 → idle
    assert gpu_idle(idle_for=60, info=_admin())["idle"] is False
    assert gpu_idle(idle_for=20, info=_admin())["idle"] is True


def test_stamp_helper_updates_value():
    import mayring_core.qwen_activity as qa
    qa.stamp(now=1000.0)
    assert qa.last_activity() == pytest.approx(1000.0)
    qa.stamp(now=2000.0)
    assert qa.last_activity() == pytest.approx(2000.0)


def test_non_admin_forbidden():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        gpu_idle(idle_for=20, info=TokenInfo(workspace_id="ws", scopes=("mcp:memory",)))
    assert exc.value.status_code == 403


def test_text_generate_stamps_but_vision_does_not(monkeypatch):
    """The chokepoint: imageless generate() stamps; an images= (vision) call must not."""
    import mayring_core.qwen_activity as qa
    import mayring_core.ollama_client as oc

    calls: list[float] = []
    monkeypatch.setattr(qa, "stamp", lambda now=None: calls.append(now or 1.0))
    monkeypatch.setattr(oc, "_generate_call", lambda *a, **k: "ok")
    monkeypatch.setattr(oc, "_should_route_cloud_primary", lambda: False)

    oc.generate("http://x", "qwen3.5-mayring:2b", "hi", stream=False)
    assert len(calls) == 1  # text route stamped

    oc.generate("http://x", "qwen2.5vl:3b", "caption", images=["b64"], stream=False)
    assert len(calls) == 1  # vision route did NOT stamp
