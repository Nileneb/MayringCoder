import asyncio


def _maybe_run(c):
    return asyncio.run(c) if asyncio.iscoroutine(c) else c
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from src.api.jwt_auth import TokenInfo
from src.api.routes import ambient_admin


def _admin():
    return TokenInfo(workspace_id="ws1", scopes=("*",))


def _user():
    return TokenInfo(workspace_id="ws1", scopes=("mcp:memory",))


def test_ambient_refresh_requires_admin():
    with pytest.raises(HTTPException) as e:
        _maybe_run(ambient_admin.trigger_ambient_refresh(info=_user(), repo_slug="x", model="m"))
    assert e.value.status_code == 403


def test_ambient_refresh_calls_generate_and_returns_chars():
    with patch("mayring_core.memory.ambient.generate_ambient_snapshot",
               return_value="snapshot text 123") as gen:
        out = _maybe_run(ambient_admin.trigger_ambient_refresh(
            info=_admin(), repo_slug="myrepo", model="m", workspace_id="ws1"))
    assert gen.called
    assert out["generated"] is True
    assert out["chars"] == len("snapshot text 123")
    assert out["repo_slug"] == "myrepo"
