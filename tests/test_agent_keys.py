import pytest

from src.api import agent_keys as ak


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(ak, "_store_path", lambda: tmp_path / "agent_keys.json")
    return tmp_path


def test_mint_returns_short_plaintext_and_verifies(store):
    plaintext, rec = ak.mint("ws1", label="research")
    assert plaintext.startswith("mca_")
    assert len(plaintext) <= 60          # well under Langdock's 1000-char limit
    assert rec["label"] == "research" and rec["workspace_id"] == "ws1"
    # plaintext verifies → resolves workspace + label, never stored in clear
    res = ak.verify(plaintext)
    assert res is not None
    assert res["workspace_id"] == "ws1" and res["label"] == "research"


def test_verify_unknown_key_is_none(store):
    ak.mint("ws1", label="research")
    assert ak.verify("mca_bogus") is None


def test_revoke_disables_verify(store):
    plaintext, rec = ak.mint("ws1", label="research")
    ak.revoke("ws1", rec["key_id"])
    assert ak.verify(plaintext) is None  # revoked → no longer valid


def test_list_excludes_secret_material(store):
    p1, r1 = ak.mint("ws1", label="research")
    listed = ak.list_keys("ws1")
    assert len(listed) == 1
    row = listed[0]
    assert row["label"] == "research" and row["key_id"] == r1["key_id"]
    assert "key_hash" not in row and "plaintext" not in row  # never leak the secret
    # the plaintext is NOT recoverable from the list
    assert all("mca_" not in str(v) for v in row.values())


def test_workspace_isolation_on_revoke(store):
    p1, r1 = ak.mint("ws1", label="a")
    # ws2 cannot revoke ws1's key
    ak.revoke("ws2", r1["key_id"])
    assert ak.verify(p1) is not None  # still valid — foreign revoke ignored


def test_route_mint_list_revoke(store):
    import asyncio
    from src.api.routes import agent_keys as route

    out = asyncio.run(route.mint_agent_key(route.MintRequest(label="research"), workspace_id="ws1"))
    assert out["api_key"].startswith("mca_")
    kid = out["key"]["key_id"]
    listed = asyncio.run(route.list_agent_keys(workspace_id="ws1"))
    assert any(k["key_id"] == kid and k["label"] == "research" for k in listed["keys"])
    # the list endpoint never returns the plaintext
    assert all("api_key" not in k and "mca_" not in str(k.values()) for k in listed["keys"])
    rev = asyncio.run(route.revoke_agent_key(kid, workspace_id="ws1"))
    assert rev["ok"] and ak.verify(out["api_key"]) is None


def test_route_revoke_foreign_404(store):
    import asyncio
    import pytest as _pytest
    from fastapi import HTTPException
    from src.api.routes import agent_keys as route
    out = asyncio.run(route.mint_agent_key(route.MintRequest(label="x"), workspace_id="ws1"))
    with _pytest.raises(HTTPException) as e:
        asyncio.run(route.revoke_agent_key(out["key"]["key_id"], workspace_id="ws2"))
    assert e.value.status_code == 404
