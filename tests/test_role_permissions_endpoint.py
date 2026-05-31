import asyncio
from src.api.routes import role_permissions as rp
from src.api.jwt_auth import TokenInfo, Membership


def _admin_info(ws):
    return TokenInfo(workspace_id=ws, scopes=(), memberships=(
        Membership(id=ws, type="organization", role="owner", name="Acme"),))


def test_get_returns_effective_matrix(tmp_path, monkeypatch):
    from mayring_core.memory import store
    conn = store.init_memory_db(tmp_path / "m.db")
    monkeypatch.setattr(rp, "_get_conn", lambda: conn)
    res = asyncio.run(rp.get_role_permissions_endpoint("ws-1", workspace_id="ws-1", info=_admin_info("ws-1")))
    assert res["matrix"]["editor"]["share_org"] is True


def test_put_sets_override_and_blocks_lockout(tmp_path, monkeypatch):
    from mayring_core.memory import store
    conn = store.init_memory_db(tmp_path / "m.db")
    monkeypatch.setattr(rp, "_get_conn", lambda: conn)
    asyncio.run(rp.put_role_permission_endpoint("ws-1", rp.RolePermissionUpdate(role="editor", permission="share_public", allowed=False), workspace_id="ws-1", info=_admin_info("ws-1")))
    res = asyncio.run(rp.get_role_permissions_endpoint("ws-1", workspace_id="ws-1", info=_admin_info("ws-1")))
    assert res["matrix"]["editor"]["share_public"] is False
    import fastapi
    try:
        asyncio.run(rp.put_role_permission_endpoint("ws-1", rp.RolePermissionUpdate(role="admin", permission="manage_members", allowed=False), workspace_id="ws-1", info=_admin_info("ws-1")))
        assert False, "expected HTTPException"
    except fastapi.HTTPException as e:
        assert e.status_code == 422


def test_put_denied_for_non_manage_member_caller(tmp_path, monkeypatch):
    from mayring_core.memory import store
    conn = store.init_memory_db(tmp_path / "m.db")
    monkeypatch.setattr(rp, "_get_conn", lambda: conn)
    user_info = TokenInfo(workspace_id="ws-1", scopes=(), memberships=(
        Membership(id="ws-1", type="organization", role="user", name="Acme"),))
    import fastapi
    try:
        asyncio.run(rp.put_role_permission_endpoint("ws-1", rp.RolePermissionUpdate(role="editor", permission="write", allowed=False), workspace_id="ws-1", info=user_info))
        assert False
    except fastapi.HTTPException as e:
        assert e.status_code == 403
