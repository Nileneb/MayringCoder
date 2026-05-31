"""Tests for the JWT→authz bridge (tenancy Phase B, Task 4)."""
from src.api.authz_helpers import caller_role, caller_can
from src.api.jwt_auth import TokenInfo, Membership


def test_caller_role_from_membership():
    info = TokenInfo(workspace_id="ws-1", scopes=(), memberships=(
        Membership(id="ws-1", type="organization", role="owner", name="Acme"),))
    assert caller_role(info, "ws-1") == "admin"
    assert caller_role(info, "ws-other") == "user"


def test_caller_can_uses_default_matrix():
    info = TokenInfo(workspace_id="ws-1", scopes=(), memberships=(
        Membership(id="ws-1", type="organization", role="editor", name="Acme"),))
    assert caller_can(info, "ws-1", "share_org", overrides={}) is True
    assert caller_can(info, "ws-1", "manage_members", overrides={}) is False


def test_caller_can_super_admin():
    info = TokenInfo(workspace_id="ws-1", scopes=("admin",), memberships=())
    assert caller_can(info, "ws-1", "manage_foreign", overrides={}) is True


def test_workspace_owner_is_admin(monkeypatch):
    import src.api.authz_helpers as ah
    import mayring_core.identity.workspace_resolver as wr
    monkeypatch.setattr(wr, "workspace_owner", lambda conn, ws: "owner-sub")
    # also patch _get_db_conn so no real DB is needed
    monkeypatch.setattr(ah, "_get_db_conn", lambda: None)
    info = TokenInfo(workspace_id="ws-1", scopes=(), sub="owner-sub", memberships=())
    assert ah.caller_role(info, "ws-1") == "admin"
