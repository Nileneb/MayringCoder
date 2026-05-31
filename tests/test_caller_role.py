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


def test_act_as_role_grants_workspace_membership(monkeypatch):
    """X-Act-As-Role (tenancy phase B) makes the synthetic identity a member with
    that role for its active workspace, so role-gated paths are testable."""
    monkeypatch.setenv("MAYRING_ALLOW_ACT_AS", "1")
    from src.api.auth import _maybe_act_as
    from src.api.authz_helpers import caller_role
    from src.api.jwt_auth import TokenInfo
    priv = TokenInfo(workspace_id="system", scopes=("*",))
    synth = _maybe_act_as(priv, sub="A", orgs="org-x", workspace="ws-A", role="admin")
    assert synth.sub == "A"
    assert "*" not in synth.scopes and "admin" not in synth.scopes  # privilege dropped
    assert caller_role(synth, "ws-A") == "admin"


def test_act_as_role_ignored_for_unprivileged(monkeypatch):
    monkeypatch.setenv("MAYRING_ALLOW_ACT_AS", "1")
    from src.api.auth import _maybe_act_as
    from src.api.jwt_auth import TokenInfo
    plain = TokenInfo(workspace_id="ws-1", scopes=("mcp:memory",))
    out = _maybe_act_as(plain, sub="A", orgs=None, workspace="ws-A", role="admin")
    assert out is plain
