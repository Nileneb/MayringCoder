"""Tests für die Workspace-Identity-Foundation.

Kernkontrakt: ein gegebener Eingang produziert IMMER denselben kanonischen
Workspace, egal ob CLI-Argument, JWT-Claim, Alias oder None mit
default_user_id. Das ist die Grundlage für Multi-Tenant + Multi-User.
"""
from __future__ import annotations

import pytest

from mayring_core.identity.workspace_resolver import (
    IdentityRequiredError,
    UnknownWorkspaceError,
    add_alias,
    email_to_slug,
    ensure_project_workspace,
    ensure_user_workspace,
    list_workspaces_for_user,
    resolve_workspace,
)
from mayring_core.memory.db_adapter import DBAdapter
from mayring_core.memory.store import _init_schema


@pytest.fixture
def conn(tmp_path):
    a = DBAdapter.create(tmp_path / "test.db", check_same_thread=False)
    _init_schema(a)
    return a


# ─── kind=user Pattern ──────────────────────────────────────────────


def test_email_to_slug_basic():
    assert email_to_slug("bene@linn.games") == "bene"
    assert email_to_slug("foo.bar@example.com") == "foo-bar"
    assert email_to_slug("admin+tag@firma.de") == "admin-tag"
    assert email_to_slug("") == ""
    assert email_to_slug(None) == ""


def test_resolve_with_email_creates_slug_workspace(conn):
    """Email als default → workspace.id = slug, NICHT 'user-N'."""
    result = resolve_workspace(
        conn, None,
        default_user_id=2, default_email="bene@linn.games",
        default_display_name="Benedikt Linn",
    )
    assert result == "bene"
    row = conn.execute(
        "SELECT kind, owner_user_id, email, display_name FROM workspaces WHERE id='bene'"
    ).fetchone()
    assert tuple(row) == ("user", 2, "bene@linn.games", "Benedikt Linn")


def test_resolve_without_email_raises(conn):
    """Pre-launch: kein user-N-Fallback. Email PFLICHT für Workspace-Resolution."""
    with pytest.raises(IdentityRequiredError):
        resolve_workspace(conn, None, default_user_id=2)


def test_system_workspace_resolves_without_seed(conn):
    """'system'-bucket: Service-Tokens, Cron-Jobs, ambient. Resolver
    akzeptiert das Wort direkt + ensures the row (idempotent)."""
    result = resolve_workspace(conn, "system")
    assert result == "system"
    row = conn.execute("SELECT kind FROM workspaces WHERE id='system'").fetchone()
    assert tuple(row) == ("system",)


def test_user_n_input_no_longer_auto_created(conn):
    """Pre-launch: 'user-N' als input-string wird NICHT mehr auto-created."""
    with pytest.raises(UnknownWorkspaceError):
        resolve_workspace(conn, "user-2")


def test_project_workspace_under_email_slug(conn):
    """Project-Sub-Workspaces hängen unter dem email-slug."""
    sub = ensure_project_workspace(
        conn, 2, "mayringcoder", email="bene@linn.games",
    )
    assert sub == "bene:mayringcoder"
    assert tuple(conn.execute(
        "SELECT id, kind FROM workspaces WHERE id='bene'"
    ).fetchone()) == ("bene", "user")
    assert tuple(conn.execute(
        "SELECT id, kind, parent_id, owner_user_id FROM workspaces "
        "WHERE id='bene:mayringcoder'"
    ).fetchone()) == ("bene:mayringcoder", "project", "bene", 2)


# ─── None + default_user_id ─────────────────────────────────────────


def test_none_with_email_resolves_to_slug(conn):
    result = resolve_workspace(
        conn, None, default_user_id=5, default_email="alice@x.de",
    )
    assert result == "alice"
    row = conn.execute(
        "SELECT kind, email FROM workspaces WHERE id='alice'"
    ).fetchone()
    assert tuple(row) == ("user", "alice@x.de")


def test_none_without_default_raises(conn):
    """Kein silent 'default'-Fallback mehr — der Auth-relevante Pfad
    muss eindeutig identifizierbar sein."""
    with pytest.raises(IdentityRequiredError):
        resolve_workspace(conn, None)


def test_empty_string_treated_as_none(conn):
    with pytest.raises(IdentityRequiredError):
        resolve_workspace(conn, "")
    with pytest.raises(IdentityRequiredError):
        resolve_workspace(conn, "   ")


# ─── Aliases ────────────────────────────────────────────────────────


def test_alias_resolves_to_canonical(conn):
    ensure_user_workspace(conn, 2, email="bene@linn.games")
    add_alias(conn, "nileneb-mayringcoder", "bene")
    assert resolve_workspace(conn, "nileneb-mayringcoder") == "bene"


def test_alias_to_unknown_workspace_rejects(conn):
    with pytest.raises(UnknownWorkspaceError):
        add_alias(conn, "anything", "ghost-workspace")


def test_resolve_from_token_canonicalizes_alias(conn):
    """workspace-repoint: alte Tokens (alias) lösen via conn transparent auf die
    kanonische Workspace auf; ohne conn bleibt das Verhalten unverändert."""
    from types import SimpleNamespace

    from mayring_core.identity.workspace_resolver import resolve_workspace_from_token

    ensure_user_workspace(conn, 1, email="bene@linn.games")  # slug 'bene'
    add_alias(conn, "old-ws", "bene")

    tok = SimpleNamespace(workspace_id="old-ws", scopes=("mcp:memory",))
    assert resolve_workspace_from_token(tok, conn=conn) == "bene"   # alias → canonical
    assert resolve_workspace_from_token(tok) == "old-ws"            # no conn → unchanged

    svc = SimpleNamespace(workspace_id="system", scopes=("*",))
    assert resolve_workspace_from_token(svc, override_header="old-ws", conn=conn) == "bene"


# ─── Unknown workspace ──────────────────────────────────────────────


def test_unknown_string_raises(conn):
    with pytest.raises(UnknownWorkspaceError):
        resolve_workspace(conn, "some-random-name")


# ─── list_workspaces_for_user ───────────────────────────────────────


def test_list_for_user_includes_user_and_projects(conn):
    ensure_user_workspace(conn, 3, email="alice@x.de")
    ensure_project_workspace(conn, 3, "alpha", email="alice@x.de")
    ensure_project_workspace(conn, 3, "beta", email="alice@x.de")
    ensure_user_workspace(conn, 4, email="bob@x.de")

    items = list_workspaces_for_user(conn, 3)
    ids = sorted(i["id"] for i in items)
    assert ids == ["alice", "alice:alpha", "alice:beta"]


# ─── Idempotency ─────────────────────────────────────────────────────


def test_resolve_is_idempotent(conn):
    ensure_user_workspace(conn, 2, email="bene@linn.games")
    a = resolve_workspace(conn, "bene")
    b = resolve_workspace(conn, "bene")
    assert a == b == "bene"
    cnt = conn.execute(
        "SELECT COUNT(*) FROM workspaces WHERE id='bene'"
    ).fetchone()[0]
    assert cnt == 1


def test_ensure_user_workspace_without_email_or_slug_raises(conn):
    """Pre-launch: keine user-N-Buckets mehr generierbar."""
    with pytest.raises(IdentityRequiredError):
        ensure_user_workspace(conn, 2)
