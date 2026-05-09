"""Tests für die Workspace-Identity-Foundation.

Kernkontrakt: ein gegebener Eingang produziert IMMER denselben kanonischen
Workspace, egal ob CLI-Argument, JWT-Claim, Alias oder None mit
default_user_id. Das ist die Grundlage für Multi-Tenant + Multi-User.
"""
from __future__ import annotations

import pytest

from src.identity.workspace_resolver import (
    IdentityRequiredError,
    UnknownWorkspaceError,
    add_alias,
    ensure_project_workspace,
    ensure_user_workspace,
    list_workspaces_for_user,
    resolve_workspace,
)
from src.memory.db_adapter import DBAdapter
from src.memory.store import _init_schema


@pytest.fixture
def conn(tmp_path):
    a = DBAdapter.create(tmp_path / "test.db", check_same_thread=False)
    _init_schema(a)
    return a


# ─── kind=user Pattern ──────────────────────────────────────────────


def test_system_workspace_resolves_without_seed(conn):
    """'system'-bucket: Service-Tokens, Cron-Jobs, ambient. Resolver
    akzeptiert das Wort direkt + ensures the row (idempotent)."""
    result = resolve_workspace(conn, "system")
    assert result == "system"
    row = conn.execute("SELECT kind FROM workspaces WHERE id='system'").fetchone()
    assert tuple(row) == ("system",)


def test_user_n_pattern_resolves_to_canonical(conn):
    """JWT-Pfad: 'user-2' ist immer kanonisch und wird auto-created."""
    result = resolve_workspace(conn, "user-2")
    assert result == "user-2"
    row = conn.execute(
        "SELECT kind, owner_user_id FROM workspaces WHERE id='user-2'"
    ).fetchone()
    assert tuple(row) == ("user", 2)


def test_user_pattern_with_subslug_creates_project_workspace(conn):
    """user-2:mayringcoder → kind=project mit parent=user-2."""
    result = resolve_workspace(conn, "user-2:mayringcoder")
    assert result == "user-2:mayringcoder"
    parent = conn.execute(
        "SELECT id, kind FROM workspaces WHERE id='user-2'"
    ).fetchone()
    sub = conn.execute(
        "SELECT id, kind, parent_id, owner_user_id FROM workspaces "
        "WHERE id='user-2:mayringcoder'"
    ).fetchone()
    assert tuple(parent) == ("user-2", "user")
    assert tuple(sub) == ("user-2:mayringcoder", "project", "user-2", 2)


# ─── None + default_user_id ─────────────────────────────────────────


def test_none_with_default_user_id_resolves_and_creates(conn):
    result = resolve_workspace(conn, None, default_user_id=5)
    assert result == "user-5"
    row = conn.execute(
        "SELECT kind FROM workspaces WHERE id='user-5'"
    ).fetchone()
    assert tuple(row) == ("user",)


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
    ensure_user_workspace(conn, 2)
    add_alias(conn, "default", "user-2")
    add_alias(conn, "nileneb-mayringcoder", "user-2")

    assert resolve_workspace(conn, "default") == "user-2"
    assert resolve_workspace(conn, "nileneb-mayringcoder") == "user-2"


def test_alias_to_unknown_workspace_rejects(conn):
    """Foreign-key-style validation: alias darf nur auf existierende
    canonical workspaces zeigen."""
    with pytest.raises(UnknownWorkspaceError):
        add_alias(conn, "anything", "user-99")


# ─── Unknown workspace ──────────────────────────────────────────────


def test_unknown_string_raises(conn):
    """Kein Auto-Create für free-form Strings — sonst wäre die Mapping-
    Tabelle nutzlos."""
    with pytest.raises(UnknownWorkspaceError):
        resolve_workspace(conn, "some-random-name")


# ─── ensure_project_workspace ───────────────────────────────────────


def test_ensure_project_creates_parent_implicitly(conn):
    """Aufruf nur mit project-slug → parent user-N wird mitangelegt."""
    sub = ensure_project_workspace(conn, 7, "myproject")
    assert sub == "user-7:myproject"
    assert tuple(conn.execute(
        "SELECT kind FROM workspaces WHERE id='user-7'"
    ).fetchone()) == ("user",)


# ─── list_workspaces_for_user ───────────────────────────────────────


def test_list_for_user_includes_user_and_projects(conn):
    ensure_user_workspace(conn, 3)
    ensure_project_workspace(conn, 3, "alpha")
    ensure_project_workspace(conn, 3, "beta")
    ensure_user_workspace(conn, 4)  # different user, must not appear

    items = list_workspaces_for_user(conn, 3)
    ids = sorted(i["id"] for i in items)
    assert ids == ["user-3", "user-3:alpha", "user-3:beta"]


# ─── Idempotency ─────────────────────────────────────────────────────


def test_resolve_is_idempotent(conn):
    """Wiederholte Aufrufe ändern nichts und behalten dieselbe ID."""
    a = resolve_workspace(conn, "user-2")
    b = resolve_workspace(conn, "user-2")
    c = resolve_workspace(conn, "user-2")
    assert a == b == c == "user-2"
    cnt = conn.execute(
        "SELECT COUNT(*) FROM workspaces WHERE id='user-2'"
    ).fetchone()[0]
    assert cnt == 1


def test_auto_create_can_be_disabled(conn):
    """Für Read-only-Pfade: auto_create=False prüft nur Existenz."""
    with pytest.raises(UnknownWorkspaceError):
        resolve_workspace(conn, "user-99", auto_create_user_workspace=False)
