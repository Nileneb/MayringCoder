"""Active-workspace → team-sharing wiring.

MayringCoder kept its home bucket per user (email-slug) but discarded the
`workspace_id` UUID claim app.linn.games ships (the *active* workspace incl.
the #195 switch). This wires that claim through: when the caller's active
app.linn.games workspace is an organization they belong to, new memory writes
are stamped visibility='org' + org_id=<active workspace> so the whole team
sees them — reusing the existing _scope_filter org path. The home bucket
(TokenInfo.workspace_id = email-slug) is intentionally untouched (no re-key).
"""
from __future__ import annotations

import time

import pytest

from src.api.jwt_auth import Membership, TokenInfo, validate_jwt_token


# ---------------------------------------------------------------------------
# TokenInfo defaults
# ---------------------------------------------------------------------------

def test_token_info_active_workspace_defaults():
    ti = TokenInfo(workspace_id="bene", scopes=("mcp:memory",))
    assert ti.active_workspace_id is None
    assert ti.active_workspace_kind == "personal"


# ---------------------------------------------------------------------------
# validate_jwt_token consumes the workspace_id claim
# ---------------------------------------------------------------------------

@pytest.fixture
def keypair():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return priv_pem, pub_pem


def _mint(payload, priv, pub, monkeypatch, tmp_path):
    import jwt as _jwt
    token = _jwt.encode(payload, priv, algorithm="RS256")
    keypath = tmp_path / "pub.pem"
    keypath.write_text(pub)
    monkeypatch.setenv("JWT_PUBLIC_KEY_PATH", str(keypath))
    monkeypatch.setenv("JWT_ISSUER", "https://app.linn.games")
    monkeypatch.setenv("JWT_AUDIENCE", "mayringcoder")
    from src.api import jwt_auth as _ja
    _ja.reset_public_key_cache()
    return token


def _base_payload():
    now = int(time.time())
    return {
        "sub": "42",
        "email": "bene@linn.games",
        "iat": now,
        "exp": now + 3600,
        "iss": "https://app.linn.games",
        "aud": "mayringcoder",
        "scope": ["mcp:memory"],
    }


def test_active_workspace_id_extracted_from_claim(monkeypatch, tmp_path, keypair):
    priv, pub = keypair
    payload = _base_payload()
    payload["workspace_id"] = "org-acme-uuid"
    payload["memberships"] = [
        {"id": "ws-personal", "type": "personal", "role": "owner"},
        {"id": "org-acme-uuid", "type": "organization", "role": "editor"},
    ]
    info = validate_jwt_token(_mint(payload, priv, pub, monkeypatch, tmp_path))
    assert info is not None
    # Home bucket stays the email-slug — NOT re-keyed to the UUID.
    assert info.workspace_id == "bene"
    # Active workspace is the org the user switched into.
    assert info.active_workspace_id == "org-acme-uuid"
    assert info.active_workspace_kind == "organization"


def test_active_workspace_kind_personal(monkeypatch, tmp_path, keypair):
    priv, pub = keypair
    payload = _base_payload()
    payload["workspace_id"] = "ws-personal"
    payload["memberships"] = [
        {"id": "ws-personal", "type": "personal", "role": "owner"},
    ]
    info = validate_jwt_token(_mint(payload, priv, pub, monkeypatch, tmp_path))
    assert info is not None
    assert info.active_workspace_id == "ws-personal"
    assert info.active_workspace_kind == "personal"


def test_active_workspace_none_when_claim_absent(monkeypatch, tmp_path, keypair):
    """Legacy JWT without workspace_id claim → no active workspace."""
    priv, pub = keypair
    info = validate_jwt_token(_mint(_base_payload(), priv, pub, monkeypatch, tmp_path))
    assert info is not None
    assert info.active_workspace_id is None
    assert info.active_workspace_kind == "personal"


# ---------------------------------------------------------------------------
# Pure decision helper: resolve_write_visibility
# ---------------------------------------------------------------------------

def test_write_visibility_org_when_active_org_member():
    from src.api.mcp_auth import resolve_write_visibility
    vis, org_id, uid = resolve_write_visibility(
        active_workspace_id="org-acme",
        active_workspace_kind="organization",
        org_ids=("org-acme", "org-lab"),
        user_id="42",
    )
    assert (vis, org_id, uid) == ("org", "org-acme", "42")


def test_write_visibility_falls_back_when_active_org_not_member():
    """Defensive: active claims org but caller isn't a member → never write
    into a foreign org bucket. Falls back to the per-user 'user' bucket."""
    from src.api.mcp_auth import resolve_write_visibility
    vis, org_id, uid = resolve_write_visibility(
        active_workspace_id="org-foreign",
        active_workspace_kind="organization",
        org_ids=("org-acme",),
        user_id="42",
    )
    assert (vis, org_id, uid) == ("user", None, "42")


def test_write_visibility_user_when_personal():
    from src.api.mcp_auth import resolve_write_visibility
    vis, org_id, uid = resolve_write_visibility(
        active_workspace_id="ws-personal",
        active_workspace_kind="personal",
        org_ids=(),
        user_id="42",
    )
    assert (vis, org_id, uid) == ("user", None, "42")


def test_write_visibility_private_when_no_identity():
    from src.api.mcp_auth import resolve_write_visibility
    vis, org_id, uid = resolve_write_visibility(
        active_workspace_id=None,
        active_workspace_kind="personal",
        org_ids=(),
        user_id=None,
    )
    assert (vis, org_id, uid) == ("private", None, None)


# ---------------------------------------------------------------------------
# Context accessors
# ---------------------------------------------------------------------------

def test_effective_active_workspace_accessors():
    import src.api.mcp_auth as _ma
    info = TokenInfo(
        workspace_id="bene",
        scopes=("mcp:memory",),
        active_workspace_id="org-acme",
        active_workspace_kind="organization",
    )
    token = _ma._TOKEN_CTX.set(info)
    try:
        assert _ma._effective_active_workspace_id() == "org-acme"
        assert _ma._effective_active_workspace_kind() == "organization"
    finally:
        _ma._TOKEN_CTX.reset(token)


def test_effective_active_workspace_accessors_no_context():
    import src.api.mcp_auth as _ma
    token = _ma._TOKEN_CTX.set(None)
    try:
        assert _ma._effective_active_workspace_id() is None
        assert _ma._effective_active_workspace_kind() == "personal"
    finally:
        _ma._TOKEN_CTX.reset(token)


# ---------------------------------------------------------------------------
# ensure_team_workspace — org workspace as first-class local row
# ---------------------------------------------------------------------------

def _conn(tmp_path):
    from src.memory.db_adapter import DBAdapter
    from src.memory.store import _init_schema
    a = DBAdapter.create(tmp_path / "team.db", check_same_thread=False)
    _init_schema(a)
    return a


def test_ensure_team_workspace_upserts_kind_team(tmp_path):
    from src.identity.workspace_resolver import ensure_team_workspace
    conn = _conn(tmp_path)
    ws = ensure_team_workspace(conn, "org-acme", display_name="ACME Inc")
    assert ws == "org-acme"
    row = conn.execute(
        "SELECT id, kind, display_name FROM workspaces WHERE id='org-acme'"
    ).fetchone()
    assert tuple(row) == ("org-acme", "team", "ACME Inc")


def test_ensure_team_workspace_idempotent(tmp_path):
    from src.identity.workspace_resolver import ensure_team_workspace
    conn = _conn(tmp_path)
    ensure_team_workspace(conn, "org-acme")
    ensure_team_workspace(conn, "org-acme")
    cnt = conn.execute(
        "SELECT COUNT(*) FROM workspaces WHERE id='org-acme'"
    ).fetchone()[0]
    assert cnt == 1
