"""Iter 1 + Iter 2 V2-Workspaces — JWT memberships + 4 visibility-bug-fixes.

Ref: docs/v2-workspaces-spec.md
- Iter 1: TokenInfo.memberships, Membership-NamedTuple, _scope_filter mit org_ids-list
- Iter 2: L7 (search passes user/org), L8 (PATCH owner-check), L6 (sync respects
  org/user), L3 (ingest sets org_id from membership when visibility='org')
"""
from __future__ import annotations

import datetime
from unittest.mock import patch

import pytest

from mayring_core.memory.db_adapter import DBAdapter
from mayring_core.memory.store import _init_schema, upsert_source, insert_chunk
from mayring_core.memory.schema import Source, Chunk
from src.api.jwt_auth import Membership, TokenInfo


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _db() -> DBAdapter:
    adapter = DBAdapter.memory()
    _init_schema(adapter)
    return adapter


def _insert_chunk(
    db: DBAdapter,
    source_id: str,
    workspace_id: str,
    visibility: str = "private",
    org_id: str | None = None,
    user_id: str | None = None,
) -> str:
    src = Source(source_id=source_id, source_type="note", repo="r", path="p")
    upsert_source(
        db, src, workspace_id=workspace_id, visibility=visibility,
        org_id=org_id, user_id=user_id,
    )
    chunk_id = f"chk-{source_id}"
    now = datetime.datetime.utcnow().isoformat()
    chunk = Chunk(
        chunk_id=chunk_id, source_id=source_id, text="hello",
        text_hash="h1", dedup_key="d1", created_at=now,
        workspace_id=workspace_id,
    )
    insert_chunk(db, chunk)
    return chunk_id


@pytest.fixture
def mock_token_info() -> TokenInfo:
    """User mit 1 personal + 1 organization membership."""
    return TokenInfo(
        workspace_id="ws-bene",
        sub="42",
        scopes=("mcp:memory",),
        memberships=(
            Membership(id="ws-bene", type="personal", role="owner"),
            Membership(id="org-acme", type="organization", role="editor"),
        ),
    )


# ---------------------------------------------------------------------------
# Iter 1 — TokenInfo / Membership
# ---------------------------------------------------------------------------

class TestMembershipParsing:
    """JWT-Claim-Vertrag V2: memberships[]."""

    def test_membership_is_namedtuple_like(self):
        m = Membership(id="org-1", type="organization", role="editor")
        assert m.id == "org-1"
        assert m.type == "organization"
        assert m.role == "editor"

    def test_token_info_memberships_default_empty(self):
        ti = TokenInfo(workspace_id="ws-1", scopes=("mcp:memory",))
        assert ti.memberships == ()

    def test_token_info_org_ids_extracted_from_memberships(self):
        ti = TokenInfo(
            workspace_id="ws-1",
            scopes=("mcp:memory",),
            memberships=(
                Membership(id="ws-personal", type="personal", role="owner"),
                Membership(id="org-acme", type="organization", role="editor"),
                Membership(id="org-lab", type="organization", role="viewer"),
            ),
        )
        assert ti.org_ids == ("org-acme", "org-lab")

    def test_token_info_org_ids_empty_when_no_memberships(self):
        ti = TokenInfo(workspace_id="ws-1", scopes=("mcp:memory",))
        assert ti.org_ids == ()

    def test_token_info_org_ids_empty_when_only_personal(self):
        ti = TokenInfo(
            workspace_id="ws-1",
            scopes=("mcp:memory",),
            memberships=(Membership(id="ws-1", type="personal", role="owner"),),
        )
        assert ti.org_ids == ()


class TestValidateJwtMemberships:
    """validate_jwt_token parst memberships-claim, backward-compat ohne Feld."""

    def _make_jwt(self, payload: dict, private_key, public_key, monkeypatch, tmp_path):
        import jwt as _jwt
        token = _jwt.encode(payload, private_key, algorithm="RS256")
        # Public key on disk + env vars so validate_jwt_token can pick them up
        keypath = tmp_path / "pub.pem"
        keypath.write_text(public_key)
        monkeypatch.setenv("JWT_PUBLIC_KEY_PATH", str(keypath))
        monkeypatch.setenv("JWT_ISSUER", "https://app.linn.games")
        monkeypatch.setenv("JWT_AUDIENCE", "mayringcoder")
        from src.api import jwt_auth as _ja
        _ja.reset_public_key_cache()
        return token

    @pytest.fixture
    def keypair(self):
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

    def test_validate_jwt_parses_memberships(self, monkeypatch, tmp_path, keypair):
        from src.api.jwt_auth import validate_jwt_token
        priv, pub = keypair
        import time as _time
        now = int(_time.time())
        payload = {
            "sub": "42",
            "email": "bene@linn.games",
            "iat": now,
            "exp": now + 3600,
            "iss": "https://app.linn.games",
            "aud": "mayringcoder",
            "scope": ["mcp:memory"],
            "workspace_id": "ws-personal",
            "memberships": [
                {"id": "ws-personal", "type": "personal", "role": "owner"},
                {"id": "org-acme", "type": "organization", "role": "editor"},
            ],
        }
        tok = self._make_jwt(payload, priv, pub, monkeypatch, tmp_path)
        info = validate_jwt_token(tok)
        assert info is not None
        assert len(info.memberships) == 2
        assert info.memberships[0].id == "ws-personal"
        assert info.memberships[1].type == "organization"
        assert info.org_ids == ("org-acme",)

    def test_validate_jwt_backward_compat_no_memberships(
        self, monkeypatch, tmp_path, keypair,
    ):
        """Alte JWTs ohne memberships-Feld funktionieren weiter."""
        from src.api.jwt_auth import validate_jwt_token
        priv, pub = keypair
        import time as _time
        now = int(_time.time())
        payload = {
            "sub": "42",
            "email": "bene@linn.games",
            "iat": now,
            "exp": now + 3600,
            "iss": "https://app.linn.games",
            "aud": "mayringcoder",
            "scope": ["mcp:memory"],
            "workspace_id": "ws-personal",
        }
        tok = self._make_jwt(payload, priv, pub, monkeypatch, tmp_path)
        info = validate_jwt_token(tok)
        assert info is not None
        assert info.memberships == ()
        assert info.org_ids == ()


# ---------------------------------------------------------------------------
# Iter 1 — _scope_filter mit org_ids-Liste
# ---------------------------------------------------------------------------

class TestScopeFilterOrgIds:
    """V2: _scope_filter akzeptiert Liste von org_ids (statt single)."""

    def test_scope_filter_includes_all_org_memberships(self):
        from mayring_core.memory.retrieval import _scope_filter
        db = _db()
        cid_a = _insert_chunk(db, "src-a", "ws-owner-a", visibility="org", org_id="org-acme")
        cid_b = _insert_chunk(db, "src-b", "ws-owner-b", visibility="org", org_id="org-lab")
        cid_other = _insert_chunk(db, "src-c", "ws-owner-c", visibility="org", org_id="org-other")

        results = _scope_filter(
            db, workspace_id="ws-bene", org_ids=("org-acme", "org-lab"),
        )
        assert cid_a in results
        assert cid_b in results
        assert cid_other not in results

    def test_scope_filter_excludes_non_member_org(self):
        from mayring_core.memory.retrieval import _scope_filter
        db = _db()
        cid = _insert_chunk(db, "src-x", "ws-other", visibility="org", org_id="org-acme")
        results = _scope_filter(
            db, workspace_id="ws-bene", org_ids=("org-different",),
        )
        assert cid not in results

    def test_scope_filter_backward_compat_no_memberships(self):
        """Caller ohne org-memberships sieht public + private mit passender user_id."""
        from mayring_core.memory.retrieval import _scope_filter
        db = _db()
        cid_pub = _insert_chunk(db, "pub", "ws-other", visibility="public")
        cid_priv = _insert_chunk(db, "own", "ws-bene", visibility="private", user_id="u1")
        cid_org = _insert_chunk(db, "org", "ws-other", visibility="org", org_id="org-x")

        # user_id provided → private chunk with matching user_id is visible
        results = _scope_filter(db, workspace_id="ws-bene", user_id="u1")
        assert cid_pub in results
        assert cid_priv in results
        assert cid_org not in results

        # No user_id → private chunks never match (new 3-value model)
        results_no_user = _scope_filter(db, workspace_id="ws-bene")
        assert cid_pub in results_no_user
        assert cid_priv not in results_no_user

    def test_scope_filter_legacy_org_id_single_still_works(self):
        """Backward-Compat: single org_id-Argument bleibt akzeptiert."""
        from mayring_core.memory.retrieval import _scope_filter
        db = _db()
        cid = _insert_chunk(db, "s", "ws-other", visibility="org", org_id="org-acme")
        # Legacy callsite passes org_id (singular)
        results = _scope_filter(db, workspace_id="ws-bene", org_id="org-acme")
        assert cid in results


# ---------------------------------------------------------------------------
# Iter 2 — L7: REST /memory/search passes user_sub + org_ids
# ---------------------------------------------------------------------------

class TestSearchPassesUserAndOrgIds:
    """L7: /memory/search-route muss user_sub + org_ids aus TokenInfo durchreichen."""

    def test_search_passes_user_id_and_org_ids(self, monkeypatch):
        from fastapi.testclient import TestClient
        from src.api.server import app
        from src.api.auth import get_token_info, get_workspace
        import src.api.dependencies as _deps
        import src.api.routes.memory as _mod

        ti = TokenInfo(
            workspace_id="ws-bene",
            sub="42",
            scopes=("mcp:memory",),
            memberships=(
                Membership(id="ws-bene", type="personal", role="owner"),
                Membership(id="org-acme", type="organization", role="editor"),
                Membership(id="org-lab", type="organization", role="viewer"),
            ),
        )
        app.dependency_overrides[get_token_info] = lambda: ti
        app.dependency_overrides[get_workspace] = lambda: "ws-bene"
        db = _db()
        _deps._conn = db

        captured: dict = {}

        def _fake_search(query, conn, chroma, ollama_url, opts, char_budget):
            captured["opts"] = dict(opts)
            return {"results": [], "prompt_context": "", "diagnostics": {}}

        monkeypatch.setattr(_mod, "_run_search", _fake_search)

        client = TestClient(app)
        resp = client.post(
            "/memory/search",
            json={"query": "x", "top_k": 3},
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200, resp.text
        assert captured["opts"]["workspace_id"] == "ws-bene"
        assert captured["opts"]["user_id"] == "42"
        # org_ids forwarded so 'org' visibility chunks surface
        assert set(captured["opts"]["org_ids"]) == {"org-acme", "org-lab"}

        app.dependency_overrides.clear()
        _deps._conn = None


# ---------------------------------------------------------------------------
# Iter 2 — L8: PATCH /sources/{id}/visibility owner-check
# ---------------------------------------------------------------------------

class TestPatchVisibilityAuthorization:
    """L8: PATCH /sources/{id}/visibility — owner/admin-only, sonst 403."""

    def _setup_app_with_token(self, ti: TokenInfo, db: DBAdapter):
        from src.api.server import app
        from src.api.auth import get_token_info, get_workspace
        import src.api.dependencies as _deps

        app.dependency_overrides[get_token_info] = lambda: ti
        app.dependency_overrides[get_workspace] = lambda: ti.workspace_id
        _deps._conn = db
        return app

    def test_patch_visibility_owner_ok(self):
        from fastapi.testclient import TestClient
        # WHY(tenancy phase B): owner-of-workspace carries the admin role in the
        # membership → caller_can(share_public) for the promotion to public.
        ti = TokenInfo(
            workspace_id="ws-owner", sub="42", scopes=("mcp:memory",),
            memberships=(Membership(id="ws-owner", type="personal", role="owner"),),
        )
        db = _db()
        upsert_source(
            db, Source(source_id="s1", source_type="note", repo="", path="x"),
            workspace_id="ws-owner", visibility="private",
        )
        app = self._setup_app_with_token(ti, db)
        client = TestClient(app)
        resp = client.patch(
            "/sources/s1/visibility",
            json={"visibility": "public", "org_id": None},
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["visibility"] == "public"
        app.dependency_overrides.clear()
        import src.api.dependencies as _deps
        _deps._conn = None

    def test_patch_visibility_other_403(self):
        """Cross-tenant: non-owner non-admin → 403."""
        from fastapi.testclient import TestClient
        # Source belongs to ws-owner / user-1; caller is ws-other / user-2
        db = _db()
        upsert_source(
            db, Source(source_id="s2", source_type="note", repo="", path="x"),
            workspace_id="ws-owner", visibility="private", user_id="1",
        )
        ti = TokenInfo(workspace_id="ws-other", sub="2", scopes=("mcp:memory",))
        app = self._setup_app_with_token(ti, db)
        client = TestClient(app)
        resp = client.patch(
            "/sources/s2/visibility",
            json={"visibility": "public", "org_id": None},
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 403, resp.text
        app.dependency_overrides.clear()
        import src.api.dependencies as _deps
        _deps._conn = None

    def test_patch_visibility_admin_ok(self):
        """Admin (scope='*' or 'admin') kann alles ändern."""
        from fastapi.testclient import TestClient
        db = _db()
        upsert_source(
            db, Source(source_id="s3", source_type="note", repo="", path="x"),
            workspace_id="ws-someone", visibility="private",
        )
        # WHY(tenancy phase B): the wildcard '*' scope (MCP_SERVICE_TOKEN) must
        # still bypass the role gate via info.is_admin → caller_can(is_super_admin).
        ti = TokenInfo(workspace_id="system", sub="0", scopes=("*",))
        app = self._setup_app_with_token(ti, db)
        client = TestClient(app)
        resp = client.patch(
            "/sources/s3/visibility",
            json={"visibility": "public", "org_id": None},
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200, resp.text
        app.dependency_overrides.clear()
        import src.api.dependencies as _deps
        _deps._conn = None

    def test_patch_visibility_user_value_accepted(self):
        """V2: 'private' (cross-device-same-human) must be a valid visibility option."""
        from fastapi.testclient import TestClient
        db = _db()
        upsert_source(
            db, Source(source_id="s4", source_type="note", repo="", path="x"),
            workspace_id="ws-owner", visibility="public",
        )
        ti = TokenInfo(workspace_id="ws-owner", sub="42", scopes=("mcp:memory",))
        app = self._setup_app_with_token(ti, db)
        client = TestClient(app)
        resp = client.patch(
            "/sources/s4/visibility",
            json={"visibility": "private", "org_id": None},
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["visibility"] == "private"
        app.dependency_overrides.clear()
        import src.api.dependencies as _deps
        _deps._conn = None


# ---------------------------------------------------------------------------
# Iter 2 — L6: sync respects org/user visibility
# ---------------------------------------------------------------------------

class TestSyncRespectsOrgUserVisibility:
    """L6: GET /memory/changes muss org/user-visibility ebenfalls liefern."""

    def test_sync_respects_org_user_visibility(self):
        from fastapi.testclient import TestClient
        from src.api.server import app
        from src.api.auth import get_token_info, get_workspace
        import src.api.dependencies as _deps

        db = _db()
        # 5 chunks: public, private-own-ws, private-other-ws, org-member, private-same-user-other-ws
        cid_pub = _insert_chunk(db, "pub", "ws-other", visibility="public")
        cid_own = _insert_chunk(db, "own", "ws-bene", visibility="private", user_id="42")
        cid_other = _insert_chunk(db, "other", "ws-other", visibility="private", user_id="99")
        cid_org = _insert_chunk(db, "org", "ws-acme", visibility="org", org_id="org-acme")
        cid_user = _insert_chunk(db, "user", "ws-other-of-bene", visibility="private", user_id="42")

        ti = TokenInfo(
            workspace_id="ws-bene",
            sub="42",
            scopes=("mcp:memory",),
            memberships=(
                Membership(id="ws-bene", type="personal", role="owner"),
                Membership(id="org-acme", type="organization", role="editor"),
            ),
        )
        app.dependency_overrides[get_token_info] = lambda: ti
        app.dependency_overrides[get_workspace] = lambda: "ws-bene"
        _deps._conn = db

        # Stub chroma.get to a no-op so the sync route doesn't hit a real Chroma
        with patch.object(_deps, "get_chroma", lambda: None):
            client = TestClient(app)
            resp = client.get(
                "/memory/changes",
                params={"since": "1970-01-01T00:00:00", "limit": 100},
                headers={"Authorization": "Bearer test"},
            )
            assert resp.status_code == 200, resp.text
            ids = {c["chunk_id"] for c in resp.json()["chunks"]}
            assert cid_pub in ids
            assert cid_own in ids
            assert cid_other not in ids   # other-tenant private must not leak
            assert cid_org in ids          # member of org-acme
            assert cid_user in ids         # private+user_id=42 visible for same user across workspaces

        app.dependency_overrides.clear()
        _deps._conn = None


# ---------------------------------------------------------------------------
# Iter 2 — L3: ingest sets org_id from membership when visibility='org'
# ---------------------------------------------------------------------------

class TestIngestOrgIdResolution:
    """L3: visibility='org' bei /memory/put resolved org_id aus membership."""

    def test_ingest_org_visibility_uses_first_membership(self, monkeypatch):
        """Wenn visibility='org' aber kein org_id im body → erste org-membership default."""
        from fastapi.testclient import TestClient
        from src.api.server import app
        from src.api.auth import get_token_info, get_workspace
        import src.api.dependencies as _deps
        import src.api.routes.memory as _mod

        db = _db()
        ti = TokenInfo(
            workspace_id="ws-bene",
            sub="42",
            scopes=("mcp:memory",),
            memberships=(
                Membership(id="ws-bene", type="personal", role="owner"),
                Membership(id="org-acme", type="organization", role="editor"),
            ),
        )
        app.dependency_overrides[get_token_info] = lambda: ti
        app.dependency_overrides[get_workspace] = lambda: "ws-bene"
        _deps._conn = db

        captured: dict = {}

        def _fake_ingest(source_dict, content, conn, chroma, ollama_url, model, opts, ws):
            captured["source_dict"] = dict(source_dict)
            return {"source_id": source_dict["source_id"], "chunk_ids": []}

        monkeypatch.setattr(_mod, "_run_ingest", _fake_ingest)

        client = TestClient(app)
        resp = client.post(
            "/memory/put",
            json={
                "source_id": "s-ingest-1",
                "content": "hello",
                "visibility": "org",
            },
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200, resp.text
        assert captured["source_dict"].get("visibility") == "org"
        assert captured["source_dict"].get("org_id") == "org-acme"

        app.dependency_overrides.clear()
        _deps._conn = None

    def test_ingest_org_id_must_be_member(self, monkeypatch):
        """Wenn org_id aus body kommt → caller muss member dieser org sein, sonst 403."""
        from fastapi.testclient import TestClient
        from src.api.server import app
        from src.api.auth import get_token_info, get_workspace
        import src.api.dependencies as _deps
        import src.api.routes.memory as _mod

        db = _db()
        ti = TokenInfo(
            workspace_id="ws-bene",
            sub="42",
            scopes=("mcp:memory",),
            memberships=(
                Membership(id="ws-bene", type="personal", role="owner"),
                Membership(id="org-acme", type="organization", role="editor"),
            ),
        )
        app.dependency_overrides[get_token_info] = lambda: ti
        app.dependency_overrides[get_workspace] = lambda: "ws-bene"
        _deps._conn = db

        # Fake ingest exists so we know route doesn't fall through to it
        def _fake_ingest(*args, **kwargs):
            return {"source_id": "x", "chunk_ids": []}
        monkeypatch.setattr(_mod, "_run_ingest", _fake_ingest)

        client = TestClient(app)
        resp = client.post(
            "/memory/put",
            json={
                "source_id": "s-ingest-2",
                "content": "hi",
                "visibility": "org",
                "org_id": "org-not-a-member",
            },
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 403, resp.text

        app.dependency_overrides.clear()
        _deps._conn = None


class TestIngestUserIdResolution:
    """L3 FIX1: visibility='private' bei /memory/put muss user_id aus TokenInfo.sub stempeln.

    Ohne diesen Stamp: _scope_filter's `s.visibility='private' AND s.user_id=?`
    findet nie einen Match → 'private'-Visibility (cross-device-of-same-human) ist
    für REST-Caller komplett gebrochen. MCP-Pfad tut dies korrekt via
    resolve_write_visibility; REST-Pfad war blind dafür.
    """

    def test_ingest_user_visibility_stamps_user_id(self, monkeypatch):
        """visibility='private' → source_dict bekommt user_id == caller.sub."""
        from fastapi.testclient import TestClient
        from src.api.server import app
        from src.api.auth import get_token_info, get_workspace
        import src.api.dependencies as _deps
        import src.api.routes.memory as _mod

        db = _db()
        ti = TokenInfo(
            workspace_id="ws-bene",
            sub="42",
            scopes=("mcp:memory",),
            memberships=(
                Membership(id="ws-bene", type="personal", role="owner"),
            ),
        )
        app.dependency_overrides[get_token_info] = lambda: ti
        app.dependency_overrides[get_workspace] = lambda: "ws-bene"
        _deps._conn = db

        captured: dict = {}

        def _fake_ingest(source_dict, content, conn, chroma, ollama_url, model, opts, ws):
            captured["source_dict"] = dict(source_dict)
            return {"source_id": source_dict["source_id"], "chunk_ids": []}

        monkeypatch.setattr(_mod, "_run_ingest", _fake_ingest)

        client = TestClient(app)
        resp = client.post(
            "/memory/put",
            json={
                "source_id": "s-user-vis-1",
                "content": "cross-device note",
                "visibility": "private",
            },
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200, resp.text
        assert captured["source_dict"].get("visibility") == "private"
        assert captured["source_dict"].get("user_id") == "42", (
            "user_id must be stamped from TokenInfo.sub so _scope_filter "
            "'s.visibility=private AND s.user_id=?' can match on cross-device reads"
        )

        app.dependency_overrides.clear()
        _deps._conn = None

    def test_ingest_non_private_visibility_does_not_stamp_user_id(self, monkeypatch):
        """visibility='public' must NOT auto-stamp user_id — only 'private' does."""
        from fastapi.testclient import TestClient
        from src.api.server import app
        from src.api.auth import get_token_info, get_workspace
        import src.api.dependencies as _deps
        import src.api.routes.memory as _mod

        db = _db()
        # WHY(tenancy phase B): exercises the user_id-stamp mechanism for
        # visibility='public' → super-admin so caller_can(share_public) passes.
        ti = TokenInfo(
            workspace_id="ws-bene",
            sub="42",
            scopes=("admin",),
        )
        app.dependency_overrides[get_token_info] = lambda: ti
        app.dependency_overrides[get_workspace] = lambda: "ws-bene"
        _deps._conn = db

        captured: dict = {}

        def _fake_ingest(source_dict, content, conn, chroma, ollama_url, model, opts, ws):
            captured["source_dict"] = dict(source_dict)
            return {"source_id": source_dict["source_id"], "chunk_ids": []}

        monkeypatch.setattr(_mod, "_run_ingest", _fake_ingest)

        client = TestClient(app)
        resp = client.post(
            "/memory/put",
            json={
                "source_id": "s-public-1",
                "content": "public note",
                "visibility": "public",
            },
            headers={"Authorization": "Bearer test"},
        )
        assert resp.status_code == 200, resp.text
        assert "user_id" not in captured["source_dict"], (
            "visibility='public' must not add user_id (only 'private' needs it)"
        )

        app.dependency_overrides.clear()
        _deps._conn = None
