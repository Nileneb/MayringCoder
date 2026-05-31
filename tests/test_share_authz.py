"""Task 7 (tenancy phase B) — manage_foreign + share_* in share/patch.

Mutating the visibility of a FOREIGN source (not owned by the caller, neither
by same-workspace nor same-sub) requires the manage_foreign permission. A
plain 'editor' (search/write/share_org/share_public, but NOT manage_foreign)
must therefore be 403'd when it patches someone else's source.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from mayring_core.memory.schema import Source
from mayring_core.memory.store import init_memory_db, upsert_source
from src.api.jwt_auth import Membership, TokenInfo


@pytest.fixture
def seeded_conn(tmp_path: Path):
    """Test DB seeded with a source owned by a FOREIGN user/workspace."""
    conn = init_memory_db(tmp_path / "m.db")
    upsert_source(
        conn,
        Source(source_id="note:foreign", source_type="note", repo="", path="",
               visibility="private", user_id="999", content_hash="h:foreign"),
        workspace_id="other-ws",
    )
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def client_editor(seeded_conn):
    """Caller = 'editor' in workspace 'bene', sub='1' (does NOT own the source)."""
    from fastapi.testclient import TestClient
    from src.api import auth as auth_module
    from src.api import server as srv

    async def _fake_ws():
        return "bene"

    info = TokenInfo(
        workspace_id="bene",
        scopes=("mcp:memory",),
        sub="1",
        memberships=(Membership(id="bene", type="organization", role="editor"),),
    )

    async def _fake_info():
        return info

    srv.app.dependency_overrides[auth_module.get_workspace] = _fake_ws
    srv.app.dependency_overrides[auth_module.get_token_info] = _fake_info
    with (
        patch("src.api.routes.memory._get_conn", return_value=seeded_conn),
        patch("src.api.routes.memory._get_chroma"),
    ):
        yield TestClient(srv.app)
    srv.app.dependency_overrides.clear()


def test_editor_cannot_patch_foreign_visibility(client_editor):
    """editor lacks manage_foreign → patching a foreign source must be 403."""
    r = client_editor.patch(
        "/sources/note:foreign/visibility",
        json={"visibility": "public"},
        headers={"Authorization": "Bearer tst"},
    )
    assert r.status_code == 403, r.text
    assert "manage_foreign" in r.json()["detail"]


def test_editor_cannot_share_foreign(client_editor):
    """Same gate on POST /sources/{id}/share."""
    r = client_editor.post(
        "/sources/note:foreign/share",
        json={},
        headers={"Authorization": "Bearer tst"},
    )
    assert r.status_code == 403, r.text
    assert "manage_foreign" in r.json()["detail"]
