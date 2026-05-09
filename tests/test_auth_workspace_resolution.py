"""Tests für src.api.auth.get_workspace — Multi-Tenant-Workspace-Resolution.

Pre-launch wie multi-tenant: Service-Token (scope='*') darf via
X-Workspace-Id-Header in einen anderen Workspace schreiben (admin-
operation), User-JWT darf das NICHT (User schreibt nur in eigene
workspace).
"""
from __future__ import annotations

import asyncio
import os

import pytest

from src.api import auth as auth_mod
from src.api.jwt_auth import TokenInfo


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_user_token_ignores_workspace_header():
    """User-JWT darf KEIN X-Workspace-Id setzen — sonst könnte alice
    in bene's Workspace schreiben. workspace_id aus dem Token ist
    autoritativ."""
    info = TokenInfo(workspace_id="alice", scopes=("mcp:memory",))
    ws = _run(auth_mod.get_workspace(info=info, x_workspace_id="bene"))
    assert ws == "alice"


def test_user_token_no_header_returns_token_workspace():
    info = TokenInfo(workspace_id="alice", scopes=("mcp:memory",))
    ws = _run(auth_mod.get_workspace(info=info, x_workspace_id=None))
    assert ws == "alice"


def test_service_token_default_is_system():
    """Service-Token ohne Header → 'system'-Maintenance-Bucket,
    NICHT der Workspace eines spezifischen Users."""
    info = TokenInfo(workspace_id="system", scopes=("*",))
    ws = _run(auth_mod.get_workspace(info=info, x_workspace_id=None))
    assert ws == "system"


def test_service_token_can_target_workspace_via_header():
    """Service-Token darf via X-Workspace-Id in Tenant-Workspace
    schreiben — z.B. post-deploy-ingest für Tenant 'bene'."""
    info = TokenInfo(workspace_id="system", scopes=("*",))
    ws = _run(auth_mod.get_workspace(info=info, x_workspace_id="bene"))
    assert ws == "bene"


def test_service_token_empty_header_falls_back_to_system():
    """Whitespace/empty header zählt als nicht-gesetzt."""
    info = TokenInfo(workspace_id="system", scopes=("*",))
    ws = _run(auth_mod.get_workspace(info=info, x_workspace_id="   "))
    assert ws == "system"


def test_admin_user_jwt_does_not_get_admin_workspace_override():
    """Auch Admin-User-JWTs (scope: mcp:memory + admin) sollen NICHT
    via X-Workspace-Id in fremde Buckets schreiben — nur das pure
    Service-Token (scope='*') darf das. Sonst könnte ein admin-
    User-Token die Tenant-Isolation aufbrechen."""
    info = TokenInfo(workspace_id="alice", scopes=("mcp:memory", "admin"))
    ws = _run(auth_mod.get_workspace(info=info, x_workspace_id="bene"))
    assert ws == "alice"
