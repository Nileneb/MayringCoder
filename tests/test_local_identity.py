"""Tests für die lokale Identity-Cache-Schicht."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from src.identity import local_identity as li
from src.identity.local_identity import (
    LocalIdentity,
    clear_identity,
    get_local_user_id,
    load_identity,
    save_identity,
)


@pytest.fixture(autouse=True)
def _isolated_config_home(tmp_path, monkeypatch):
    """Zwinge XDG_CONFIG_HOME auf ein tmp-Verzeichnis, damit Tests nicht
    in das echte ~/.config/mayring schreiben."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.delenv("MAYRING_USER_ID", raising=False)
    yield


# ─── save → load roundtrip ─────────────────────────────────────────


def test_save_and_load_roundtrip():
    saved = save_identity(LocalIdentity(user_id=42, email="u@x.com", token="abc"))
    assert saved.exists()
    loaded = load_identity()
    assert loaded is not None
    assert loaded.user_id == 42
    assert loaded.email == "u@x.com"
    assert loaded.token == "abc"


def test_save_sets_0600_permissions():
    """Token landet in der Datei → nur User darf lesen."""
    path = save_identity(LocalIdentity(user_id=1, token="secret"))
    mode = path.stat().st_mode & 0o777
    assert mode == 0o600


def test_save_is_atomic_via_tempfile(tmp_path, monkeypatch):
    """Bei Crash mitten im Schreiben darf die alte Datei nicht halb-
    überschrieben sein. Wir prüfen den Tempfile-Pattern."""
    save_identity(LocalIdentity(user_id=1, token="first"))
    save_identity(LocalIdentity(user_id=2, token="second"))
    loaded = load_identity()
    assert loaded.user_id == 2  # neuer Wert vollständig durchgekommen


# ─── env-vs-file Priorität ─────────────────────────────────────────


def test_env_takes_priority_over_file(monkeypatch):
    save_identity(LocalIdentity(user_id=99, token="t"))
    monkeypatch.setenv("MAYRING_USER_ID", "7")
    assert get_local_user_id() == 7


def test_file_used_when_env_missing():
    save_identity(LocalIdentity(user_id=99, token="t"))
    assert get_local_user_id() == 99


def test_returns_none_when_neither_env_nor_file():
    """Documented contract: kein silent 'default' fallback."""
    assert get_local_user_id() is None


def test_malformed_env_falls_back_to_file(monkeypatch):
    save_identity(LocalIdentity(user_id=42, token="t"))
    monkeypatch.setenv("MAYRING_USER_ID", "not-a-number")
    assert get_local_user_id() == 42


def test_missing_file_returns_none():
    assert load_identity() is None
    assert get_local_user_id() is None


def test_corrupt_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    path = tmp_path / ".config" / "mayring" / "identity.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not valid json")
    assert load_identity() is None


def test_missing_user_id_field_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    path = tmp_path / ".config" / "mayring" / "identity.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"email": "u@x.com"}')
    assert load_identity() is None


# ─── Token expiry check ────────────────────────────────────────────


def test_is_token_valid_with_future_expiry():
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    ident = LocalIdentity(user_id=1, token="t", expires_at=future)
    assert ident.is_token_valid() is True


def test_is_token_valid_with_past_expiry():
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    ident = LocalIdentity(user_id=1, token="t", expires_at=past)
    assert ident.is_token_valid() is False


def test_is_token_valid_without_expiry_uses_token_presence():
    assert LocalIdentity(user_id=1, token="abc").is_token_valid() is True
    assert LocalIdentity(user_id=1, token=None).is_token_valid() is False


# ─── clear_identity ────────────────────────────────────────────────


def test_clear_removes_file():
    save_identity(LocalIdentity(user_id=1, token="t"))
    assert clear_identity() is True
    assert load_identity() is None


def test_clear_when_missing_returns_false():
    assert clear_identity() is False
