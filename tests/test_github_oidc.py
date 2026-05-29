"""Security-Tests für GitHub-OIDC-Verifikation (#1, secretlose repo-watch-Auth).

Diese gaten ein PROD-Auth: ein Bug hier = Auth-Bypass auf /repo-events. Daher explizit:
gültiges owner-Token akzeptiert; falsche aud/iss/owner/Signatur → abgelehnt; non-JWT → None.
"""
from __future__ import annotations

import time
from unittest.mock import patch

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from src.api import github_oidc

_GH_ISS = "https://token.actions.githubusercontent.com"


@pytest.fixture(scope="module")
def keys():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv_pem = priv.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    pub_pem = priv.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv_pem, pub_pem


def _mint(priv_pem, **overrides) -> str:
    now = int(time.time())
    payload = {
        "iss": _GH_ISS,
        "aud": "mcp.linn.games",
        "iat": now,
        "exp": now + 300,
        "repository": "Nileneb/mayring-core",
        "repository_owner": "Nileneb",
        "repository_owner_id": "12345",
    }
    payload.update(overrides)
    return jwt.encode(payload, priv_pem, algorithm="RS256")


class _FakeKey:
    def __init__(self, pub_pem):
        self.key = pub_pem


def _patch_jwks(pub_pem):
    class _FakeClient:
        def get_signing_key_from_jwt(self, token):
            return _FakeKey(pub_pem)
    return patch.object(github_oidc, "_client", lambda: _FakeClient())


def test_valid_owner_token_accepted(keys):
    priv, pub = keys
    with _patch_jwks(pub):
        claims = github_oidc.verify_github_oidc(_mint(priv))
    assert claims is not None
    assert claims["repository"] == "Nileneb/mayring-core"


def test_wrong_audience_rejected(keys):
    priv, pub = keys
    with _patch_jwks(pub):
        assert github_oidc.verify_github_oidc(_mint(priv, aud="evil.example.com")) is None


def test_wrong_owner_rejected(keys):
    priv, pub = keys
    with _patch_jwks(pub):
        tok = _mint(priv, repository_owner="Attacker", repository_owner_id="999")
        assert github_oidc.verify_github_oidc(tok) is None


def test_wrong_issuer_no_network(keys):
    priv, pub = keys
    # iss != GitHub → must bail BEFORE touching the JWKS client (assert it's never called)
    with patch.object(github_oidc, "_client", side_effect=AssertionError("JWKS fetched for non-GH token")):
        assert github_oidc.verify_github_oidc(_mint(priv, iss="https://evil.example.com")) is None


def test_bad_signature_rejected(keys):
    priv, pub = keys
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other_pem = other.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    forged = _mint(other_pem)  # signed by a DIFFERENT key
    with _patch_jwks(pub):     # server only trusts `pub`
        assert github_oidc.verify_github_oidc(forged) is None


def test_owner_id_fallback_on_rename(keys, monkeypatch):
    priv, pub = keys
    monkeypatch.setattr(github_oidc, "_OWNER", "OldName")
    monkeypatch.setattr(github_oidc, "_OWNER_ID", "12345")
    with _patch_jwks(pub):
        # repository_owner mismatch but owner_id matches → accepted (rename-stable)
        tok = _mint(priv, repository_owner="RenamedOwner", repository_owner_id="12345")
        assert github_oidc.verify_github_oidc(tok) is not None


def test_non_jwt_returns_none():
    assert github_oidc.verify_github_oidc("static-service-token") is None
    assert github_oidc.verify_github_oidc("") is None
