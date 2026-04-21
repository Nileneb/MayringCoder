"""Tests für BYO-LLM-Provider-Routing (Phase 2).

Deckt ab:
  - TokenInfo liest Provider-Claims aus dem JWT-Payload (B1)
  - key_callback.fetch_user_key mit Cache + graceful-404 (B2)
  - get_endpoint_for_request Provider-Precedence + Hardfail (B3)
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from src.api import jwt_auth
from src.llm import key_callback
from src.llm.endpoint import (
    LLMEndpoint,
    get_endpoint_for_request,
    invalidate_cache,
)
from src.llm.key_callback import LlmKeyUnavailableError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rsa_keys():
    pk = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv = pk.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    pub = pk.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return priv, pub


@pytest.fixture
def configured_env(rsa_keys, tmp_path: Path, monkeypatch):
    _, pub = rsa_keys
    key_file = tmp_path / "jwt_public.pem"
    key_file.write_text(pub)
    monkeypatch.setenv("JWT_PUBLIC_KEY_PATH", str(key_file))
    monkeypatch.setenv("JWT_ISSUER", "https://app.linn.games")
    monkeypatch.setenv("JWT_AUDIENCE", "mayringcoder")
    monkeypatch.setenv("MCP_SERVICE_TOKEN", "test-service-token")
    monkeypatch.setenv("LARAVEL_INTERNAL_URL", "http://web.test")
    monkeypatch.setenv("JWT_TTL_SECONDS", "3600")
    monkeypatch.setenv("OLLAMA_MODEL", "")
    jwt_auth.reset_public_key_cache()
    key_callback.invalidate_cache()
    invalidate_cache()
    yield rsa_keys[0]
    jwt_auth.reset_public_key_cache()
    key_callback.invalidate_cache()
    invalidate_cache()


def _sign(priv: str, **claims) -> str:
    claims.setdefault("scope", ["mcp:memory"])
    claims.setdefault("iss", "https://app.linn.games")
    claims.setdefault("aud", "mayringcoder")
    claims.setdefault("exp", int(time.time()) + 300)
    claims.setdefault("iat", int(time.time()))
    claims.setdefault("workspace_id", "ws_test")
    claims.setdefault("sub", "42")
    return pyjwt.encode(claims, priv, algorithm="RS256")


# ---------------------------------------------------------------------------
# B1: TokenInfo Provider-Claims
# ---------------------------------------------------------------------------

def test_token_info_defaults_to_platform_without_provider_claim(configured_env):
    token = _sign(configured_env)
    info = jwt_auth.validate_jwt_token(token)
    assert info is not None
    assert info.llm_provider == "platform"
    assert info.llm_model is None
    assert info.llm_endpoint is None
    assert info.llm_requires_key is False
    assert info.uses_custom_provider is False


def test_token_info_extracts_anthropic_byo_claims(configured_env):
    token = _sign(
        configured_env,
        llm_provider="anthropic-byo",
        llm_model="claude-sonnet-4-6",
        llm_requires_key=True,
    )
    info = jwt_auth.validate_jwt_token(token)
    assert info is not None
    assert info.llm_provider == "anthropic-byo"
    assert info.llm_model == "claude-sonnet-4-6"
    assert info.llm_endpoint is None
    assert info.llm_requires_key is True
    assert info.uses_custom_provider is True


def test_token_info_extracts_openai_compatible_claims(configured_env):
    token = _sign(
        configured_env,
        llm_provider="openai-compatible",
        llm_model="llama3.2",
        llm_endpoint="http://three.linn.games:11434",
        llm_requires_key=False,
        sub="99",
        iat=1700000000,
    )
    info = jwt_auth.validate_jwt_token(token)
    assert info is not None
    assert info.llm_provider == "openai-compatible"
    assert info.llm_endpoint == "http://three.linn.games:11434"
    assert info.sub == "99"
    assert info.iat == 1700000000


# ---------------------------------------------------------------------------
# B2: key_callback
# ---------------------------------------------------------------------------

def _mock_response(status: int, json_body: dict | None = None, text: str = ""):
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    resp.json = MagicMock(return_value=json_body or {})
    return resp


def test_fetch_user_key_caches_by_sub_and_iat(configured_env):
    with patch("src.llm.key_callback.httpx.post") as mock_post:
        mock_post.return_value = _mock_response(200, {"api_key": "sk-123", "provider": "anthropic-byo"})

        key1 = key_callback.fetch_user_key("42", 1700000000, "jwt-string")
        key2 = key_callback.fetch_user_key("42", 1700000000, "jwt-string")

        assert key1 == "sk-123"
        assert key2 == "sk-123"
        # Second call is served from cache — only one HTTP request.
        assert mock_post.call_count == 1


def test_fetch_user_key_bypasses_cache_on_different_iat(configured_env):
    with patch("src.llm.key_callback.httpx.post") as mock_post:
        mock_post.return_value = _mock_response(200, {"api_key": "sk-123"})

        key_callback.fetch_user_key("42", 1700000000, "jwt-1")
        key_callback.fetch_user_key("42", 1700000100, "jwt-2")

        assert mock_post.call_count == 2


def test_fetch_user_key_returns_none_on_404(configured_env):
    with patch("src.llm.key_callback.httpx.post") as mock_post:
        mock_post.return_value = _mock_response(404, text="no key")

        assert key_callback.fetch_user_key("42", 1700000000, "jwt") is None


def test_fetch_user_key_returns_none_when_service_token_missing(configured_env, monkeypatch):
    monkeypatch.setenv("MCP_SERVICE_TOKEN", "")
    with patch("src.llm.key_callback.httpx.post") as mock_post:
        result = key_callback.fetch_user_key("42", 1700000000, "jwt")
        assert result is None
        mock_post.assert_not_called()


def test_fetch_user_key_handles_network_errors(configured_env):
    with patch("src.llm.key_callback.httpx.post", side_effect=Exception("connection refused")):
        assert key_callback.fetch_user_key("42", 1700000000, "jwt") is None


# ---------------------------------------------------------------------------
# B3: get_endpoint_for_request — Provider-Precedence
# ---------------------------------------------------------------------------

def test_platform_provider_falls_back_to_workspace_endpoint(configured_env, monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://platform-default:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:30b")
    token = _sign(configured_env)
    info = jwt_auth.validate_jwt_token(token)

    # No workspace endpoint configured → falls through to platform default.
    endpoint = get_endpoint_for_request(info, workspace_id="ws_test", user_jwt=None)
    assert endpoint.provider == "platform"
    assert endpoint.base_url == "http://platform-default:11434"


def test_openai_compatible_with_user_endpoint_skips_key_callback(configured_env):
    token = _sign(
        configured_env,
        llm_provider="openai-compatible",
        llm_model="llama3.2",
        llm_endpoint="http://three.linn.games:11434",
        llm_requires_key=False,
    )
    info = jwt_auth.validate_jwt_token(token)

    with patch("src.llm.key_callback.httpx.post") as mock_post:
        endpoint = get_endpoint_for_request(info, workspace_id="ws_test", user_jwt=token)

    assert endpoint.provider == "openai"
    assert endpoint.base_url == "http://three.linn.games:11434"
    assert endpoint.model == "llama3.2"
    assert endpoint.api_key is None
    mock_post.assert_not_called()


def test_anthropic_byo_calls_key_callback_and_builds_endpoint(configured_env):
    token = _sign(
        configured_env,
        llm_provider="anthropic-byo",
        llm_model="claude-sonnet-4-6",
        llm_requires_key=True,
        sub="42",
    )
    info = jwt_auth.validate_jwt_token(token)

    with patch("src.llm.key_callback.httpx.post") as mock_post:
        mock_post.return_value = _mock_response(200, {"api_key": "sk-ant-user"})
        endpoint = get_endpoint_for_request(info, workspace_id="ws_test", user_jwt=token)

    assert endpoint.provider == "anthropic"
    assert endpoint.base_url == "https://api.anthropic.com"
    assert endpoint.model == "claude-sonnet-4-6"
    assert endpoint.api_key == "sk-ant-user"


def test_hardfails_when_key_required_but_callback_returns_404(configured_env):
    token = _sign(
        configured_env,
        llm_provider="anthropic-byo",
        llm_model="claude-sonnet-4-6",
        llm_requires_key=True,
        sub="42",
    )
    info = jwt_auth.validate_jwt_token(token)

    with patch("src.llm.key_callback.httpx.post") as mock_post:
        mock_post.return_value = _mock_response(404)
        with pytest.raises(LlmKeyUnavailableError):
            get_endpoint_for_request(info, workspace_id="ws_test", user_jwt=token)


def test_hardfails_when_openai_compatible_has_no_endpoint(configured_env):
    token = _sign(
        configured_env,
        llm_provider="openai-compatible",
        llm_model="llama3.2",
        llm_requires_key=False,
        # llm_endpoint intentionally missing
    )
    info = jwt_auth.validate_jwt_token(token)

    with pytest.raises(LlmKeyUnavailableError):
        get_endpoint_for_request(info, workspace_id="ws_test", user_jwt=token)


def test_hardfails_when_custom_provider_has_no_model(configured_env, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "")
    token = _sign(
        configured_env,
        llm_provider="anthropic-byo",
        llm_requires_key=False,
        # llm_model missing
    )
    info = jwt_auth.validate_jwt_token(token)

    with pytest.raises(LlmKeyUnavailableError):
        get_endpoint_for_request(info, workspace_id="ws_test", user_jwt=token)


def test_none_token_info_falls_back_to_workspace_endpoint(configured_env):
    # stdio-mode (no JWT) → direct workspace lookup.
    endpoint = get_endpoint_for_request(None, workspace_id=None, user_jwt=None)
    assert isinstance(endpoint, LLMEndpoint)
    assert endpoint.provider == "platform"
