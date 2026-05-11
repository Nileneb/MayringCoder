"""Tests for /admin/logs endpoint (Phase A of Issue #213)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from src.api.jwt_auth import TokenInfo
from src.api.routes.admin_logs import (
    _check_rate_limit,
    _is_admin,
    _parse_line,
    _redact,
    _RATE_BUCKET,
    admin_logs,
    admin_logs_services,
)


def _admin_info(user_id: str = "user_a") -> TokenInfo:
    return TokenInfo(workspace_id="default", scopes=("admin",), sub=user_id)


def _user_info(user_id: str = "user_b") -> TokenInfo:
    return TokenInfo(workspace_id="default", scopes=("user",), sub=user_id)


@pytest.fixture(autouse=True)
def reset_rate_bucket():
    _RATE_BUCKET.clear()
    yield
    _RATE_BUCKET.clear()


def test_is_admin_recognizes_admin_scope():
    assert _is_admin(_admin_info()) is True


def test_is_admin_recognizes_wildcard_scope():
    info = TokenInfo(workspace_id="default", scopes=("*",), sub="x")
    assert _is_admin(info) is True


def test_is_admin_rejects_user_scope():
    assert _is_admin(_user_info()) is False


def test_redact_jwt_in_log_line():
    line = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.signature_part_xxx"
    out = _redact(line)
    assert "eyJ" not in out or "REDACTED" in out


def test_redact_postgres_password_in_conn_string():
    line = "postgres://user:supersecret@db:5432/linn"
    out = _redact(line)
    assert "supersecret" not in out
    assert "REDACTED" in out


def test_redact_password_param():
    line = "request: password=mySecret123 next-param"
    out = _redact(line)
    assert "mySecret123" not in out
    assert "REDACTED" in out


def test_redact_hex_token():
    line = "token: a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4 in line"
    out = _redact(line)
    assert "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4" not in out


def test_parse_line_detects_error_level():
    out = _parse_line("[2026-05-11] ERROR something broke")
    assert out["level"] == "error"


def test_parse_line_detects_warning_level():
    # _parse_line looks for ' WARN' (space-prefixed). Match real log format.
    out = _parse_line("[timestamp] WARN: slow query")
    assert out["level"] == "warning"


def test_parse_line_defaults_to_info():
    out = _parse_line("just a regular log line")
    assert out["level"] == "info"


def test_rate_limit_allows_5_calls_then_blocks():
    user = "user_rate_test"
    for _ in range(5):
        _check_rate_limit(user)  # no raise
    with pytest.raises(HTTPException) as exc_info:
        _check_rate_limit(user)
    assert exc_info.value.status_code == 429


def test_rate_limit_isolated_per_user():
    _check_rate_limit("user_x")
    _check_rate_limit("user_x")
    _check_rate_limit("user_x")
    _check_rate_limit("user_x")
    _check_rate_limit("user_x")
    # user_y should have its own bucket
    _check_rate_limit("user_y")  # no raise


def test_endpoint_rejects_non_admin():
    with pytest.raises(HTTPException) as exc_info:
        admin_logs(service="api", since="5m", grep=None, limit=200, info=_user_info())
    assert exc_info.value.status_code == 403


def test_endpoint_rejects_unknown_service():
    with pytest.raises(HTTPException) as exc_info:
        admin_logs(service="evil-container", since="5m", grep=None, limit=200, info=_admin_info())
    assert exc_info.value.status_code == 400


def test_endpoint_rejects_unknown_since():
    with pytest.raises(HTTPException) as exc_info:
        admin_logs(service="api", since="1ns", grep=None, limit=200, info=_admin_info())
    assert exc_info.value.status_code == 400


def test_endpoint_returns_lines_on_success():
    with patch("src.api.routes.admin_logs.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="INFO request handled\n[ts] ERROR boom\n",
            stderr="",
        )
        result = admin_logs(service="api", since="5m", grep=None, limit=200, info=_admin_info("user_success"))
    assert result["service"] == "api"
    assert result["total"] == 2
    assert result["lines"][1]["level"] == "error"


def test_endpoint_applies_grep_filter():
    with patch("src.api.routes.admin_logs.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="line1 hello\nline2 world\nline3 hello again\n",
            stderr="",
        )
        result = admin_logs(service="api", since="5m", grep="hello", limit=200, info=_admin_info("user_grep"))
    assert result["total"] == 2
    assert all("hello" in l["raw"].lower() for l in result["lines"])


def test_endpoint_redacts_secrets():
    with patch("src.api.routes.admin_logs.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="auth header: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.signature\n",
            stderr="",
        )
        result = admin_logs(service="api", since="5m", grep=None, limit=200, info=_admin_info("user_redact"))
    line = result["lines"][0]["raw"]
    assert "signature" not in line or "REDACTED" in line


def test_services_endpoint_lists_whitelist():
    out = admin_logs_services(info=_admin_info())
    assert "api" in out["services"]
    assert "pi" in out["services"]
    assert "5/min" in out["rate_limit"]


def test_services_endpoint_rejects_non_admin():
    with pytest.raises(HTTPException) as exc_info:
        admin_logs_services(info=_user_info())
    assert exc_info.value.status_code == 403
