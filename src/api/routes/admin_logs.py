"""Admin-only log-tail endpoint — Phase A of Issue #213.

User-Use-Case: claude-code agent (= ich) muss heute mehrfach via
`ssh nileneb@u-server 'docker logs mayring-mayring-api-1 --since 5m'`
für deploy-fail-diagnose / smoke-debug / cloud-routing-check. Das
skaliert nicht und der agent kann's nicht selbst.

Endpoint: GET /admin/logs?service=api|pi|mcp|webui&since=5m&grep=...

Auth: admin scope im JWT (TokenInfo.scopes enthält '*' oder 'admin').
Rate: 5 calls/min/admin (in-memory token bucket).
Quelle: `docker logs --since=...` als subprocess (whitelisted service-name).
Secret-redaction: env-vars + JWT-payload patterns vor return maskiert.
"""

from __future__ import annotations

import re
import subprocess
import time
from collections import deque
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.auth import get_token_info
from src.api.jwt_auth import TokenInfo

router = APIRouter()


# WHY(#213): nur whitelisted services. Beliebige docker-container-namen
# wären command-injection oder remote-host-exfil-risiko.
_ALLOWED_SERVICES = {
    "api": "mayring-mayring-api-1",
    "pi": "mayring-mayring-pi-1",
    "mcp": "mayring-mayring-mcp-1",
    "webui": "mayring-mayring-webui-1",
    "nginx": "mayring-mayring-nginx-1",
}

# Allowed --since-values (per docker-logs syntax). Beliebige strings
# wären erneut injection-vektor. Auch hier whitelisted statt regex.
_ALLOWED_SINCE = {"1m", "5m", "15m", "30m", "1h", "6h", "24h"}

_MAX_LINES = 500


def _is_admin(info: TokenInfo) -> bool:
    return "*" in info.scopes or "admin" in info.scopes


# Rate-limiter — per-user token bucket in-memory.
# 5 calls/min/user = window von 12s zwischen calls (smooth) ODER burst.
# Process-local: ein api-replica pro u-server reicht, kein Redis nötig.
_RATE_BUCKET: dict[str, deque[float]] = {}
_RATE_WINDOW = 60.0
_RATE_LIMIT = 5


def _check_rate_limit(user_id: str) -> None:
    now = time.monotonic()
    bucket = _RATE_BUCKET.setdefault(user_id, deque(maxlen=_RATE_LIMIT))
    # Drop expired
    while bucket and bucket[0] < now - _RATE_WINDOW:
        bucket.popleft()
    if len(bucket) >= _RATE_LIMIT:
        oldest = bucket[0]
        retry_after = int(_RATE_WINDOW - (now - oldest)) + 1
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit: {_RATE_LIMIT} calls/min. Retry in {retry_after}s.",
        )
    bucket.append(now)


# Secret-redaction patterns. Conservative — bei doppelt-anwendung lieber
# falsch maskiert als secret leakage.
_REDACT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # eyJ... = JWT header (Base64 of {"alg":...}). Maskiere full token.
    (re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]+"),
     "<JWT_REDACTED>"),
    # Bearer + 20+ char token
    (re.compile(r"(Bearer|Authorization:\s*Bearer)\s+[A-Za-z0-9_\-]{20,}", re.IGNORECASE),
     r"\1 <REDACTED>"),
    # Generic 32+ char hex (potential api-key or service-token)
    (re.compile(r"\b[A-Fa-f0-9]{32,}\b"), "<HEX_REDACTED>"),
    # password=... / pwd=... / secret=... / token=... in URL/log-context
    (re.compile(r"(password|pwd|secret|token|api[_-]?key)=([^\s&'\"]+)", re.IGNORECASE),
     r"\1=<REDACTED>"),
    # postgres connection strings with password
    (re.compile(r"://([^:]+):([^@]+)@"), r"://\1:<REDACTED>@"),
]


def _redact(text: str) -> str:
    for pat, repl in _REDACT_PATTERNS:
        text = pat.sub(repl, text)
    return text


def _parse_line(raw: str) -> dict[str, Any]:
    """Try to extract structured fields from a docker-log line."""
    # docker logs prefix: timestamp container-name | rest
    # If line starts with ISO timestamp (--timestamps not used here so
    # rarely), try to parse. Otherwise return raw text.
    line = _redact(raw.rstrip("\n"))
    # Detect level by keyword
    level = "info"
    upper = line.upper()
    if " ERROR" in upper or " CRITICAL" in upper or " FATAL" in upper:
        level = "error"
    elif " WARN" in upper:
        level = "warning"
    elif " DEBUG" in upper:
        level = "debug"
    return {"raw": line, "level": level}


@router.get("/admin/logs")
def admin_logs(
    service: str = Query(..., description="api | pi | mcp | webui | nginx"),
    since: str = Query("5m", description="1m, 5m, 15m, 30m, 1h, 6h, 24h"),
    grep: str | None = Query(None, description="case-insensitive substring filter"),
    limit: int = Query(200, ge=1, le=_MAX_LINES),
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """Return docker-container logs for an admin caller.

    WHY(#213): replaces manual `ssh u-server docker logs ...` workflow.
    Secrets in log lines are redacted (JWT/Bearer/hex/password/conn-strs).
    """
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")

    if service not in _ALLOWED_SERVICES:
        raise HTTPException(
            status_code=400,
            detail=f"service must be one of: {sorted(_ALLOWED_SERVICES)}",
        )
    if since not in _ALLOWED_SINCE:
        raise HTTPException(
            status_code=400,
            detail=f"since must be one of: {sorted(_ALLOWED_SINCE)}",
        )

    _check_rate_limit(info.sub or "anon")

    container = _ALLOWED_SERVICES[service]
    try:
        proc = subprocess.run(
            ["docker", "logs", "--since", since, "--tail", str(limit * 2), container],
            capture_output=True, text=True, timeout=8,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="docker logs timed out")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="docker CLI not available")

    # docker logs schreibt stdout + stderr getrennt — beide einsammeln
    raw = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 and not raw:
        raise HTTPException(
            status_code=500,
            detail=f"docker logs failed (exit {proc.returncode})",
        )

    lines = [l for l in raw.split("\n") if l.strip()]

    if grep:
        needle = grep.lower()
        lines = [l for l in lines if needle in l.lower()]

    # tail-style: last N lines
    truncated = len(lines) > limit
    lines = lines[-limit:]

    parsed = [{"container": container, **_parse_line(l)} for l in lines]

    return {
        "service": service,
        "container": container,
        "since": since,
        "grep": grep,
        "total": len(parsed),
        "truncated": truncated,
        "lines": parsed,
    }


@router.get("/admin/logs/services")
def admin_logs_services(info: TokenInfo = Depends(get_token_info)) -> dict:
    """List available service names + container mappings."""
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    return {
        "services": _ALLOWED_SERVICES,
        "since_options": sorted(_ALLOWED_SINCE),
        "max_lines": _MAX_LINES,
        "rate_limit": f"{_RATE_LIMIT}/min",
    }
