"""JWT auth middleware and context helpers for MCP HTTP transport."""

from __future__ import annotations

import contextvars
import os
from typing import Any

from src.api.jwt_auth import TokenInfo, validate_jwt_token

_AUTH_ENABLED  = os.getenv("MCP_AUTH_ENABLED", "false").lower() in ("true", "1", "yes")
_OAUTH_BASE_URL = os.getenv("MCP_OAUTH_BASE_URL", "https://mcp.linn.games")

_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

_TOKEN_CTX: contextvars.ContextVar["TokenInfo | None"] = contextvars.ContextVar(
    "token_info", default=None
)
_RAW_JWT_CTX: contextvars.ContextVar["str | None"] = contextvars.ContextVar(
    "raw_jwt", default=None
)


def _current_token_info() -> "TokenInfo | None":
    return _TOKEN_CTX.get(None)


def _current_raw_jwt() -> "str | None":
    return _RAW_JWT_CTX.get(None)


def _effective_workspace_id(caller_default: str = "default") -> str:
    info = _TOKEN_CTX.get(None)
    if info is None:
        return caller_default or "default"
    return info.workspace_id


def _enforce_tenant(requested: str | None) -> str | None:
    info = _TOKEN_CTX.get(None)
    if info is None:
        return requested
    if info.is_admin:
        return requested
    return info.workspace_id


class JWTAuthMiddleware:
    """RS256 JWT auth for MCP HTTP transport.

    Token via: Authorization: Bearer <jwt>  or  X-Auth-Token: <jwt>
    Admin access: JWT claim `scope: ["admin"]`.
    """

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        if not _AUTH_ENABLED:
            _TOKEN_CTX.set(None)
            _RAW_JWT_CTX.set(None)
            await self._app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        token: str = ""
        raw = headers.get(b"x-auth-token", b"").decode().strip()
        if raw:
            token = raw
        else:
            auth_header = headers.get(b"authorization", b"").decode().strip()
            if auth_header.lower().startswith("bearer "):
                token = auth_header[7:].strip()

        if not token:
            await self._send_401(send, "Missing authentication token")
            return

        info = validate_jwt_token(token)
        if info is None:
            await self._send_401(send, "Invalid or expired token")
            return

        _TOKEN_CTX.set(info)
        _RAW_JWT_CTX.set(token)
        scope["workspace_id"] = info.workspace_id
        await self._app(scope, receive, send)

    @staticmethod
    async def _send_401(send: Any, message: str) -> None:
        body = message.encode()
        metadata_url = f"{_OAUTH_BASE_URL}/.well-known/oauth-authorization-server"
        www_auth = f'Bearer realm="{_OAUTH_BASE_URL}", resource_metadata="{metadata_url}"'
        await send({
            "type": "http.response.start", "status": 401,
            "headers": [
                [b"content-type", b"text/plain; charset=utf-8"],
                [b"content-length", str(len(body)).encode()],
                [b"www-authenticate", www_auth.encode()],
            ],
        })
        await send({"type": "http.response.body", "body": body})
