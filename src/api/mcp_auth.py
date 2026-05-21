"""JWT auth middleware and context helpers for MCP HTTP transport."""

from __future__ import annotations

import contextvars
import os
from typing import Any

from src.api.jwt_auth import TokenInfo, validate_jwt_token

# Default ON. Forgotten env var on a prod deploy must NOT silently expose
# unauthenticated MCP. Local dev opts out via MCP_AUTH_ENABLED=false.
_AUTH_ENABLED  = os.getenv("MCP_AUTH_ENABLED", "true").lower() in ("true", "1", "yes")
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
    """Backward-compat shim — delegates to resolve_workspace_from_token().

    Vor V2 hatte dies eigenen TokenInfo-Read; jetzt ist
    workspace_resolver.resolve_workspace_from_token die einzige SoT.
    Wenn kein TokenInfo (Tests / manueller MCP-Call), fallback auf
    caller_default ('default').
    """
    info = _TOKEN_CTX.get(None)
    if info is None:
        return caller_default or "default"
    from src.identity.workspace_resolver import resolve_workspace_from_token
    return resolve_workspace_from_token(info, override_header=None)


def _enforce_tenant(requested: str | None) -> str | None:
    """MCP-tool-arg-driven workspace override.

    Anders als get_workspace (HTTP-Header-Pfad), ist `requested` hier ein
    explizit von der Tool-Signature übergebener Wert (z.B.
    `search_memory(workspace_id='other')`). Admin-USER UND Service-Token
    dürfen damit cross-workspace lesen — der Tool-Caller hat den Wert
    bewusst gesetzt, kein silent header-Override. Reguläre Token-User
    werden auf ihren Workspace gepinnt.

    WHY(v2-stufe1.1): NICHT via resolve_workspace_from_token, weil das
    nur Service-Token-Override erlaubt. Hier ist die Semantik anders
    (MCP-explicit-arg vs HTTP-header).
    """
    info = _TOKEN_CTX.get(None)
    if info is None:
        return requested
    if info.is_admin:
        return requested
    return info.workspace_id


def _effective_user_id() -> str | None:
    """Caller's user_id (JWT.sub). Same value across all workspaces of the
    same human user, so it's the right key for visibility='user' sharing."""
    info = _TOKEN_CTX.get(None)
    return info.sub if info is not None else None


def _effective_org_id() -> str | None:
    """Legacy single-org accessor. Prefer _effective_org_ids() in new code."""
    info = _TOKEN_CTX.get(None)
    if info is None:
        return None
    if info.org_id:
        return info.org_id
    # Fall back to first membership-derived org so legacy callsites still
    # surface at least one org bucket post-V2-JWT-rollout.
    org_ids = info.org_ids
    return org_ids[0] if org_ids else None


def _effective_org_ids() -> tuple[str, ...]:
    """V2: all organization-workspace ids the caller is a member of."""
    info = _TOKEN_CTX.get(None)
    return info.org_ids if info is not None else ()


def _effective_active_workspace_id() -> str | None:
    """The app.linn.games *active* workspace (its UUID), or None."""
    info = _TOKEN_CTX.get(None)
    return info.active_workspace_id if info is not None else None


def _effective_active_workspace_kind() -> str:
    """'personal' | 'organization' for the active workspace."""
    info = _TOKEN_CTX.get(None)
    return info.active_workspace_kind if info is not None else "personal"


def resolve_write_visibility(
    *,
    active_workspace_id: str | None,
    active_workspace_kind: str,
    org_ids: tuple[str, ...] | list[str] | None,
    user_id: str | None,
) -> tuple[str, str | None, str | None]:
    """Decide (visibility, org_id, user_id) for a memory write.

    Pure — no context — so the rule is unit-testable. Drives the MCP
    remember/ingest path: when the caller's active app.linn.games workspace is
    an organization they belong to, the write is stamped visibility='org' so
    the whole team sees it (read-side already handled by retrieval._scope_filter).
    Otherwise it stays per-user ('user', cross-app same human) or 'private'.
    Membership is re-checked here so an active='org' claim the caller isn't a
    member of can never silently write into a foreign org bucket.
    """
    if (
        active_workspace_kind == "organization"
        and active_workspace_id
        and active_workspace_id in (org_ids or ())
    ):
        return "org", active_workspace_id, user_id
    if user_id:
        return "user", None, user_id
    return "private", None, None


class JWTAuthMiddleware:
    """RS256 JWT auth for MCP HTTP transport.

    Token via: Authorization: Bearer <jwt>  or  X-Auth-Token: <jwt>
    Admin access: JWT claim `scope: ["admin"]`.

    auth_enabled overrides the module-level _AUTH_ENABLED (useful for tests).
    """

    def __init__(self, app: Any, *, auth_enabled: bool | None = None) -> None:
        self._app = app
        self._auth_enabled = _AUTH_ENABLED if auth_enabled is None else auth_enabled

    # Paths that the docker healthcheck and OAuth flow need *without* a
    # bearer token. Without this, the container's `curl /health` probe gets
    # 401, exits non-zero, and Docker marks the container "unhealthy" — the
    # MCP-mayring container ran in that state silently for weeks because
    # nothing actually depends on the unhealthy flag.
    _PUBLIC_PATHS: frozenset[str] = frozenset({
        "/health",
        "/healthz",
        "/.well-known/oauth-authorization-server",
        "/.well-known/oauth-protected-resource",
    })

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in self._PUBLIC_PATHS:
            _TOKEN_CTX.set(None)
            _RAW_JWT_CTX.set(None)
            await self._app(scope, receive, send)
            return

        if not self._auth_enabled:
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
