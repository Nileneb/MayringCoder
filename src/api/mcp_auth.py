"""JWT auth middleware and context helpers for MCP HTTP transport."""

from __future__ import annotations

import contextvars
import logging
import os
import uuid
from typing import Any

from src.api.jwt_auth import TokenInfo, validate_jwt_token

_DEVICE_WARNED = False


def _local_device_id() -> str:
    """Stable per-device id (decoupled from hostname) for the unclaimed bucket of a
    device that has no valid hook.jwt. Persisted to <config>/mayring/device_id."""
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    p = os.path.join(base, "mayring", "device_id")
    try:
        with open(p, encoding="utf-8") as fh:
            v = fh.read().strip()
            if v:
                return v
    except OSError:
        pass
    v = uuid.uuid4().hex[:12]
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(v)
    os.chmod(p, 0o600)
    return v

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

# WHY(pi-observability): the stdio MCP server (local_mcp) has no per-HTTP-request
# token, so the contextvars above stay None → every tool fell back to workspace
# 'default' and had no token to authenticate cloud calls. A process-wide identity
# loaded once from the local hook.jwt fixes both (works across the worker thread,
# unlike a contextvar). Prod HTTP never calls init_stdio_identity() → unaffected.
_STDIO_TOKEN_INFO: "TokenInfo | None" = None
_STDIO_RAW_JWT: "str | None" = None


def init_stdio_identity() -> "TokenInfo | None":
    """Seed a process-wide identity from the local hook.jwt for the stdio MCP
    server. Validates the token; on missing/expired/invalid it silently leaves
    the identity unset (callers then fall back to 'default' as before)."""
    global _STDIO_TOKEN_INFO, _STDIO_RAW_JWT
    path = os.environ.get("MAYRING_HOOK_JWT") or os.path.expanduser(
        "~/.config/mayring/hook.jwt"
    )
    try:
        with open(path, encoding="utf-8") as fh:
            raw = fh.read().strip()
    except OSError:
        return None
    if not raw:
        return None
    # Prefer full RS256 validation when a public key is configured (server-side).
    # The plugin has no JWT_PUBLIC_KEY_PATH, so fall back to decoding the claims
    # unverified. Safe: this token only tags the LOCAL workspace and is forwarded
    # verbatim on cloud pushes, where the server validates its real signature and
    # scopes by the verified claims — a tampered local token cannot reach another
    # tenant's cloud data.
    info = validate_jwt_token(raw) or _decode_identity_unverified(raw)
    if info is not None:
        _STDIO_TOKEN_INFO = info
        _STDIO_RAW_JWT = raw
    return info


def _decode_identity_unverified(token: str) -> "TokenInfo | None":
    """Build a TokenInfo from a JWT's claims WITHOUT signature verification.
    Returns None for an expired or workspace-less token (so we never seed a
    stale identity). Signature is intentionally not checked here — see
    init_stdio_identity for why that is safe."""
    import time

    try:
        import jwt  # PyJWT

        payload = jwt.decode(token, options={"verify_signature": False})
    except Exception:  # noqa: BLE001 — malformed token → no identity, never raise
        return None
    exp = payload.get("exp")
    try:
        if exp and time.time() >= float(exp):
            return None
    except (TypeError, ValueError):
        return None
    ws = str(payload.get("workspace_id") or "").strip()
    if not ws:
        return None
    raw_scopes = payload.get("scope", [])
    if isinstance(raw_scopes, str):
        scopes = tuple(s for s in raw_scopes.split() if s)
    elif isinstance(raw_scopes, list):
        scopes = tuple(str(s) for s in raw_scopes if s)
    else:
        scopes = ()
    return TokenInfo(
        workspace_id=ws,
        scopes=scopes,
        sub=str(payload["sub"]) if payload.get("sub") else None,
        iat=payload.get("iat"),
    )


def _current_token_info() -> "TokenInfo | None":
    return _TOKEN_CTX.get(None) or _STDIO_TOKEN_INFO


def _current_raw_jwt() -> "str | None":
    return _RAW_JWT_CTX.get(None) or _STDIO_RAW_JWT


def _effective_workspace_id(caller_default: str = "default") -> str:
    """Backward-compat shim — delegates to resolve_workspace_from_token().

    Vor V2 hatte dies eigenen TokenInfo-Read; jetzt ist
    workspace_resolver.resolve_workspace_from_token die einzige SoT.
    Wenn kein TokenInfo (Tests / manueller MCP-Call), fallback auf
    caller_default ('default').
    """
    info = _current_token_info()
    if info is None:
        # No valid local token → DON'T silently misfile into the shared 'default'
        # bucket (the per-device leak). Write to a clearly-marked, claimable
        # unclaimed:<device> bucket and warn loudly once (CLAUDE.md: no silent errors).
        global _DEVICE_WARNED
        dev = _local_device_id()
        if not _DEVICE_WARNED:
            logging.getLogger(__name__).warning(
                "Kein gültiges hook.jwt — Memory landet in 'unclaimed:%s'. "
                "oauth_install ausführen oder im Dashboard claimen.", dev)
            _DEVICE_WARNED = True
        return f"unclaimed:{dev}"
    from mayring_core.identity.workspace_resolver import resolve_workspace_from_token
    # conn → Alias-Auflösung (workspace-repoint): alte Tokens (019d6933 — z.B. der
    # claude.ai-Memory-Connector, der vor der Migration ausgestellt wurde) lösen
    # transparent auf die kanonische 019e14d6 auf. OHNE conn (wie bisher) blieb der
    # MCP-/Connector-Ingest auf 019d6933 hängen → Goal/Memory im verwaisten
    # Workspace. Ohne DB (Tests) bleibt der Resolver alias-los, bricht aber nicht.
    try:
        from src.api.dependencies import get_conn
        conn = get_conn()
    except Exception:  # noqa: BLE001 — ohne DB kein Alias, aber MCP-Auth darf nicht brechen
        conn = None
    return resolve_workspace_from_token(info, override_header=None, conn=conn)


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
    info = _current_token_info()
    if info is None:
        return requested
    target = requested if info.is_admin else info.workspace_id
    # Alias-Canonicalization auch auf dem MCP-tool-arg-Pfad (workspace-repoint):
    # ob admin-override oder gepinnter User-Workspace — 019d6933 → 019e14d6.
    if not target:
        return target
    from mayring_core.identity.workspace_resolver import _canonicalize_alias
    try:
        from src.api.dependencies import get_conn
        conn = get_conn()
    except Exception:  # noqa: BLE001 — ohne DB kein Alias, aber MCP-Auth darf nicht brechen
        conn = None
    return _canonicalize_alias(conn, target)


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


def _effective_active_workspace_name() -> str | None:
    """Display name of the active workspace (from JWT memberships), or None."""
    info = _TOKEN_CTX.get(None)
    return info.membership_name(info.active_workspace_id) if info is not None else None


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
