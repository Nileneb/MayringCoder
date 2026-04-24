"""OAuth 2.0 / PKCE endpoints and path-normalizer middleware for MCP HTTP transport."""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import time
from pathlib import Path
from typing import Any

from src.api.mcp_auth import _OAUTH_BASE_URL

_SERVICE_TOKEN = os.getenv("MCP_SERVICE_TOKEN", "")

_auth_codes: dict[str, dict[str, Any]] = {}

_LANDING_HTML_PATH = Path(__file__).parent / "templates" / "landing.html"


def _pkce_verify(verifier: str, challenge: str, method: str) -> bool:
    if method == "S256":
        digest = hashlib.sha256(verifier.encode()).digest()
        expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        return secrets.compare_digest(expected, challenge)
    return secrets.compare_digest(verifier, challenge)


async def _oauth_metadata(request: Any) -> Any:
    from starlette.responses import JSONResponse
    base = _OAUTH_BASE_URL
    return JSONResponse({
        "issuer": base,
        "authorization_endpoint": f"{base}/authorize",
        "token_endpoint": f"{base}/token",
        "registration_endpoint": f"{base}/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
    })


async def _oauth_authorize(request: Any) -> Any:
    from starlette.requests import Request
    from starlette.responses import RedirectResponse
    req: Request = request
    params = dict(req.query_params)
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    return RedirectResponse(
        f"https://app.linn.games/mcp/authorize?{qs}",
        status_code=302,
    )


async def _oauth_register_code(request: Any) -> Any:
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    req: Request = request

    auth_header = req.headers.get("authorization", "")
    token = auth_header[7:].strip() if auth_header.lower().startswith("bearer ") else ""
    if not _SERVICE_TOKEN or not secrets.compare_digest(token, _SERVICE_TOKEN):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    code = str(body.get("code", "")).strip()
    if not code:
        return JSONResponse({"error": "code required"}, status_code=400)

    _auth_codes[code] = {
        "token":                 str(body.get("token", "")),
        "workspace_id":          str(body.get("workspace_id", "default")),
        "code_challenge":        str(body.get("code_challenge", "")),
        "code_challenge_method": str(body.get("code_challenge_method", "S256")),
        "redirect_uri":          str(body.get("redirect_uri", "")),
        "state":                 str(body.get("state", "")),
        "expires_at":            time.time() + 300,
    }
    return JSONResponse({"ok": True})


async def _oauth_token(request: Any) -> Any:
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    req: Request = request

    ct = req.headers.get("content-type", "")
    if "application/json" in ct:
        body = await req.json()
    else:
        form = await req.form()
        body = dict(form)

    grant_type = body.get("grant_type", "")
    if grant_type != "authorization_code":
        return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

    code = str(body.get("code", ""))
    code_verifier = str(body.get("code_verifier", ""))
    redirect_uri = str(body.get("redirect_uri", ""))

    entry = _auth_codes.pop(code, None)
    if not entry:
        return JSONResponse({"error": "invalid_grant", "error_description": "Unknown code"}, 400)
    if time.time() > entry["expires_at"]:
        return JSONResponse({"error": "invalid_grant", "error_description": "Code expired"}, 400)
    if redirect_uri and redirect_uri != entry["redirect_uri"]:
        return JSONResponse({"error": "invalid_grant", "error_description": "redirect_uri mismatch"}, 400)
    if code_verifier and not _pkce_verify(code_verifier, entry["code_challenge"], entry["code_challenge_method"]):
        return JSONResponse({"error": "invalid_grant", "error_description": "PKCE verification failed"}, 400)

    return JSONResponse({
        "access_token": entry["token"],
        "token_type": "bearer",
        "workspace_id": entry["workspace_id"],
    })


async def _oauth_register(request: Any) -> Any:
    from starlette.responses import JSONResponse
    client_id = secrets.token_urlsafe(16)
    try:
        body = await request.json()
    except Exception:
        body = {}
    return JSONResponse({"client_id": client_id, "client_secret": None, **body}, status_code=201)


async def _landing_page(request: Any) -> Any:
    from starlette.responses import HTMLResponse
    try:
        html = _LANDING_HTML_PATH.read_text(encoding="utf-8")
    except OSError:
        return HTMLResponse("<h1>MayringCoder</h1>", status_code=200)
    return HTMLResponse(
        html,
        status_code=200,
        headers={"Cache-Control": "public, max-age=300"},
    )


class PathNormMiddleware:
    """Rewrite / and /sse → /mcp so the streamable_http_app Route('/mcp') matches."""

    _REWRITE = frozenset(("/", "/sse", ""))

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if (
            scope.get("type") == "http"
            and scope.get("path", "/") in self._REWRITE
            and scope.get("method", "") != "GET"
        ):
            scope = {**scope, "path": "/mcp", "raw_path": b"/mcp"}
        await self._app(scope, receive, send)


def build_starlette_routes() -> list:
    from starlette.routing import Route
    return [
        Route("/.well-known/oauth-authorization-server", _oauth_metadata),
        Route("/.well-known/oauth-protected-resource", _oauth_metadata),
        Route("/.well-known/oauth-protected-resource/sse", _oauth_metadata),
        Route("/register", _oauth_register, methods=["POST"]),
        Route("/authorize", _oauth_authorize, methods=["GET"]),
        Route("/authorize/register-code", _oauth_register_code, methods=["POST"]),
        Route("/token", _oauth_token, methods=["POST"]),
        Route("/", _landing_page, methods=["GET"]),
    ]
