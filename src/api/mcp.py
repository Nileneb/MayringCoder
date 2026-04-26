"""MCP Memory Server for Claude Code.

Provides persistent, local memory via MCP tools over stdio/HTTP transport.

Add to Claude Code MCP settings (.claude/settings.json or user settings):

Option 1: Direct Python venv (development):
{
    "mcpServers": {
        "memory": {
            "command": "/path/to/.venv/bin/python",
            "args": ["-m", "src.api.mcp"],
            "cwd": "/path/to/MayringCoder"
        }
    }
}

Option 2: Docker (production):
{
    "mcpServers": {
        "memory": {
            "command": "docker",
            "args": ["run", "-i", "--rm",
                     "-v", "cache:/app/cache",
                     "--env-file", "/path/to/.env",
                     "mayrингcoder-mcp"]
        }
    }
}

Build Docker image: docker build -t mayrингcoder-mcp .
Or with compose:  docker-compose up -d mcp-memory
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_ROOT / ".env")

from src.api.mcp_auth import JWTAuthMiddleware, _AUTH_ENABLED, _OAUTH_BASE_URL
from src.api.mcp_memory_tools import register_memory_tools
from src.api.mcp_oauth import PathNormMiddleware, build_starlette_routes

# Backward-compat aliases for tests that import with underscore prefix
_JWTAuthMiddleware = JWTAuthMiddleware
_PathNormMiddleware = PathNormMiddleware

_TRANSPORT  = os.getenv("MCP_TRANSPORT", "stdio")
_HTTP_PORT  = int(os.getenv("MCP_HTTP_PORT", "8000"))
_HTTP_HOST  = os.getenv("MCP_HTTP_HOST", "0.0.0.0")

mcp = FastMCP(
    "memory",
    host=_HTTP_HOST,
    port=_HTTP_PORT,
)

register_memory_tools(mcp)
# Agent tools (pi_task, duel, etc.) live in local_mcp.py — see Issue #107


def main() -> None:
    if _TRANSPORT in ("http", "sse"):
        import uvicorn
        from starlette.applications import Starlette
        from starlette.routing import Mount

        _mcp_http_app = mcp.streamable_http_app()
        _inner = JWTAuthMiddleware(PathNormMiddleware(_mcp_http_app))

        _auth_label = "rs256-jwt" if _AUTH_ENABLED else "disabled"
        print(
            f"[mcp-memory] HTTP/streamable on {_HTTP_HOST}:{_HTTP_PORT}"
            f" | auth={_auth_label}"
            f" | oauth={_OAUTH_BASE_URL}"
        )

        routes = build_starlette_routes()
        routes.append(Mount("/", app=_inner))

        app = Starlette(
            routes=routes,
            lifespan=_mcp_http_app.router.lifespan_context,
        )
        uvicorn.run(app, host=_HTTP_HOST, port=_HTTP_PORT)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
