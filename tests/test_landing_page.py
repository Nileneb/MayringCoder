"""Landing page routing: GET / → HTML, POST / → MCP forward, /ui link present."""
from __future__ import annotations

from pathlib import Path

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from src.api.mcp import _landing_page


def _app() -> TestClient:
    app = Starlette(routes=[Route("/", _landing_page, methods=["GET"])])
    return TestClient(app)


def test_landing_returns_html():
    client = _app()
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")


def test_landing_mentions_mayringcoder():
    r = _app().get("/")
    assert "MayringCoder" in r.text


def test_landing_links_to_register():
    r = _app().get("/")
    assert "https://www.linn.games/register" in r.text
    assert "einstellungen/mayring-abo" in r.text


def test_landing_links_to_dashboard():
    r = _app().get("/")
    assert 'href="/ui/"' in r.text


def test_landing_template_file_exists():
    tpl = Path("src/api/templates/landing.html")
    assert tpl.exists() and tpl.stat().st_size > 500


def test_path_norm_middleware_skips_get():
    from src.api.mcp import _PathNormMiddleware
    from unittest.mock import AsyncMock
    import anyio

    mw = _PathNormMiddleware(AsyncMock())
    # GET /  → must NOT be rewritten
    scope_get = {"type": "http", "method": "GET", "path": "/", "headers": []}

    async def _run():
        downstream = AsyncMock()
        mw._app = downstream
        await mw(scope_get, AsyncMock(), AsyncMock())
        # downstream received the scope unchanged
        assert downstream.await_args.args[0]["path"] == "/"

    anyio.run(_run)


def test_path_norm_middleware_rewrites_post():
    from src.api.mcp import _PathNormMiddleware
    from unittest.mock import AsyncMock
    import anyio

    mw = _PathNormMiddleware(AsyncMock())
    scope_post = {"type": "http", "method": "POST", "path": "/", "headers": []}

    async def _run():
        downstream = AsyncMock()
        mw._app = downstream
        await mw(scope_post, AsyncMock(), AsyncMock())
        assert downstream.await_args.args[0]["path"] == "/mcp"

    anyio.run(_run)
