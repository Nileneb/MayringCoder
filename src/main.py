"""All-in-one entry point: starts API, MCP, Pi-Agent and Web UI in one process.

Usage:
    python -m src.main

Ports:
    7860 — Gradio Web UI
    8080 — FastAPI API server
    8090 — MCP server (HTTP/streamable-http)
    8091 — Pi-Agent HTTP server
"""
from __future__ import annotations

import os
import threading

import uvicorn


def _thread(target: str, host: str, port: int, **env: str) -> threading.Thread:
    for k, v in env.items():
        os.environ.setdefault(k, v)

    def _run() -> None:
        import importlib
        mod_path, attr = target.rsplit(":", 1)
        mod = importlib.import_module(mod_path)
        asgi_app = getattr(mod, attr)
        config = uvicorn.Config(asgi_app, host=host, port=port, log_level="warning")
        uvicorn.Server(config).run()

    return threading.Thread(target=_run, daemon=True, name=target)


def main() -> None:
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    api_url = os.getenv("API_URL", "http://localhost:8080")

    os.environ.setdefault("MCP_TRANSPORT", "http")
    os.environ.setdefault("MCP_HTTP_HOST", "0.0.0.0")
    os.environ.setdefault("MCP_HTTP_PORT", "8090")

    # Build MCP ASGI app before threading to avoid FastMCP init races
    from src.api.mcp import mcp, _JWTAuthMiddleware  # noqa: PLC0415
    _mcp_asgi = _JWTAuthMiddleware(mcp.streamable_http_app())

    def _run_mcp() -> None:
        config = uvicorn.Config(_mcp_asgi, host="0.0.0.0", port=8090, log_level="warning")
        uvicorn.Server(config).run()

    threads = [
        _thread("src.api.server:app", "0.0.0.0", 8080),
        threading.Thread(target=_run_mcp, daemon=True, name="mcp"),
        _thread("src.agents.pi_server:app", "0.0.0.0", 8091),
    ]
    for t in threads:
        t.start()

    # Web UI blocks main thread
    from src.api.web_ui import build_app  # noqa: PLC0415
    gradio_app = build_app(ollama_url=ollama_url, api_url=api_url)
    gradio_app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("UI_PORT", "7860")),
        prevent_thread_lock=False,
        theme=__import__("gradio").themes.Soft(),
    )


if __name__ == "__main__":
    main()
