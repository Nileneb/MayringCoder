"""Lightweight HTTP server that exposes Pi-agent (run_task_with_memory) via REST.

Start:
    .venv/bin/python pi_server.py          # port 8091
    PI_PORT=8099 .venv/bin/python pi_server.py

POST /pi-task
    Body: {"task": "...", "repo_slug": "optional", "system_prompt": "optional"}
    Returns: {"content": "..."}

GET /health
    Returns: {"status": "ok", "chunks": N}
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.memory_store import init_memory_db
from src.pi_agent import run_task_with_memory

_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
_PORT = int(os.getenv("PI_PORT", "8091"))
_REPO_SLUG = os.getenv("PI_REPO_SLUG", "Nileneb/app.linn.games")


def _chunk_count() -> int:
    conn = init_memory_db()
    n = conn.execute("SELECT COUNT(*) FROM chunks WHERE is_active=1").fetchone()[0]
    conn.close()
    return n


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:  # suppress default access log
        pass

    def _send_json(self, code: int, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "model": _MODEL, "chunks": _chunk_count()})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/pi-task":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return

        task = payload.get("task", "").strip()
        if not task:
            self._send_json(400, {"error": "'task' is required"})
            return

        repo_slug = payload.get("repo_slug") or _REPO_SLUG
        system_prompt = payload.get("system_prompt") or None
        timeout = float(payload.get("timeout", 180))

        try:
            result = run_task_with_memory(
                task=task,
                ollama_url=_OLLAMA_URL,
                model=_MODEL,
                repo_slug=repo_slug,
                system_prompt=system_prompt,
                timeout=timeout,
            )
            self._send_json(200, {"content": result})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})


if __name__ == "__main__":
    print(f"[pi-server] model={_MODEL}  ollama={_OLLAMA_URL}  port={_PORT}  chunks={_chunk_count()}")
    server = HTTPServer(("0.0.0.0", _PORT), _Handler)
    print(f"[pi-server] listening on http://0.0.0.0:{_PORT}")
    server.serve_forever()
