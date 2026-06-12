"""AST-Gate gegen Event-Loop-Blocker in API-Route-Handlern.

WHY(#363, prod-outage-klasse): FastAPI führt ``async def``-Handler direkt auf
dem Event-Loop aus. Ein Handler ohne ``await``, der sync-Arbeit macht
(Ollama-HTTP, SQLite, Chroma), blockiert damit den EINZIGEN Uvicorn-Worker
für die gesamte Dauer — alle parallelen Requests laufen in Timeouts (http=0).
Genau das hat smoke #363 produziert: EIN langsamer micro-batch-Summarize
unter Last → micro_batch_indexes UND project_link_boost_roundtrip
gleichzeitig rot. #343 hat dieselbe Klasse in devices.py gefixt; dieser Test
verhindert die Wiederkehr in ALLEN Route-Files: sync-Arbeit gehört in
``def``-Handler (FastAPI-Threadpool).

Ein async-Handler ist nur dann legitim, wenn er den Loop wirklich braucht:
``await`` / ``async for`` / ``async with`` / ``asyncio.create_task`` /
nested ``async def`` (Background-Coroutine).
"""

from __future__ import annotations

import ast
from pathlib import Path

ROUTES_DIR = Path(__file__).resolve().parent.parent / "src" / "api" / "routes"

_LOOP_CALLS = {
    "create_task", "ensure_future", "get_event_loop", "run_in_executor",
    # Helpers, die intern asyncio.create_task aufrufen — Handler, die sie
    # rufen, brauchen den Loop transitiv (CI-Fund nach der Konvertierung:
    # trigger_populate/set_watch_repo/repo_events → "no running event loop").
    "enqueue_populate", "_handle_event",
}


def _needs_event_loop(fn: ast.AsyncFunctionDef) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, (ast.Await, ast.AsyncFor, ast.AsyncWith)):
            return True
        if isinstance(node, ast.AsyncFunctionDef) and node is not fn:
            return True
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name in _LOOP_CALLS:
                return True
    return False


def test_no_async_route_handler_without_await():
    violations: list[str] = []
    for path in sorted(ROUTES_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.AsyncFunctionDef) and not _needs_event_loop(node):
                violations.append(f"{path.name}:{node.lineno} {node.name}")
    assert not violations, (
        "async-def-Handler ohne await blockieren den Event-Loop (sync-Arbeit "
        "läuft auf dem Loop statt im Threadpool). 'async def' → 'def' machen:\n  "
        + "\n  ".join(violations)
    )
