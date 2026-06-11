"""Backward-compat regression: /pi-task response-schema unchanged after T3.

All tests are async (anyio/asyncio) so the event loop is consistent across
queue creation, handler setup, and endpoint invocation. Using _maybe_run()
in a non-async test would create multiple isolated event loops and cause the
worker tasks to be stranded on the wrong loop.
"""
from __future__ import annotations

import asyncio


def _maybe_run(c):
    return asyncio.run(c) if asyncio.iscoroutine(c) else c

import pytest

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def reset_singleton():
    from mayring_pi_agent import pi_queue
    pi_queue._SINGLETON = None
    yield
    pi_queue._SINGLETON = None


async def test_pi_task_response_schema_unchanged():
    """Backward-compat: /pi-task returns {workspace_id, content}."""
    from src.api.routes.memory import pi_task
    from src.api.routes.models import PiTaskRequest
    from mayring_pi_agent import pi_queue

    q = pi_queue.get_pi_queue()

    async def _fake(job):
        return {"echo": job.task_text}

    q.set_handler(_fake)
    await q.start()
    try:
        request = PiTaskRequest(task="hello", system_prompt="", timeout=10.0)
        result = await pi_task(request, workspace_id="bene")
        assert result["workspace_id"] == "bene"
        assert "content" in result
    finally:
        await q.shutdown()


async def test_pi_task_propagates_exception():
    """Exceptions from the handler surface as HTTPException 500."""
    from fastapi import HTTPException
    from src.api.routes.memory import pi_task
    from src.api.routes.models import PiTaskRequest
    from mayring_pi_agent import pi_queue

    q = pi_queue.get_pi_queue()

    async def _fail(job):
        raise RuntimeError("deliberate failure")

    q.set_handler(_fail)
    await q.start()
    try:
        request = PiTaskRequest(task="boom", timeout=10.0)
        with pytest.raises(HTTPException) as exc_info:
            await pi_task(request, workspace_id="bene")
        assert exc_info.value.status_code == 500
    finally:
        await q.shutdown()


async def test_pi_task_uses_repo_slug_from_request():
    """repo_slug from request is forwarded through the job."""
    from src.api.routes.memory import pi_task
    from src.api.routes.models import PiTaskRequest
    from mayring_pi_agent import pi_queue
    from mayring_pi_agent.pi_jobs import PiJob

    captured: list[PiJob] = []
    q = pi_queue.get_pi_queue()

    async def _capture(job: PiJob):
        captured.append(job)
        return "ok"

    q.set_handler(_capture)
    await q.start()
    try:
        request = PiTaskRequest(task="check slug", repo_slug="my-repo", timeout=10.0)
        await pi_task(request, workspace_id="ws1")
        assert len(captured) == 1
        assert captured[0].repo_slug == "my-repo"
    finally:
        await q.shutdown()


async def test_pi_task_job_class_classified():
    """Job class is classified based on task length."""
    from src.api.routes.memory import pi_task
    from src.api.routes.models import PiTaskRequest
    from mayring_pi_agent import pi_queue
    from mayring_pi_agent.pi_jobs import PiJob

    captured: list[PiJob] = []
    q = pi_queue.get_pi_queue()

    async def _capture(job: PiJob):
        captured.append(job)
        return "ok"

    q.set_handler(_capture)
    await q.start()
    try:
        request = PiTaskRequest(task="hi", timeout=10.0)
        await pi_task(request, workspace_id="ws1")
        assert len(captured) == 1
        assert captured[0].job_class == "mini"
    finally:
        await q.shutdown()
