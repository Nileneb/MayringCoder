"""Tests for PiQueue (Issue #183 T2)."""
from __future__ import annotations

import asyncio

import pytest

from src.agents.pi_jobs import PiJob

# PiQueue uses asyncio primitives (asyncio.Queue, asyncio.Future, asyncio.Event)
# which are not compatible with trio — restrict to asyncio backend.
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def reset_singleton():
    """Each test gets a fresh queue (singleton would otherwise leak state)."""
    from src.agents import pi_queue
    pi_queue._SINGLETON = None
    yield
    pi_queue._SINGLETON = None


def _make_job(text: str = "ping", **kw) -> PiJob:
    return PiJob(job_id=f"j-{text[:5]}", task_text=text, **kw)


async def test_enqueue_returns_future_resolving_with_result():
    """Caller enqueues, awaits future, gets result. Worker calls handler."""
    from src.agents.pi_queue import PiQueue

    q = PiQueue(concurrency=2)
    async def _handler(job: PiJob) -> dict:
        return {"echo": job.task_text}
    q.set_handler(_handler)
    await q.start()
    try:
        fut = q.enqueue(_make_job("hello"))
        result = await asyncio.wait_for(fut, timeout=2.0)
        assert result == {"echo": "hello"}
    finally:
        await q.shutdown()


async def test_concurrency_cap_respected():
    """With concurrency=2 and 5 enqueued, max 2 are in-flight at any time."""
    from src.agents.pi_queue import PiQueue

    q = PiQueue(concurrency=2)
    in_flight = 0
    peak = 0
    async def _handler(job: PiJob) -> dict:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return {"id": job.job_id}
    q.set_handler(_handler)
    await q.start()
    try:
        futs = [q.enqueue(_make_job(f"j{i}")) for i in range(5)]
        await asyncio.gather(*futs)
    finally:
        await q.shutdown()
    assert peak == 2, f"expected peak in-flight = 2, got {peak}"


async def test_handler_exception_propagates_to_future():
    """Handler-error → future.set_exception, caller sees the error."""
    from src.agents.pi_queue import PiQueue

    q = PiQueue(concurrency=1)
    async def _handler(job: PiJob) -> dict:
        raise RuntimeError("boom")
    q.set_handler(_handler)
    await q.start()
    try:
        fut = q.enqueue(_make_job("x"))
        with pytest.raises(RuntimeError, match="boom"):
            await asyncio.wait_for(fut, timeout=2.0)
    finally:
        await q.shutdown()


async def test_shutdown_cancels_pending_jobs():
    """Pending (un-started) jobs see future cancelled on shutdown."""
    from src.agents.pi_queue import PiQueue

    q = PiQueue(concurrency=1)
    started = asyncio.Event()
    async def _handler(job: PiJob) -> dict:
        if job.task_text == "long":
            started.set()
            await asyncio.sleep(10.0)  # long-running, cancelled by shutdown
        return {"id": job.job_id}
    q.set_handler(_handler)
    await q.start()

    long_fut = q.enqueue(_make_job("long"))
    pending_fut = q.enqueue(_make_job("pending"))
    await started.wait()  # ensure long-job started
    await q.shutdown()  # should cancel both

    # pending future was never started → cancelled
    assert pending_fut.cancelled(), "pending future must be cancelled on shutdown"


async def test_singleton_get_pi_queue_returns_same_instance():
    """Module-level singleton — same FastAPI process gets same queue."""
    from src.agents.pi_queue import get_pi_queue
    q1 = get_pi_queue()
    q2 = get_pi_queue()
    assert q1 is q2


async def test_stats_after_jobs():
    """Stats ring-buffer tracks completed jobs per class."""
    from src.agents.pi_queue import PiQueue
    q = PiQueue(concurrency=1)

    async def _h(job):
        return {}

    q.set_handler(_h)
    await q.start()
    try:
        for i in range(3):
            await q.enqueue(_make_job(f"j{i}", job_class="mini"))
        await asyncio.sleep(0.2)
        s = q.stats()
        assert s["completed_total"] == 3
        assert "mini" in s["by_class"]
        assert s["by_class"]["mini"]["count"] == 3
        assert s["in_flight"] == 0
    finally:
        await q.shutdown()


async def test_stats_fallback_rate():
    """fallback_rate is 1.0 when model_used differs from model_requested."""
    from src.agents.pi_queue import PiQueue

    q = PiQueue(concurrency=1)

    async def _h(job):
        job.model_used = "other-model"
        return {}

    q.set_handler(_h)
    await q.start()
    try:
        job = _make_job("x", model="requested-model", job_class="standard")
        await q.enqueue(job)
        await asyncio.sleep(0.2)
        s = q.stats()
        assert s["by_class"]["standard"]["fallback_rate"] == 1.0
    finally:
        await q.shutdown()


async def test_stats_empty_queue():
    """stats() on a fresh queue returns zero counts."""
    from src.agents.pi_queue import PiQueue

    q = PiQueue(concurrency=1)

    async def _h(job):
        return {}

    q.set_handler(_h)
    await q.start()
    try:
        s = q.stats()
        assert s["in_flight"] == 0
        assert s["completed_total"] == 0
        assert s["by_class"] == {}
    finally:
        await q.shutdown()
