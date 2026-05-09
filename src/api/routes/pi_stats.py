"""GET /pi-jobs/stats — read-only stats over the in-process PiQueue.

Issue #183 T5: local telemetry for the in-process Pi-Queue.
Admin-only restriction comes later when the /stats/admin pattern is
wanted for pi as well.
"""
from __future__ import annotations

from fastapi import APIRouter

from src.agents.pi_queue import get_pi_queue

router = APIRouter(tags=["pi"])


@router.get("/pi-jobs/stats")
async def pi_jobs_stats() -> dict:
    """Read-only counters + p50/p95-latency per job-class."""
    return get_pi_queue().stats()
