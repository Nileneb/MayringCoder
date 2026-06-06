"""Device-Registry + Hook-Events + Write-Job-Routing (#274).

Cloud-side counterpart to mayring-claude-plugin#5. The plugin / Pi-worker sends
a stable ``device_id`` via the ``X-Device-Id`` header on every call; the cloud
persists and serves device / hook / worker data, scoped by
``(workspace_id from JWT, device_id from header)``.

Three concerns, one router:
  * **Device registry** — register / heartbeat / list devices + capabilities.
  * **Hook events** — best-effort firing log for the Observability dashboard;
    NEVER 5xx back to the hook (it runs on every prompt/stop).
  * **Write-job routing** — the cloud-claim path (``/pi_task_claim_cloud`` +
    ``/pi_task_complete_cloud``, the previously-removed pull-model endpoints,
    see ``src/api/mcp.py``) reactivated. A claiming worker's capabilities are
    resolved from the *registry* (``devices.effective_capabilities``), not its
    self-report, so a ``capability_required='write'`` job only ever reaches a
    device registered with ``'write'``.

``device_id`` is orthogonal to the JWT (JWT = user/workspace, header = device) —
no new auth surface.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel

from mayring_core.memory import devices as device_store
from mayring_pi_agent import pi_jobs
from src.api.auth import get_workspace
from src.api.dependencies import get_conn as _get_conn

router = APIRouter(tags=["devices"])
logger = logging.getLogger(__name__)

# WHY(#343 db-lock-outage 2026-06-06): touch_last_seen lief im pi_task_claim-
# Hotpath bei JEDEM Poll → SQLite-Write-Contention → "database is locked"; weil
# sqlite synchron im async-Handler läuft, blockierte ein 10s-Lock-Wait den
# Event-Loop inkl. /health → mcp-Outage. Heartbeat ist unkritisch → In-Memory
# gedrosselt (max 1 Write/30s/Gerät) + best-effort (Lock killt den Request nicht).
_LAST_TOUCH: dict[str, float] = {}
_TOUCH_INTERVAL_S = 30.0


def _touch_best_effort(conn: Any, device_id: str, workspace_id: str) -> None:
    key = f"{workspace_id}:{device_id}"
    now = time.monotonic()
    if now - _LAST_TOUCH.get(key, 0.0) < _TOUCH_INTERVAL_S:
        return
    try:
        device_store.touch_last_seen(conn, device_id, workspace_id)
        _LAST_TOUCH[key] = now
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower():
            logger.warning("touch_last_seen skipped (db locked): %s", e)
            return
        raise


# WHY(#343): fail_stale_cloud_jobs ist ein TTL-Sweep (WRITE) der im Claim-Hotpath
# bei JEDEM Poll lief → zweite Write-Amplifikation neben touch_last_seen. Der Sweep
# ist unkritisch + idempotent → global gedrosselt (max 1×/60s) + best-effort (ein
# DB-Lock killt den Claim nicht). Der Claim-Handler selbst läuft jetzt im Threadpool
# (def statt async def), sodass ein 10s-busy_timeout-Wait nicht mehr den Event-Loop
# inkl. /health blockiert.
_LAST_STALE_SWEEP: list[float] = [0.0]
_STALE_SWEEP_INTERVAL_S = 60.0


def _fail_stale_best_effort() -> None:
    now = time.monotonic()
    if now - _LAST_STALE_SWEEP[0] < _STALE_SWEEP_INTERVAL_S:
        return
    try:
        pi_jobs.fail_stale_cloud_jobs(
            max_age_s=float(os.getenv("MAYRING_CLOUD_JOB_TTL_S", "1800")),
            db_path=_job_db_path(),
        )
        _LAST_STALE_SWEEP[0] = now
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower():
            logger.warning("fail_stale_cloud_jobs skipped (db locked): %s", e)
            return
        raise


def _job_db_path() -> Path:
    """DB file the pi_jobs helpers should hit — mirror dependencies.get_conn so
    the per-call job connection and the shared device connection point at the
    SAME file (MAYRING_LOCAL_DB override in tests, else MEMORY_DB_PATH)."""
    from mayring_core.memory.store import MEMORY_DB_PATH
    local = os.environ.get("MAYRING_LOCAL_DB", "")
    return Path(local) if local else MEMORY_DB_PATH


def _resolve_device_id(header_id: str | None, body_id: str | None) -> str:
    """X-Device-Id header wins; fall back to a body-supplied id."""
    return (header_id or body_id or "").strip()


# --- request models ---------------------------------------------------------

class DeviceRegisterRequest(BaseModel):
    device_id: str | None = None
    name: str = ""
    os: str = ""
    capabilities: list[str] | str = []


class HeartbeatRequest(BaseModel):
    device_id: str | None = None


class HookEventRequest(BaseModel):
    device_id: str | None = None
    hook_type: str = ""
    fired_at: str = ""
    summary: str = ""


class CloudClaimRequest(BaseModel):
    worker_id: str | None = None
    capabilities: list[str] = []


class CloudCompleteRequest(BaseModel):
    job_id: str
    result: Any | None = None
    error: str | None = None


# --- device registry --------------------------------------------------------

@router.post("/devices/register")
def register_device(
    req: DeviceRegisterRequest,
    workspace_id: str = Depends(get_workspace),
    x_device_id: str | None = Header(default=None, alias="X-Device-Id"),
) -> dict:
    device_id = _resolve_device_id(x_device_id, req.device_id)
    if not device_id:
        raise HTTPException(status_code=400, detail="device_id required (X-Device-Id header or body)")
    conn = _get_conn()
    device_store.upsert_device(
        conn,
        device_id=device_id,
        workspace_id=workspace_id,
        name=req.name,
        os=req.os,
        capabilities=req.capabilities,
    )
    # Echo back the persisted, normalised capabilities (single source of truth)
    # rather than re-deriving from the request body.
    caps = device_store.device_capabilities(conn, device_id, workspace_id) or []
    return {"registered": True, "device_id": device_id, "capabilities": caps}


@router.post("/devices/heartbeat")
def device_heartbeat(
    req: HeartbeatRequest,
    workspace_id: str = Depends(get_workspace),
    x_device_id: str | None = Header(default=None, alias="X-Device-Id"),
) -> dict:
    device_id = _resolve_device_id(x_device_id, req.device_id)
    if not device_id:
        raise HTTPException(status_code=400, detail="device_id required (X-Device-Id header or body)")
    _touch_best_effort(_get_conn(), device_id, workspace_id)
    return {"ok": True, "device_id": device_id}


@router.get("/devices")
def list_devices(workspace_id: str = Depends(get_workspace)) -> dict:
    items = device_store.list_devices(_get_conn(), workspace_id)
    return {"devices": items, "count": len(items)}


# --- hook events ------------------------------------------------------------

@router.post("/hooks/events")
def record_hook_event(
    req: HookEventRequest,
    workspace_id: str = Depends(get_workspace),
    x_device_id: str | None = Header(default=None, alias="X-Device-Id"),
) -> dict:
    """Best-effort insert — NEVER 5xx back to the hook (it fires on every
    prompt/stop). A DB error is logged loudly (not swallowed) and surfaced as
    recorded:false with 200, matching the sync.py hook-facing convention."""
    device_id = _resolve_device_id(x_device_id, req.device_id)
    try:
        rid = device_store.record_hook_event(
            _get_conn(),
            workspace_id=workspace_id,
            device_id=device_id,
            hook_type=req.hook_type,
            fired_at=req.fired_at,
            summary=req.summary,
        )
        return {"recorded": True, "id": rid}
    except Exception:  # noqa: BLE001 — hook must never see a 5xx
        logger.exception("hooks/events: insert failed")
        return {"recorded": False, "error": "failed to record hook event"}


@router.get("/hooks/events")
def list_hook_events(
    since: str | None = Query(default=None),
    limit: int = Query(default=200, le=2000),
    workspace_id: str = Depends(get_workspace),
) -> dict:
    items = device_store.list_hook_events(_get_conn(), workspace_id, since=since, limit=limit)
    return {"events": items, "count": len(items)}


# --- write-job routing (cloud-claim path, reactivated #274) -----------------

@router.post("/pi_task_claim_cloud")
def pi_task_claim_cloud(
    req: CloudClaimRequest,
    workspace_id: str = Depends(get_workspace),
    x_device_id: str | None = Header(default=None, alias="X-Device-Id"),
) -> dict:
    """Atomically claim the oldest queued cloud job this device is allowed to
    run. Capabilities come from the REGISTRY (authoritative), not the body —
    so write-jobs only route to registry-confirmed write devices."""
    device_id = _resolve_device_id(x_device_id, req.worker_id)
    if not device_id:
        raise HTTPException(status_code=400, detail="device_id/worker_id required")
    conn = _get_conn()
    caps = device_store.effective_capabilities(
        conn, device_id, workspace_id, req.capabilities,
    )
    _touch_best_effort(conn, device_id, workspace_id)
    # Cheap piggy-backed TTL sweep: fail cloud jobs no worker claimed in time so
    # an A2A client does not poll forever. No cron needed. Throttled + best-effort
    # (#343) so it can't write-amplify the claim hotpath.
    _fail_stale_best_effort()
    job = pi_jobs.claim_cloud_next(
        device_id,
        capabilities=caps,
        workspace_id=workspace_id,
        db_path=_job_db_path(),
    )
    if job is None:
        return {"job": None}
    return {"job": job.to_dict(), "effective_capabilities": caps}


@router.post("/pi_task_complete_cloud")
def pi_task_complete_cloud(
    req: CloudCompleteRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Report a cloud job's outcome. Workspace-scoped: a worker can only finish
    a job that belongs to its own tenant (foreign job_id → 404)."""
    db_path = _job_db_path()
    existing = pi_jobs.get_job(req.job_id, workspace_id=workspace_id, db_path=db_path)
    if existing is None:
        raise HTTPException(status_code=404, detail="job not found in workspace")
    if req.error is not None:
        pi_jobs.fail_job(req.job_id, req.error, db_path=db_path)
        return {"ok": True, "status": "failed", "job_id": req.job_id}
    pi_jobs.complete_job(req.job_id, req.result, db_path=db_path)
    return {"ok": True, "status": "completed", "job_id": req.job_id}
