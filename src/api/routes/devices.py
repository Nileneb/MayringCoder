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
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel

from mayring_core.memory import devices as device_store
from src.agents import pi_jobs
from src.api.auth import get_workspace
from src.api.dependencies import get_conn as _get_conn

router = APIRouter(tags=["devices"])
logger = logging.getLogger(__name__)


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
async def register_device(
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
    return {"registered": True, "device_id": device_id,
            "capabilities": device_store._split_caps(device_store._join_caps(req.capabilities))}


@router.post("/devices/heartbeat")
async def device_heartbeat(
    req: HeartbeatRequest,
    workspace_id: str = Depends(get_workspace),
    x_device_id: str | None = Header(default=None, alias="X-Device-Id"),
) -> dict:
    device_id = _resolve_device_id(x_device_id, req.device_id)
    if not device_id:
        raise HTTPException(status_code=400, detail="device_id required (X-Device-Id header or body)")
    device_store.touch_last_seen(_get_conn(), device_id, workspace_id)
    return {"ok": True, "device_id": device_id}


@router.get("/devices")
async def list_devices(workspace_id: str = Depends(get_workspace)) -> dict:
    items = device_store.list_devices(_get_conn(), workspace_id)
    return {"devices": items, "count": len(items)}


# --- hook events ------------------------------------------------------------

@router.post("/hooks/events")
async def record_hook_event(
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
async def list_hook_events(
    since: str | None = Query(default=None),
    limit: int = Query(default=200, le=2000),
    workspace_id: str = Depends(get_workspace),
) -> dict:
    items = device_store.list_hook_events(_get_conn(), workspace_id, since=since, limit=limit)
    return {"events": items, "count": len(items)}


# --- write-job routing (cloud-claim path, reactivated #274) -----------------

@router.post("/pi_task_claim_cloud")
async def pi_task_claim_cloud(
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
    device_store.touch_last_seen(conn, device_id, workspace_id)
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
async def pi_task_complete_cloud(
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
