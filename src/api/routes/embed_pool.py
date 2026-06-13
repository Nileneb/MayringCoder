"""Distributed embedding pool routes (#365 Schicht 3).

Eligibility mirrors devices.effective_capabilities: a claiming device must be
REGISTERED with the 'embed' capability (no self-grant), heartbeat-fresh, and not
quarantined. On a divergence both devices are quarantined + flagged; the caller
that enqueued polls GET /embed_pool/{embed_id} for the verified vector."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from mayring_core import config as cfg
from mayring_core.memory import devices as device_store
from mayring_core.memory import embed_pool as ep
from src.api.auth import get_workspace
from src.api.dependencies import get_conn as _get_conn

router = APIRouter(tags=["embed-pool"])
logger = logging.getLogger(__name__)

_GOLDEN_FIXTURE = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "golden_embed.json"


def _golden_sample() -> tuple[str, list[float]]:
    data = json.loads(_GOLDEN_FIXTURE.read_text())
    return data["text"], data["reference"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class EnqueueRequest(BaseModel):
    projekt_id: str = ""
    text: str
    chunk_ref: str = ""
    model: str = "bge-m3"


class ClaimRequest(BaseModel):
    capabilities: list[str] = []  # advisory only — registry is authoritative


class CompleteRequest(BaseModel):
    embed_id: str
    vector: list[float]


def _device_id(header_id: str | None) -> str:
    did = (header_id or "").strip()
    if not did:
        raise HTTPException(status_code=400, detail="X-Device-Id required")
    return did


@router.post("/embed_pool/enqueue")
def enqueue(req: EnqueueRequest, workspace_id: str = Depends(get_workspace)) -> dict:
    eid = ep.enqueue(_get_conn(), workspace_id=workspace_id, projekt_id=req.projekt_id,
                     text=req.text, chunk_ref=req.chunk_ref, model=req.model)
    return {"embed_id": eid}


@router.post("/embed_pool/claim")
def claim(req: ClaimRequest, workspace_id: str = Depends(get_workspace),
          x_device_id: str | None = Header(default=None, alias="X-Device-Id")) -> dict:
    device_id = _device_id(x_device_id)
    conn = _get_conn()
    if not device_store.is_eligible_embed(
        conn, device_id, workspace_id, now=_now_iso(),
        fresh_seconds=cfg.EMBED_HEARTBEAT_FRESH_SECONDS,
    ):
        return {"job": None, "reason": "not eligible (register with 'embed', heartbeat, not quarantined)"}
    job = ep.claim_replica(conn, device_id=device_id, workspace_id=workspace_id)
    return {"job": job}


@router.post("/embed_pool/complete")
def complete(req: CompleteRequest, workspace_id: str = Depends(get_workspace),
             x_device_id: str | None = Header(default=None, alias="X-Device-Id")) -> dict:
    device_id = _device_id(x_device_id)
    conn = _get_conn()
    out = ep.submit_result(conn, embed_id=req.embed_id, device_id=device_id,
                           vector=req.vector, threshold=cfg.EMBED_VERIFY_THRESHOLD)
    if out.get("verdict") == "agreement":
        for d in out["devices"]:
            device_store.record_embed_verified(conn, d, workspace_id)
    elif out.get("verdict") == "divergence":
        until = (datetime.now(timezone.utc)
                 + timedelta(seconds=cfg.EMBED_QUARANTINE_SECONDS)
                 ).strftime("%Y-%m-%dT%H:%M:%SZ")
        for d in out["devices"]:
            device_store.record_embed_divergence(conn, d, workspace_id)
            device_store.set_quarantine(conn, d, workspace_id, until=until)
        text, ref = _golden_sample()
        for d in out["devices"]:
            ep.enqueue_golden(conn, workspace_id=workspace_id, text=text, reference=ref)
    return out


@router.get("/embed_pool/{embed_id}")
def status(embed_id: str, workspace_id: str = Depends(get_workspace)) -> dict:
    job = ep.get(_get_conn(), embed_id)
    if job is None or job["workspace_id"] != workspace_id:
        raise HTTPException(status_code=404, detail="embed job not found in workspace")
    out = {"embed_id": embed_id, "status": job["status"], "verdict": job["verdict"],
           "cosine": job["cosine"], "chunk_ref": job["chunk_ref"]}
    if job["status"] == "verified" and not job["is_golden"] and job["result_a"]:
        out["agreed_vector"] = json.loads(job["result_a"])
    return out


class GoldenClaimRequest(BaseModel):
    capabilities: list[str] = []


class GoldenCompleteRequest(BaseModel):
    embed_id: str
    vector: list[float]


@router.post("/embed_pool/golden/claim")
def golden_claim(req: GoldenClaimRequest, workspace_id: str = Depends(get_workspace),
                 x_device_id: str | None = Header(default=None, alias="X-Device-Id")) -> dict:
    device_id = _device_id(x_device_id)
    job = ep.claim_golden(_get_conn(), device_id=device_id, workspace_id=workspace_id)
    return {"job": job}


@router.post("/embed_pool/golden/complete")
def golden_complete(req: GoldenCompleteRequest, workspace_id: str = Depends(get_workspace),
                    x_device_id: str | None = Header(default=None, alias="X-Device-Id")) -> dict:
    device_id = _device_id(x_device_id)
    conn = _get_conn()
    out = ep.submit_golden(conn, embed_id=req.embed_id, device_id=device_id,
                           vector=req.vector, threshold=cfg.EMBED_VERIFY_THRESHOLD)
    if out["passed"]:
        device_store.set_quarantine(conn, device_id, workspace_id, until="")  # rehabilitate
    return out
