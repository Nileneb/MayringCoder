"""Admin endpoints for runtime text-model selection.

GET  /stats/admin/text-models  — list available Ollama models + active override
POST /stats/admin/text-model   — write CACHE_DIR/text_model.txt (admin only)

Mounted under /stats/ so the production nginx allowlist already covers it.
POST is admin-only; GET is read-only and also requires auth (admin gate mirrors
igio_admin.py so the pattern is consistent across admin routes).
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from mayring_core.config import _resolve_cache_dir, BASE_DIR
from mayring_core.model_router import ModelRouter
from src.api.auth import get_token_info
from src.api.jwt_auth import TokenInfo

router = APIRouter()
_log = logging.getLogger(__name__)

_DEFAULT_CLOUD_MODELS = "mistral:7b-instruct"


def _is_admin(info: TokenInfo) -> bool:
    return "*" in info.scopes or "admin" in info.scopes


def _ollama_tags() -> list[str]:
    """Fetch model names from Ollama /api/tags. Returns [] on error."""
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    try:
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", []) if m.get("name")]
    except Exception as exc:
        _log.warning("ollama /api/tags unavailable: %s", exc)
        return []


def _cloud_allowlist() -> set[str]:
    raw = os.getenv("OLLAMA_CLOUD_MODELS", _DEFAULT_CLOUD_MODELS)
    return {m.strip() for m in raw.split(",") if m.strip()}


def _active_text_model() -> str:
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    try:
        return ModelRouter(ollama_url).resolve("text") or ""
    except Exception:
        return ""


class TextModelReq(BaseModel):
    model: str


@router.get("/stats/admin/text-models")
def list_text_models(
    info: TokenInfo = Depends(get_token_info),
) -> dict[str, Any]:
    """List models available in Ollama and the currently active text model."""
    names = _ollama_tags()
    cloud = _cloud_allowlist()
    active = _active_text_model()
    return {
        "active": active,
        "models": [
            {
                "name": n,
                "is_active": n == active,
                "cloud_available": n in cloud,
            }
            for n in names
        ],
    }


@router.post("/stats/admin/text-model")
def set_text_model(
    req: TextModelReq,
    info: TokenInfo = Depends(get_token_info),
) -> dict[str, str]:
    """Write a runtime override for the text model.

    Validates that the model is available in Ollama (HTTP 422 otherwise).
    Only admin tokens (scope '*' or 'admin') may call this endpoint.
    """
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")

    available = _ollama_tags()
    if req.model not in available:
        raise HTTPException(
            status_code=422,
            detail=f"Model '{req.model}' not in Ollama /api/tags. Available: {available}",
        )

    cache_dir = _resolve_cache_dir(BASE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "text_model.txt").write_text(req.model, encoding="utf-8")
    _log.info("text model override set to %s", req.model)

    return {"active": req.model}
