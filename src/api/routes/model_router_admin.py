"""Admin endpoints to inspect and update ModelRouter routes at runtime.

Closes Issue #140: ``ModelRouter`` reads from ``config/model_routes.yaml`` on
every instantiation, but the YAML lived next to the code with no HTTP path
to mutate it. Updating a route required ``docker exec`` + restart.

These endpoints write the YAML atomically via ``os.replace``. Because every
caller instantiates a fresh ``ModelRouter()`` (see git grep), the change is
picked up without any restart on the very next call site.

Mounted under ``/stats/`` so the production nginx whitelist already covers it.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Literal

import yaml
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.auth import get_token_info
from src.api.jwt_auth import TokenInfo
from mayring_core.model_router import ModelRouter

router = APIRouter()
_log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent.parent
_CONFIG_PATH = _ROOT / "config" / "model_routes.yaml"


def _is_admin(info: TokenInfo) -> bool:
    return "*" in info.scopes or "admin" in info.scopes


def _load_yaml() -> dict:
    if not _CONFIG_PATH.exists():
        return {}
    with _CONFIG_PATH.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _save_yaml_atomic(data: dict) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".model_routes.", suffix=".yaml.tmp", dir=str(_CONFIG_PATH.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        os.replace(tmp, _CONFIG_PATH)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


class RouteUpdateRequest(BaseModel):
    task: Literal["text", "vision", "embedding"]
    model: str = Field(..., min_length=1, max_length=128)
    fallback: str = Field("", max_length=128)
    timeout: int = Field(240, ge=10, le=3600)


@router.get("/stats/admin/model-routes")
def get_model_routes(
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """Return the active ModelRouter configuration.

    ``resolved_models`` reports the actual string each call site would
    receive from ``ModelRouter.resolve(task)`` right now. This is the
    canonical proof that admin overrides took effect — it is what
    smoke check ``model_identity_check`` (Issue #106) probes.
    """
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    router = ModelRouter(ollama_url=ollama_url)
    return {
        "routes": router.to_dict(),
        "resolved_models": {
            "text":      router.resolve("text"),
            "vision":    router.resolve("vision"),
            "embedding": router.resolve("embedding"),
        },
        "config_path": str(_CONFIG_PATH.relative_to(_ROOT)),
        "config_exists": _CONFIG_PATH.exists(),
    }


@router.post("/stats/admin/model-routes")
def update_model_route(
    request: RouteUpdateRequest,
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """Persist a single task→model assignment to ``config/model_routes.yaml``.

    Subsequent calls to ``ModelRouter()`` pick up the new mapping without a
    restart, because ``__init__`` re-reads the YAML on every instance.
    """
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    data = _load_yaml()
    data[request.task] = {
        "model": request.model,
        "fallback": request.fallback,
        "timeout": request.timeout,
    }
    _save_yaml_atomic(data)
    _log.info(
        "model_router admin update: task=%s model=%s by workspace=%s",
        request.task, request.model, info.workspace_id,
    )
    return {
        "updated": request.task,
        "model": request.model,
        "fallback": request.fallback,
        "timeout": request.timeout,
        "saved_to": str(_CONFIG_PATH.relative_to(_ROOT)),
    }


@router.post("/stats/admin/model-routes/reset")
def reset_model_routes(
    info: TokenInfo = Depends(get_token_info),
) -> dict:
    """Drop ``config/model_routes.yaml`` so ``ModelRouter`` falls back to
    its hardcoded defaults. Useful when an admin pushed a bad config."""
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    if _CONFIG_PATH.exists():
        _CONFIG_PATH.unlink()
        deleted = True
    else:
        deleted = False
    _log.info("model_router admin reset by workspace=%s", info.workspace_id)
    return {"reverted_to": "defaults", "config_deleted": deleted}
