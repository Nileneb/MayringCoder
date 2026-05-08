"""REST mirror of the ``pi_second_opinion`` MCP tool.

The MCP tool can only be exercised from an MCP client. This REST endpoint
exposes the same WikiSecondOpinion validator path so smoke tests can prove
the production validator works end-to-end with one HTTP probe — no MCP
client wiring needed.
"""
from __future__ import annotations

import os
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.auth import get_workspace

router = APIRouter()


class SecondOpinionRequest(BaseModel):
    target_id: str = Field(..., min_length=1, max_length=512)
    scope: Literal["cluster", "edge"] = "cluster"
    repo_slug: str = "default"
    model: str | None = None
    dry_run: bool = False


@router.post("/wiki/second-opinion")
async def wiki_second_opinion(
    request: SecondOpinionRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Run a single-target validation pass.

    With ``dry_run=true``: only validates the target exists in the graph,
    does not fire an Ollama call. Used by smoke to confirm the route +
    lookup pipeline without burning LLM time.
    """
    from src.config import CACHE_DIR
    from src.wiki_v2.graph import WikiGraph
    from src.wiki_v2.second_opinion import WikiSecondOpinion

    graph = WikiGraph(
        workspace_id=workspace_id,
        repo_slug=request.repo_slug,
        db_path=CACHE_DIR / "wiki_v2.db",
    )
    try:
        if request.scope == "cluster":
            clusters = graph.get_clusters()
            target = next(
                (c for c in clusters if c.cluster_id == request.target_id),
                None,
            )
            if target is None:
                raise HTTPException(
                    status_code=404, detail="cluster not found",
                )
            if request.dry_run:
                return {
                    "scope": "cluster",
                    "target_id": request.target_id,
                    "members_count": len(target.members),
                    "dry_run": True,
                }
            ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            chosen_model = request.model or _resolve_default_model(ollama_url)
            verdicts = WikiSecondOpinion().validate_clusters(
                [target], graph, chosen_model, ollama_url,
            )
            v = verdicts[0]
            return {
                "scope": "cluster",
                "target_id": request.target_id,
                "action": v.action,
                "rationale": v.rationale,
                "members_count": len(target.members),
                "model": chosen_model,
            }

        sep = "→" if "→" in request.target_id else "->"
        if sep not in request.target_id or ":" not in request.target_id:
            raise HTTPException(
                status_code=422,
                detail="edge target_id must look like 'src→tgt:type' or 'src->tgt:type'",
            )
        src_tgt, _, etype = request.target_id.rpartition(":")
        src, tgt = src_tgt.split(sep, 1)
        edges = [
            e for e in graph.get_edges(node_id=src.strip())
            if e.source == src.strip() and e.target == tgt.strip()
            and e.type == etype.strip()
        ]
        if not edges:
            raise HTTPException(status_code=404, detail="edge not found")
        if request.dry_run:
            return {
                "scope": "edge",
                "target_id": request.target_id,
                "edges_matched": len(edges),
                "dry_run": True,
            }
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        chosen_model = request.model or _resolve_default_model(ollama_url)
        verdicts = WikiSecondOpinion().validate_edges(
            edges, graph, chosen_model, ollama_url,
            edge_types=[etype.strip()],
        )
        v = verdicts[0]
        return {
            "scope": "edge",
            "target_id": request.target_id,
            "action": v.action,
            "rationale": v.rationale,
            "validated": v.action == "BESTÄTIGT",
            "corrected_type": v.corrected_type,
            "model": chosen_model,
        }
    finally:
        try:
            graph.close()
        except Exception:
            pass


def _resolve_default_model(ollama_url: str) -> str:
    """Issue #88: ModelRouter is the only allowed path. Never read
    OLLAMA_MODEL from env or admin overrides won't take effect."""
    from src.model_router import ModelRouter
    return ModelRouter(ollama_url=ollama_url).resolve("text") or "qwen2.5-coder:7b"
