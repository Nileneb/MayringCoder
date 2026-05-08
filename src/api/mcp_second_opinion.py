"""MCP tool: pi_second_opinion — runtime cross-validation for clusters/edges.

Closes Issue #139: ``--wiki-second-opinion`` lived only as a CLI flag.
The Pi-Agent could never trigger second-opinion validation mid-task, so
clusters/edges with low confidence stayed unvalidated until a human ran
the CLI manually. This tool exposes the same logic via the FastMCP
``memory-agents`` server so the agent can call it during analysis.
"""
from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.api.mcp_auth import (
    _OLLAMA_URL,
    _effective_workspace_id,
    _enforce_tenant,
)


def _resolve_model(override: str | None) -> str:
    if override:
        return override
    from src.model_router import ModelRouter
    routed = ModelRouter(_OLLAMA_URL).resolve("text")
    return routed or os.getenv("OLLAMA_MODEL") or "qwen2.5-coder:7b"


def register_second_opinion_tools(mcp: FastMCP) -> None:

    @mcp.tool()
    def pi_second_opinion(
        target_id: str,
        scope: str = "cluster",
        model: str | None = None,
        repo_slug: str = "default",
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Run a second-opinion validation pass on ONE cluster or edge.

        Args:
            target_id: For scope='cluster', the ``cluster_id``.
                       For scope='edge', the format ``"src→tgt:type"`` or
                       ``"src->tgt:type"`` (any of '->', '→' separates nodes).
            scope: ``"cluster"`` or ``"edge"``.
            model: Override Ollama model (default: ``ModelRouter.resolve("text")``).
            repo_slug: Repo slug for graph lookup. Defaults to 'default'.
            workspace_id: Tenant scope (default: from JWT).

        Returns:
            cluster scope: {action, rationale, members_count, target_id, model, validator_action}
            edge scope:    {action, rationale, validated, corrected_type, target_id, model}
            on error:      {error, hint, target_id}
        """
        from src.config import CACHE_DIR
        from src.wiki_v2.graph import WikiGraph
        from src.wiki_v2.second_opinion import WikiSecondOpinion

        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        if scope not in ("cluster", "edge"):
            return {"error": f"unknown scope: {scope}",
                    "hint": "use 'cluster' or 'edge'", "target_id": target_id}
        chosen_model = _resolve_model(model)

        graph = WikiGraph(
            workspace_id=ws, repo_slug=repo_slug,
            db_path=CACHE_DIR / "wiki_v2.db",
        )
        validator = WikiSecondOpinion()

        try:
            if scope == "cluster":
                clusters = graph.get_clusters()
                target = next((c for c in clusters if c.cluster_id == target_id), None)
                if target is None:
                    return {"error": "cluster not found",
                            "hint": f"no cluster with id={target_id} in workspace={ws}",
                            "target_id": target_id}
                verdicts = validator.validate_clusters(
                    [target], graph, chosen_model, _OLLAMA_URL,
                )
                if not verdicts:
                    return {"error": "validator returned no verdict",
                            "target_id": target_id}
                v = verdicts[0]
                return {
                    "scope": "cluster",
                    "target_id": target_id,
                    "action": v.action,
                    "rationale": v.rationale,
                    "members_count": len(target.members),
                    "model": chosen_model,
                }

            sep = "→" if "→" in target_id else ("->" if "->" in target_id else None)
            if sep is None or ":" not in target_id:
                return {"error": "edge target_id must look like 'src→tgt:type'",
                        "hint": "use the unicode arrow or '->' as separator",
                        "target_id": target_id}
            src_tgt, _, etype = target_id.rpartition(":")
            src, tgt = src_tgt.split(sep, 1)
            edges = [
                e for e in graph.get_edges(node_id=src.strip())
                if e.source == src.strip() and e.target == tgt.strip()
                and e.type == etype.strip()
            ]
            if not edges:
                return {"error": "edge not found",
                        "hint": f"no {etype} edge {src}→{tgt} in workspace={ws}",
                        "target_id": target_id}
            verdicts = validator.validate_edges(
                edges, graph, chosen_model, _OLLAMA_URL,
                edge_types=[etype.strip()],
            )
            if not verdicts:
                return {"error": "validator returned no verdict",
                        "target_id": target_id}
            v = verdicts[0]
            return {
                "scope": "edge",
                "target_id": target_id,
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


__all__ = ("register_second_opinion_tools",)
