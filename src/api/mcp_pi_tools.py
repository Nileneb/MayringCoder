"""Cloud-side MCP tools for the pi-task queue.

Registered onto the cloud MCP server (`src/api/mcp.py`). Backed by the same
`pi_jobs` table as the local Phase-1 worker, but with `scope='cloud'` and
worker_id tracking so multiple user devices can pull from one queue.

Tool surface:
    pi_task_submit_cloud(task, capability_required, ...)   → {job_id}
    pi_task_claim_cloud(worker_id, capabilities)            → next job | None
    pi_task_complete_cloud(job_id, result | error)
    pi_task_status_cloud(job_id)                            → status snapshot
    pi_task_list_cloud(only_active, limit)                  → recent jobs
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from src.api.mcp_auth import _enforce_tenant, _effective_workspace_id


def register_pi_queue_tools(mcp: FastMCP) -> None:
    """Register the pi-task cloud queue tools onto the FastMCP instance."""

    @mcp.tool()
    def pi_task_submit_cloud(
        task: str,
        capability_required: str = "",
        repo_slug: str | None = None,
        prefer: str = "auto",
        ollama_url: str = "",
        model: str = "",
        timeout: float = 180.0,
        workspace_id: str | None = None,
    ) -> dict:
        """Submit a pi-task to the CLOUD queue. Any capable worker may claim.

        `capability_required` is a comma-separated whitelist (e.g.
        "local-gpu,ollama-models:qwen2.5-coder:7b"). Empty string matches
        any worker.

        Returns: {"job_id": "pij_...", "status": "queued", "workspace_id": "..."}
        """
        from src.agents import pi_jobs
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        try:
            job = pi_jobs.insert_cloud_job(
                task_text=task,
                repo_slug=repo_slug or "",
                workspace_id=ws,
                prefer=prefer,
                ollama_url=ollama_url,
                model=model,
                timeout_s=timeout,
                capability_required=capability_required,
            )
        except Exception as exc:
            return {"error": str(exc), "workspace_id": ws}
        return {"job_id": job.job_id, "status": "queued", "workspace_id": ws}

    @mcp.tool()
    def pi_task_claim_cloud(
        worker_id: str,
        capabilities: list[str] | None = None,
        workspace_id: str | None = None,
    ) -> dict:
        """Worker-side: atomically claim the next pending CLOUD job.

        Returns the job to execute, or {"job": null} when nothing matches.
        Capabilities is a list of strings the worker advertises (e.g.
        ["local-gpu", "ollama-models:qwen2.5-coder:7b"]).
        """
        from src.agents import pi_jobs
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        try:
            job = pi_jobs.claim_cloud_next(
                worker_id=worker_id,
                capabilities=capabilities or [],
                workspace_id=ws,
            )
        except Exception as exc:
            return {"error": str(exc), "workspace_id": ws}
        if job is None:
            return {"job": None, "workspace_id": ws}
        d = job.to_dict()
        return {
            "job": {
                "job_id": d["job_id"],
                "task_text": d["task_text"],
                "repo_slug": d["repo_slug"],
                "ollama_url": d["ollama_url"],
                "model": d["model"],
                "timeout_s": d["timeout_s"],
                "capability_required": d["capability_required"],
                "claimed_at": d["claimed_at"],
            },
            "workspace_id": ws,
        }

    @mcp.tool()
    def pi_task_complete_cloud(
        job_id: str,
        result: Any = None,
        error: str | None = None,
        workspace_id: str | None = None,
    ) -> dict:
        """Worker-side: report job outcome. Provide `result` OR `error`."""
        from src.agents import pi_jobs
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        try:
            if error:
                pi_jobs.fail_job(job_id, error)
            else:
                pi_jobs.complete_job(job_id, result if result is not None else "")
        except Exception as exc:
            return {"error": str(exc), "workspace_id": ws}
        return {"job_id": job_id, "status": "ack", "workspace_id": ws}

    @mcp.tool()
    def pi_task_status_cloud(
        job_id: str, workspace_id: str | None = None,
    ) -> dict:
        """Read-only snapshot of a cloud-routed pi-task job."""
        from src.agents import pi_jobs
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        job = pi_jobs.get_job(job_id)
        if job is None:
            return {"error": "not found", "job_id": job_id, "workspace_id": ws}
        d = job.to_dict()
        return {
            "job_id": d["job_id"],
            "scope": d["scope"],
            "status": d["status"],
            "result": d["result"],
            "error": d["error"],
            "claimed_by": d["claimed_by"],
            "claimed_at": d["claimed_at"],
            "started_at": d["started_at"],
            "finished_at": d["finished_at"],
            "workspace_id": ws,
        }

    @mcp.tool()
    def pi_task_list_cloud(
        only_active: bool = False,
        limit: int = 10,
        workspace_id: str | None = None,
    ) -> dict:
        """List recent CLOUD-scope pi-task jobs (default: 10 newest)."""
        from src.agents import pi_jobs
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        jobs = pi_jobs.list_recent(
            only_active=only_active, scope="cloud", limit=max(1, min(50, limit)),
        )
        return {
            "jobs": [
                {
                    "job_id": j.job_id,
                    "status": j.status,
                    "task_preview": j.task_text[:120],
                    "capability_required": j.capability_required,
                    "claimed_by": j.claimed_by,
                    "created_at": j.created_at,
                    "started_at": j.started_at,
                    "finished_at": j.finished_at,
                }
                for j in jobs
            ],
            "workspace_id": ws,
        }


__all__ = ("register_pi_queue_tools",)
