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

import logging
import sqlite3
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.api.mcp_auth import _enforce_tenant, _effective_workspace_id

logger = logging.getLogger(__name__)


_PI_JOBS_PHASE2_COLUMNS = (
    ("scope", "TEXT NOT NULL DEFAULT 'local'"),
    ("capability_required", "TEXT NOT NULL DEFAULT ''"),
    ("claimed_by", "TEXT NOT NULL DEFAULT ''"),
    ("claimed_at", "TEXT NOT NULL DEFAULT ''"),
)


def _ensure_pi_jobs_phase2_columns(db) -> None:
    """Lazily add Plan-C Phase-2 columns to pi_jobs.

    Defence-in-depth: production hit "no such column: scope" because the
    cloud DB pre-dates the migration, and the central `_migrate_schema`
    only fires when `init_memory_db` runs against that DB. The startup
    hook in server.py covers the happy path; this hook covers the case
    where the connection was already cached before that hook landed.

    Idempotent: PRAGMA skips if the column is present, ALTER TABLE
    fails are logged but never raised — so `pi_task_*_cloud` requests
    never 500 on a half-migrated schema.
    """
    try:
        cols = {r[1] for r in db.execute("PRAGMA table_info(pi_jobs)").fetchall()}
    except sqlite3.Error:
        return  # table itself missing — caller will fail visibly, not silently.
    if not cols:
        # PRAGMA on a non-existent table returns no rows. Surface that
        # explicitly: the deployment lost the pi_jobs table entirely and
        # needs init_memory_db to recreate it. Don't try ALTER TABLE.
        logger.error(
            "mcp_pi_tools: pi_jobs table is missing — run init_memory_db()"
        )
        return
    for col_name, col_def in _PI_JOBS_PHASE2_COLUMNS:
        if col_name not in cols:
            try:
                db.execute(f"ALTER TABLE pi_jobs ADD COLUMN {col_name} {col_def}")
                logger.warning("mcp_pi_tools: applied missing pi_jobs.%s", col_name)
            except sqlite3.Error as e:
                logger.error("mcp_pi_tools: failed to add pi_jobs.%s: %s", col_name, e)


def register_pi_queue_tools(mcp: FastMCP) -> None:
    """Register the pi-task cloud queue tools onto the FastMCP instance."""

    # One-shot migration at registration time, against the same connection
    # the tools below will use. Cheap: ALTER TABLE on already-present
    # columns is a no-op via the PRAGMA pre-check.
    try:
        from src.api.dependencies import get_conn as _get_conn
        _ensure_pi_jobs_phase2_columns(_get_conn())
    except Exception:
        logger.exception("mcp_pi_tools: registration-time migration skipped")

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
        """Read-only snapshot of a cloud-routed pi-task job, scoped to the
        caller's tenant. Foreign jobs report as "not found" so the row's
        existence cannot be probed cross-tenant."""
        from src.agents import pi_jobs
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        job = pi_jobs.get_job(job_id, workspace_id=ws)
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
            "workspace_id": d["workspace_id"],
        }

    @mcp.tool()
    def pi_task_list_cloud(
        only_active: bool = False,
        limit: int = 10,
        workspace_id: str | None = None,
    ) -> dict:
        """List recent CLOUD-scope pi-task jobs scoped to the caller's tenant."""
        from src.agents import pi_jobs
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        jobs = pi_jobs.list_recent(
            only_active=only_active,
            scope="cloud",
            workspace_id=ws,
            limit=max(1, min(50, limit)),
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
