"""Agent tools registered onto the FastMCP instance."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from src.api.mcp_auth import (
    _OLLAMA_URL,
    _MODEL,
    _enforce_tenant,
    _effective_workspace_id,
    _current_raw_jwt,
)
from src.api.dependencies import get_conn as _get_conn, get_chroma as _get_chroma
from src.api.memory_service import run_ingest as _run_ingest


def register_agent_tools(mcp: FastMCP) -> None:

    @mcp.tool()
    def pi_task(
        task: str,
        repo_slug: str | None = None,
        system_prompt: str | None = None,
        timeout: float = 180.0,
        workspace_id: str | None = None,
    ) -> dict:
        """Run a free-form task using the Pi-Agent with memory-augmented reasoning.

        The Pi-Agent searches workspace memory for relevant context, injects it
        into the system prompt (ambient context), and can issue up to 5
        search_memory tool calls during reasoning. Requires Ollama to be reachable.

        Args:
            task: Free-form task or question
            repo_slug: Repository slug to scope memory retrieval
            system_prompt: Custom system prompt (default: Pi-Agent task prompt)
            timeout: Per-request HTTP timeout in seconds (default: 180)
            workspace_id: Tenant namespace for memory scope (default: from JWT)

        Returns:
            {result: str, workspace_id: str} or {error: str}
        """
        try:
            from src.agents.pi import run_task_with_memory
            ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
            result = run_task_with_memory(
                task=task,
                ollama_url=_OLLAMA_URL,
                model=_MODEL,
                repo_slug=repo_slug,
                system_prompt=system_prompt,
                timeout=timeout,
            )
            return {"result": result, "workspace_id": ws}
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def conversation_ingest(
        turns: list[dict],
        session_id: str,
        workspace_slug: str = "default",
        presumarized: str | None = None,
    ) -> dict:
        """Ingest a batch of conversation turns into workspace memory.

        Summarizes (via Ollama) and stores as a conversation_summary source.
        Dedup: same session_id + same content is skipped automatically.

        Args:
            turns: List of {"role": "user"|"assistant", "content": str, "timestamp": str}
            session_id: Unique session identifier (used for dedup)
            workspace_slug: Workspace namespace (default: "default")
            presumarized: Pre-written summary to skip Ollama summarization

        Returns:
            {workspace_id, source_id, chunk_ids, indexed, deduped, skipped}
        """
        try:
            import hashlib
            from tools.ingest_conversations import _summarize as _summarize_turns

            if not turns:
                return {"error": "turns must not be empty"}

            ws = _enforce_tenant(workspace_slug) or workspace_slug
            batch_key = f"{session_id}:{len(turns)}:{turns[-1].get('timestamp', '')}"
            content_hash = "sha256:" + hashlib.sha256(batch_key.encode()).hexdigest()[:16]
            source_id = f"conversation:{workspace_slug}:{session_id[:16]}"

            summary = presumarized or _summarize_turns(turns, "", _OLLAMA_URL, _MODEL)

            if not summary or not summary.strip():
                return {"workspace_id": ws, "source_id": source_id,
                        "chunk_ids": [], "indexed": False, "skipped": True}

            source_dict = {
                "source_id": source_id,
                "source_type": "conversation_summary",
                "repo": workspace_slug,
                "path": f"conversations/{session_id[:16]}",
                "content_hash": content_hash,
            }
            result = _run_ingest(
                source_dict, summary, _get_conn(), _get_chroma(),
                _OLLAMA_URL, _MODEL, {"categorize": False}, ws,
            )
            return {"workspace_id": ws, **result}
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def analyze(
        repo: str,
        full: bool = False,
        no_pi: bool = False,
        budget: int | None = None,
        workspace_id: str | None = None,
    ) -> dict:
        """Start a full code analysis pipeline for a repository (async).

        Runs src.pipeline as a subprocess. Returns immediately with a pid.
        Analysis output is written to reports/. No job_id — fire and forget.

        Args:
            repo: Repository URL or local path
            full: Force full re-analysis (ignore incremental cache)
            no_pi: Disable Pi-Agent (faster, no memory-augmented analysis)
            budget: Max files to analyse (None = no limit)
            workspace_id: Tenant namespace (default: from JWT)

        Returns:
            {status: "started", pid: int, repo: str} or {error: str}
        """
        try:
            import subprocess

            ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
            cmd = [sys.executable, "-m", "src.pipeline", "--repo", repo, "--workspace-id", ws]
            if full:
                cmd.append("--full")
            if no_pi:
                cmd.append("--no-pi")
            if budget is not None:
                cmd.extend(["--budget", str(budget)])

            proc = subprocess.Popen(
                cmd,
                cwd=str(Path(__file__).parent.parent.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return {"status": "started", "pid": proc.pid, "repo": repo, "workspace_id": ws}
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def duel(
        task: str,
        model_a: str,
        model_b: str,
        repo_slug: str | None = None,
        judge: bool = True,
        judge_model: str | None = None,
        no_memory_baseline: bool = False,
        timeout: float = 180.0,
        workspace_id: str | None = None,
    ) -> dict:
        """Run the same task on two models and get an automatic verdict.

        Both models run through the Pi-Agent with full memory injection.
        An auto-judge LLM rates both answers (0-10) and picks a winner.

        Args:
            task: The task/question both models should answer
            model_a: First model name (e.g. "gemma4:e4b")
            model_b: Second model name (e.g. "qwen3:2b")
            repo_slug: Optional repo scope for memory search
            judge: Whether to auto-judge both answers (default True)
            judge_model: Model for judging (default: server OLLAMA_MODEL)
            no_memory_baseline: Also run both models WITHOUT memory for comparison
            timeout: Per-model timeout in seconds
            workspace_id: Tenant namespace (default: from JWT)

        Returns:
            Job dict with job_id — poll /jobs/{id} for result_a, result_b, verdict
        """
        import httpx
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        _api = os.getenv("MAYRING_API_URL", "http://localhost:8090").rstrip("/")
        _jwt = _current_raw_jwt()
        headers = {"Authorization": f"Bearer {_jwt}"} if _jwt else {}
        try:
            resp = httpx.post(
                f"{_api}/duel",
                json={
                    "task": task,
                    "model_a": model_a,
                    "model_b": model_b,
                    "repo_slug": repo_slug,
                    "judge": judge,
                    "judge_model": judge_model,
                    "no_memory_baseline": no_memory_baseline,
                    "timeout": timeout,
                },
                headers=headers,
                timeout=30.0,
            )
            resp.raise_for_status()
            return {**resp.json(), "workspace_id": ws}
        except Exception as exc:
            return {"error": str(exc), "workspace_id": ws}

    @mcp.tool()
    def benchmark_tasks(
        model_a: str,
        model_b: str,
        category: str | None = None,
        repo_slug: str | None = None,
        judge_model: str | None = None,
        timeout: float = 180.0,
        workspace_id: str | None = None,
    ) -> dict:
        """Run the predefined task suite on two models and get a quality comparison.

        Tasks are defined in benchmarks/task_suite.yaml (categories: context_injection,
        pico, code_review, conversation_summary). Each task is scored by keyword hits
        and an auto-judge LLM.

        Args:
            model_a: First model name
            model_b: Second model name
            category: Filter by category (None = all tasks)
            repo_slug: Optional repo scope for memory search
            judge_model: Model used for judging (default: server OLLAMA_MODEL)
            timeout: Per-task per-model timeout in seconds
            workspace_id: Tenant namespace (default: from JWT)

        Returns:
            Report with wins_a, wins_b, avg_score_a, avg_score_b, per-task results
        """
        import httpx
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        _api = os.getenv("MAYRING_API_URL", "http://localhost:8090").rstrip("/")
        _jwt = _current_raw_jwt()
        headers = {"Authorization": f"Bearer {_jwt}"} if _jwt else {}
        try:
            resp = httpx.post(
                f"{_api}/benchmark/tasks",
                json={
                    "model_a": model_a,
                    "model_b": model_b,
                    "category": category,
                    "repo_slug": repo_slug,
                    "judge_model": judge_model,
                    "timeout": timeout,
                },
                headers=headers,
                timeout=600.0,
            )
            resp.raise_for_status()
            return {**resp.json(), "workspace_id": ws}
        except Exception as exc:
            return {"error": str(exc), "workspace_id": ws}
