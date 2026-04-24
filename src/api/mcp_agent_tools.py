"""Agent tools registered onto the FastMCP instance."""

from __future__ import annotations

import os
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
    def ingest(
        source: str,
        source_type: str = "auto",
        source_id: str | None = None,
        workspace_id: str | None = None,
    ) -> dict:
        """Ingest any source into memory and run the full analysis pipeline.

        One call handles everything: chunk → embed → categorize (Mayring) →
        wiki update → ambient snapshot → predictive rebuild. Dedup via content
        hash — unchanged content is skipped automatically.

        source_type auto-detection (when "auto"):
        - GitHub/GitLab URL or git@ → repo pipeline (async, returns job_id)
        - Everything else → text pipeline (sync, returns chunk info)

        Args:
            source:      Repo URL (e.g. "https://github.com/nileneb/MayringCoder")
                         or text content (conversation summary, file content, insight)
            source_type: "auto" | "repo" | "text"  (default: "auto")
            source_id:   Dedup key for text sources (e.g. "session-memory:2026-04-25-topic")
            workspace_id: Tenant namespace (default: from JWT)

        Returns:
            repo: {job_id, status, repo, workspace_id}
            text: {source_id, chunk_ids, workspace_id}
        """
        import httpx
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        _api = os.getenv("MAYRING_API_URL", "http://localhost:8090").rstrip("/")
        _jwt = _current_raw_jwt()
        headers = {"Authorization": f"Bearer {_jwt}"} if _jwt else {}

        is_repo = source_type == "repo" or (
            source_type == "auto" and (
                source.startswith("https://github.com")
                or source.startswith("https://gitlab.com")
                or source.startswith("git@")
            )
        )

        if is_repo:
            try:
                resp = httpx.post(
                    f"{_api}/populate",
                    json={"repo": source},
                    headers=headers,
                    timeout=30.0,
                )
                resp.raise_for_status()
                return {**resp.json(), "workspace_id": ws}
            except Exception as exc:
                return {"error": str(exc), "workspace_id": ws}
        else:
            try:
                import hashlib
                sid = source_id or f"text:{ws}:{hashlib.sha256(source[:64].encode()).hexdigest()[:12]}"
                source_dict = {
                    "source_id": sid,
                    "source_type": "knowledge",
                    "repo": ws,
                    "path": sid,
                    "content_hash": "sha256:" + hashlib.sha256(source.encode()).hexdigest()[:16],
                }
                result = _run_ingest(
                    source_dict, source, _get_conn(), _get_chroma(),
                    _OLLAMA_URL, _MODEL, {"categorize": True}, ws,
                )
                # Post-ingest: wiki + ambient (fire-and-forget, non-critical)
                try:
                    httpx.post(f"{_api}/wiki/generate",
                               json={"workspace_id": ws},
                               headers=headers, timeout=5.0)
                    httpx.post(f"{_api}/ambient/snapshot",
                               json={"repo": ws},
                               headers=headers, timeout=5.0)
                except Exception:
                    pass
                return {"source_id": sid, "workspace_id": ws, **result}
            except Exception as exc:
                return {"error": str(exc), "workspace_id": ws}

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
