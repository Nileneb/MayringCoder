"""Agent tools registered onto the FastMCP instance."""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from src.api.mcp_auth import (
    _OLLAMA_URL,
    _enforce_tenant,
    _effective_workspace_id,
    _current_raw_jwt,
)


def _model(task: str = "text") -> str:
    from mayring_core.model_router import ModelRouter
    return ModelRouter(_OLLAMA_URL).resolve(task) or "qwen2.5-coder:7b"
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
        """DEFAULT delegation tool for implementation tasks — use this FIRST before Claude subagents.

        Use pi_task for: concrete code changes, file analysis, bug fixes, iterative test loops,
        repo convention lookups, TODOs. Only fall back to Claude subagents when pi_task fails
        or the task requires frontier-level reasoning / direct file writes.

        Runs via local Ollama + synced memory DB (PI_AGENT_URL=direct, no pi_server.py needed).

        Args:
            task: Free-form task or question
            repo_slug: Repository slug to scope memory retrieval
            system_prompt: Custom system prompt (default: Pi-Agent task prompt)
            timeout: Per-request HTTP timeout in seconds (default: 180)
            workspace_id: Tenant namespace for memory scope (default: from JWT)

        Returns:
            {result: str, workspace_id: str} or {error: str, hint: str}
        """
        import httpx
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        pi_url = os.getenv("PI_AGENT_URL", "http://localhost:8091")

        if pi_url.lower() != "direct":
            try:
                resp = httpx.post(
                    f"{pi_url}/task",
                    json={
                        "task": task,
                        "repo_slug": repo_slug,
                        "system_prompt": system_prompt,
                        "timeout": timeout,
                    },
                    headers={"Authorization": f"Bearer {_current_raw_jwt() or ''}"},
                    timeout=timeout + 10,
                )
                resp.raise_for_status()
                return {**resp.json(), "workspace_id": ws}
            except httpx.ConnectError:
                return {
                    "error": f"Pi-Server nicht erreichbar ({pi_url})",
                    "hint": "pi_server.py starten: cd MayringCoder && .venv/bin/python -m mayring_pi_agent.pi_server",
                }
            except Exception as exc:
                return {"error": str(exc), "hint": "Pi-Server-Fehler"}

        # Fallback: direct Ollama call (only when PI_AGENT_URL=direct)
        try:
            resp = httpx.get(f"{_OLLAMA_URL}/api/tags", timeout=3.0)
            if resp.status_code != 200:
                raise httpx.HTTPStatusError("", request=resp.request, response=resp)
        except Exception:
            return {
                "error": "Ollama nicht erreichbar",
                "hint": f"Ollama starten: ollama serve  (erwartet unter {_OLLAMA_URL})",
            }
        try:
            from mayring_pi_agent.pi import run_task_with_memory
            result = run_task_with_memory(
                task=task,
                ollama_url=_OLLAMA_URL,
                model=_model("text"),
                repo_slug=repo_slug,
                system_prompt=system_prompt,
                timeout=timeout,
            )
            return {"result": result, "workspace_id": ws}
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def pi_optimize_searchstring(
        searchstring: str,
        database: str,
        forschungsfrage: str,
        repo_slug: str | None = None,
        timeout: float = 120.0,
        workspace_id: str | None = None,
    ) -> dict:
        """Überarbeite einen P4-Suchstring kostengünstig via Pi-Agent (#261).

        Läuft über die PI_AGENT_URL-Boundary (Cloud-Distributor wenn gesetzt,
        sonst in-process Ollama) — KEIN eigener Direct-Ollama-Pfad. Von
        app.linn.games via PI_AGENT_URL aufrufbar.

        Args:
            searchstring: Bestehender Suchstring.
            database: Datenbankname (z.B. 'PubMed', 'Scopus') — bestimmt die Syntax.
            forschungsfrage: Forschungsfrage als Kontext für die Optimierung.
            repo_slug: Optionaler Memory-Scope.
            timeout: Pi-Call-Timeout (s).
            workspace_id: Tenant-Scope.

        Returns:
            #260-konformes Paar: {input: {searchstring, database, forschungsfrage},
            output: {revised, reasoning}, parsed: bool, workspace_id} oder {error}.
        """
        from src.workflows.searchstring_review import build_prompt, parse_response

        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        if not searchstring or not searchstring.strip():
            return {"error": "searchstring leer", "workspace_id": ws}

        prompt = build_prompt(searchstring, database, forschungsfrage)
        # Delegiert über pi_task → respektiert PI_AGENT_URL (Distributor/in-process).
        res = pi_task(
            task=prompt,
            repo_slug=repo_slug,
            system_prompt="Du bist ein Suchstring-Optimierer. Antworte NUR mit dem geforderten JSON.",
            timeout=timeout,
            workspace_id=workspace_id,
        )
        if "error" in res:
            return res
        parsed = parse_response(res.get("result", ""))
        return {
            "input": {
                "searchstring": searchstring,
                "database": database,
                "forschungsfrage": forschungsfrage,
            },
            "output": {"revised": parsed["revised"], "reasoning": parsed["reasoning"]},
            "parsed": parsed["parsed"],
            "workspace_id": ws,
        }

    @mcp.tool()
    def pi_task_start(
        task: str,
        repo_slug: str | None = None,
        prefer: str = "auto",
        ollama_url: str | None = None,
        model: str | None = None,
        timeout: float = 180.0,
        workspace_id: str | None = None,
    ) -> dict:
        """Submit a pi-task asynchronously. Returns immediately with a job_id.

        The local worker thread (started at MCP-server boot) picks the job up,
        runs `run_task_with_memory()` on the configured Ollama backend, and
        persists the result. Poll `pi_task_status(job_id)` to retrieve it.

        Args:
            task: Free-form task description.
            repo_slug: Repository slug for memory scope filtering.
            prefer: "auto" | "local" | "cloud" — routing hint (Phase 2 honours it;
                Phase 1 always runs locally).
            ollama_url: Override the default OLLAMA_URL for this single job.
            model: Override the resolved model for this single job.
            timeout: Per-LLM-request timeout (seconds).
            workspace_id: Tenant namespace (default: from JWT).

        Returns:
            {"job_id": "pij_...", "status": "queued", "workspace_id": "..."}
        """
        from mayring_pi_agent import pi_jobs
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        if prefer not in pi_jobs.VALID_PREFER:
            return {
                "error": f"prefer must be one of {pi_jobs.VALID_PREFER}",
                "workspace_id": ws,
            }
        try:
            job = pi_jobs.insert_job(
                task_text=task,
                repo_slug=repo_slug or "",
                workspace_id=ws,
                prefer=prefer,
                ollama_url=ollama_url or "",
                model=model or _model("text"),
                timeout_s=timeout,
            )
        except Exception as exc:
            return {"error": str(exc), "workspace_id": ws}
        return {"job_id": job.job_id, "status": "queued", "workspace_id": ws}

    @mcp.tool()
    def pi_task_status(job_id: str, workspace_id: str | None = None) -> dict:
        """Return the current state of a pi-task job.

        Scoped to the caller's effective workspace — a job belonging to
        another tenant is reported as "not found" so its existence cannot
        be probed cross-tenant.

        Result envelope:
            {"job_id", "status", "result", "error",
             "started_at", "finished_at", "workspace_id"}
        """
        from mayring_pi_agent import pi_jobs
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        job = pi_jobs.get_job(job_id, workspace_id=ws)
        if job is None:
            return {"error": "not found", "job_id": job_id, "workspace_id": ws}
        d = job.to_dict()
        return {
            "job_id": d["job_id"],
            "status": d["status"],
            "result": d["result"],
            "error": d["error"],
            "started_at": d["started_at"],
            "finished_at": d["finished_at"],
            # The row is now scoped to `ws`, so its workspace_id matches.
            # Echo the row's value explicitly so the response can never claim
            # a workspace the row doesn't belong to.
            "workspace_id": d["workspace_id"],
        }

    @mcp.tool()
    def pi_task_list(
        only_active: bool = False,
        limit: int = 10,
        workspace_id: str | None = None,
    ) -> dict:
        """List recent pi-task jobs scoped to the caller's workspace.

        Set `only_active=True` to see only queued+running jobs. Only jobs
        belonging to the caller's tenant are returned.
        """
        from mayring_pi_agent import pi_jobs
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        jobs = pi_jobs.list_recent(
            only_active=only_active,
            workspace_id=ws,
            limit=max(1, min(50, limit)),
        )
        return {
            "jobs": [
                {
                    "job_id": j.job_id,
                    "status": j.status,
                    "task_preview": j.task_text[:120],
                    "created_at": j.created_at,
                    "started_at": j.started_at,
                    "finished_at": j.finished_at,
                    "model": j.model,
                    "prefer": j.prefer,
                }
                for j in jobs
            ],
            "workspace_id": ws,
        }

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
            source_type: "auto" | "repo" | "text" | "conversation_summary" | "session_knowledge"
                         | "note" | "paper" (default: "auto")
            source_id:   Dedup key for text sources (e.g. "session-memory:2026-04-25-topic")
            workspace_id: Tenant namespace (default: from JWT)

        Returns:
            repo: {job_id, status, repo, workspace_id}
            text: {source_id, chunk_ids, workspace_id}
        """
        import httpx
        from urllib.parse import urlparse
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        _api = os.getenv("MAYRING_API_URL", "http://localhost:8090").rstrip("/")
        _jwt = _current_raw_jwt()
        headers = {"Authorization": f"Bearer {_jwt}"} if _jwt else {}

        parsed = urlparse(source) if source_type == "auto" else None
        host = (parsed.hostname or "").lower() if parsed else ""
        is_repo = source_type == "repo" or (
            source_type == "auto"
            and (
                (parsed and parsed.scheme == "https" and host in {"github.com", "gitlab.com"})
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
                _stype = source_type if source_type not in ("auto", "text") else "knowledge"
                source_dict = {
                    "source_id": sid,
                    "source_type": _stype,
                    "repo": ws,
                    "path": sid,
                    "content_hash": "sha256:" + hashlib.sha256(source.encode()).hexdigest()[:16],
                }
                result = _run_ingest(
                    source_dict, source, _get_conn(), _get_chroma(),
                    _OLLAMA_URL, _model(), {"categorize": True}, ws,
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
            judge_model: Model for judging (default: ModelRouter 'text' task)
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
            judge_model: Model used for judging (default: ModelRouter 'text' task)
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

    @mcp.tool()
    def diff_history(
        file: str,
        commits: int = 15,
        model: str | None = None,
    ) -> dict:
        """Architektur-Trajektorie einer Datei via lokalem Pi-Agent.

        Lädt die letzten N commits einer Datei (`git log --follow -p`) und
        lässt qwen2.5-coder:7b (oder ModelRouter('text')) eine drei-teilige
        Synthese erzeugen: Trajektorie / Obsolete / Aktiv. Verwendung BEVOR
        Du große refactor-/issue-Entscheidungen über eine Datei triffst —
        verhindert dass entfernte konzepte (z.B. Langdock, deprecated APIs)
        wieder reingebaut werden.

        Args:
            file: Repo-relativer oder absoluter Pfad zur Datei
            commits: Anzahl letzte commits (default 15)
            model: Override Ollama-Modell

        Returns:
            {file, commits, model, summary} oder {error}
        """
        from mayring_pi_agent.diff_history import DiffHistoryError, run as _run
        try:
            return _run(
                file,
                commits=commits,
                model=model,
                ollama_url=_OLLAMA_URL,
            )
        except DiffHistoryError as exc:
            return {"error": str(exc), "file": file}

    # ─── Specialized Pi-Sub-Tools (#211) ─────────────────────────────
    # WHY(2026-05-11): pi_task ist generischer free-form-dispatch. Diese
    # spezialisierten tools haben enge schemas + focused system-prompts,
    # sodass claude-code sie für spezifische sub-tasks dispatchen kann
    # ohne den Pi-Agent jedes Mal komplett zu re-instruct. Asymmetric
    # job-distribution: 90% der categorize/judge/summarize-calls
    # können hier laufen statt in Claude.

    @mcp.tool()
    def pi_categorize(
        text: str,
        task: str = "",
        codebook: list[str] | None = None,
        mode: str = "hybrid",
        max_labels: int = 5,
        workspace_id: str | None = None,
        timeout: float = 60.0,
    ) -> dict:
        """Mayring-categorize a piece of text via local Pi-Agent.

        Nutzt die kanonischen prompts/mayring_{deduktiv,induktiv,hybrid}.md
        templates — gleiche source-of-truth wie die inline-ingest-pipeline.
        {{task}} = das Thema worauf untersucht wird (Mayring Selektions-
        kriterium); {{categories}} = anchor-codebook (deductive/hybrid).

        Args:
            text: The text to categorize (chunk content, max ~4000 chars).
            task: Topic(s) to look for. Empty → derive from text.
            codebook: anchor categories (deductive/hybrid). None + deductive/
                      hybrid → fallback auf universal_v1.yaml.
            mode: 'inductive' | 'deductive' | 'hybrid' (default).
            max_labels: Cap on returned labels (default 5).
            workspace_id: Tenant scope. Optional.
            timeout: Pi-Agent call timeout in seconds.

        Returns:
            {labels: [str], mode, model} oder {error}.
        """
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        if not text or len(text.strip()) < 10:
            return {"error": "text too short (<10 chars)", "workspace_id": ws}

        mode = mode if mode in ("inductive", "deductive", "hybrid") else "hybrid"

        # Codebook only needed for deductive/hybrid. Fall back to universal.
        if mode != "inductive" and not codebook:
            try:
                import yaml
                from pathlib import Path
                cb_path = Path("config/codebooks/universal_v1.yaml")
                if cb_path.exists():
                    cb_data = yaml.safe_load(cb_path.read_text())
                    codebook = list((cb_data or {}).get("categories", {}).keys())
            except Exception:
                codebook = None

        # Load the CANONICAL mayring template — single source of truth.
        try:
            from mayring_core.memory.ingestion.categorization import _load_mayring_template
            template = _load_mayring_template(mode)
        except Exception:
            template = ("Categorize this text chunk for topic {{task}} using "
                        "these categories if applicable: {{categories}}. "
                        "Respond with ONLY a comma-separated list of labels.")
        cats_str = ", ".join(codebook) if codebook else "(none — derive inductively)"
        task_str = task.strip() if task and task.strip() else "(kein Task angegeben)"
        prompt = (
            template
            .replace("{{categories}}", cats_str)
            .replace("{{task}}", task_str)
            + "\n\nText:\n" + text[:4000]
        )

        import httpx
        try:
            resp = httpx.post(
                f"{_OLLAMA_URL}/api/generate",
                json={
                    "model": _model("text"),
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 200},
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            # mayring templates return a comma-separated list, not JSON.
            labels = [
                lbl.strip().lower()
                for lbl in raw.split(",")
                if lbl.strip() and len(lbl.strip()) <= 50
            ][:max_labels]
            return {
                "labels": labels,
                "mode": mode,
                "model": _model("text"),
                "workspace_id": ws,
            }
        except Exception as exc:
            return {"error": str(exc), "workspace_id": ws}

    @mcp.tool()
    def pi_mark_categories(
        text: str,
        task: str = "",
        codebook: list[str] | None = None,
        source_id: str | None = None,
        chunk_id: str = "",
        persist: bool = False,
        workspace_id: str | None = None,
        timeout: float = 90.0,
    ) -> dict:
        """Textmarker — Mayring-categorize a chunk with SPAN-LEVEL evidence.

        Anders als pi_categorize (gibt nur labels für den ganzen chunk):
        dieses tool markiert konkrete Textabschnitte und ordnet jedem
        Abschnitt eine Kategorie zu, MIT der paraphrase-begründung. So ist
        jede Kategorie LOGISCH am Text nachvollziehbar — der "Beweis" für
        die Kategorie ist der markierte Abschnitt.

        Wenn `persist=True` und `source_id` gesetzt: die markings landen in
        wiki_category_evidence (vorhandene evidence für source_id wird
        zuerst gelöscht — clean re-mark). Abrufbar via pi_category_evidence.

        Args:
            text: The chunk text (max ~4000 chars).
            task: Topic(s) to look for — Mayring Selektionskriterium.
                  Empty → derive from text.
            codebook: optional anchor categories (hybrid mode); None → induktiv.
            source_id: which source/chunk this came from (für persist).
            chunk_id: optional chunk-id (für persist).
            persist: True + source_id → schreibt markings ins wiki.
            workspace_id: tenant scope.
            timeout: Pi-Agent call timeout.

        Returns:
            {markings: [{span: [start, end], excerpt, category, reasoning}],
             model, persisted: int} oder {error}. span = char offsets into `text`.
        """
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        if not text or len(text.strip()) < 10:
            return {"error": "text too short (<10 chars)", "workspace_id": ws}

        task_str = task.strip() if task and task.strip() else "(kein Task angegeben)"
        anchors = ", ".join(codebook) if codebook else "(keine — bilde induktiv)"

        sys_prompt = (
            "Du bist ein qualitativer Inhaltsanalytiker (Mayring). Aus dem "
            f"Thema bekommst du, worauf du den Text untersuchen sollst.\n"
            f"Thema: {task_str}\n"
            f"Anker-Kategorien (nutze wenn passend, sonst induktiv neu bilden): {anchors}\n\n"
            "Markiere konkrete Textabschnitte, paraphrasiere jeden bezogen "
            "aufs Thema, generalisiere, reduziere zu einer Kategorie. Jede "
            "Kategorie MUSS sich logisch am markierten Abschnitt nachvollziehen "
            "lassen — keine Kategorie ohne Beleg. Ein Abschnitt kann mehrere "
            "Kategorien haben; Kategorien können sich überlappen. Wenn der Text "
            "nichts zum Thema beiträgt: leeres markings-array.\n\n"
            "Output STRICT JSON: "
            '{"markings":[{"excerpt":"<wortwörtliches textstück>",'
            '"category":"<lowercase-label>","reasoning":"<paraphrase: was sagt '
            'der abschnitt bzgl. des themas, warum diese kategorie>"}]}\n'
            "Kein prosa, kein markdown. excerpt MUSS wortwörtlich im Text "
            "vorkommen (wir berechnen die char-offsets selbst)."
        )

        import httpx
        import json as _json
        try:
            resp = httpx.post(
                f"{_OLLAMA_URL}/api/generate",
                json={
                    "model": _model("text"),
                    "prompt": sys_prompt + "\n\nText:\n" + text[:4000],
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 800},
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            data = _json.loads(raw)
            markings_in = data.get("markings", []) or []
            markings_out = []
            for m in markings_in[:20]:
                if not isinstance(m, dict):
                    continue
                excerpt = str(m.get("excerpt", "")).strip()
                category = str(m.get("category", "")).strip().lower()
                reasoning = str(m.get("reasoning", "")).strip()
                if not excerpt or not category:
                    continue
                # Resolve char-offsets ourselves — don't trust LLM-reported spans.
                idx = text.find(excerpt)
                span = [idx, idx + len(excerpt)] if idx >= 0 else None
                markings_out.append({
                    "span": span,  # None = excerpt not found verbatim (LLM paraphrased it)
                    "excerpt": excerpt[:300],
                    "category": category[:50],
                    "reasoning": reasoning[:400],
                })
            persisted = 0
            if persist and source_id and markings_out:
                try:
                    from src.wiki_v2.store import init_wiki_db, persist_category_evidence, delete_category_evidence
                    from mayring_core.memory.store import MEMORY_DB_PATH
                    wdb = init_wiki_db(MEMORY_DB_PATH.parent / "wiki_v2.db")
                    try:
                        delete_category_evidence(wdb, ws, source_id)
                        persisted = persist_category_evidence(
                            wdb, ws, source_id, markings_out,
                            task=task_str if task_str != "(kein Task angegeben)" else "",
                            chunk_id=chunk_id,
                        )
                    finally:
                        wdb.close()
                except Exception as exc:
                    import logging as _logging
                    _logging.getLogger(__name__).warning("category-evidence persist failed: %s", exc)

            return {
                "markings": markings_out,
                "model": _model("text"),
                "persisted": persisted,
                "workspace_id": ws,
            }
        except _json.JSONDecodeError as exc:
            return {"error": f"JSON parse fail: {exc}", "raw": raw[:200], "workspace_id": ws}
        except Exception as exc:
            return {"error": str(exc), "workspace_id": ws}

    @mcp.tool()
    def pi_category_evidence(
        category: str | None = None,
        source_id: str | None = None,
        limit: int = 100,
        workspace_id: str | None = None,
    ) -> dict:
        """Read Mayring category-evidence (from pi_mark_categories persist).

        Beantwortet "zeig mir alle Belege für Kategorie X" oder "welche
        Kategorien wurden in Source Y gefunden, mit welchem Beleg".

        Args:
            category: filter by category label (lowercase). None = all.
            source_id: filter by source. None = all.
            limit: max rows.
            workspace_id: tenant scope.

        Returns:
            {evidence: [{source_id, chunk_id, category, span, excerpt,
             reasoning, task, created_at}], count} oder {error}.
        """
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        try:
            from src.wiki_v2.store import init_wiki_db, get_category_evidence
            from mayring_core.memory.store import MEMORY_DB_PATH
            wdb = init_wiki_db(MEMORY_DB_PATH.parent / "wiki_v2.db")
            try:
                rows = get_category_evidence(
                    wdb, ws, category=category, source_id=source_id, limit=limit,
                )
            finally:
                wdb.close()
            return {"evidence": rows, "count": len(rows), "workspace_id": ws}
        except Exception as exc:
            return {"error": str(exc), "workspace_id": ws}

    @mcp.tool()
    def pi_judge_relevance(
        query: str,
        chunks: list[dict],
        workspace_id: str | None = None,
        timeout: float = 90.0,
    ) -> dict:
        """Score chunk relevance to a query (0..1) via local Pi-Agent.

        Replacement for the LLM-judge call in stop_hook + retrieval rerank.
        Returns score per chunk so the caller can re-rank / filter.

        Args:
            query: The user query / task description.
            chunks: [{chunk_id, text}, ...]. Max 20 chunks per call.
            workspace_id: Tenant scope.
            timeout: Pi-Agent call timeout.

        Returns:
            {scores: {chunk_id: 0..1}, model} oder {error}.
        """
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        if not query or not chunks:
            return {"error": "query and chunks required", "workspace_id": ws}

        chunks = chunks[:20]
        items = []
        for c in chunks:
            cid = c.get("chunk_id") or ""
            text = (c.get("text") or "")[:600]
            if cid and text:
                items.append({"id": cid, "text": text})

        if not items:
            return {"scores": {}, "workspace_id": ws}

        import json as _json
        sys_prompt = (
            "You are a relevance judge. For each chunk, decide how relevant it is "
            "to the query on a 0..1 scale:\n"
            "  0.0 = not relevant at all\n"
            "  0.3 = tangential\n"
            "  0.6 = relevant context\n"
            "  0.9 = primary source\n"
            "  1.0 = directly answers the query\n\n"
            'Output STRICT JSON only: {"scores":{"<chunk_id>":<0..1>,...}}\n'
            "No prose, no explanation. ALL chunks MUST appear in the output."
        )
        chunks_text = "\n\n".join(
            f"[{c['id']}]\n{c['text']}" for c in items
        )
        prompt = sys_prompt + f"\n\nQuery: {query}\n\nChunks:\n{chunks_text}"

        import httpx
        try:
            resp = httpx.post(
                f"{_OLLAMA_URL}/api/generate",
                json={
                    "model": _model("text"),
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 600},
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            data = _json.loads(raw)
            scores = {
                str(k): max(0.0, min(1.0, float(v)))
                for k, v in (data.get("scores") or {}).items()
            }
            return {"scores": scores, "model": _model("text"), "workspace_id": ws}
        except _json.JSONDecodeError as exc:
            return {"error": f"JSON parse fail: {exc}", "raw": raw[:200], "workspace_id": ws}
        except Exception as exc:
            return {"error": str(exc), "workspace_id": ws}

    @mcp.tool()
    def pi_summarize_for_memory(
        content: str,
        style: str = "paraphrase_generalize_reduce",
        workspace_id: str | None = None,
        timeout: float = 60.0,
    ) -> dict:
        """Mayring-style summarization for memory-ingestion.

        Three reductions in one call:
          1. Paraphrase — core statement without filler
          2. Generalize — meta-category (architecture / debug / config / ...)
          3. Reduce — one-sentence essence

        Suitable as the pre-ingest step for `mcp__claude_ai_Memory__ingest`.

        Args:
            content: Raw text to summarize (transcript, file content, etc).
            style: Reserved for future variants. Currently always uses the
                   3-step Mayring pipeline.
            workspace_id: Tenant scope.
            timeout: Pi-Agent call timeout.

        Returns:
            {paraphrase, generalize, reduce, suggested_source_id, model} oder
            {error}.
        """
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        if not content or len(content.strip()) < 30:
            return {"error": "content too short (<30 chars)", "workspace_id": ws}

        sys_prompt = (
            "You are a Mayring-style summarizer. Reduce the input to three forms:\n"
            "1. paraphrase — core statement without filler (≤2 sentences)\n"
            "2. generalize — pick ONE meta-category from: "
            "[architecture, debug, config, decision, session-memory, context]\n"
            "3. reduce — one-sentence essence (≤120 chars)\n\n"
            "Also propose suggested_source_id: <generalize>:<yyyy-mm-dd>-<short-slug>\n\n"
            "Output STRICT JSON only:\n"
            '{"paraphrase":"...","generalize":"...","reduce":"...","suggested_source_id":"..."}\n'
            "No prose, no markdown."
        )

        import httpx
        import json as _json
        from datetime import datetime as _dt
        date_str = _dt.utcnow().strftime("%Y-%m-%d")
        prompt = sys_prompt + f"\n\nDate (for source_id): {date_str}\n\nContent:\n{content[:6000]}"

        try:
            resp = httpx.post(
                f"{_OLLAMA_URL}/api/generate",
                json={
                    "model": _model("text"),
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 400},
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            data = _json.loads(raw)
            return {
                "paraphrase": data.get("paraphrase", ""),
                "generalize": data.get("generalize", ""),
                "reduce": data.get("reduce", ""),
                "suggested_source_id": data.get("suggested_source_id", ""),
                "model": _model("text"),
                "workspace_id": ws,
            }
        except _json.JSONDecodeError as exc:
            return {"error": f"JSON parse fail: {exc}", "raw": raw[:200], "workspace_id": ws}
        except Exception as exc:
            return {"error": str(exc), "workspace_id": ws}
