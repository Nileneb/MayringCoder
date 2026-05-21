"""Memory tools registered onto the FastMCP instance."""

from __future__ import annotations

import os
import sqlite3
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP

from src.api.mcp_auth import (
    _enforce_tenant,
    _effective_workspace_id,
    _effective_user_id,
    _effective_org_ids,
    _effective_active_workspace_id,
    _effective_active_workspace_kind,
    _current_raw_jwt,
    resolve_write_visibility,
)
from src.api.dependencies import get_conn as _get_conn, get_chroma as _get_chroma
from src.api.memory_service import run_search as _run_search, run_ingest as _run_ingest
from src.memory.retrieval import invalidate_query_cache
from src.memory.store import (
    add_feedback,
    deactivate_chunks_by_source,
    log_ingestion_event,
)


def register_memory_tools(mcp: FastMCP) -> None:

    @mcp.tool()
    def search_memory(
        query: str,
        repo: str | None = None,
        categories: list[str] | None = None,
        source_type: str | None = None,
        top_k: int = 8,
        include_text: bool = True,
        source_affinity: str | None = None,
        char_budget: int = 6000,
        compacted: bool = False,
        workspace_id: str | None = None,
        task_context: str | None = None,
    ) -> dict:
        """Hybrid 4-stage memory search (scope filter → symbolic → vector → rerank).

        Args:
            query: Natural language search query
            repo: Filter by repository (e.g. "owner/name")
            categories: Filter by any of these Mayring category labels
            source_type: Filter by source type (e.g. "repo_file")
            top_k: Maximum number of results (default 8)
            include_text: Include chunk text in results (default True)
            source_affinity: source_id to boost in affinity scoring
            char_budget: Max chars for prompt_context output
            compacted: Set True after /compact to boost conversation_summary chunks
            workspace_id: Tenant namespace filter (None = no filter)
            task_context: Optional richer task description (e.g. plan/todo content).
                Used as additional symbolic-scoring signal and as PI-advisor prompt
                input for sharper relevance ranking. Recommended when invoking from
                a plan-driven workflow.

        Returns:
            {results: list[RetrievalRecord], prompt_context: str}
        """
        try:
            from src.memory.schema import canonicalize_url
            ws = _enforce_tenant(workspace_id)
            opts = {
                "repo": canonicalize_url(repo) if repo else repo,
                "categories": categories,
                "source_type": source_type,
                "top_k": top_k,
                "include_text": include_text,
                "source_affinity": source_affinity,
                "workspace_id": ws,
                # user_id + org_ids let visibility='user' / 'org' chunks
                # surface across workspaces of the same human user / orgs.
                "user_id": _effective_user_id(),
                "org_ids": _effective_org_ids(),
                "task_context": task_context,
            }
            result = _run_search(query, _get_conn(), _get_chroma(),
                                 os.getenv("OLLAMA_URL", "http://three.linn.games:11434"),
                                 opts, char_budget, session_compacted=compacted)
            try:
                import json as _json
                from datetime import datetime, timezone
                _ids = _json.dumps([r.get("chunk_id", "") for r in result.get("results", [])])
                _conn = _get_conn()
                _conn.execute(
                    "INSERT INTO context_feedback_log"
                    " (trigger_ids,context_text,was_referenced,led_to_retrieval,relevance_score,captured_at)"
                    " VALUES (?,?,0,0,0.0,?)",
                    (_ids, result.get("prompt_context", "")[:2000],
                     datetime.now(timezone.utc).isoformat()),
                )
                _conn.commit()
            except (sqlite3.OperationalError, sqlite3.IntegrityError) as log_exc:
                # WHY(v2-stufe2.1): das ist Telemetrie-Insert für IGIO,
                # niemals der hot-path. Schlucken nur konkreten DB-Klassen,
                # alles andere (z.B. unerwartetes JSON-Encoding) muss laut.
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "context_feedback_log insert skipped: %s", log_exc,
                )
            return result
        except Exception as exc:
            return {"error": str(exc), "results": [], "prompt_context": ""}

    @mcp.tool()
    def ingest_status(job_id: str, workspace_id: str | None = None) -> dict:
        """Poll the status of a repo-ingest job (returned by `ingest()` for repo URLs).

        Returns the same payload as GET /jobs/{job_id} but tenant-scoped via the
        MCP session JWT — no Bearer token wrangling. Use this to verify whether
        an ingest job is queued, running, done, or has errored.

        Args:
            job_id: The job_id from a prior `ingest(<repo_url>)` call.
            workspace_id: Tenant namespace (default: from JWT).

        Returns:
            {
              "job_id": str,
              "status": "queued" | "running" | "done" | "error",
              "started_at": ISO timestamp,
              "progress": {"current": int, "total": int} | None,
              "stages": {<stage_name>: {"detail": str, "ts": ISO}, ...},
              "v2_jobs": {label: child_job_id, ...},   # populate triggers wiki/ambient/etc
              "output": str (last 4kb on done/error)
            }
        """
        try:
            from src.api.job_queue import get_job as _get_job
            ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
            job = _get_job(job_id)
            if not job:
                return {"error": "job not found", "job_id": job_id}
            if job.get("workspace_id") != ws:
                return {"error": "job not found", "job_id": job_id}
            output = job.get("output") or ""
            return {
                "job_id": job_id,
                "status": job.get("status"),
                "started_at": job.get("started_at"),
                "progress": job.get("progress"),
                "stages": job.get("stages") or {},
                "v2_jobs": job.get("v2_jobs") or {},
                "output": output[-4096:] if output else "",
                "workspace_id": ws,
            }
        except Exception as exc:
            return {"error": str(exc), "job_id": job_id}

    @mcp.tool()
    def ingest_jobs(limit: int = 20, workspace_id: str | None = None) -> dict:
        """List recent ingest/analysis jobs for the caller's workspace.

        Returns the most recent jobs sorted by started_at desc. Use this when
        you launched multiple `ingest()` calls in parallel and want a single
        overview instead of polling each job_id individually.

        Args:
            limit: Max jobs to return (default 20, max 100).
            workspace_id: Tenant namespace (default: from JWT).

        Returns:
            {"jobs": [{job_id, status, started_at, progress, stages_summary}, ...],
             "total": int}
        """
        try:
            from src.api.job_queue import _JOBS
            ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
            n = max(1, min(int(limit), 100))
            mine = [
                {**j, "job_id": jid}
                for jid, j in _JOBS.items()
                if j.get("workspace_id") == ws
            ]
            mine.sort(key=lambda j: j.get("started_at") or "", reverse=True)
            top = mine[:n]
            jobs = []
            for j in top:
                stages = j.get("stages") or {}
                stages_summary = ", ".join(
                    f"{name}:{(s.get('ts') or '')[11:19]}"
                    for name, s in list(stages.items())[-3:]
                )
                jobs.append({
                    "job_id": j["job_id"],
                    "status": j.get("status"),
                    "started_at": j.get("started_at"),
                    "progress": j.get("progress"),
                    "stages_summary": stages_summary,
                    "v2_jobs": j.get("v2_jobs") or {},
                })
            return {"jobs": jobs, "total": len(mine), "workspace_id": ws}
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def invalidate(source_id: str) -> dict:
        """Deactivate all memory chunks for a source (use when source is deleted/outdated).

        Returns:
            {source_id, deactivated_count}
        """
        try:
            count = deactivate_chunks_by_source(_get_conn(), source_id)
            log_ingestion_event(_get_conn(), source_id, "invalidated", {"count": count})
            invalidate_query_cache()
            return {"source_id": source_id, "deactivated_count": count}
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
        wiki update → ambient snapshot. Dedup via content hash — unchanged
        content is skipped automatically.

        source_type auto-detection (when "auto"):
        - GitHub/GitLab URL or git@ → repo pipeline (async, returns job_id)
        - Everything else → text pipeline (sync, returns chunk info)

        Args:
            source:      Repo URL (e.g. "https://github.com/nileneb/MayringCoder")
                         or text content (conversation summary, file content, insight)
            source_type: "auto" | "repo" | "text" | "conversation_summary" |
                         "session_knowledge" | "note" | "paper" (default: "auto")
            source_id:   Dedup key for text sources (e.g. "session-memory:2026-04-25-topic")
            workspace_id: Tenant namespace (default: from JWT)

        Returns:
            repo: {job_id, status, repo, workspace_id}
            text: {source_id, chunk_ids, workspace_id}
        """
        import hashlib
        import httpx
        from src.memory.schema import canonicalize_url

        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        _api = os.getenv("MAYRING_API_URL", "http://localhost:8090").rstrip("/")
        _jwt = _current_raw_jwt()
        headers = {"Authorization": f"Bearer {_jwt}"} if _jwt else {}

        parsed_source = urlparse(source)
        source_host = (parsed_source.hostname or "").lower()
        is_repo = source_type == "repo" or (
            source_type == "auto" and (
                source.startswith("git@")
                or (
                    parsed_source.scheme in {"http", "https"}
                    and source_host in {"github.com", "gitlab.com"}
                )
            )
        )

        if is_repo:
            try:
                resp = httpx.post(
                    f"{_api}/populate",
                    json={"repo": canonicalize_url(source)},
                    headers=headers,
                    timeout=30.0,
                )
                resp.raise_for_status()
                return {**resp.json(), "workspace_id": ws}
            except Exception as exc:
                return {"error": str(exc), "workspace_id": ws}
        else:
            try:
                ollama_url = os.getenv("OLLAMA_URL", "http://three.linn.games:11434")
                model = os.getenv("MAYRING_MODEL", "qwen2.5-coder:7b")
                sid = source_id or f"text:{ws}:{hashlib.sha256(source[:64].encode()).hexdigest()[:12]}"
                _stype = source_type if source_type not in ("auto", "text") else "knowledge"
                # Visibility follows the caller's ACTIVE app.linn.games
                # workspace: an org → 'org' (the whole team sees it), else the
                # per-user 'user' bucket (same human across claude.ai web vs
                # claude-cli) or 'private'. resolve_write_visibility re-checks
                # org membership so an active='org' claim can't write into a
                # foreign bucket.
                _vis, _org, _uid = resolve_write_visibility(
                    active_workspace_id=_effective_active_workspace_id(),
                    active_workspace_kind=_effective_active_workspace_kind(),
                    org_ids=_effective_org_ids(),
                    user_id=_effective_user_id(),
                )
                source_dict = {
                    "source_id": sid,
                    "source_type": _stype,
                    "repo": ws,
                    "path": sid,
                    "content_hash": "sha256:" + hashlib.sha256(source.encode()).hexdigest()[:16],
                    "visibility": _vis,
                    "user_id": _uid,
                    "org_id": _org,
                }
                result = _run_ingest(
                    source_dict, source, _get_conn(), _get_chroma(),
                    ollama_url, model, {"categorize": True}, ws,
                )
                # WHY(v2-stufe2.1): wiki/generate + ambient/snapshot sind
                # post-ingest-side-effects. Netzwerk/Timeout sind erwartbar
                # und blocken den Caller nicht — aber wir loggen statt schlucken.
                for endpoint, body in (
                    ("/wiki/generate", {"workspace_id": ws}),
                    ("/ambient/snapshot", {"repo": ws}),
                ):
                    try:
                        httpx.post(f"{_api}{endpoint}",
                                   json=body, headers=headers, timeout=5.0)
                    except (httpx.HTTPError, httpx.TimeoutException) as side_exc:
                        import logging as _logging
                        _logging.getLogger(__name__).warning(
                            "post-ingest %s skipped: %s", endpoint, side_exc,
                        )
                return {"source_id": sid, "workspace_id": ws, **result}
            except Exception as exc:
                return {"error": str(exc), "workspace_id": ws}

    @mcp.tool()
    def cleanup_hallucinated_categories(
        workspace_id: str | None = None,
        dry_run: bool = True,
        strict: bool = False,
    ) -> dict:
        """Remove implausible ``[neu]X`` category labels from chunks.

        Mistral and similar weak models hallucinate new categories during the
        hybrid-mode categorization. This tool strips obviously-bad ones (length,
        special chars, single-char dominance for length≥5). Chunks that lose
        all their valid labels are flagged with category_source='cleanup-pending'
        for next re-categorization run.

        Args:
            workspace_id: Limit to one workspace (default: all)
            dry_run: Preview only, no DB writes (default: True for safety)
            strict: Remove ALL [neu]X labels regardless of validity (use after
                    a model upgrade for full re-categorization)

        Returns:
            {scanned, affected, labels_removed, marked_for_recategorize,
             dry_run, samples: [{chunk_id, before, after, removed}, ...]}
        """
        try:
            from tools.cleanup_hallucinated_categories import strip_neu_labels
            ws = _enforce_tenant(workspace_id)
            conn = _get_conn()

            sql = ("SELECT chunk_id, category_labels FROM chunks "
                   "WHERE category_labels LIKE '%[neu]%' AND is_active = 1")
            params: list = []
            if ws:
                sql += " AND workspace_id = ?"
                params.append(ws)

            rows = conn.execute(sql, params).fetchall()
            scanned = len(rows)
            affected = 0
            labels_removed = 0
            pending = 0
            samples: list[dict] = []

            for chunk_id, label_csv in rows:
                cleaned, removed = strip_neu_labels(label_csv or "", strict)
                if removed == 0:
                    continue
                affected += 1
                labels_removed += removed
                recategorize = not cleaned.strip()
                if recategorize:
                    pending += 1
                if len(samples) < 20:
                    samples.append({
                        "chunk_id": chunk_id,
                        "before": label_csv,
                        "after": cleaned,
                        "removed": removed,
                        "pending_recategorize": recategorize,
                    })
                if not dry_run:
                    if recategorize:
                        conn.execute(
                            "UPDATE chunks SET category_labels = ?, category_confidence = 0.0,"
                            " category_source = 'cleanup-pending' WHERE chunk_id = ?",
                            (cleaned, chunk_id),
                        )
                    else:
                        conn.execute(
                            "UPDATE chunks SET category_labels = ? WHERE chunk_id = ?",
                            (cleaned, chunk_id),
                        )

            if not dry_run:
                conn.commit()
                invalidate_query_cache()

            return {
                "scanned": scanned,
                "affected": affected,
                "labels_removed": labels_removed,
                "marked_for_recategorize": pending,
                "dry_run": dry_run,
                "strict": strict,
                "samples": samples,
            }
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def reduce_categories(
        workspace_id: str | None = None,
        source_id: str | None = None,
        threshold: int = 15,
        dry_run: bool = True,
    ) -> dict:
        """Mayring S7 — Workspace-weite induktive Reduktion.

        Konsolidiert semantisch äquivalente Labels über alle Chunks des Workspace
        (oder einer einzelnen Source). Der automatische S3-Injektion-Schritt in
        ingest() verhindert Label-Explosion proaktiv; dieses Tool macht den
        Backfill-Pass für bestehende Workspaces oder Edge Cases.

        dry_run=True (Default): zeigt das Mapping, schreibt nichts in die DB.
        """
        try:
            import os
            from src.memory.ingestion.categorization import reduce_categories as _reduce
            ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
            conn = _get_conn()

            if source_id:
                rows = conn.execute(
                    "SELECT chunk_id FROM chunks WHERE source_id = ? "
                    "AND workspace_id = ? AND is_active = 1",
                    (source_id, ws),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT chunk_id FROM chunks WHERE workspace_id = ? AND is_active = 1",
                    (ws,),
                ).fetchall()
            chunk_ids = [r[0] for r in rows]

            ollama_url = os.getenv("OLLAMA_URL", "http://three.linn.games:11434")
            model = os.getenv("MAYRING_MODEL", "qwen2.5-coder:7b")

            if dry_run:
                conn.execute("SAVEPOINT s7_dry")
                try:
                    result = _reduce(
                        chunk_ids=chunk_ids, conn=conn, chroma_collection=None,
                        ollama_url=ollama_url, model=model,
                        workspace_id=ws, threshold=threshold,
                    )
                finally:
                    conn.execute("ROLLBACK TO SAVEPOINT s7_dry")
                    conn.execute("RELEASE SAVEPOINT s7_dry")
                result["dry_run"] = True
                return result

            chroma = _get_chroma()
            result = _reduce(
                chunk_ids=chunk_ids, conn=conn, chroma_collection=chroma,
                ollama_url=ollama_url, model=model,
                workspace_id=ws, threshold=threshold,
            )
            result["dry_run"] = False
            if not result.get("skipped"):
                log_ingestion_event(conn, ws, "s7_reduction_mcp", {
                    "unique_before": result.get("unique_before", 0),
                    "unique_after":  result.get("unique_after", 0),
                    "chunks_updated": result.get("chunks_updated", 0),
                    "source_id": source_id,
                })
            return result
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def feedback(
        chunk_id: str,
        signal: str,
        metadata: dict | None = None,
    ) -> dict:
        """Record rating-feedback for a memory chunk (training signal).

        WHY(2026-05-10 rating-migration): binary positive/negative entfernt.
        Nur noch rating 1..5 als skalares signal.

        Args:
            chunk_id: The chunk that was used
            signal: rating '1'..'5'. 1=schadhaft / verwirrend, 2=irrelevant,
                3=neutral, 4=wichtig, 5=primärquelle der antwort.
            metadata: Optional context (e.g. {"query": "...", "task": "..."})

        Returns:
            {chunk_id, recorded: True}
        """
        try:
            enriched = dict(metadata or {})
            conn = _get_conn()
            # WHY(v2-stufe2.1): Anreichern mit query_context ist optional;
            # konkrete sqlite-Fehler werden geloggt, nicht geschluckt.
            try:
                row = conn.execute(
                    "SELECT id, context_text FROM context_feedback_log"
                    " WHERE json_extract(trigger_ids, '$') LIKE ?"
                    " ORDER BY id DESC LIMIT 1",
                    (f"%{chunk_id}%",),
                ).fetchone()
                if row:
                    log_id, context_text = row
                    if "query_context" not in enriched and context_text:
                        enriched["query_context"] = context_text[:500]
                    conn.execute(
                        "UPDATE context_feedback_log SET was_referenced=1 WHERE id=?",
                        (log_id,),
                    )
                    conn.commit()
            except sqlite3.OperationalError as log_exc:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "feedback enrich skipped: %s", log_exc,
                )
            add_feedback(conn, chunk_id, signal, enriched)
            invalidate_query_cache()
            # WHY(2026-05-11): /stats/summary cached die feedback-count 30s
            # → smoke `feedback_count_delta` würde 0 sehen. Cache busten.
            try:
                from src.api.server import bust_stats_cache
                bust_stats_cache()
            except Exception as exc:
                # Non-fatal (stats serve a stale count for ≤30s) but NOT
                # silent-swallowed — #255 / project rule "no silent errors".
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "mcp_memory_tools.feedback: stats-cache bust failed: %s", exc
                )
            return {"chunk_id": chunk_id, "recorded": True}
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def live_logs(
        service: str,
        since: str = "5m",
        grep: str | None = None,
        limit: int = 200,
    ) -> dict:
        """Tail docker-container logs without SSH — Phase B of Issue #213.

        Replaces manual ``ssh u-server docker logs ...`` calls for deploy-
        debug / smoke-fail / cloud-routing investigations.

        Args:
            service: api | pi | mcp | webui | nginx
            since:   1m | 5m | 15m | 30m | 1h | 6h | 24h (whitelist)
            grep:    case-insensitive substring filter (optional)
            limit:   max lines returned (1..500)

        Returns:
            {service, container, since, grep, total, truncated, lines:
             [{container, raw, level}]}

        Auth: admin scope required. Secret-redaction applied
        (JWT/Bearer/hex/password/conn-strs maskiert).

        Rate: 5 calls/min/admin.
        """
        from fastapi import HTTPException as _HTTPException
        from src.api.routes.admin_logs import admin_logs as _impl
        from src.api.jwt_auth import TokenInfo as _TokenInfo

        # The MCP session already validated the JWT — we synthesize a
        # TokenInfo with admin scope inferred from the same auth context
        # that registered this tool. Without a per-session token-info
        # here, fall through to the same _is_admin check by re-reading
        # the session header.
        try:
            from src.api.mcp_auth import _current_token_info
            info = _current_token_info()
            if info is None:
                return {"error": "no token info — MCP session not authenticated"}
        except Exception:
            return {"error": "no token info available — call via authenticated MCP session"}

        try:
            return _impl(
                service=service, since=since, grep=grep, limit=limit, info=info,
            )
        except _HTTPException as e:
            return {"error": str(e.detail), "status": e.status_code}
        except Exception as exc:
            return {"error": str(exc)}
