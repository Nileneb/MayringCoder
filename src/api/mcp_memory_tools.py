"""Memory tools registered onto the FastMCP instance."""

from __future__ import annotations

import os
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP

from src.api.mcp_auth import _enforce_tenant, _effective_workspace_id, _current_raw_jwt
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
            ws = _enforce_tenant(workspace_id)
            opts = {
                "repo": repo,
                "categories": categories,
                "source_type": source_type,
                "top_k": top_k,
                "include_text": include_text,
                "source_affinity": source_affinity,
                "workspace_id": ws,
                "task_context": task_context,
            }
            result = _run_search(query, _get_conn(), _get_chroma(), None,
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
            except Exception:
                pass  # non-critical; never block the search result
            return result
        except Exception as exc:
            return {"error": str(exc), "results": [], "prompt_context": ""}

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
                ollama_url = os.getenv("OLLAMA_URL", "http://three.linn.games:11434")
                model = os.getenv("MAYRING_MODEL", "qwen2.5-coder:7b")
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
                    ollama_url, model, {"categorize": True}, ws,
                )
                try:
                    httpx.post(f"{_api}/wiki/generate",
                               json={"workspace_id": ws}, headers=headers, timeout=5.0)
                    httpx.post(f"{_api}/ambient/snapshot",
                               json={"repo": ws}, headers=headers, timeout=5.0)
                except Exception:
                    pass
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
    def feedback(
        chunk_id: str,
        signal: str,
        metadata: dict | None = None,
    ) -> dict:
        """Record usage feedback for a memory chunk (training signal).

        Args:
            chunk_id: The chunk that was used
            signal: "positive" | "negative" | "1"–"5" (star rating)
            metadata: Optional context (e.g. {"query": "...", "task": "..."})

        Returns:
            {chunk_id, recorded: True}
        """
        try:
            import json as _json
            enriched = dict(metadata or {})
            conn = _get_conn()
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
            except Exception:
                pass
            add_feedback(conn, chunk_id, signal, enriched)
            invalidate_query_cache()
            return {"chunk_id": chunk_id, "recorded": True}
        except Exception as exc:
            return {"error": str(exc)}
