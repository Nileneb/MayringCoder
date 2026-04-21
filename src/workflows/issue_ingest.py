"""GitHub-Issues-Ingest Workflow — pulls issues via gh CLI, ingests to memory."""
from __future__ import annotations

from datetime import datetime

from src.config import CACHE_DIR, repo_slug as _repo_slug
from src.model_router import ModelRouter


def run_ingest_issues(args, ollama_url: str, model: str, router: ModelRouter | None = None) -> None:
    if router is not None and not model:
        if router.is_available("embedding"):
            model = router.resolve("embedding")

    from src.gpu_metrics import format_summary, parse_metrics, start_monitoring, stop_monitoring
    from src.memory.ingest import fetch_issues, get_or_create_chroma_collection, ingest, issues_to_sources
    from src.memory.store import init_memory_db

    issues_repo = args.ingest_issues
    state = getattr(args, "issues_state", "open")
    limit = getattr(args, "issues_limit", 100)
    do_multiview = getattr(args, "multiview", False)

    _gpu_proc = None
    _gpu_csv = None
    if getattr(args, "gpu_metrics", False):
        _gpu_csv = CACHE_DIR / f"gpu_metrics_issues_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        _gpu_proc = start_monitoring(_gpu_csv)
        if _gpu_proc:
            print(f"[ingest-issues] GPU-Monitoring aktiv → {_gpu_csv}")

    print(f"[ingest-issues] Issues laden von {issues_repo!r} (state={state}, limit={limit}) ...")
    issues = fetch_issues(issues_repo, state=state, limit=limit)
    if not issues:
        print("[ingest-issues] Keine Issues gefunden oder gh CLI nicht verfügbar.")
        return

    print(f"[ingest-issues] {len(issues)} Issues gefunden")
    sources = issues_to_sources(issues, issues_repo)

    do_force = getattr(args, "force_reingest", False)
    conn = init_memory_db()
    chroma = get_or_create_chroma_collection()
    ok_count = 0
    error_count = 0
    dedup_count = 0

    if do_force:
        from src.memory.retrieval import invalidate_query_cache
        from src.memory.store import deactivate_chunks_by_source, get_chunks_by_source
        print("[ingest-issues] --force-reingest: Bestehende Chunks werden invalidiert ...")
        old_chunk_ids: list[str] = []
        for source, _ in sources:
            old_chunks = get_chunks_by_source(conn, source.source_id, active_only=False)
            old_chunk_ids.extend(c.chunk_id for c in old_chunks)
            deactivate_chunks_by_source(conn, source.source_id)
        if old_chunk_ids and chroma is not None:
            try:
                chroma.delete(ids=old_chunk_ids)
            except Exception:
                pass
        invalidate_query_cache()

    try:
        for source, content in sources:
            try:
                result = ingest(
                    source,
                    content,
                    conn,
                    chroma,
                    ollama_url,
                    model,
                    opts={"multiview": do_multiview} if not do_multiview else None,
                    workspace_id=getattr(args, "workspace_id", "default"),
                )
                dedup_count += result.get("deduped", 0)
                ok_count += 1
            except Exception as exc:
                print(f"[ingest-issues] FEHLER bei Issue {source.path!r}: {exc}")
                error_count += 1
    finally:
        conn.close()

    if _gpu_proc:
        stop_monitoring(_gpu_proc)
        if _gpu_csv:
            gpu_summary = parse_metrics(_gpu_csv)
            print(f"[ingest-issues] {format_summary(gpu_summary)}")

    print(
        f"\n[ingest-issues] Fertig: {len(sources)} Issues total, "
        f"{ok_count} OK, {error_count} Fehler, {dedup_count} Dedup."
    )

    if ok_count > 0 and not getattr(args, "dry_run", False):
        print("\nKnowledge Graph aktualisieren …")
        try:
            from tools.generate_knowledge_graph import generate
            slug = _repo_slug(getattr(args, "ingest_issues", ""))
            generate(project_filter=slug)
        except Exception as _kg_exc:
            print(f"  Knowledge Graph Fehler: {_kg_exc}")
