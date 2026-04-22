"""Memory-Population Workflow — Repo → Chunks → Memory-DB + ChromaDB."""
from __future__ import annotations

import hashlib as _hashlib
import os
from datetime import datetime
from pathlib import Path

from src.analysis.categorizer import (
    filter_excluded_files,
    load_codebook,
    load_exclude_patterns,
    load_mayringignore,
)
from src.analysis.fetcher import fetch_repo
from src.analysis.splitter import split_into_files
from src.config import CACHE_DIR, CODEBOOK_PATH, repo_slug as _repo_slug
from src.model_router import ModelRouter


def _content_hash(text: str) -> str:
    return "sha256:" + _hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_populate_memory(args, repo_url: str, ollama_url: str, model: str, router: ModelRouter | None = None) -> None:
    if router is not None and not model:
        if router.is_available("embedding"):
            model = router.resolve("embedding")

    from src.gpu_metrics import format_summary, parse_metrics, start_monitoring, stop_monitoring
    from src.memory.ingest import get_or_create_chroma_collection, ingest
    from src.memory.schema import Source
    from src.memory.store import init_memory_db

    token = os.getenv("GITHUB_TOKEN") or None
    codebook_path = Path(args.codebook) if getattr(args, "codebook", None) else CODEBOOK_PATH

    _gpu_proc = None
    _gpu_csv = None
    if getattr(args, "gpu_metrics", False):
        _gpu_csv = CACHE_DIR / f"gpu_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        _gpu_proc = start_monitoring(_gpu_csv)
        if _gpu_proc:
            print(f"[populate-memory] GPU-Monitoring aktiv → {_gpu_csv}")

    print(f"[populate-memory] Repository laden: {repo_url} ...")
    _summary, _tree, _repo_content = fetch_repo(repo_url, token)

    files = split_into_files(_repo_content)
    print(f"[populate-memory] {len(files)} Dateien gefunden")

    _profile = getattr(args, "codebook_profile", None)
    if not _profile and _tree:
        from src.analysis.categorizer import detect_profile_from_tree
        _profile = detect_profile_from_tree(_tree)
        print(f"[populate-memory] Profil auto-detected: {_profile}")

    if _profile:
        from src.analysis.categorizer import load_codebook_modular
        exclude_pats, codebook = load_codebook_modular(_profile)
        exclude_pats = list(exclude_pats) + load_mayringignore()
        print(f"[populate-memory] Codebook-Profil: {_profile} ({len(codebook)} Kategorien, {len(exclude_pats)} Excludes)")
    else:
        codebook = load_codebook(codebook_path)
        exclude_pats = load_exclude_patterns(codebook_path) + load_mayringignore()

    files, excluded = filter_excluded_files(files, exclude_pats)
    print(f"[populate-memory] {len(excluded)} Dateien ausgeschlossen, {len(files)} verbleiben")

    if not files:
        print("[populate-memory] Keine Dateien nach Filter — Abbruch.")
        return

    conn = init_memory_db()
    chroma = get_or_create_chroma_collection()

    total = len(files)
    ok_count = 0
    error_count = 0
    dedup_count = 0

    do_force = bool(getattr(args, "force_reingest", False))
    if do_force:
        print("[populate-memory] --force-reingest: alte Chunks werden invalidiert "
              "und Kategorisierung/Embedding läuft komplett neu.")
        from src.memory.retrieval import invalidate_query_cache
        from src.memory.store import deactivate_chunks_by_source, get_chunks_by_source
        old_chunk_ids: list[str] = []
        for f in files:
            src_id = f"repo:{repo_url}:{f['filename']}"
            old = get_chunks_by_source(conn, src_id, active_only=False)
            old_chunk_ids.extend(c.chunk_id for c in old)
            deactivate_chunks_by_source(conn, src_id)
        if old_chunk_ids and chroma is not None:
            try:
                chroma.delete(ids=old_chunk_ids)
            except Exception:
                pass
        invalidate_query_cache()
        print(f"[populate-memory] --force-reingest: {len(old_chunk_ids)} "
              f"alte Chunks invalidiert.")

    try:
        for f in files:
            content_hash = _content_hash(f["content"])
            source = Source(
                source_id=f"repo:{repo_url}:{f['filename']}",
                source_type="repo_file",
                repo=repo_url,
                path=f["filename"],
                content_hash=content_hash,
                branch="",
                commit="",
            )
            try:
                _opts: dict = {}
                # Mayring-Kategorisierung ist default an (siehe cli.py); Opt-Out
                # nur wenn ein Caller args.memory_categorize explizit auf False
                # setzt. Kein "silent off" mehr.
                if getattr(args, "memory_categorize", True) is False:
                    _opts["categorize"] = False
                codebook_profile = getattr(args, "codebook_profile", None)
                if codebook_profile:
                    _opts["codebook"] = codebook_profile
                if do_force:
                    _opts["force"] = True
                result = ingest(
                    source,
                    f["content"],
                    conn,
                    chroma,
                    ollama_url,
                    model,
                    opts=_opts or None,
                    workspace_id=getattr(args, "workspace_id", "default"),
                )
                dedup_count += result.get("deduped", 0)
                ok_count += 1
            except Exception as exc:
                print(f"[populate-memory] FEHLER bei {f['filename']!r}: {exc}")
                error_count += 1
    finally:
        conn.close()

    if _gpu_proc:
        stop_monitoring(_gpu_proc)
        if _gpu_csv:
            gpu_summary = parse_metrics(_gpu_csv)
            print(f"[populate-memory] {format_summary(gpu_summary)}")

    print(
        f"\n[populate-memory] Fertig: {total} Dateien total, "
        f"{ok_count} OK, {error_count} Fehler, {dedup_count} Dedup."
    )

    if ok_count > 0 and not getattr(args, "dry_run", False):
        print("\nKnowledge Graph aktualisieren …")
        try:
            from tools.generate_knowledge_graph import generate
            slug = _repo_slug(repo_url)
            generate(project_filter=slug)
        except Exception as _kg_exc:
            print(f"  Knowledge Graph Fehler: {_kg_exc}")
