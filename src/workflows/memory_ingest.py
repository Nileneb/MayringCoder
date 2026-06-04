"""Memory-Population Workflow — Repo → Chunks → Memory-DB + ChromaDB."""
from __future__ import annotations

import hashlib as _hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from src.analysis.categorizer import (
    filter_excluded_files,
    load_codebook,
    load_exclude_patterns,
    load_mayringignore,
)
from mayring_core.identity.cli import resolve_cli_workspace
from src.analysis.fetcher import fetch_repo
from src.analysis.splitter import split_into_files
from mayring_core.config import CACHE_DIR, CODEBOOK_PATH, repo_slug as _repo_slug
from mayring_core.model_router import ModelRouter


def _content_hash(text: str) -> str:
    return "sha256:" + _hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ollama_ps(ollama_url: str) -> dict:
    import json as _json
    import urllib.request
    try:
        with urllib.request.urlopen(f"{ollama_url}/api/ps", timeout=3) as r:
            return _json.loads(r.read())
    except Exception:
        return {}


def run_populate_memory(args, repo_url: str, ollama_url: str, model: str, router: ModelRouter | None = None) -> None:
    if router is None:
        router = ModelRouter(ollama_url=ollama_url)

    _ingest_model = router.resolve_with_fallback("text") or model
    print(f"[populate-memory] Modell: {_ingest_model} (Task: text)")

    from src.gpu_metrics import format_summary, parse_metrics, start_monitoring, stop_monitoring
    from mayring_core.memory.ingest import get_or_create_chroma_collection, ingest
    from mayring_core.memory.schema import Source, canonicalize_url
    from mayring_core.memory.store import init_memory_db

    # Collapse case-only typo variants (Nileneb/X vs nileneb/X) into one
    # source_id namespace so vector + symbolic stages see the same chunks.
    repo_url = canonicalize_url(repo_url)

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
    try:
        _summary, _tree, _repo_content = fetch_repo(repo_url, token)
    except Exception as exc:
        # Domänen-saubere Behandlung: kein Stack-Trace im Job-Output
        # für nicht-existente / nicht-erreichbare Repos. Smoke-Tests
        # gegen 'smoke-nonexistent-test-repo' landen sonst als
        # 30-Zeilen-Crash in der job-history.
        from src.analysis.fetcher import RepoNotFoundError
        if isinstance(exc, RepoNotFoundError):
            print(f"[populate-memory] Repository nicht erreichbar: {exc}")
            return  # Job endet sauber mit info-message statt RuntimeError
        raise

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
    print(f"[STAGE] fetch_repo done files={len(files) + len(excluded)}")
    print(f"[STAGE] categorize done included={len(files)} excluded={len(excluded)}")

    if not files:
        print("[populate-memory] Keine Dateien nach Filter — Abbruch.")
        return

    conn = init_memory_db()
    chroma = get_or_create_chroma_collection()

    total = len(files)
    _budget = getattr(args, "budget", None)
    if isinstance(_budget, int) and _budget < total:
        print(f"[populate-memory] Budget: {_budget} von {total} Dateien werden verarbeitet.")
        files = files[:_budget]
        total = _budget

    _ps = _ollama_ps(ollama_url)
    for _m in _ps.get("models", []):
        print(f"[GPU] {_m['name']} in VRAM: {_m.get('size_vram', 0) // 1024**2} MB")

    ok_count = 0
    error_count = 0
    dedup_count = 0

    do_force = bool(getattr(args, "force_reingest", False))
    if do_force:
        print("[populate-memory] --force-reingest: alte Chunks werden invalidiert "
              "und Kategorisierung/Embedding läuft komplett neu.")
        from mayring_core.memory.retrieval import invalidate_query_cache
        from mayring_core.memory.store import deactivate_chunks_by_source, get_chunks_by_source
        old_chunk_ids: list[str] = []
        for f in files:
            src_id = Source.make_id(repo_url, f['filename'])
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
        from tqdm import tqdm as _tqdm
    except ImportError:  # pragma: no cover
        def _tqdm(it, **_kw):
            return it

    _workers = max(1, int(getattr(args, "workers", 1) or 1))
    _delay = float(getattr(args, "batch_delay", None) or 0)
    _workspace = resolve_cli_workspace(args)
    _do_categorize = getattr(args, "memory_categorize", True) is not False
    _codebook_profile = getattr(args, "codebook_profile", None)

    # WHY(C3 project-link producer-A populate): resolve project_id once before
    # the loop — same for every file in this populate run (1 repo → 1 project).
    # Fail-soft: if no project exists, _project_id stays None and linking is skipped.
    _canon_url = repo_url  # already canonicalized above
    _project_id: str | None = None
    try:
        from mayring_core.memory.schema import canonicalize_url as _canon_fn
        from mayring_core.memory.store import init_memory_db as _idb_proj
        _pconn = _idb_proj()
        try:
            _canon_ref = _canon_fn(_canon_url)
            _proj_row = _pconn.execute(
                "SELECT id FROM projects WHERE source_type='github' AND source_ref=?",
                (_canon_ref,),
            ).fetchone()
            if _proj_row:
                _project_id = _proj_row[0]
                print(f"[populate-memory] Project-Link: project_id={_project_id} ({_canon_ref})")
            else:
                print(f"[populate-memory] Project-Link: no project for {_canon_ref!r} — links skipped")
        finally:
            _pconn.close()
    except Exception as _proj_exc:
        import logging as _pl
        _pl.getLogger(__name__).warning("populate: project_id lookup failed: %s", _proj_exc)

    def _ingest_one(f: dict) -> dict:
        """Verarbeitet eine Datei — thread-safe (eigene DB-Connection pro Worker)."""
        from mayring_core.memory.store import init_memory_db
        _conn = init_memory_db()
        try:
            _src = Source(
                source_id=Source.make_id(repo_url, f['filename']),
                source_type="repo_file",
                repo=repo_url,
                path=f["filename"],
                content_hash=_content_hash(f["content"]),
                branch="",
                commit="",
            )
            _opts: dict = {}
            if not _do_categorize:
                _opts["categorize"] = False
            if _codebook_profile:
                _opts["codebook"] = _codebook_profile
            if do_force:
                _opts["force"] = True
            r = ingest(_src, f["content"], _conn, chroma, ollama_url, _ingest_model,
                       opts=_opts or None, workspace_id=_workspace)
            # WHY(C3 project-link producer-A): link every newly ingested chunk to
            # the project that owns this repo. Idempotent (INSERT OR IGNORE). Fail-soft.
            if _project_id:
                try:
                    from mayring_core.memory.store import get_chunks_by_source, link_chunk_to_project
                    from mayring_core.memory.schema import canonicalize_url as _cu
                    _src_id = Source.make_id(repo_url, f['filename'])
                    for _ch in get_chunks_by_source(_conn, _src_id, active_only=True):
                        link_chunk_to_project(
                            _conn, _ch.chunk_id, _project_id,
                            origin_ref=_cu(repo_url),
                            source="repo-analyze",
                            workspace_id=_workspace,
                        )
                except Exception as _le:
                    import logging as _ll
                    _ll.getLogger(__name__).warning(
                        "populate: chunk→project link failed for %r: %s", f["filename"], _le
                    )
            return {"ok": True, "deduped": r.get("deduped", 0)}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "file": f["filename"]}
        finally:
            _conn.close()

    print(f"[STAGE] ingest_loop start total={total} workers={_workers} delay={_delay}s")
    pbar = _tqdm(total=total, desc="populate-memory", unit="file")
    try:
        batch_size = _workers
        for batch_start in range(0, total, batch_size):
            batch = files[batch_start:batch_start + batch_size]
            if _workers == 1:
                for f in batch:
                    r = _ingest_one(f)
                    if r["ok"]:
                        ok_count += 1
                        dedup_count += r["deduped"]
                    else:
                        print(f"[populate-memory] FEHLER bei {r['file']!r}: {r['error']}")
                        error_count += 1
                    pbar.update(1)
            else:
                with ThreadPoolExecutor(max_workers=_workers) as pool:
                    futures = {pool.submit(_ingest_one, f): f for f in batch}
                    for fut in as_completed(futures):
                        r = fut.result()
                        if r["ok"]:
                            ok_count += 1
                            dedup_count += r["deduped"]
                        else:
                            print(f"[populate-memory] FEHLER bei {r['file']!r}: {r['error']}")
                            error_count += 1
                        pbar.update(1)
            if _delay > 0 and batch_start + batch_size < total:
                time.sleep(_delay)
    finally:
        pbar.close()
        print(f"[STAGE] ingest_loop done ok={ok_count} errors={error_count} dedup={dedup_count}")
        for _m in _ollama_ps(ollama_url).get("models", []):
            print(f"[GPU] Post-Run: {_m['name']} VRAM: {_m.get('size_vram', 0) // 1024**2} MB")
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

    try:
        from src.wiki_v2.watcher import on_post_ingest
        wid = resolve_cli_workspace(args)
        slug = _repo_slug(repo_url)
        for f in files:
            on_post_ingest(wid, slug, Source.make_id(repo_url, f['filename']), chroma=chroma)
    except Exception as _wiki_exc:
        import logging as _wlog
        _wlog.warning("wiki_hook failed: %s", _wiki_exc)
        print(f"[STAGE] wiki_hook_error detail={_wiki_exc}")
