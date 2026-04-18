"""Ausführungslogik der MayringCoder-Pipeline.

Alle run_*-Funktionen sind hier. CLI-Einstiegspunkt (parse_args / main) wurde in src/cli.py ausgelagert.
"""

import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from src.analysis.aggregator import aggregate_findings
from src.analysis.analyzer import analyze_files, configure_training_log, overview_files
from src.analysis.cache import find_changed_files, init_db, mark_files_analyzed, reset_repo
from src.analysis.categorizer import (
    categorize_files,
    filter_excluded_files,
    load_codebook,
    load_exclude_patterns,
    load_mayringignore,
)
from src.model_router import ModelRouter
from src.config import (
    CACHE_DIR,
    CODEBOOK_PATH,
    DEFAULT_PROMPT,
    EMBEDDING_MODEL,
    MAX_FILES_PER_RUN,
    OVERVIEW_PROMPT,
    PROMPTS_DIR,
    REPORTS_DIR,
    repo_slug as _repo_slug,
    set_batch_delay,
    set_batch_size,
    set_max_chars_per_file,
)
from src.analysis.context import (
    _chroma_dir,
    index_overview_to_vectordb,
    load_overview_cache_raw,
    load_overview_context,
    query_similar_context,
    save_overview_context,
    set_max_context_chars,
)
from src.analysis.history import cleanup_runs, compare_runs, generate_run_id, list_runs, save_run
from src.model_selector import resolve_model
from src.analysis.exporter import export_results
from src.analysis.fetcher import fetch_repo
from src.analysis.report import generate_overview_report, generate_report
from src.analysis.splitter import split_into_files


def is_test_file(filename: str) -> bool:
    import re as _re
    patterns = [
        _re.compile(r"(?:^|/)(?:test[s]?[_\-].*|tests?|spec|__tests?__)", _re.IGNORECASE),
        _re.compile(r"_test\.(?:py|php|js|ts|go|java)$", _re.IGNORECASE),
        _re.compile(r"(?:^|/)(?:test)\.\w+$", _re.IGNORECASE),
    ]
    return any(p.search(filename) for p in patterns)


def load_prompt(path: Path | str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_turbulence_cache(repo_url: str) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    cache_path = CACHE_DIR / f"{_repo_slug(repo_url)}_turbulence.json"
    if not cache_path.exists():
        return None, None
    try:
        report = json.loads(cache_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None, None

    hot_zone_map: dict[str, str] = {}
    tier_map: dict[str, str] = {}

    for cf in report.get("all_files", report.get("critical_files", [])):
        path = cf.get("path", "")
        tier = cf.get("tier", "")
        tier_map[path] = tier

        zones = cf.get("hot_zones", [])
        if not zones:
            continue

        lines = ["## Hot-Zone-Kontext (aus Turbulenz-Analyse)"]
        for zone in zones:
            start = zone.get("start_line", "?")
            end = zone.get("end_line", "?")
            cats = zone.get("categories", [])
            peak = zone.get("peak_score", 0)
            cats_str = " × ".join(cats) if isinstance(cats, list) else str(cats)
            lines.append(
                f"ACHTUNG: Hot-Zone bei Zeile {start}-{end} "
                f"({cats_str}, Peak-Score: {peak:.0%})"
            )
            affected = zone.get("affected_functions", [])
            for fn_info in affected[:5]:
                if isinstance(fn_info, dict):
                    name = fn_info.get("name", "")
                    inputs = ", ".join(fn_info.get("inputs", []))
                    calls = ", ".join(fn_info.get("calls", []))
                    lines.append(f"  Betroffene Funktion: {name}({inputs}) → calls: {calls}")

        hot_zone_map[path] = "\n".join(lines)

    return hot_zone_map, tier_map


def run_turbulence(args, repo_url: str, ollama_url: str, turb_model: str, router: ModelRouter | None = None) -> None:
    from src.turbulence import analyze_repo, build_markdown

    print(f"\n{'='*60}")
    print("  Stufe 2: Turbulenz-Analyse")
    print(f"  Repo:    {repo_url}")
    print(f"  Modus:   {'LLM (' + turb_model + ')' if args.llm else 'Heuristik (schnell)'}")
    if args.use_overview_cache:
        print("  Overview-Cache: aktiv")
    print(f"{'='*60}")

    start = time.perf_counter()

    overview_cache = None
    if args.use_overview_cache:
        overview_cache = load_overview_cache_raw(repo_url)
        if overview_cache:
            print(f"  Overview-Cache geladen: {len(overview_cache)} Dateien")
        else:
            print("  Overview-Cache nicht gefunden — Fallback auf Standard-Kategorisierung")

    print(f"\nRepository laden: {repo_url} ...")
    _, _, content = fetch_repo(repo_url, os.getenv("GITHUB_TOKEN") or None)

    files = split_into_files(content)
    print(f"{len(files)} Dateien gefunden")

    if not files:
        print("Keine analysierbaren Dateien — Turbulenz-Analyse übersprungen.")
        return

    def _write_files(file_list: list[dict], target_dir: str) -> int:
        base = Path(target_dir)
        written = 0
        for f in file_list:
            p = base / f["filename"]
            p.parent.mkdir(parents=True, exist_ok=True)
            try:
                p.write_text(f["content"], encoding="utf-8")
                written += 1
            except OSError:
                pass
        return written

    with tempfile.TemporaryDirectory(prefix="turb_") as tmpdir:
        written = _write_files(files, tmpdir)
        print(f"{written} Dateien ins Arbeitsverzeichnis geschrieben\n")
        report = analyze_repo(
            tmpdir,
            use_llm=args.llm,
            model=turb_model if args.llm else None,
            overview_cache=overview_cache,
        )

    report["full_scan"] = args.full
    elapsed = time.perf_counter() - start

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")

    json_path = REPORTS_DIR / f"turbulence-{ts}.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    cache_path = CACHE_DIR / f"{_repo_slug(repo_url)}_turbulence.json"
    cache_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    md_path = REPORTS_DIR / f"turbulence-{ts}.md"
    md_path.write_text(
        build_markdown(report, repo_url, turb_model, elapsed, full_scan=args.full),
        encoding="utf-8",
    )

    print(f"\nReport (JSON): {json_path}")
    print(f"Report (MD):   {md_path}")
    print(f"Cache:         {cache_path}")
    print(f"Fertig in {elapsed:.0f}s")


def run_populate_memory(args, repo_url: str, ollama_url: str, model: str, router: ModelRouter | None = None) -> None:
    # Model-Auflösung über Router, falls verfügbar
    if router is not None and not model:
        if router.is_available("embedding"):
            model = router.resolve("embedding")

    from src.memory.ingest import get_or_create_chroma_collection, ingest
    from src.memory.store import init_memory_db
    from src.memory.schema import Source
    from src.gpu_metrics import start_monitoring, stop_monitoring, parse_metrics, format_summary
    import hashlib as _hashlib

    def _content_hash(text: str) -> str:
        return "sha256:" + _hashlib.sha256(text.encode("utf-8")).hexdigest()

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
                if not args.memory_categorize:
                    _opts["categorize"] = False
                codebook_profile = getattr(args, "codebook_profile", None)
                if codebook_profile:
                    _opts["codebook"] = codebook_profile
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


def run_ingest_issues(args, ollama_url: str, model: str, router: ModelRouter | None = None) -> None:
    # Model-Auflösung über Router, falls verfügbar
    if router is not None and not model:
        if router.is_available("embedding"):
            model = router.resolve("embedding")

    from src.memory.ingest import fetch_issues, issues_to_sources
    from src.memory.ingest import get_or_create_chroma_collection, ingest
    from src.memory.store import init_memory_db
    from src.gpu_metrics import start_monitoring, stop_monitoring, parse_metrics, format_summary

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
        from src.memory.store import deactivate_chunks_by_source, get_chunks_by_source
        from src.memory.retrieval import invalidate_query_cache
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


def run_ingest_images(args, ollama_url: str, model: str, router: ModelRouter | None = None) -> None:
    # Model-Auflösung über Router, falls verfügbar
    if router is not None and not model:
        if router.is_available("vision"):
            model = router.resolve("vision")

    from src.memory.ingest import run_image_ingest

    repo_url = args.ingest_images
    vision_model = getattr(args, "vision_model", "qwen2.5vl:3b")
    max_images = getattr(args, "max_images", 50)
    do_force = getattr(args, "force_reingest", False)

    print(f"[ingest-images] Starte Bild-Ingest für: {repo_url}")
    print(f"[ingest-images] Vision-Modell: {vision_model}, Max-Bilder: {max_images}")

    result = run_image_ingest(
        repo_url=repo_url,
        ollama_url=ollama_url,
        vision_model=vision_model,
        embed_model=model,
        max_images=max_images,
        force_reingest=do_force,
        workspace_id=getattr(args, "workspace_id", "default"),
    )

    print(
        f"\n[ingest-images] Fertig: {result['images_found']} Bilder total, "
        f"{result['images_captioned']} captioniert, "
        f"{result['images_skipped']} Dedup, "
        f"{result['images_failed']} Fehler."
    )


def run_pi_task(args, ollama_url: str, model: str, router: ModelRouter | None = None) -> None:
    # Model-Auflösung über Router, falls verfügbar
    if router is not None and not model:
        if router.is_available("analysis"):
            model = router.resolve("analysis")

    from src.agents.pi import run_task_with_memory

    task = args.pi_task
    repo_url = getattr(args, "repo", None)
    repo_slug_val = None
    if repo_url:
        repo_slug_val = _repo_slug(repo_url)

    print(f"[pi-task] Auftrag: {task[:80]}{'...' if len(task) > 80 else ''}")
    if repo_slug_val:
        print(f"[pi-task] Memory-Scope: {repo_slug_val}")
    print()

    result = run_task_with_memory(
        task=task,
        ollama_url=ollama_url,
        model=model,
        repo_slug=repo_slug_val,
    )

    print(result)


def run_analysis(args, repo_url: str, ollama_url: str, model: str, start_time: float, router: ModelRouter | None = None) -> None:
    """Haupt-Analyse-Pipeline: fetch → split → filter → categorize → diff → analyze → report."""
    # Model-Auflösung über Router, falls verfügbar
    if router is not None and not model:
        if router.is_available("analysis"):
            model = router.resolve("analysis")

    token = os.getenv("GITHUB_TOKEN") or None
    prompt_path = Path(args.prompt) if args.prompt else None
    from src.config import DEFAULT_PROMPT
    if prompt_path is None:
        prompt_path = DEFAULT_PROMPT
    codebook_path = Path(args.codebook) if args.codebook else CODEBOOK_PATH
    cache_run_key = args.run_id or (model if args.cache_by_model else "default")

    _modular_codebook: list[dict] | None = None
    _modular_exclude_pats: list[str] | None = None
    if getattr(args, "codebook_profile", None):
        from src.analysis.categorizer import load_codebook_modular
        _modular_exclude_pats, _modular_codebook = load_codebook_modular(args.codebook_profile)

    print(f"Repository laden: {repo_url} ...")
    summary, tree, content = fetch_repo(repo_url, token)

    if args.debug:
        from urllib.parse import urlparse
        import re as _re
        parsed = urlparse(repo_url)
        slug = parsed.path.strip("/").replace("/", "-").lower()
        slug = _re.sub(r"[^a-z0-9\-]", "", slug) or "repo"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        raw_path = CACHE_DIR / f"{slug}_raw_latest.txt"
        raw_path.write_text(content, encoding="utf-8")
        print(f"  [debug] Raw-Snapshot: {raw_path}")

    files = split_into_files(content)
    print(f"{len(files)} Dateien gefunden")

    if not files:
        print("Keine analysierbaren Dateien im Repository.")
        sys.exit(0)

    if _modular_exclude_pats is not None:
        exclude_pats = _modular_exclude_pats + load_mayringignore()
    else:
        exclude_pats = load_exclude_patterns(codebook_path) + load_mayringignore()
    files, excluded = filter_excluded_files(files, exclude_pats)
    if excluded:
        print(f"  ✗ {len(excluded)} Dateien ausgeschlossen (exclude patterns)")
    print(f"  → {len(files)} Dateien nach Filter")

    if not files:
        print("Alle Dateien wurden durch Exclude-Patterns herausgefiltert.")
        sys.exit(0)

    codebook = _modular_codebook if _modular_codebook is not None else load_codebook(codebook_path)
    files = categorize_files(files, codebook)
    categories = {f["filename"]: f["category"] for f in files}

    if args.full or args.no_limit:
        max_files = 0
    elif args.budget is not None:
        max_files = args.budget
    else:
        max_files = MAX_FILES_PER_RUN

    embedding_prefilter_meta: dict | None = None
    if args.embedding_prefilter:
        from src.analysis.context import filter_by_embedding

        embed_model = args.embedding_model or EMBEDDING_MODEL

        if args.embedding_query:
            embed_query = args.embedding_query
        else:
            prompt_label = prompt_path.stem.replace("_", " ")
            cb_label = codebook_path.stem.replace("_", " ")
            embed_query = f"{prompt_label} {cb_label} code quality analysis"

        print(
            f"\nEmbedding-Vorfilter aktiv"
            f" (Modell: {embed_model}, Top-K: {args.embedding_top_k}"
            + (f", Threshold: {args.embedding_threshold}" if args.embedding_threshold is not None else "")
            + f")"
        )
        print(f"  Query: {embed_query!r}")

        selected_by_embed, filtered_out_by_embed = filter_by_embedding(
            files=files,
            query=embed_query,
            ollama_url=ollama_url,
            top_k=args.embedding_top_k,
            threshold=args.embedding_threshold,
            embedding_model=embed_model,
            repo_url=repo_url,
        )

        n_before = len(files)
        files = [f for f in files if f["filename"] in set(selected_by_embed)]
        categories = {fn: cat for fn, cat in categories.items() if fn in set(selected_by_embed)}
        print(
            f"  → {len(files)} Dateien nach Embedding-Vorfilter"
            f" ({n_before - len(files)} herausgefiltert)"
        )

        embedding_prefilter_meta = {
            "model": embed_model,
            "top_k": args.embedding_top_k,
            "threshold": args.embedding_threshold,
            "query": embed_query,
            "files_before": n_before,
            "files_after": len(files),
            "filtered_out": filtered_out_by_embed,
        }

        if not files:
            print("Embedding-Vorfilter hat alle Dateien herausgefiltert. Abbruch.")
            sys.exit(0)

    if args.mode == "overview":
        _run_overview(args, repo_url, ollama_url, model, files, categories, codebook_path,
                      cache_run_key, start_time)
        return

    # Turbulenz-Modus wird direkt in checker.py vor run_analysis dispatched — hier nicht erreichbar.

    # Diff
    if args.full:
        filenames_to_check = [f["filename"] for f in files]
        diff: dict = {
            "changed": [], "added": filenames_to_check, "removed": [],
            "unchanged": [], "unanalyzed": filenames_to_check,
            "selected": filenames_to_check, "skipped": [],
            "snapshot_id": None,
        }
        conn = None
        print(f"--full (analyze): {len(filenames_to_check)} Dateien werden analysiert")
    else:
        conn = init_db(repo_url, workspace_id=getattr(args, "workspace_id", "default"))
        diff = find_changed_files(conn, repo_url, files, categories, max_files, run_key=cache_run_key)

        n_c, n_a, n_r, n_u = (
            len(diff["changed"]), len(diff["added"]),
            len(diff["removed"]), len(diff["unchanged"]),
        )
        n_queue = len(diff["unanalyzed"])
        print(f"{n_c} geändert, {n_a} neu, {n_r} gelöscht, {n_u} unverändert | {n_queue} in Analyse-Queue")

        filenames_to_check = diff["selected"]
        if diff.get("skipped"):
            print(
                f"  → {len(filenames_to_check)} ausgewählt"
                f" (Budget-Limit: {max_files}),"
                f" {len(diff['skipped'])} verbleiben auf nächste Runs"
            )

    if not filenames_to_check:
        elapsed = time.perf_counter() - start_time
        print(f"Keine Änderungen seit dem letzten Run. Fertig in {elapsed:.0f}s")
        if conn:
            conn.close()
        sys.exit(0)

    if args.show_selection or args.dry_run:
        print("\nDateien zur Analyse:")
        for fn in filenames_to_check:
            print(f"  • [{categories.get(fn, 'uncategorized')}] {fn}")
        if diff.get("skipped"):
            print(f"\nÜbersprungen ({len(diff['skipped'])}):")
            for fn in diff["skipped"]:
                print(f"  ○ {fn}")

    if args.dry_run:
        elapsed = time.perf_counter() - start_time
        print(f"\nFertig in {elapsed:.0f}s")
        sys.exit(0)

    # RAG-Kontext
    rag_context_fn = None
    project_context = None

    if _chroma_dir(repo_url).exists():
        def _rag_ctx(file: dict) -> str | None:
            query = f"[{file.get('category', '?')}] {file['filename']}"
            return query_similar_context(query, repo_url, ollama_url)
        rag_context_fn = _rag_ctx
        print("  RAG-Kontext aktiv (ChromaDB → Similarity Search)")
    else:
        project_context = load_overview_context(repo_url)
        if project_context:
            print("  Projektkontext aus Overview-Cache geladen (Phase 1 Fallback)")

    hot_zone_context_map: dict[str, str] | None = None
    if args.use_turbulence_cache:
        hot_zone_context_map, _turb_tiers = load_turbulence_cache(repo_url)
        if hot_zone_context_map is not None:
            n_hz = sum(1 for v in hot_zone_context_map.values() if v)
            print(f"  Hot-Zone-Kontext geladen: {n_hz} Dateien mit Hot-Zones")
            if _turb_tiers:
                stable_files = {fn for fn, tier in _turb_tiers.items() if tier == "stable"}
                before = len(filenames_to_check)
                filenames_to_check = [fn for fn in filenames_to_check if fn not in stable_files]
                skipped = before - len(filenames_to_check)
                if skipped:
                    print(f"  → {skipped} stabile Dateien übersprungen (tier=stable)")
        else:
            print("  Turbulence-Cache nicht gefunden — kein Hot-Zone-Kontext")

    test_prompt_path = PROMPTS_DIR / "test_inspector.md"
    use_test_prompt = test_prompt_path.exists()

    test_files, non_test_files = [], []
    for fn in filenames_to_check:
        if is_test_file(fn):
            test_files.append(fn)
        else:
            non_test_files.append(fn)

    if test_files and use_test_prompt:
        print(f"  → {len(test_files)} Test-Datei(en) werden mit test_inspector.md analysiert")

    results: list[dict] = []

    _use_pi = not getattr(args, "no_pi", False)
    _pi_repo_slug = _repo_slug(repo_url)
    if _use_pi:
        try:
            from src.memory.store import init_memory_db as _init_mem_db
            _mem_conn = _init_mem_db()
            _chunk_count = _mem_conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            _mem_conn.close()
            if _chunk_count == 0:
                print(
                    "  Warnung: Pi-Agent aktiv, aber Memory-DB enthält keine Chunks. "
                    "Starte zuvor --populate-memory, --ingest-issues oder --ingest-images."
                )
            else:
                print(f"  Pi-Agent aktiv: {_chunk_count} Memory-Chunks verfügbar")
        except Exception as _pi_check_exc:
            print(f"  Warnung: Pi-Agent Memory-Check fehlgeschlagen: {_pi_check_exc}")

    _time_budget_hit = False
    if non_test_files:
        print(f"\nAnalysiere {len(non_test_files)} Nicht-Test-Dateien mit {model} ...")
        batch_results, _time_budget_hit = analyze_files(
            files, non_test_files, prompt_path, ollama_url, model,
            project_context=project_context,
            context_fn=rag_context_fn,
            hot_zone_context_map=hot_zone_context_map,
            time_budget=args.time_budget,
            use_pi=_use_pi,
            pi_repo_slug=_pi_repo_slug,
        )
        results.extend(batch_results)

    if test_files and use_test_prompt and not _time_budget_hit:
        print(f"\nAnalysiere {len(test_files)} Test-Dateien mit {model} (test_inspector.md) ...")
        batch_results, _tbh = analyze_files(
            files, test_files, test_prompt_path, ollama_url, model,
            project_context=None,
            context_fn=None,
            time_budget=args.time_budget,
            use_pi=_use_pi,
            pi_repo_slug=_pi_repo_slug,
        )
        results.extend(batch_results)
        _time_budget_hit = _time_budget_hit or _tbh

    if conn is not None and diff.get("snapshot_id") is not None:
        analyzed_ok = [r["filename"] for r in results if "error" not in r]
        mark_files_analyzed(conn, diff["snapshot_id"], analyzed_ok, run_key=cache_run_key)
        remaining = len(diff["unanalyzed"]) - len(analyzed_ok)
        print(f"  → {len(analyzed_ok)} analysiert, {max(0, remaining)} verbleiben in Queue")
    if conn:
        conn.close()

    if args.rag_enrichment:
        from src.analysis.context import enrich_findings_with_rag
        if _chroma_dir(repo_url).exists():
            n_findings = sum(
                len(r.get("potential_smells", []) + r.get("codierungen", []))
                for r in results if "error" not in r
            )
            print(f"\nRAG-Enrichment: bereichere {n_findings} Findings ...", flush=True)
            results = enrich_findings_with_rag(results, repo_url, ollama_url)
            n_enriched = sum(
                1 for r in results if "error" not in r
                for s in (r.get("potential_smells", []) + r.get("codierungen", []))
                if s.get("_rag_context")
            )
            print(f"  → {n_enriched}/{n_findings} Findings mit RAG-Kontext angereichert")
        else:
            print("  RAG-Enrichment: Vektor-DB nicht gefunden (--mode overview zuerst ausführen)")

    adversarial_stats: dict | None = None
    if args.adversarial:
        print(f"\nAdvocatus Diaboli: prüfe {len(results)} Findings ...", flush=True)
        from src.analysis.extractor import validate_findings
        all_findings: list[dict] = []
        for r in results:
            if "error" in r:
                continue
            for smell in r.get("potential_smells", []):
                all_findings.append({**smell, "_filename": r["filename"]})
        if all_findings:
            validated, adversarial_stats = validate_findings(
                all_findings, results, ollama_url, model,
                min_confidence=args.min_confidence,
            )
            n_total = len(all_findings)
            n_rejected = adversarial_stats["rejected"]
            print(
                f"  Adversarial: {adversarial_stats['validated']}/{n_total} BESTÄTIGT, "
                f"{n_rejected} ABGELEHNT, "
                f"{adversarial_stats.get('below_confidence', 0)} unter Confidence-Schwelle"
            )
            rejected_filenames: set[str] = {f["_filename"] for f in validated}
            for r in results:
                if r["filename"] not in rejected_filenames:
                    r["potential_smells"] = [
                        s for s in r.get("potential_smells", [])
                        if s.get("_adversarial_verdict") != "ABGELEHNT"
                    ]

    second_opinion_model_name = (
        args.second_opinion
        or os.getenv("SECOND_OPINION_MODEL")
    )
    second_opinion_stats: dict | None = None
    if second_opinion_model_name:
        from src.analysis.extractor import second_opinion_validate
        all_findings_so: list[dict] = []
        for r in results:
            if "error" in r:
                continue
            for smell in r.get("potential_smells", []):
                all_findings_so.append({**smell, "_filename": r["filename"]})
        if all_findings_so:
            print(
                f"\nSecond Opinion ({second_opinion_model_name}):"
                f" prüfe {len(all_findings_so)} Findings ...",
                flush=True,
            )
            validated_so, second_opinion_stats = second_opinion_validate(
                all_findings_so, results, ollama_url, second_opinion_model_name
            )
            second_opinion_stats["model"] = second_opinion_model_name
            print(
                f"  Second Opinion: "
                f"{second_opinion_stats['confirmed']} BESTÄTIGT, "
                f"{second_opinion_stats['rejected']} ABGELEHNT, "
                f"{second_opinion_stats['refined']} PRÄZISIERT, "
                f"{second_opinion_stats['errors']} Fehler"
            )
            for r in results:
                r["potential_smells"] = [
                    s for s in r.get("potential_smells", [])
                    if s.get("_second_opinion_verdict") != "ABGELEHNT"
                ]

    min_conf = args.min_confidence
    aggregation = aggregate_findings(
        results,
        min_confidence=min_conf,
        adversarial_stats=adversarial_stats if args.adversarial else None,
        second_opinion_stats=second_opinion_stats,
    )

    elapsed = time.perf_counter() - start_time
    report_path = generate_report(
        repo_url, model, results, aggregation, diff, elapsed,
        run_id=cache_run_key,
        embedding_prefilter_meta=embedding_prefilter_meta,
        full_scan=args.full,
        time_budget_hit=_time_budget_hit,
        workspace_id=getattr(args, "workspace_id", "default"),
    )
    n_filtered = aggregation.get("_below_confidence_filtered", 0)
    print(f"\nReport: {report_path}")
    if n_filtered > 0:
        print(f"  → {n_filtered} Findings unter Confidence-Schwelle '{min_conf}' herausgefiltert")
    if args.adversarial_cost_report or args.adversarial:
        stats = aggregation.get("adversarial_stats", {})
        if stats:
            print(
                f"  Adversarial: {stats.get('validated', 0)} BESTÄTIGT, "
                f"{stats.get('rejected', 0)} ABGELEHNT, "
                f"{stats.get('errors', 0)} Fehler"
            )
    if second_opinion_stats:
        print(
            f"  Second Opinion: "
            f"{second_opinion_stats.get('confirmed', 0)} BESTÄTIGT, "
            f"{second_opinion_stats.get('rejected', 0)} ABGELEHNT, "
            f"{second_opinion_stats.get('refined', 0)} PRÄZISIERT"
        )
    if args.export:
        ep = export_results(results, args.export, codebook_path.name, "analyze")
        print(f"Export: {ep}")

    rid = args.run_id or generate_run_id()
    run_path = save_run(rid, repo_url, model, "analyze", results, diff, elapsed, aggregation,
                        extra={"time_budget_hit": _time_budget_hit},
                        workspace_id=getattr(args, "workspace_id", "default"))
    print(f"Run-History: {run_path.name}")
    print(f"Fertig in {elapsed:.0f}s")


def _run_overview(
    args, repo_url: str, ollama_url: str, model: str,
    files: list[dict], categories: dict, codebook_path: Path,
    cache_run_key: str, start_time: float,
    router: ModelRouter | None = None,
) -> None:
    # Model-Auflösung über Router, falls verfügbar
    if router is not None and not model:
        if router.is_available("overview"):
            model = router.resolve("overview")

    filenames_to_check = [f["filename"] for f in files]
    diff = {
        "changed": [], "added": filenames_to_check, "removed": [],
        "unchanged": [], "unanalyzed": filenames_to_check,
        "selected": filenames_to_check, "skipped": [],
        "snapshot_id": None,
    }

    if args.show_selection or args.dry_run:
        print("\nDateien zur Analyse:")
        for fn in filenames_to_check:
            print(f"  • [{categories.get(fn, 'uncategorized')}] {fn}")
    if args.dry_run:
        elapsed = time.perf_counter() - start_time
        print(f"\nFertig in {elapsed:.0f}s")
        sys.exit(0)

    print(f"\nErstelle Übersicht für {len(filenames_to_check)} Dateien mit {model} ...")
    results, _time_budget_hit = overview_files(
        files, filenames_to_check, OVERVIEW_PROMPT, ollama_url, model,
        time_budget=args.time_budget,
    )

    ctx_path = save_overview_context(results, repo_url)
    print(f"  Kontext gespeichert: {ctx_path}")

    try:
        n_indexed = index_overview_to_vectordb(repo_url, ollama_url)
        if n_indexed:
            print(f"  Vektor-DB indiziert: {n_indexed} Einträge ({EMBEDDING_MODEL})")
    except Exception as exc:
        print(f"  Vektor-DB Indexierung fehlgeschlagen (RAG deaktiviert): {exc}")

    elapsed = time.perf_counter() - start_time
    report_path = generate_overview_report(repo_url, model, results, diff, elapsed,
                                           run_id=cache_run_key, full_scan=args.full)
    print(f"\nReport: {report_path}")
    if args.export:
        ep = export_results(results, args.export, codebook_path.name, "overview")
        print(f"Export: {ep}")

    rid = args.run_id or generate_run_id()
    run_path = save_run(rid, repo_url, model, "overview", results, diff, elapsed,
                        workspace_id=getattr(args, "workspace_id", "default"))
    print(f"Run-History: {run_path.name}")
    print(f"Fertig in {elapsed:.0f}s")


if __name__ == "__main__":
    from src.cli import main
    main()
