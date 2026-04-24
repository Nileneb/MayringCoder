"""Haupt-Analyse-Workflow — fetch → split → filter → categorize → diff → analyze → report.

Enthält sowohl den vollen Analyse-Modus als auch den Overview-Modus (Helper
`_run_overview`, der nur innerhalb von run_analysis aufgerufen wird).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from src.analysis.aggregator import aggregate_findings
from src.analysis.analyzer import analyze_files, overview_files
from src.analysis.cache import find_changed_files, init_db, mark_files_analyzed
from src.analysis.categorizer import (
    categorize_files,
    filter_excluded_files,
    load_codebook,
    load_exclude_patterns,
    load_mayringignore,
)
from src.analysis.context import (
    _chroma_dir,
    index_overview_to_vectordb,
    load_overview_context,
    query_similar_context,
    save_overview_context,
)
from src.analysis.exporter import export_results
from src.analysis.fetcher import fetch_repo
from src.analysis.history import generate_run_id, save_run
from src.analysis.report import generate_overview_report, generate_report
from src.analysis.splitter import split_into_files
from src.config import (
    CACHE_DIR,
    CODEBOOK_PATH,
    DEFAULT_PROMPT,
    EMBEDDING_MODEL,
    MAX_FILES_PER_RUN,
    OVERVIEW_PROMPT,
    PROMPTS_DIR,
    repo_slug as _repo_slug,
)
from src.model_router import ModelRouter
from src.workflows._common import is_test_file, load_turbulence_cache


def run_analysis(
    args,
    repo_url: str,
    ollama_url: str,
    model: str,
    start_time: float,
    router: ModelRouter | None = None,
) -> None:
    """Haupt-Analyse-Pipeline."""
    if router is not None and not model:
        if router.is_available("analysis"):
            model = router.resolve("analysis")

    token = os.getenv("GITHUB_TOKEN") or None
    prompt_path = Path(args.prompt) if args.prompt else None
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
        _run_overview(
            args, repo_url, ollama_url, model, files, categories, codebook_path,
            cache_run_key, start_time,
        )
        return

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
    _turb_tiers: dict[str, str] | None = None
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

    # Load wiki context map for injection (skipped when --no-wiki-inject)
    _wiki_context_map: dict[str, str] = {}
    if not getattr(args, "no_wiki_inject", False):
        try:
            from src.config import CACHE_DIR
            from src.wiki_v2.graph import WikiGraph
            from src.wiki_v2.injection import WikiContextInjector
            _wid = getattr(args, "workspace_id", "default")
            _wg = WikiGraph(_wid, _pi_repo_slug, CACHE_DIR / "wiki_v2.db")
            if _wg.node_count() > 0:
                _inj = WikiContextInjector()
                for _fn in non_test_files:
                    _ctx = _inj.build_context(_fn, _wg)
                    if _ctx:
                        _wiki_context_map[_fn] = _ctx
                if _wiki_context_map:
                    print(f"  Wiki-Kontext aktiv: {len(_wiki_context_map)} Dateien angereichert")
            _wg.close()
        except Exception:
            pass

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
            wiki_context_map=_wiki_context_map,
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
    run_path = save_run(
        rid, repo_url, model, "analyze", results, diff, elapsed, aggregation,
        extra={"time_budget_hit": _time_budget_hit},
        workspace_id=getattr(args, "workspace_id", "default"),
    )
    print(f"Run-History: {run_path.name}")
    print(f"Fertig in {elapsed:.0f}s")

    try:
        from src.wiki_v2.watcher import on_post_analyze, on_post_finding
        wid = getattr(args, "workspace_id", "default")
        slug = _repo_slug(repo_url)
        for r in results:
            if "error" in r or not r.get("filename"):
                continue
            on_post_analyze(wid, slug, r["filename"],
                            turbulence_tier=(_turb_tiers or {}).get(r["filename"], ""))
            texts = []
            for item in r.get("potential_smells", []) + r.get("codierungen", []):
                for key in ("reasoning", "text", "description", "summary"):
                    if item.get(key):
                        texts.append(str(item[key]))
                        break
            if texts:
                on_post_finding(wid, slug, r["filename"], " ".join(texts))
    except Exception:
        pass


def _run_overview(
    args,
    repo_url: str,
    ollama_url: str,
    model: str,
    files: list[dict],
    categories: dict,
    codebook_path: Path,
    cache_run_key: str,
    start_time: float,
    router: ModelRouter | None = None,
) -> None:
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
    report_path = generate_overview_report(
        repo_url, model, results, diff, elapsed,
        run_id=cache_run_key, full_scan=args.full,
    )
    print(f"\nReport: {report_path}")
    if args.export:
        ep = export_results(results, args.export, codebook_path.name, "overview")
        print(f"Export: {ep}")

    rid = args.run_id or generate_run_id()
    run_path = save_run(
        rid, repo_url, model, "overview", results, diff, elapsed,
        workspace_id=getattr(args, "workspace_id", "default"),
    )
    print(f"Run-History: {run_path.name}")
    print(f"Fertig in {elapsed:.0f}s")
