#!/usr/bin/env python3
"""RepoChecker — Einstiegspunkt. Orchestriert die Mayring-Analyse-Pipeline."""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from src.aggregator import aggregate_findings
from src.analyzer import analyze_files, overview_files, set_max_chars_per_file as set_analyzer_max_chars
from src.cache import find_changed_files, init_db, mark_files_analyzed, reset_repo
from src.categorizer import (
    categorize_files,
    filter_excluded_files,
    load_codebook,
    load_exclude_patterns,
    load_mayringignore,
)
from src.config import CACHE_DIR, CODEBOOK_PATH, DEFAULT_PROMPT, EMBEDDING_MODEL, MAX_FILES_PER_RUN, OVERVIEW_PROMPT
from src.context import (
    index_overview_to_vectordb,
    load_overview_context,
    query_similar_context,
    save_overview_context,
    set_max_context_chars,
)
from src.exporter import export_results
from src.fetcher import fetch_repo
from src.history import cleanup_runs, compare_runs, generate_run_id, list_runs, save_run
from src.model_selector import resolve_model
from src.report import generate_report, generate_overview_report, set_max_chars_per_file as set_report_max_chars
from src.splitter import split_into_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RepoChecker — lokale Code-Analyse mit Ollama")
    p.add_argument("--repo", help="GitHub-Repo URL (überschreibt .env)")
    p.add_argument("--model", help="Ollama-Modell (überschreibt .env)")
    p.add_argument("--full", action="store_true", help="Cache ignorieren, alle Dateien analysieren")
    p.add_argument("--dry-run", action="store_true", help="Nur Diff + Selektion zeigen, keine Analyse")
    p.add_argument("--show-selection", action="store_true", help="Zeigt ausgewählte Dateien inkl. Kategorie")
    p.add_argument("--prompt", help="Pfad zu einem alternativen Prompt")
    p.add_argument("--debug", action="store_true", help="Speichert Raw-Snapshot lokal unter cache/")
    p.add_argument("--reset", action="store_true", help="Cache-DB für das Repo löschen (alle Analysen zurücksetzen)")
    p.add_argument("--mode", choices=["analyze", "overview"], default="analyze",
                   help="Modus: 'overview' = nur Funktions-Übersicht, 'analyze' = volle Fehlersuche (Standard)")
    p.add_argument("--no-limit", action="store_true", help="Kein Datei-Limit pro Lauf (alle Dateien verarbeiten)")
    p.add_argument("--max-chars", type=int, metavar="N", help="Zeichenlimit pro Datei überschreiben (Kontextlimit wird automatisch angepasst)")
    p.add_argument("--budget", type=int, metavar="N", help="Datei-Limit pro Lauf überschreiben (Standard: 20)")
    p.add_argument("--run-id", help="Logischer Run-Key für Cache + Report (ermöglicht Modell-/Run-Vergleiche)")
    p.add_argument("--cache-by-model", action="store_true", help="Modellnamen als Cache-Key verwenden (wenn kein --run-id gesetzt ist)")
    p.add_argument("--codebook", help="Pfad zu einem alternativen Codebook (YAML)")
    p.add_argument("--export", metavar="DATEI", help="Ergebnisse exportieren (.csv oder .json)")
    p.add_argument("--history", action="store_true", help="Vergangene Runs anzeigen")
    p.add_argument("--compare", nargs=2, metavar="RUN_ID", help="Zwei Runs vergleichen (alt neu)")
    p.add_argument("--cleanup", type=int, metavar="N", help="Nur die N neuesten Runs behalten, Rest löschen")
    p.add_argument("--resolve-model-only", action="store_true",
                   help="Gibt nur den aufgelösten Modellnamen aus und beendet (für Shell-Skripting)")
    return p.parse_args()


def _repo_slug_for_debug(repo_url: str) -> str:
    parsed = urlparse(repo_url)
    slug = parsed.path.strip("/").replace("/", "-").lower()
    return re.sub(r"[^a-z0-9\-]", "", slug) or "repo"


def main() -> None:
    load_dotenv()
    args = parse_args()

    repo_url = args.repo or os.getenv("GITHUB_REPO", "")
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    _env_model = (os.getenv("OLLAMA_MODEL") or "").strip() or None
    model = resolve_model(ollama_url, args.model, _env_model)

    # Shell-helper: print model name and exit (used by run.sh to avoid double-prompting).
    if args.resolve_model_only:
        print(model)
        sys.exit(0)

    token = os.getenv("GITHUB_TOKEN") or None
    prompt_path = Path(args.prompt) if args.prompt else DEFAULT_PROMPT
    codebook_path = Path(args.codebook) if args.codebook else CODEBOOK_PATH
    cache_run_key = args.run_id or (model if args.cache_by_model else "default")

    if args.max_chars is not None:
        if args.max_chars < 500:
            print("Fehler: --max-chars muss mindestens 500 sein.")
            sys.exit(1)

        # Keep one control variable: context budget follows per-file budget (2/3 ratio).
        max_chars_per_file = args.max_chars
        max_context_chars = max(500, (max_chars_per_file * 2) // 3)
        set_analyzer_max_chars(max_chars_per_file)
        set_report_max_chars(max_chars_per_file)
        set_max_context_chars(max_context_chars)
        print(
            f"Limits aktiv: Datei={max_chars_per_file} Zeichen, "
            f"Kontext={max_context_chars} Zeichen"
        )

    if not repo_url:
        print("Fehler: Kein Repository angegeben. Nutze --repo oder setze GITHUB_REPO in .env")
        sys.exit(1)

    # --reset: delete cache DB and exit
    if args.reset:
        removed = reset_repo(repo_url, run_key=(cache_run_key if args.run_id or args.cache_by_model else None))
        if removed:
            print(f"Cache gelöscht: {removed}")
            print("Nächster Lauf analysiert alle Dateien von vorn.")
        else:
            print("Kein Cache vorhanden — nichts zu löschen.")
        sys.exit(0)

    # --history: show past runs and exit
    if args.history:
        runs = list_runs(repo_url)
        if not runs:
            print("Keine Run-History vorhanden.")
        else:
            print(f"{'Run-ID':<20} {'Modus':<10} {'Modell':<18} {'Dateien':>7} {'Zeit (s)':>8}  Zeitstempel")
            print("-" * 90)
            for r in runs:
                print(
                    f"{r['run_id']:<20} {r['mode']:<10} {r['model']:<18} "
                    f"{r['files_checked']:>7} {r['timing_seconds']:>8.1f}  {r['timestamp']}"
                )
        sys.exit(0)

    # --compare: diff two runs and exit
    if args.compare:
        try:
            cmp = compare_runs(args.compare[0], args.compare[1], repo_url)
        except FileNotFoundError as exc:
            print(f"Fehler: {exc}")
            sys.exit(1)
        s = cmp["summary"]
        print(f"Vergleich: {cmp['run_a']} → {cmp['run_b']}")
        print(f"  Dateien: {s['files_a']} → {s['files_b']}")
        print(f"  Neu:      {s['new_count']}")
        print(f"  Behoben:  {s['resolved_count']}")
        print(f"  Severity: {s['severity_changed_count']} geändert")
        if cmp["new"]:
            print("\nNeue Findings:")
            for f in cmp["new"]:
                print(f"  + [{f.get('severity','?')}] {f['_filename']}: {f.get('type','')}")
        if cmp["resolved"]:
            print("\nBehobene Findings:")
            for f in cmp["resolved"]:
                print(f"  - [{f.get('severity','?')}] {f['_filename']}: {f.get('type','')}")
        if cmp["severity_changed"]:
            print("\nSeverity geändert:")
            for f in cmp["severity_changed"]:
                print(f"  ~ {f['_filename']}: {f.get('type','')} {f['_old_severity']} → {f['_new_severity']}")
        sys.exit(0)

    # --cleanup: keep only N newest runs and exit
    if args.cleanup is not None:
        deleted = cleanup_runs(repo_url, keep=args.cleanup)
        print(f"{deleted} alte Runs gelöscht, {args.cleanup} behalten.")
        sys.exit(0)

    if not prompt_path.exists():
        print(f"Fehler: Prompt-Datei nicht gefunden: {prompt_path}")
        sys.exit(1)

    # Codebook/Prompt-Kompatibilitätscheck
    _COMPAT_MAP = {
        "codebook_sozialforschung.yaml": {"mayring_deduktiv.md", "mayring_induktiv.md"},
        "codebook.yaml": {"file_inspector.md", "smell_inspector.md", "explainer.md"},
    }
    cb_name = codebook_path.name
    pr_name = prompt_path.name
    for cb_pattern, prompt_set in _COMPAT_MAP.items():
        if cb_name == cb_pattern and pr_name not in prompt_set:
            print(
                f"⚠ Warnung: Codebook '{cb_name}' passt üblicherweise zu"
                f" {', '.join(sorted(prompt_set))} — nicht zu '{pr_name}'."
                f" Ergebnisse könnten unbrauchbar sein."
            )

    start = time.perf_counter()

    if cache_run_key != "default":
        print(f"Cache-Run-Key aktiv: {cache_run_key}")

    # 1. Fetch
    print(f"Repository laden: {repo_url} ...")
    summary, tree, content = fetch_repo(repo_url, token)

    if args.debug:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        raw_path = CACHE_DIR / f"{_repo_slug_for_debug(repo_url)}_raw_latest.txt"
        raw_path.write_text(content, encoding="utf-8")
        print(f"  [debug] Raw-Snapshot: {raw_path}")

    # 2. Split
    files = split_into_files(content)
    print(f"{len(files)} Dateien gefunden")

    if not files:
        print("Keine analysierbaren Dateien im Repository.")
        sys.exit(0)

    # 2b. Exclude-Patterns anwenden (vor Kategorisierung)
    exclude_pats = load_exclude_patterns(codebook_path) + load_mayringignore()
    files, excluded = filter_excluded_files(files, exclude_pats)
    if excluded:
        print(f"  ✗ {len(excluded)} Dateien ausgeschlossen (exclude patterns)")
    print(f"  → {len(files)} Dateien nach Filter")

    if not files:
        print("Alle Dateien wurden durch Exclude-Patterns herausgefiltert.")
        sys.exit(0)

    # 3. Categorize (Mayring Stufe 1: Strukturierung)
    codebook = load_codebook(codebook_path)
    files = categorize_files(files, codebook)
    categories = {f["filename"]: f["category"] for f in files}

    # Determine effective file limit
    if args.no_limit:
        max_files = 0
    elif args.budget is not None:
        max_files = args.budget
    else:
        max_files = MAX_FILES_PER_RUN

    # 4. Diff
    if args.full:
        filenames_to_check = [f["filename"] for f in files]
        diff: dict = {
            "changed": [], "added": filenames_to_check, "removed": [],
            "unchanged": [], "unanalyzed": filenames_to_check,
            "selected": filenames_to_check, "skipped": [],
            "snapshot_id": None,
        }
        conn = None
        print(f"--full: {len(filenames_to_check)} Dateien werden analysiert")
    elif args.mode == "overview":
        # Overview processes ALL files, no cache interaction
        filenames_to_check = [f["filename"] for f in files]
        diff = {
            "changed": [], "added": filenames_to_check, "removed": [],
            "unchanged": [], "unanalyzed": filenames_to_check,
            "selected": filenames_to_check, "skipped": [],
            "snapshot_id": None,
        }
        conn = None
        print(f"--mode overview: {len(filenames_to_check)} Dateien kartieren (keine Fehlersuche)")
    else:
        conn = init_db(repo_url)
        diff = find_changed_files(conn, repo_url, files, categories, max_files, run_key=cache_run_key)
        # conn stays open — needed for mark_files_analyzed after analysis

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
        elapsed = time.perf_counter() - start
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
        elapsed = time.perf_counter() - start
        print(f"\nFertig in {elapsed:.0f}s")
        sys.exit(0)

    # 5. Run the selected mode
    if args.mode == "overview":
        overview_prompt = OVERVIEW_PROMPT
        print(f"\nErstelle Übersicht für {len(filenames_to_check)} Dateien mit {model} ...")
        results = overview_files(files, filenames_to_check, overview_prompt, ollama_url, model)

        # Save overview context for later analyze runs
        ctx_path = save_overview_context(results, repo_url)
        print(f"  Kontext gespeichert: {ctx_path}")

        # Index into vector DB for RAG (Phase 2)
        n_indexed = index_overview_to_vectordb(repo_url, ollama_url)
        if n_indexed:
            print(f"  Vektor-DB indiziert: {n_indexed} Einträge ({EMBEDDING_MODEL})")

        # 7. Overview Report (kein Aggregate nötig)
        elapsed = time.perf_counter() - start
        report_path = generate_overview_report(repo_url, model, results, diff, elapsed, run_id=cache_run_key)
        print(f"\nReport: {report_path}")
        if args.export:
            ep = export_results(results, args.export, codebook_path.name, "overview")
            print(f"Export: {ep}")

        # Save run history
        rid = args.run_id or generate_run_id()
        run_path = save_run(rid, repo_url, model, "overview", results, diff, elapsed)
        print(f"Run-History: {run_path.name}")
        print(f"Fertig in {elapsed:.0f}s")
    else:
        # 5. Analyze (Mayring Stufe 2: Reduktion + Stufe 3: Explikations-Markierung)
        # Try RAG context (Phase 2), fall back to flat overview context (Phase 1)
        rag_context_fn = None
        project_context = None

        # Check if vector DB exists for this repo
        from src.context import _chroma_dir
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

        print(f"\nAnalysiere {len(filenames_to_check)} Dateien mit {model} ...")
        results = analyze_files(
            files, filenames_to_check, prompt_path, ollama_url, model,
            project_context=project_context,
            context_fn=rag_context_fn,
        )

        # Mark successfully analyzed files so they leave the backlog queue
        if conn is not None and diff.get("snapshot_id") is not None:
            analyzed_ok = [r["filename"] for r in results if "error" not in r]
            mark_files_analyzed(conn, diff["snapshot_id"], analyzed_ok, run_key=cache_run_key)
            remaining = len(diff["unanalyzed"]) - len(analyzed_ok)
            print(f"  → {len(analyzed_ok)} analysiert, {max(0, remaining)} verbleiben in Queue")
        if conn:
            conn.close()

        # 6. Aggregate (Mayring Stufe 4: Zusammenführung)
        aggregation = aggregate_findings(results)

        # 7. Report
        elapsed = time.perf_counter() - start
        report_path = generate_report(repo_url, model, results, aggregation, diff, elapsed, run_id=cache_run_key)
        print(f"\nReport: {report_path}")
        if args.export:
            ep = export_results(results, args.export, codebook_path.name, "analyze")
            print(f"Export: {ep}")

        # Save run history
        rid = args.run_id or generate_run_id()
        run_path = save_run(rid, repo_url, model, "analyze", results, diff, elapsed, aggregation)
        print(f"Run-History: {run_path.name}")
        print(f"Fertig in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
