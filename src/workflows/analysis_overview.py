"""Overview-Modus für die Analyse-Pipeline."""
from __future__ import annotations

import sys
import time
from pathlib import Path

from src.analysis.analyzer import overview_files
from src.analysis.context import index_overview_to_vectordb, save_overview_context
from src.analysis.exporter import export_results
from src.analysis.history import generate_run_id, save_run
from src.analysis.report import generate_overview_report
from src.config import EMBEDDING_MODEL, OVERVIEW_PROMPT
from src.model_router import ModelRouter


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
