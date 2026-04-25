"""Overview-Modus für die Analyse-Pipeline."""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

from src.analysis.analyzer import overview_files
from src.analysis.context import index_overview_to_vectordb, save_overview_context
from src.analysis.context import load_overview_cache_raw
from src.analysis.exporter import export_results
from src.analysis.history import generate_run_id, save_run
from src.analysis.report import generate_overview_report
from src.config import EMBEDDING_MODEL, OVERVIEW_PROMPT
from src.model_router import ModelRouter


def _content_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


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

    existing_cache = load_overview_cache_raw(repo_url) or {}

    # Only re-analyze files whose content changed since the last overview run.
    changed_files: list[dict] = []
    unchanged_filenames: list[str] = []
    for f in files:
        h = _content_hash(f["content"])
        cached = existing_cache.get(f["filename"])
        if cached and cached.get("content_hash") == h:
            unchanged_filenames.append(f["filename"])
        else:
            f["content_hash"] = h
            changed_files.append(f)

    filenames_to_check = [f["filename"] for f in changed_files]
    all_filenames = [f["filename"] for f in files]
    diff = {
        "changed": filenames_to_check, "added": filenames_to_check, "removed": [],
        "unchanged": unchanged_filenames, "unanalyzed": filenames_to_check,
        "selected": filenames_to_check, "skipped": unchanged_filenames,
        "snapshot_id": None,
    }

    if args.show_selection or args.dry_run:
        print("\nDateien zur Analyse:")
        for fn in all_filenames:
            marker = "~" if fn in unchanged_filenames else "+"
            print(f"  • {marker} [{categories.get(fn, 'uncategorized')}] {fn}")
    if args.dry_run:
        elapsed = time.perf_counter() - start_time
        print(f"\nFertig in {elapsed:.0f}s")
        sys.exit(0)

    print(f"\nÜbersicht: {len(changed_files)} geändert / {len(unchanged_filenames)} unverändert (übersprungen)")
    print(f"[STAGE] overview_start changed={len(changed_files)} unchanged={len(unchanged_filenames)} budget={args.time_budget}")

    results, _time_budget_hit = overview_files(
        changed_files, filenames_to_check, OVERVIEW_PROMPT, ollama_url, model,
        time_budget=args.time_budget,
    )
    print(f"[STAGE] overview_done analyzed={len(results)} skipped={len(unchanged_filenames)} budget_hit={_time_budget_hit}")

    ctx_path = save_overview_context(results, repo_url, existing=existing_cache)
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
