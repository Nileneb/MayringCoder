"""Turbulence-Analyse Workflow — CLI-Wrapper um src/turbulence.py (die Engine).

Beachte: `src.workflows.turbulence` (dieses Modul, CLI-Glue) ist NICHT zu
verwechseln mit `src.turbulence` (der Analyse-Engine ~484 Zeilen).
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

from src.analysis.context import load_overview_cache_raw
from src.analysis.fetcher import fetch_repo
from src.analysis.splitter import split_into_files
from src.config import CACHE_DIR, REPORTS_DIR, repo_slug as _repo_slug
from src.model_router import ModelRouter


def run_turbulence(
    args,
    repo_url: str,
    ollama_url: str,
    turb_model: str,
    router: ModelRouter | None = None,
) -> None:
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
