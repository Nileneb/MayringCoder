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
from src.analyzer import analyze_files
from src.cache import find_changed_files, init_db, mark_files_analyzed
from src.categorizer import categorize_files, load_codebook
from src.config import CACHE_DIR, DEFAULT_PROMPT, MAX_FILES_PER_RUN
from src.fetcher import fetch_repo
from src.report import generate_report
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
    model = args.model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    token = os.getenv("GITHUB_TOKEN") or None
    prompt_path = Path(args.prompt) if args.prompt else DEFAULT_PROMPT

    if not repo_url:
        print("Fehler: Kein Repository angegeben. Nutze --repo oder setze GITHUB_REPO in .env")
        sys.exit(1)

    if not prompt_path.exists():
        print(f"Fehler: Prompt-Datei nicht gefunden: {prompt_path}")
        sys.exit(1)

    start = time.perf_counter()

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

    # 3. Categorize (Mayring Stufe 1: Strukturierung)
    codebook = load_codebook()
    files = categorize_files(files, codebook)
    categories = {f["filename"]: f["category"] for f in files}

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
    else:
        conn = init_db(repo_url)
        diff = find_changed_files(conn, repo_url, files, categories)
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
                f" (Budget-Limit: {MAX_FILES_PER_RUN}),"
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

    # 5. Analyze (Mayring Stufe 2: Reduktion + Stufe 3: Explikations-Markierung)
    print(f"\nAnalysiere {len(filenames_to_check)} Dateien mit {model} ...")
    results = analyze_files(files, filenames_to_check, prompt_path, ollama_url, model)

    # Mark successfully analyzed files so they leave the backlog queue
    if conn is not None and diff.get("snapshot_id") is not None:
        analyzed_ok = [r["filename"] for r in results if "error" not in r]
        mark_files_analyzed(conn, diff["snapshot_id"], analyzed_ok)
        remaining = len(diff["unanalyzed"]) - len(analyzed_ok)
        print(f"  → {len(analyzed_ok)} analysiert, {max(0, remaining)} verbleiben in Queue")
    conn and conn.close()

    # 6. Aggregate (Mayring Stufe 4: Zusammenführung)
    aggregation = aggregate_findings(results)

    # 7. Report
    elapsed = time.perf_counter() - start
    report_path = generate_report(repo_url, model, results, aggregation, diff, elapsed)
    print(f"\nReport: {report_path}")
    print(f"Fertig in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
