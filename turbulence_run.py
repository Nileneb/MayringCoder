#!/usr/bin/env python3
"""
turbulence_run.py — Stufe 3 des MayringCoder-Pipelines
=======================================================

Holt das Repo via gitingest, schreibt die Dateien in ein
temporäres Verzeichnis und führt die Turbulenz-Analyse durch.
Der Report wird als JSON + Markdown in reports/ gespeichert.

Aufruf:
    python turbulence_run.py [--llm] [--repo URL]
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.fetcher import fetch_repo
from src.splitter import split_into_files
from src.turbulence_analyzer import analyze_repo
from src.turbulence_report import build_markdown
from src.config import REPORTS_DIR


def _write_files_to_tmpdir(files: list[dict], tmpdir: str) -> int:
    """Schreibt die gitingest-Dateien ins temporäre Verzeichnis."""
    base = Path(tmpdir)
    written = 0
    for f in files:
        target = base / f["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_text(f["content"], encoding="utf-8")
            written += 1
        except OSError:
            pass
    return written


def main() -> None:
    import argparse
    import time

    load_dotenv()

    parser = argparse.ArgumentParser(description="Turbulenz-Analyse (Stufe 3)")
    parser.add_argument("--repo",  default=None, help="GitHub-Repo URL (überschreibt .env)")
    parser.add_argument("--llm",   action="store_true", help="LLM-Modus (Ollama muss laufen)")
    args = parser.parse_args()

    repo_url = args.repo or os.getenv("GITHUB_REPO", "")
    if not repo_url:
        print("Fehler: Kein Repository angegeben. Nutze --repo oder setze GITHUB_REPO in .env")
        sys.exit(1)

    ollama_url  = os.getenv("OLLAMA_URL",  "http://localhost:11434")
    turb_model  = os.getenv("TURB_MODEL",  "mistral:7b-instruct")

    # Env-Vars für turbulence_analyzer.py sichtbar machen
    os.environ["OLLAMA_URL"]  = ollama_url
    os.environ["TURB_MODEL"]  = turb_model

    print(f"\n{'='*60}")
    print("  Stufe 3: Turbulenz-Analyse")
    print(f"  Repo:    {repo_url}")
    print(f"  Modus:   {'LLM (' + turb_model + ')' if args.llm else 'Heuristik (schnell)'}")
    print(f"{'='*60}")

    start = time.perf_counter()

    # 1. Repo holen
    print(f"\nRepository laden: {repo_url} ...")
    _, _, content = fetch_repo(repo_url, os.getenv("GITHUB_TOKEN") or None)

    # 2. In Dateien aufteilen
    files = split_into_files(content)
    print(f"{len(files)} Dateien gefunden")

    if not files:
        print("Keine analysierbaren Dateien — Turbulenz-Analyse übersprungen.")
        sys.exit(0)

    # 3. In temporäres Verzeichnis schreiben
    with tempfile.TemporaryDirectory(prefix="turb_") as tmpdir:
        written = _write_files_to_tmpdir(files, tmpdir)
        print(f"{written} Dateien ins Arbeitsverzeichnis geschrieben\n")

        # 4. Turbulenz-Analyse
        report = analyze_repo(tmpdir, use_llm=args.llm)

    elapsed = time.perf_counter() - start

    # 5. Report speichern
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")

    json_path = REPORTS_DIR / f"turbulence-{ts}.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    md_path = REPORTS_DIR / f"turbulence-{ts}.md"
    md_path.write_text(
        build_markdown(report, repo_url, turb_model, elapsed),
        encoding="utf-8",
    )

    print(f"\nReport (JSON): {json_path}")
    print(f"Report (MD):   {md_path}")
    print(f"Fertig in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
