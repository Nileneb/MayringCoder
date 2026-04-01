#!/usr/bin/env python3
"""RepoChecker — Lokales Code-Analyse-Tool mit gitingest + Ollama."""

import argparse
import hashlib
import os
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from gitingest import ingest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 48
MAX_CONTENT_LENGTH = 3000
OLLAMA_TIMEOUT = 120
BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
REPORTS_DIR = BASE_DIR / "reports"
DEFAULT_PROMPT = BASE_DIR / "prompts" / "smell_inspector.md"


# ---------------------------------------------------------------------------
# Repo fetching
# ---------------------------------------------------------------------------


def fetch_repo(repo_url: str, token: str | None = None) -> tuple[str, str, str]:
    kwargs = {"source": repo_url}
    if token:
        kwargs["token"] = token
    summary, tree, content = ingest(**kwargs)
    return summary, tree, content


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------

_FILE_BLOCK_RE = re.compile(
    rf"{re.escape(SEPARATOR)}\nFILE: (.+?)\n{re.escape(SEPARATOR)}\n(.*?)(?=\n{re.escape(SEPARATOR)}\n|$)",
    re.DOTALL,
)

SKIP_MARKERS = {"[Binary file]", "[Empty file]"}


def split_into_files(content: str) -> list[dict]:
    files = []
    for match in _FILE_BLOCK_RE.finditer(content):
        filename = match.group(1).strip()
        file_content = match.group(2).rstrip("\n")
        if file_content.strip() in SKIP_MARKERS:
            continue
        file_hash = hashlib.sha256(file_content.encode()).hexdigest()[:16]
        files.append({"filename": filename, "content": file_content, "hash": file_hash})
    return files


# ---------------------------------------------------------------------------
# SQLite Cache & Diff
# ---------------------------------------------------------------------------


def _repo_slug(repo_url: str) -> str:
    parsed = urlparse(repo_url)
    slug = parsed.path.strip("/").replace("/", "-").lower()
    slug = re.sub(r"[^a-z0-9\-]", "", slug)
    return slug or "repo"


def init_db(repo_url: str) -> sqlite3.Connection:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    db_path = CACHE_DIR / f"{_repo_slug(repo_url)}.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS file_hashes (
               filename TEXT PRIMARY KEY,
               hash     TEXT NOT NULL,
               updated_at TEXT NOT NULL
           )"""
    )
    conn.commit()
    return conn


def find_changed_files(
    conn: sqlite3.Connection, files: list[dict]
) -> dict[str, list[str]]:
    now = datetime.now().isoformat()

    old_rows = conn.execute("SELECT filename, hash FROM file_hashes").fetchall()
    old_map = {row[0]: row[1] for row in old_rows}

    new_map = {f["filename"]: f["hash"] for f in files}

    added = [fn for fn in new_map if fn not in old_map]
    removed = [fn for fn in old_map if fn not in new_map]
    changed = [fn for fn in new_map if fn in old_map and new_map[fn] != old_map[fn]]
    unchanged = [fn for fn in new_map if fn in old_map and new_map[fn] == old_map[fn]]

    # Persist new state
    conn.execute("DELETE FROM file_hashes")
    conn.executemany(
        "INSERT INTO file_hashes (filename, hash, updated_at) VALUES (?, ?, ?)",
        [(fn, h, now) for fn, h in new_map.items()],
    )
    conn.commit()

    return {"changed": changed, "added": added, "removed": removed, "unchanged": unchanged}


# ---------------------------------------------------------------------------
# LLM analysis
# ---------------------------------------------------------------------------


def _load_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8")


def analyze_file(
    filename: str,
    content: str,
    prompt_template: str,
    ollama_url: str,
    model: str,
) -> str:
    if len(content) > MAX_CONTENT_LENGTH:
        content = content[:MAX_CONTENT_LENGTH] + f"\n\n[... gekuerzt, Original {len(content)} Zeichen]"

    prompt = f"{prompt_template}\n\nDatei: {filename}\n```\n{content}\n```"

    try:
        resp = httpx.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["response"]
    except httpx.ConnectError:
        return f"FEHLER: Ollama nicht erreichbar unter {ollama_url}. Laeuft `ollama serve`?"
    except httpx.TimeoutException:
        return f"FEHLER: Timeout ({OLLAMA_TIMEOUT}s) bei Analyse von {filename}."
    except (httpx.HTTPStatusError, KeyError) as exc:
        return f"FEHLER: Ollama-Anfrage fehlgeschlagen: {exc}"


def analyze_files(
    files: list[dict],
    filenames_to_check: list[str],
    prompt_path: Path,
    ollama_url: str,
    model: str,
) -> list[dict]:
    prompt_template = _load_prompt(prompt_path)
    file_map = {f["filename"]: f["content"] for f in files}
    results = []
    total = len(filenames_to_check)

    for i, fn in enumerate(filenames_to_check, 1):
        print(f"  [{i}/{total}] {fn} ...", flush=True)
        response = analyze_file(fn, file_map[fn], prompt_template, ollama_url, model)
        results.append({"filename": fn, "analysis": response})

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def generate_report(
    repo_url: str,
    model: str,
    results: list[dict],
) -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    report_path = REPORTS_DIR / f"repo-check-{timestamp}.md"

    lines = [
        "---",
        f"repo: {repo_url}",
        f"date: {datetime.now().isoformat()}",
        f"model: {model}",
        f"files_checked: {len(results)}",
        "---",
        "",
        f"# RepoChecker Report — {timestamp}",
        "",
    ]

    for r in results:
        lines.append(f"## {r['filename']}")
        lines.append("")
        lines.append(r["analysis"])
        lines.append("")
        lines.append("---")
        lines.append("")

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    return str(report_path)


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RepoChecker — lokale Code-Analyse mit Ollama")
    p.add_argument("--repo", help="GitHub-Repo URL (ueberschreibt .env)")
    p.add_argument("--model", help="Ollama-Modell (ueberschreibt .env)")
    p.add_argument("--full", action="store_true", help="Cache ignorieren, alles analysieren")
    p.add_argument("--dry-run", action="store_true", help="Nur Diff zeigen, keine Analyse")
    p.add_argument("--prompt", help="Pfad zu einem alternativen Prompt")
    return p.parse_args()


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

    # --- Fetch ---
    print(f"Repository laden: {repo_url} ...")
    summary, tree, content = fetch_repo(repo_url, token)

    # --- Split ---
    files = split_into_files(content)
    print(f"{len(files)} Dateien gefunden")

    if not files:
        print("Keine analysierbaren Dateien im Repository.")
        sys.exit(0)

    # --- Diff ---
    if args.full:
        filenames_to_check = [f["filename"] for f in files]
        print(f"--full: Alle {len(filenames_to_check)} Dateien werden analysiert")
    else:
        conn = init_db(repo_url)
        diff = find_changed_files(conn, files)
        conn.close()

        n_changed = len(diff["changed"])
        n_added = len(diff["added"])
        n_removed = len(diff["removed"])
        n_unchanged = len(diff["unchanged"])
        print(
            f"{n_changed} geaendert, {n_added} neu, "
            f"{n_removed} geloescht, {n_unchanged} unveraendert"
        )

        filenames_to_check = diff["changed"] + diff["added"]

    if not filenames_to_check:
        elapsed = time.perf_counter() - start
        print(f"Keine Aenderungen seit dem letzten Run. Fertig in {elapsed:.0f}s")
        sys.exit(0)

    if args.dry_run:
        print("\n--dry-run: Dateien die analysiert wuerden:")
        for fn in filenames_to_check:
            print(f"  • {fn}")
        elapsed = time.perf_counter() - start
        print(f"\nFertig in {elapsed:.0f}s")
        sys.exit(0)

    # --- Analyse ---
    print(f"Analysiere {len(filenames_to_check)} Dateien mit {model} ...")
    results = analyze_files(files, filenames_to_check, prompt_path, ollama_url, model)

    # --- Report ---
    report_path = generate_report(repo_url, model, results)
    print(f"Report: {report_path}")

    elapsed = time.perf_counter() - start
    print(f"Fertig in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
