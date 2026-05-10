#!/usr/bin/env python3
"""Diff-history für Architektur-Entscheidungen pro File.

User-Feature-Request (2026-05-10): "SETZ EINEN SCHEISS SUBAGENT FÜR EINE
DIFF-ZUSAMMENFASSUNG AUF EINZELNE SKRIPTE". Hintergrund: Architektur-
entscheidungen (z.B. "wir nutzen Langdock NICHT mehr") gehen verloren
wenn der Verlauf eines Files nur via `git log` betrachtet wird —
einzelne commits sehen lokal sinnvoll aus, aber die Trajektorie über
Wochen wird unsichtbar.

Dieses Tool:
1. nimmt einen file-pfad als argument
2. holt die letzten N commits via `git log --follow --patch`
3. sendet alle commits als ein Block an Ollama (qwen3.5:9b)
4. Prompt zwingt das Modell zu einer **Architektur-Trajektorie** —
   keine commit-message-list, sondern: "diese Datei wurde von X über Y
   nach Z migriert; davon obsolet: A, B; aktuell aktiv: C"

Usage:
    python tools/diff_history.py <file-path> [--commits 20] [--model qwen3.5:9b]

Output: Markdown — kann in PR-bodies, issue-comments, oder
docs/architecture-decisions/ eingefügt werden.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request


OLLAMA_URL = os.environ.get("OLLAMA_URL", "https://three.linn.games").rstrip("/")
DEFAULT_MODEL = os.environ.get("MAYRING_DIFF_MODEL", "qwen3.5:9b")
DEFAULT_COMMITS = 15


SYSTEM_PROMPT = """Du analysierst die Architektur-Trajektorie einer einzelnen Code-Datei.

Eingabe: eine chronologische Liste von Commits (jeweils Hash, Datum,
Subject, Diff). Letzter Commit zuerst.

Aufgabe: Schreibe eine kompakte (max 250 Wörter) Architektur-Zusammen-
fassung in 3 Abschnitten:

**Trajektorie:** woher kam die Datei (was war ihr ursprünglicher Zweck),
welche Phasen hat sie durchlaufen (z.B. "von Langdock-Integration zu
Claude-API-direkt", "von REST zu MCP-Streamable-HTTP"), wo steht sie
heute.

**Obsolete:** welche Konzepte/Funktionen wurden aus der Datei entfernt
und sollten nirgendwo wieder eingebaut werden (z.B. Langdock-Calls,
veraltete Routen, deprecated APIs). Mit commit-hash als Beleg.

**Aktiv:** welche Konzepte sind heute noch produktiv (mit kurzer
Begründung, warum). Hilft zukünftigen Reviewern zu verstehen was
"intended state" der Datei ist.

KEINE generic Phrasen. KEINE Wiederholung der commit-messages. Wenn
du eine Architektur-Entscheidung erkennst (z.B. Migration), nenn sie
explizit beim Namen.
"""


def fetch_history(file_path: str, n: int) -> str:
    """git log --follow --patch -n N -- file. Returns raw output."""
    if not os.path.exists(file_path):
        sys.exit(f"file not found: {file_path}")
    result = subprocess.run(
        ["git", "log", "--follow", "--patch", f"-n{n}",
         "--pretty=format:%n=== %h | %ad | %s ===%n", "--date=short", "--",
         file_path],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        sys.exit(f"git log failed: {result.stderr}")
    return result.stdout


def summarize(history: str, file_path: str, model: str) -> str:
    """POST history to ollama, return text summary."""
    body = {
        "model": model,
        "stream": False,
        "system": SYSTEM_PROMPT,
        "prompt": f"# File: {file_path}\n\n## Commits (chronologisch)\n\n{history[:30000]}",
        "options": {"num_predict": 600, "temperature": 0.2},
        "think": False,
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            payload = json.loads(r.read())
    except Exception as exc:
        sys.exit(f"ollama call failed: {exc}")
    return payload.get("response", "").strip() or "(empty response)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="path to file (must exist or have history)")
    ap.add_argument("--commits", type=int, default=DEFAULT_COMMITS)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    print(f"# Architektur-Trajektorie: {args.file}", flush=True)
    print(f"_letzte {args.commits} commits, model={args.model}_", flush=True)
    print("", flush=True)

    history = fetch_history(args.file, args.commits)
    if not history.strip():
        print("_keine git-history gefunden._")
        return 0

    summary = summarize(history, args.file, args.model)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
