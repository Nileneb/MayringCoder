#!/usr/bin/env python3
"""CLI wrapper for src.agents.diff_history — Architektur-Trajektorie pro File.

Logik liegt in `src/agents/diff_history.py`, damit das MCP-Tool
`diff_history` (im memory-agents Plugin) und dieser CLI dieselbe
Implementierung teilen.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.diff_history import DiffHistoryError, run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="path to file (must exist + have history)")
    ap.add_argument("--commits", type=int, default=15)
    ap.add_argument("--model", default=None,
                    help="override Ollama model (default: ModelRouter -> text)")
    ap.add_argument("--ollama-url", default=None)
    args = ap.parse_args()

    try:
        result = run(
            args.file,
            commits=args.commits,
            model=args.model,
            ollama_url=args.ollama_url,
        )
    except DiffHistoryError as exc:
        sys.exit(str(exc))

    print(f"# Architektur-Trajektorie: {result['file']}")
    print(f"_letzte {result['commits']} commits, model={result['model']}_")
    print()
    if note := result.get("note"):
        print(f"_{note}_")
        return 0
    print(result["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
