"""export_training_data.py — Labeled Training-Daten in Fine-Tuning-Format exportieren.

Input:  cache/<slug>_training_log_labeled.jsonl  (Output von label.py --auto/--interactive)
Output: training_data/finetune_<date>.jsonl

Formate:
  raw      {"prompt": "...", "response": "..."}
  alpaca   {"instruction": "...", "input": "", "output": "..."}
  sharegpt {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
           (kompatibel mit Axolotl / Unsloth)

Verwendung:
  .venv/bin/python export_training_data.py [--log PATH] [--output PATH]
      [--format raw|alpaca|sharegpt] [--label positive|candidate|all]
      [--call-type analyze|overview|all]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import BASE_DIR, CACHE_DIR

TRAINING_DIR = BASE_DIR / "training_data"

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_labeled(path: Path) -> list[dict]:
    """Load all JSONL entries from a labeled training log."""
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _default_labeled_log() -> Path | None:
    """Find first labeled training log in cache/."""
    candidates = sorted(CACHE_DIR.glob("*_training_log_labeled.jsonl"))
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_entries(
    entries: list[dict],
    label: str = "positive",
    call_type: str = "analyze",
) -> list[dict]:
    """Filter entries by label and call_type.

    label="all" and call_type="all" disable the respective filter.
    """
    result = entries
    if label != "all":
        result = [e for e in result if e.get("label") == label]
    if call_type != "all":
        result = [e for e in result if e.get("call_type") == call_type]
    return result


# ---------------------------------------------------------------------------
# Formatierung
# ---------------------------------------------------------------------------

def format_entry(entry: dict, fmt: str) -> dict:
    """Convert a labeled training entry to the target fine-tuning format."""
    prompt = entry.get("prompt", "")
    response = entry.get("raw_response", "")

    if fmt == "raw":
        return {"prompt": prompt, "response": response}

    if fmt == "alpaca":
        # Split prompt into instruction + input if possible
        # Heuristic: everything before the first "Datei:" line is instruction
        parts = prompt.split("\nDatei:", 1)
        instruction = parts[0].strip()
        input_text = ("Datei:" + parts[1]).strip() if len(parts) > 1 else ""
        return {
            "instruction": instruction,
            "input": input_text,
            "output": response,
        }

    if fmt == "sharegpt":
        return {
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": response},
            ]
        }

    raise ValueError(f"Unbekanntes Format: '{fmt}'. Unterstützt: raw, alpaca, sharegpt")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(
    entries: list[dict],
    path: Path,
    fmt: str = "raw",
) -> int:
    """Write formatted entries to JSONL. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            try:
                formatted = format_entry(entry, fmt)
                fh.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                written += 1
            except (ValueError, KeyError):
                continue
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Labeled Training-Daten exportieren")
    p.add_argument("--log", type=Path, metavar="PATH",
                   help="Labeled Training-Log JSONL (Standard: erster gefundener in cache/)")
    p.add_argument("--output", type=Path, metavar="PATH",
                   help="Ausgabe-JSONL (Standard: training_data/finetune_<date>.jsonl)")
    p.add_argument("--format", choices=["raw", "alpaca", "sharegpt"], default="raw",
                   help="Fine-Tuning-Format (Standard: raw)")
    p.add_argument("--label", default="positive",
                   help="Label-Filter: positive|candidate|all (Standard: positive)")
    p.add_argument("--call-type", default="analyze",
                   help="Call-Type-Filter: analyze|overview|all (Standard: analyze)")
    p.add_argument("--stats", action="store_true",
                   help="Statistik vor dem Export anzeigen")
    args = p.parse_args()

    log_path = args.log or _default_labeled_log()
    if log_path is None or not log_path.exists():
        print("FEHLER: Kein labeled Training-Log gefunden. Bitte zuerst label.py --auto ausführen.")
        sys.exit(1)

    entries = load_labeled(log_path)
    print(f"Geladen: {len(entries)} Einträge aus {log_path}")

    filtered = filter_entries(entries, label=args.label, call_type=args.call_type)
    print(f"Gefiltert: {len(filtered)} Einträge (label={args.label}, call-type={args.call_type})")

    if not filtered:
        print("Keine Einträge nach Filter. Export übersprungen.")
        sys.exit(0)

    if args.stats:
        from collections import Counter
        models = Counter(e.get("model", "?") for e in filtered)
        print("\nModelle im Export:")
        for model, count in models.most_common():
            print(f"  {model:<40} {count:>5}")
        print()

    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    output_path = args.output or (TRAINING_DIR / f"finetune_{ts}.jsonl")

    n = export(filtered, output_path, fmt=args.format)
    print(f"Exportiert: {n} Einträge → {output_path}  (Format: {args.format})")


if __name__ == "__main__":
    main()
