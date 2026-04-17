"""label.py — Training-Daten labeln für Fine-Tuning.

Liest den Training-Log (cache/<slug>_training_log.jsonl) und annotiert
jeden Eintrag mit einem Label: positive / negative / candidate / neutral.

Auto-Labeling-Regeln:
  parsed_ok=False                          → negative
  parsed_ok=True + findings_count=0        → neutral
  Kein matching Run-JSON                   → candidate
  Run-JSON + Second-Opinion BESTÄTIGT      → positive
  Run-JSON + Second-Opinion ABGELEHNT      → negative
  Run-JSON + gemischte Verdikten           → candidate

Verwendung:
  .venv/bin/python label.py --auto [--log PATH] [--output PATH]
  .venv/bin/python label.py --interactive [--log PATH] [--output PATH]
  .venv/bin/python label.py --stats [--log PATH]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import CACHE_DIR

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_training_log(path: Path) -> list[dict]:
    """Load all JSONL entries from a training log file."""
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


def save_labeled(entries: list[dict], path: Path) -> int:
    """Write labeled entries to a JSONL file. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    return len(entries)


# ---------------------------------------------------------------------------
# Run-JSON lookup
# ---------------------------------------------------------------------------

def _find_run_json(run_id: str, cache_dir: Path = CACHE_DIR) -> dict | None:
    """Search all cache/<slug>/runs/<run_id>.json for a matching run_id."""
    for candidate in cache_dir.glob(f"*/runs/{run_id}.json"):
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return None


def _second_opinion_verdicts_for_file(run: dict, filename: str) -> list[str]:
    """Extract all _second_opinion_verdict values for a given filename in a run."""
    verdicts = []
    for r in run.get("results", []):
        if r.get("filename") != filename:
            continue
        for smell in r.get("potential_smells", []) + r.get("codierungen", []):
            v = smell.get("_second_opinion_verdict")
            if v:
                verdicts.append(v)
    return verdicts


# ---------------------------------------------------------------------------
# Auto-Labeling
# ---------------------------------------------------------------------------

def auto_label(entries: list[dict], cache_dir: Path = CACHE_DIR) -> list[dict]:
    """Annotate each entry with a 'label' field based on heuristics and run data."""
    # Cache run JSONs to avoid re-reading disk for same run_id
    _run_cache: dict[str, dict | None] = {}

    labeled = []
    for entry in entries:
        e = dict(entry)  # copy

        # Rule 1: parse failure → always negative
        if not e.get("parsed_ok", True):
            e["label"] = "negative"
            e["label_source"] = "parse_failed"
            labeled.append(e)
            continue

        # Rule 2: no findings → neutral (valid but uninformative)
        if e.get("findings_count", 0) == 0:
            e["label"] = "neutral"
            e["label_source"] = "no_findings"
            labeled.append(e)
            continue

        # Rule 3: try to find matching run JSON for second-opinion enrichment
        run_id = e.get("run_id", "")
        if run_id not in _run_cache:
            _run_cache[run_id] = _find_run_json(run_id, cache_dir)
        run = _run_cache[run_id]

        if run is None:
            e["label"] = "candidate"
            e["label_source"] = "no_run_found"
            labeled.append(e)
            continue

        verdicts = _second_opinion_verdicts_for_file(run, e.get("label", ""))
        if not verdicts:
            e["label"] = "candidate"
            e["label_source"] = "no_second_opinion"
            labeled.append(e)
            continue

        # Rule 4: all verdicts ABGELEHNT → negative
        if all(v == "ABGELEHNT" for v in verdicts):
            e["label"] = "negative"
            e["label_source"] = "second_opinion_rejected"
        # Rule 5: at least one BESTÄTIGT or PRÄZISIERT, none ABGELEHNT → positive
        elif all(v in ("BESTÄTIGT", "PRÄZISIERT") for v in verdicts):
            e["label"] = "positive"
            e["label_source"] = "second_opinion_confirmed"
        # Rule 6: mixed → candidate (needs manual review)
        else:
            e["label"] = "candidate"
            e["label_source"] = "second_opinion_mixed"

        labeled.append(e)

    return labeled


# ---------------------------------------------------------------------------
# Interactive Labeling
# ---------------------------------------------------------------------------

def interactive_label(entries: list[dict], output_path: Path) -> list[dict]:
    """CLI for manually labeling 'candidate' entries.

    Shows prompt/response preview and asks for user input.
    Writes each decision immediately to output_path.
    """
    candidates = [e for e in entries if e.get("label") == "candidate"]
    non_candidates = [e for e in entries if e.get("label") != "candidate"]

    if not candidates:
        print("Keine 'candidate'-Einträge zum Labeln gefunden.")
        return entries

    print(f"\n{len(candidates)} Einträge zum manuellen Labeln.")
    print("Tasten: [P]ositiv  [N]egativ  [S]kip  [Q]uit\n")

    labeled_candidates = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write non-candidates directly
    with output_path.open("w", encoding="utf-8") as fh:
        for e in non_candidates:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

        for i, entry in enumerate(candidates, 1):
            print(f"─── [{i}/{len(candidates)}] {entry.get('label', '?')} ───")
            print(f"  Datei:   {entry.get('label', '?')}")
            print(f"  Modell:  {entry.get('model', '?')}")
            print(f"  Run-ID:  {entry.get('run_id', '?')}")
            print(f"  Typ:     {entry.get('call_type', '?')}")
            print(f"  Findings: {entry.get('findings_count', 0)}")
            prompt_preview = (entry.get("prompt") or "")[:200].replace("\n", " ")
            response_preview = (entry.get("raw_response") or "")[:200].replace("\n", " ")
            print(f"  Prompt:  {prompt_preview}…")
            print(f"  Antwort: {response_preview}…")
            print()

            while True:
                choice = input("  Label [P/N/S/Q]: ").strip().upper()
                if choice == "P":
                    entry["label"] = "positive"
                    entry["label_source"] = "manual"
                    break
                elif choice == "N":
                    entry["label"] = "negative"
                    entry["label_source"] = "manual"
                    break
                elif choice == "S":
                    break  # keep as candidate
                elif choice == "Q":
                    # Write remaining candidates as-is
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    for remaining in candidates[i:]:
                        fh.write(json.dumps(remaining, ensure_ascii=False) + "\n")
                    print(f"\nAbgebrochen nach {i} Einträgen.")
                    return non_candidates + labeled_candidates + candidates[i - 1:]
                else:
                    print("  Ungültige Eingabe. P=Positiv, N=Negativ, S=Skip, Q=Quit")
                    continue

            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            labeled_candidates.append(entry)
            print()

    return non_candidates + labeled_candidates


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(entries: list[dict]) -> None:
    """Print a summary of label distribution and per-model breakdown."""
    from collections import Counter

    labels = Counter(e.get("label", "unlabeled") for e in entries)
    models = Counter(e.get("model", "?") for e in entries)
    call_types = Counter(e.get("call_type", "?") for e in entries)

    print(f"\nGesamt: {len(entries)} Einträge\n")
    print("Labels:")
    for label, count in sorted(labels.items()):
        pct = count / len(entries) * 100 if entries else 0
        bar = "█" * int(pct / 5)
        print(f"  {label:<12} {count:>5}  {pct:5.1f}%  {bar}")

    print("\nModelle:")
    for model, count in models.most_common():
        print(f"  {model:<40} {count:>5}")

    print("\nCall-Types:")
    for ct, count in call_types.most_common():
        print(f"  {ct:<20} {count:>5}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_log_path() -> Path | None:
    """Try to find a training log in cache/."""
    candidates = sorted(CACHE_DIR.glob("*_training_log.jsonl"))
    return candidates[0] if candidates else None


def _interactive_label_categories(output_path: Path) -> None:
    """Show uncategorized chunks from memory DB and ask for manual category assignment."""
    from src.memory_store import init_memory_db

    conn = init_memory_db()
    rows = conn.execute(
        "SELECT chunk_id, text, source_id, source_type "
        "FROM chunks WHERE is_active = 1 AND (category_labels = '' OR category_labels IS NULL) "
        "LIMIT 100"
    ).fetchall()

    if not rows:
        print("Keine unkategorisierten Chunks gefunden.")
        conn.close()
        return

    print(f"\n{len(rows)} unkategorisierte Chunks. Kategorien eingeben (comma-separated) oder leer lassen zum Überspringen.\n")
    print("Tasten: Kategorie(n) eingeben → Enter  |  leer → Skip  |  'q' → Quit\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled = 0

    with output_path.open("a", encoding="utf-8") as fh:
        for i, (chunk_id, text, source_id, source_type) in enumerate(rows, 1):
            print(f"─── [{i}/{len(rows)}] {source_id}")
            print(f"  Typ: {source_type}")
            print(f"  Text: {(text or '')[:300].replace(chr(10), ' ')}…")
            print()
            raw = input("  Kategorien: ").strip()
            if raw.lower() == "q":
                print(f"\nAbgebrochen nach {i - 1} Einträgen.")
                break
            if not raw:
                continue
            cats = [c.strip() for c in raw.split(",") if c.strip()]
            conn.execute(
                "UPDATE chunks SET category_labels = ?, category_source = 'manual', category_confidence = 1.0 "
                "WHERE chunk_id = ?",
                (",".join(cats), chunk_id),
            )
            conn.commit()
            fh.write(json.dumps({"chunk_id": chunk_id, "categories": cats, "source": "manual"}, ensure_ascii=False) + "\n")
            labeled += 1
            print(f"  → {cats}\n")

    conn.close()
    print(f"Fertig: {labeled} Chunks manuell kategorisiert → {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Training-Daten labeln")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--auto", action="store_true",
                      help="Automatisch labeln (Second-Opinion-Verdikten + Heuristik)")
    mode.add_argument("--interactive", action="store_true",
                      help="Manuell labeln (Terminal-UI für candidate-Einträge)")
    mode.add_argument("--stats", action="store_true",
                      help="Statistik anzeigen")
    p.add_argument("--mode", choices=["findings", "categories"], default="findings",
                   help="Label-Modus: findings (Standard) oder categories (Memory-Chunks ohne Kategorie labeln)")
    p.add_argument("--log", type=Path, metavar="PATH",
                   help="Training-Log JSONL (Standard: erster gefundener in cache/)")
    p.add_argument("--output", type=Path, metavar="PATH",
                   help="Ausgabe-JSONL (Standard: <log>_labeled.jsonl)")
    p.add_argument("--model", metavar="M", help="Nur Einträge für dieses Modell verarbeiten")
    p.add_argument("--run-id", metavar="R", help="Nur Einträge für diese Run-ID verarbeiten")
    p.add_argument("--call-type", metavar="T",
                   help="Nur Einträge für diesen Call-Typ (analyze/overview/extract)")
    args = p.parse_args()

    if args.mode == "categories":
        from datetime import datetime as _dt
        ts = _dt.now().strftime("%Y-%m-%d_%H%M")
        out = args.output or (CACHE_DIR / f"category_labels_{ts}.jsonl")
        _interactive_label_categories(out)
        return

    log_path = args.log or _default_log_path()
    if log_path is None or not log_path.exists():
        print(f"FEHLER: Kein Training-Log gefunden. Bitte --log PATH angeben.")
        sys.exit(1)

    entries = load_training_log(log_path)
    print(f"Geladen: {len(entries)} Einträge aus {log_path}")

    # Optional filters
    if args.model:
        entries = [e for e in entries if e.get("model") == args.model]
        print(f"  → {len(entries)} nach Modell-Filter '{args.model}'")
    if args.run_id:
        entries = [e for e in entries if e.get("run_id") == args.run_id]
        print(f"  → {len(entries)} nach Run-ID-Filter '{args.run_id}'")
    if args.call_type:
        entries = [e for e in entries if e.get("call_type") == args.call_type]
        print(f"  → {len(entries)} nach Call-Type-Filter '{args.call_type}'")

    if args.stats:
        print_stats(entries)
        return

    output_path = args.output or log_path.with_name(log_path.stem + "_labeled.jsonl")

    if args.auto:
        labeled = auto_label(entries)
        n = save_labeled(labeled, output_path)
        print(f"Geschrieben: {n} Einträge → {output_path}")
        print_stats(labeled)

    elif args.interactive:
        # Run auto-label first to pre-populate non-candidate labels
        entries = auto_label(entries)
        labeled = interactive_label(entries, output_path)
        print(f"Fertig. Datei: {output_path}")
        print_stats(labeled)


if __name__ == "__main__":
    main()
