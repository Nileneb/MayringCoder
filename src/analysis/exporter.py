"""Export analysis results to CSV or JSON for manual review / fine-tuning."""

import csv
import json
from pathlib import Path


def _flatten_results(
    results: list[dict],
    codebook_name: str,
    prompt_mode: str,
) -> list[dict]:
    """Flatten per-file results into one row per finding."""
    rows: list[dict] = []
    for r in results:
        if "error" in r:
            continue
        for smell in r.get("potential_smells", []):
            rows.append({
                "dateiname": r["filename"],
                "textstelle": smell.get("evidence_excerpt", ""),
                "kategorie": smell.get("type", ""),
                "severity": smell.get("severity", ""),
                "confidence": smell.get("confidence", ""),
                "line_hint": smell.get("line_hint", ""),
                "fix_suggestion": smell.get("fix_suggestion", ""),
                "codebook": codebook_name,
                "prompt_modus": prompt_mode,
                "bewertung": "",
            })
    return rows


_FIELDNAMES = [
    "dateiname",
    "textstelle",
    "kategorie",
    "severity",
    "confidence",
    "line_hint",
    "fix_suggestion",
    "codebook",
    "prompt_modus",
    "bewertung",
]


def export_results(
    results: list[dict],
    export_path: str | Path,
    codebook_name: str = "code.yaml",
    prompt_mode: str = "analyze",
) -> str:
    """Export findings to CSV or JSON (detected by file extension).

    Returns the resolved export path.
    """
    path = Path(export_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _flatten_results(results, codebook_name, prompt_mode)

    suffix = path.suffix.lower()
    if suffix == ".json":
        path.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    elif suffix == ".csv":
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise ValueError(
            f"Unbekanntes Export-Format '{suffix}'. Unterstützt: .csv, .json"
        )

    return str(path)
