"""benchmark_summary.py — Vergleichsmatrix aller bench_*-Runs.

Liest alle bench_*-Runs aus cache/<slug>/runs/ und gibt eine sortierte
Tabelle aus mit: Modell, Dateien, Findings, Parse-Fehler, Laufzeit, Score.

Score-Formel:
    score = (findings_per_file) * (1 - parse_error_rate) * (high_conf_ratio)
    Normiert auf 0–100 für einfache Vergleichbarkeit.

Verwendung:
    .venv/bin/python benchmark_summary.py
    .venv/bin/python benchmark_summary.py --repo https://github.com/owner/repo
    .venv/bin/python benchmark_summary.py --prefix bench_  # anderer Run-Prefix
    .venv/bin/python benchmark_summary.py --json           # JSON-Ausgabe
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running without package install
sys.path.insert(0, str(Path(__file__).parent))

from src.config import CACHE_DIR


def _find_all_run_dirs() -> list[Path]:
    """Find all cache/<slug>/runs/ directories."""
    return [p for p in CACHE_DIR.glob("*/runs") if p.is_dir()]


def _load_bench_runs(run_dirs: list[Path], prefix: str) -> list[dict]:
    """Load all run JSONs whose stem starts with *prefix*."""
    runs = []
    for d in run_dirs:
        for p in sorted(d.glob(f"{prefix}*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                data["_path"] = str(p)
                runs.append(data)
            except (json.JSONDecodeError, OSError):
                continue
    return runs


def _compute_metrics(run: dict) -> dict:
    agg = run.get("aggregation") or {}
    results = run.get("results") or []

    files_checked = run.get("files_checked", len(results))
    timing = run.get("timing_seconds", 0.0)
    time_budget_hit = run.get("time_budget_hit", False)

    # Findings
    total_findings = agg.get("total_findings", 0)
    by_sev = agg.get("by_severity") or {}
    critical = by_sev.get("critical", 0)
    warning = by_sev.get("warning", 0)

    # Parse errors
    parse_errors = len(agg.get("parse_errors") or [])
    parse_error_rate = parse_errors / files_checked if files_checked > 0 else 0.0

    # Confidence breakdown from raw findings
    high_conf = 0
    total_findable = 0
    for r in results:
        if "error" in r:
            continue
        for s in r.get("potential_smells", []) + r.get("codierungen", []):
            total_findable += 1
            if s.get("confidence", "").lower() == "high":
                high_conf += 1
    high_conf_ratio = high_conf / total_findable if total_findable > 0 else 0.0

    # Findings per file (throughput)
    findings_per_file = total_findings / files_checked if files_checked > 0 else 0.0

    # Score: throughput × quality × confidence
    raw_score = findings_per_file * (1 - parse_error_rate) * max(high_conf_ratio, 0.1)

    return {
        "model": run.get("model", "?"),
        "run_id": run.get("run_id", "?"),
        "timestamp": run.get("timestamp", "")[:16].replace("T", " "),
        "files": files_checked,
        "findings": total_findings,
        "critical": critical,
        "warning": warning,
        "parse_errors": parse_errors,
        "timing_s": round(timing, 1),
        "time_budget_hit": time_budget_hit,
        "high_conf_ratio": round(high_conf_ratio, 2),
        "raw_score": raw_score,
    }


def _normalize_scores(rows: list[dict]) -> list[dict]:
    """Normalize raw_score to 0–100."""
    max_score = max((r["raw_score"] for r in rows), default=1.0) or 1.0
    for r in rows:
        r["score"] = round(r["raw_score"] / max_score * 100)
    return rows


def _stars(score: int) -> str:
    if score >= 80:
        return "★★★★★"
    if score >= 60:
        return "★★★★☆"
    if score >= 40:
        return "★★★☆☆"
    if score >= 20:
        return "★★☆☆☆"
    return "★☆☆☆☆"


def _render_table(rows: list[dict]) -> str:
    if not rows:
        return "Keine bench_*-Runs gefunden."

    header = (
        f"{'Modell':<35} {'Dateien':>7} {'Findings':>8} {'Krit':>5} "
        f"{'Parse-Err':>9} {'Zeit(s)':>8} {'HiConf':>7} {'Score':>6}  Bewertung"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for r in rows:
        budget_marker = "⏱" if r["time_budget_hit"] else " "
        lines.append(
            f"{r['model']:<35} {r['files']:>7} {r['findings']:>8} {r['critical']:>5} "
            f"{r['parse_errors']:>9} {r['timing_s']:>8.1f} {r['high_conf_ratio']:>7.0%} "
            f"{r['score']:>5}  {_stars(r['score'])} {budget_marker}"
        )

    lines += [
        sep,
        "⏱ = Zeit-Budget wurde erreicht (vorzeitiger Stop)",
        f"Score = (Findings/Datei) × (1 - Parse-Fehler-Rate) × max(High-Conf-Rate, 10%) — normiert 0–100",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark-Auswertung aller bench_*-Runs")
    p.add_argument("--repo", help="Nur Runs für dieses Repo anzeigen (URL oder Slug)")
    p.add_argument("--prefix", default="bench_", help="Run-ID-Prefix (Standard: bench_)")
    p.add_argument("--json", action="store_true", dest="as_json", help="JSON-Ausgabe statt Tabelle")
    p.add_argument("--sort", choices=["score", "files", "findings", "timing"], default="score",
                   help="Sortierspalte (Standard: score)")
    args = p.parse_args()

    if not CACHE_DIR.exists():
        print("Kein Cache-Verzeichnis gefunden. Bitte zuerst einen Benchmark-Lauf starten.")
        sys.exit(1)

    if args.repo:
        from src.config import repo_slug
        slug = repo_slug(args.repo) if args.repo.startswith("http") else args.repo
        run_dirs = [CACHE_DIR / slug / "runs"]
    else:
        run_dirs = _find_all_run_dirs()

    runs = _load_bench_runs(run_dirs, args.prefix)
    if not runs:
        print(f"Keine Runs mit Prefix '{args.prefix}' gefunden in {CACHE_DIR}")
        sys.exit(0)

    rows = [_compute_metrics(r) for r in runs]
    rows = _normalize_scores(rows)
    rows.sort(key=lambda r: -r.get(args.sort, r["score"]))

    if args.as_json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print(f"\nBenchmark-Auswertung — {len(rows)} Runs (Prefix: '{args.prefix}')\n")
        print(_render_table(rows))
        print()

        # Winner callout
        best = rows[0]
        print(f"  Bestes Modell: {best['model']} (Score: {best['score']}, "
              f"{best['files']} Dateien, {best['findings']} Findings, "
              f"{best['parse_errors']} Parse-Fehler)")
        print()


if __name__ == "__main__":
    main()
