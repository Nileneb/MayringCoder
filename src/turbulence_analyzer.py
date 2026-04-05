#!/usr/bin/env python3
"""turbulence_analyzer.py — Orchestrator der Turbulenz-Analyse-Pipeline.

Ablauf:
  1. Dateien im Repository finden und filtern
  2. Jede Datei in Chunks zerlegen  →  turbulence_calculator.chunkify()
  3. Chunks kategorisieren           →  turbulence_calculator.categorize_*()
  4. Turbulenz-Score berechnen       →  turbulence_calculator.calculate_turbulence()
  5. Hot-Zones tiefenanalysieren     →  turbulence_calculator.deep_analyze_hotzone()
  6. Redundanzen erkennen            →  turbulence_calculator.find_redundancies()
  7. Report zusammenstellen          →  turbulence_report.build_report()

Reine Orchestrierung — keine Berechnungslogik, keine Markdown-Ausgabe.
"""

from pathlib import Path

from src.turbulence_calculator import (
    FileAnalysis,
    calculate_turbulence,
    categorize_chunk_heuristic,
    categorize_chunk_llm,
    chunkify,
    deep_analyze_hotzone,
    find_redundancies,
    THRESHOLD_SKIP,
    THRESHOLD_DEEP,
    MIN_CHUNKS_FOR_TRIAGE,
)
from src.turbulence_report import build_report


def analyze_repo(
    repo_path: str,
    use_llm: bool = False,
    model: str | None = None,
    overview_cache: dict | None = None,
) -> dict:
    """Analysiert ein ganzes Repository auf Turbulenz.

    Args:
        repo_path:      Pfad zum (temporären) Repository-Verzeichnis.
        use_llm:        True = LLM-Kategorisierung (Ollama), False = Heuristik (schnell).
        model:          Ollama-Modellname für LLM-Modus.
        overview_cache: {filename: entry_dict} aus Overview-Stage (Issue #17).
                        Wenn vorhanden, wird die Kategorie aus dem Cache übernommen
                        statt per LLM/Heuristik neu zu kategorisieren.

    Returns:
        Report-Dict mit summary, critical_files, redundancies.
    """
    repo = Path(repo_path)

    # Dateien finden und Vendor/Node ausschließen
    files = []
    for ext_pattern in ["**/*.php", "**/*.blade.php", "**/*.js", "**/*.ts"]:
        files.extend(repo.glob(ext_pattern))

    seen: set[Path] = set()
    filtered: list[Path] = []
    for f in files:
        resolved = f.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        rel = str(f.relative_to(repo))
        if any(skip in rel for skip in ["vendor/", "node_modules/", ".git/", "storage/"]):
            continue
        filtered.append(f)

    print(f"\n📁 {len(filtered)} Dateien gefunden in {repo_path}")
    print(f"🤖 Modus: {'LLM' if use_llm else 'Heuristik'}")
    if overview_cache:
        print(f"📋 Overview-Cache: {len(overview_cache)} Einträge geladen")
    print("=" * 60)

    categorize_fn = categorize_chunk_llm if use_llm else categorize_chunk_heuristic
    _overview_hits = 0
    all_analyses: list[FileAnalysis] = []
    all_chunks = []

    for filepath in sorted(filtered):
        rel_path = str(filepath.relative_to(repo))
        chunks = chunkify(str(filepath))
        if not chunks:
            continue

        # Feed-forward: use overview cache category when available (Issue #17)
        ov_entry = overview_cache.get(rel_path) if overview_cache else None

        for chunk in chunks:
            if ov_entry and ov_entry.get("category"):
                chunk.category = ov_entry["category"]
                chunk.functional_name = chunk.functional_name or ""
                _overview_hits += 1
            elif use_llm:
                categorize_chunk_llm(chunk, model=model)
            else:
                categorize_chunk_heuristic(chunk)

        all_chunks.extend(chunks)

        # Dateien mit zu wenig Chunks liefern keinen verlässlichen Score —
        # ein 2-Chunk-File mit zwei Kategorien würde immer 50% ergeben (Artefakt).
        if len(chunks) < MIN_CHUNKS_FOR_TRIAGE:
            tier = "stable"
            turb_score = 0.0
            hot_zones: list = []
        else:
            turb_score, hot_zones = calculate_turbulence(chunks)
            if turb_score < THRESHOLD_SKIP:
                tier = "skip"
            elif turb_score < THRESHOLD_DEEP:
                tier = "light"
            else:
                tier = "deep"

        analysis = FileAnalysis(
            path=rel_path,
            total_lines=sum(c.end_line - c.start_line + 1 for c in chunks),
            chunks=chunks,
            turbulence_score=turb_score,
            hot_zones=hot_zones,
            tier=tier,
        )

        # Enrich hot zones with function I/O from overview cache (Issue #17)
        if ov_entry and hot_zones:
            ov_functions = ov_entry.get("functions", [])
            for zone in hot_zones:
                zone["affected_functions"] = ov_functions

        if tier == "deep" and hot_zones:
            for zone in hot_zones[:3]:
                finding = deep_analyze_hotzone(
                    str(filepath),
                    zone["start_line"],
                    zone["end_line"],
                    zone["peak_score"],
                    use_llm=use_llm,
                    model=model,
                )
                if finding:
                    analysis.findings.append({"zone": zone, **finding})

        icon = {"stable": "⬜", "skip": "⬛", "light": "🟡", "deep": "🔴"}[tier]
        print(f"  {icon} {rel_path:50s} Turbulenz: {turb_score:.0%} [{tier}]")

        all_analyses.append(analysis)

    redundancies = find_redundancies(all_chunks)

    if overview_cache and _overview_hits:
        print(f"\n  📋 Overview-Cache: {_overview_hits} Chunks aus Cache kategorisiert")

    return build_report(all_analyses, redundancies)


# ── CLI (Direktaufruf) ─────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Turbulenz-Analyse: Findet fragile Code-Stellen "
                    "durch funktionale Farbcodierung"
    )
    parser.add_argument("repo", help="Pfad zum Repository")
    parser.add_argument("--llm", action="store_true",
                        help="LLM für Kategorisierung nutzen (langsamer, genauer)")
    parser.add_argument("--output", "-o", default=None,
                        help="JSON-Report speichern")

    args = parser.parse_args()
    report = analyze_repo(args.repo, use_llm=args.llm)

    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str)
        )
        print(f"\n💾 Report gespeichert: {args.output}")
