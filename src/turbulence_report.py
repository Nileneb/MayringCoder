"""turbulence_report.py — Report-Generierung für Turbulenz-Analyse.

Enthält:
- build_report()       → konsolidiert Analysen zu einem dict + stdout-Ausgabe
- build_markdown()     → rendert den dict-Report als Markdown-String

Kein SQLite. Keine LLM-Aufrufe. Kein Chunking.
"""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from src.turbulence_calculator import FileAnalysis, Redundancy


def build_report(
    analyses: list[FileAnalysis],
    redundancies: list[Redundancy],
) -> dict:
    """Erstellt den finalen Report-Dict und gibt eine Zusammenfassung auf stdout aus."""

    analyses.sort(key=lambda a: -a.turbulence_score)

    skipped = [a for a in analyses if a.tier == "skip"]
    light   = [a for a in analyses if a.tier == "light"]
    deep    = [a for a in analyses if a.tier == "deep"]

    total_findings = sum(len(a.findings) for a in analyses)

    print("\n" + "=" * 60)
    print("📊 TURBULENZ-REPORT")
    print("=" * 60)
    print(f"\n  Dateien gesamt:     {len(analyses)}")
    print(f"  🔴 Kritisch (>50%): {len(deep)}")
    print(f"  🟡 Mittel (20-50%): {len(light)}")
    print(f"  ⬛ Stabil (<20%):   {len(skipped)} (übersprungen)")
    print(f"  Findings:           {total_findings}")
    print(f"  Redundanzen:        {len(redundancies)}")

    if deep:
        print("\n── 🔴 Kritische Dateien ──────────────────────────────")
        for a in deep:
            print(f"\n  {a.path}")
            print(
                f"    Turbulenz: {a.turbulence_score:.0%} | "
                f"Zeilen: {a.total_lines} | "
                f"Hot-Zones: {len(a.hot_zones)}"
            )
            for f in a.findings:
                print(f"    → {f.get('problem', 'k.A.')}")
                print(f"      Empfehlung: {f.get('refactoring', 'k.A.')}")

    if redundancies:
        print("\n── 🔄 Mögliche Redundanzen ──────────────────────────")
        for r in redundancies[:10]:
            print(f"\n  \"{r.name_a}\" ({Path(r.file_a).name})")
            print(f"  ≈ \"{r.name_b}\" ({Path(r.file_b).name})")
            print(f"  Ähnlichkeit: {r.similarity:.0%}")

    if skipped:
        print(f"\n── ⬛ {len(skipped)} stabile Dateien übersprungen ──")
        for a in skipped[:5]:
            print(f"  {a.path} ({a.turbulence_score:.0%})")
        if len(skipped) > 5:
            print(f"  ... und {len(skipped) - 5} weitere")

    return {
        "summary": {
            "total_files": len(analyses),
            "critical": len(deep),
            "medium": len(light),
            "stable": len(skipped),
            "findings": total_findings,
            "redundancies": len(redundancies),
        },
        "critical_files": [
            {
                "path": a.path,
                "turbulence": a.turbulence_score,
                "hot_zones": a.hot_zones,
                "findings": a.findings,
            }
            for a in deep
        ],
        "redundancies": [asdict(r) for r in redundancies[:20]],
    }


def build_markdown(report: dict, repo_url: str, model: str, elapsed: float) -> str:
    """Rendert einen Turbulenz-Report-Dict als Markdown-String."""
    s = report["summary"]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "---",
        f"repo: {repo_url}",
        f"date: {datetime.now().isoformat()}",
        f"model: {model}",
        f"mode: turbulence",
        f"files_total: {s.get('total_files', 0)}",
        f"run_time_s: {elapsed:.1f}",
        "---",
        "",
        f"# Turbulenz-Analyse — {ts}",
        "",
        "## Zusammenfassung",
        "",
        "| Metrik | Wert |",
        "|--------|------|",
        f"| Dateien gesamt | {s.get('total_files', 0)} |",
        f"| 🔴 Kritisch (>50%) | {s.get('critical', 0)} |",
        f"| 🟡 Mittel (20-50%) | {s.get('medium', 0)} |",
        f"| ⬛ Stabil (<20%) | {s.get('stable', 0)} |",
        f"| Findings | {s.get('findings', 0)} |",
        f"| Redundanzen | {s.get('redundancies', 0)} |",
        "",
    ]

    critical = report.get("critical_files", [])
    if critical:
        lines += ["## 🔴 Kritische Dateien", ""]
        for f in critical:
            pct = round(f["turbulence"] * 100)
            lines.append(f"### `{f['path']}` — {pct}% Turbulenz")
            lines.append("")
            for hz in f.get("hot_zones", []):
                lines.append(
                    f"- Hot-Zone Zeile {hz['start_line']}–{hz['end_line']} "
                    f"(Peak: {round(hz['peak_score'] * 100)}%)"
                )
            for finding in f.get("findings", []):
                sev = finding.get("severity", "medium").upper()
                lines.append(f"\n**[{sev}]** {finding.get('problem', '—')}")
                lines.append(f"> Empfehlung: {finding.get('refactoring', '—')}")
            lines.append("")

    redundancies = report.get("redundancies", [])
    if redundancies:
        lines += ["## 🔄 Mögliche Redundanzen", ""]
        for r in redundancies:
            lines.append(
                f"- `{r['name_a']}` ≈ `{r['name_b']}` "
                f"({round(r['similarity'] * 100)}% Ähnlichkeit)"
            )
        lines.append("")

    return "\n".join(lines)
