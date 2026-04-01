"""Generate Markdown report + run_meta.json."""

import json
from datetime import datetime

from src.config import MAX_CHARS_PER_FILE, REPORTS_DIR

_SEV_EMOJI = {"critical": "🔴", "warning": "🟡", "info": "🟢"}


def generate_report(
    repo_url: str,
    model: str,
    results: list[dict],
    aggregation: dict,
    diff: dict,
    timing: float,
    commit: str | None = None,
) -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    report_path = REPORTS_DIR / f"repo-check-{timestamp}.md"
    meta_path = REPORTS_DIR / f"repo-check-{timestamp}_meta.json"

    files_total = (
        len(diff.get("changed", []))
        + len(diff.get("added", []))
        + len(diff.get("unchanged", []))
    )
    files_checked = len(results)
    bysev = aggregation.get("by_severity", {})
    findings_count = aggregation.get("total_findings", 0)

    lines: list[str] = [
        "---",
        f"repo: {repo_url}",
        f"commit: {commit or 'unknown'}",
        f"date: {datetime.now().isoformat()}",
        f"model: {model}",
        f"files_total: {files_total}",
        f"files_checked: {files_checked}",
        f"files_skipped: {len(diff.get('skipped', []))}",
        f"findings_count: {findings_count}",
        f"run_time_s: {timing:.1f}",
        "---",
        "",
        f"# RepoChecker Report — {timestamp}",
        "",
        "## Summary",
        "",
        f"- **Repo:** {repo_url}",
        f"- **Analysiert:** {files_checked} Dateien",
        f"- **Geändert/Neu:** {len(diff.get('changed', []))} / {len(diff.get('added', []))}",
        f"- **Übersprungen (Budget):** {len(diff.get('skipped', []))}",
        f"- **Findings:** {findings_count}"
        f"  (🔴 {bysev.get('critical', 0)}"
        f" · 🟡 {bysev.get('warning', 0)}"
        f" · 🟢 {bysev.get('info', 0)})",
        f"- **Laufzeit:** {timing:.1f}s",
        "",
    ]

    # --- Category Digest ---
    lines += ["## Category Digest", ""]
    cat_groups: dict[str, list[str]] = {}
    for r in results:
        cat = r.get("category", "uncategorized")
        cat_groups.setdefault(cat, []).append(r["filename"])
    for cat, fns in sorted(cat_groups.items()):
        lines.append(f"- **{cat}**: {', '.join(fns)}")
    lines += [""]

    # --- Top Findings ---
    lines += ["## Top Findings", ""]
    top = aggregation.get("top_findings", [])
    if not top:
        lines.append("Keine Findings.")
    else:
        for i, s in enumerate(top, 1):
            sev = s.get("severity", "info").lower()
            emoji = _SEV_EMOJI.get(sev, "🟢")
            lines.append(
                f"{i}. {emoji} **{s.get('type', 'unknown')}**"
                f" — `{s.get('_filename', '')}`"
                f" (Konfidenz: {s.get('confidence', '?')})"
            )
            if s.get("evidence_excerpt"):
                lines.append(f"   > {s['evidence_excerpt'][:200]}")
            if s.get("fix_suggestion"):
                lines.append(f"   → {s['fix_suggestion']}")
            lines.append("")
    lines += [""]

    # --- Per-File Findings ---
    lines += ["## Per-File Findings", ""]
    for r in results:
        if "error" in r:
            lines += [f"### ⚠ {r['filename']}", "", f"FEHLER: {r['error']}", "", "---", ""]
            continue

        lines.append(f"### {r['filename']}")
        if r.get("category"):
            lines.append(f"*Kategorie: {r['category']}*")
        if r.get("truncated"):
            lines.append(f"*⚠ Inhalt gekürzt auf {MAX_CHARS_PER_FILE} Zeichen*")
        lines.append("")

        if r.get("file_summary"):
            lines.append(r["file_summary"])
            lines.append("")

        smells = r.get("potential_smells", [])
        if not smells:
            lines.append("Keine Auffälligkeiten gefunden.")
        else:
            for s in smells:
                sev = s.get("severity", "info").lower()
                emoji = _SEV_EMOJI.get(sev, "🟢")
                explik_note = " *(Explikation empfohlen)*" if s.get("needs_explikation") else ""
                lines.append(
                    f"- {emoji} **{s.get('type', '?')}**{explik_note}"
                    f" (Zeile ~{s.get('line_hint', '?')}"
                    f", Konfidenz: {s.get('confidence', '?')})"
                )
                if s.get("evidence_excerpt"):
                    lines.append(f"  > `{s['evidence_excerpt'][:150]}`")
                if s.get("fix_suggestion"):
                    lines.append(f"  → {s['fix_suggestion']}")

        lines += ["", "---", ""]

    # --- Explikation ---
    explik_list = aggregation.get("needs_explikation", [])
    if explik_list:
        lines += [
            "## Explikation empfohlen (manueller Re-Run)",
            "",
            "Diese Findings haben niedrige Konfidenz."
            " Erneut analysieren mit `--prompt prompts/explainer.md`:",
            "",
        ]
        for e in explik_list:
            excerpt = e.get("evidence_excerpt", "")[:100]
            lines.append(f"- `{e['filename']}` — **{e['type']}**: `{excerpt}`")
        lines += [""]

    # --- Next Steps ---
    lines += ["## Empfohlene nächste Schritte", ""]
    next_steps = aggregation.get("next_steps", [])
    if next_steps:
        for i, step in enumerate(next_steps, 1):
            lines.append(f"{i}. {step}")
    else:
        lines.append("Keine konkreten Handlungsempfehlungen.")
    lines.append("")

    report_text = "\n".join(lines)
    print(report_text)
    report_path.write_text(report_text, encoding="utf-8")

    # run_meta.json
    meta = {
        "repo": repo_url,
        "timestamp": timestamp,
        "model": model,
        "diff_stats": {
            "changed": len(diff.get("changed", [])),
            "added": len(diff.get("added", [])),
            "removed": len(diff.get("removed", [])),
            "unchanged": len(diff.get("unchanged", [])),
        },
        "selected_files": diff.get("selected", []),
        "skipped_files": diff.get("skipped", []),
        "truncation_flags": [r["filename"] for r in results if r.get("truncated")],
        "parse_errors": aggregation.get("parse_errors", []),
        "run_time_s": round(timing, 2),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return str(report_path)
