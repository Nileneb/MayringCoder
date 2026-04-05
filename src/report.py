"""Generate Markdown report + run_meta.json."""

import json
from datetime import datetime

from src.config import REPORTS_DIR, get_max_chars_per_file

_SEV_EMOJI = {"critical": "🔴", "warning": "🟡", "info": "🟢"}


def generate_report(
    repo_url: str,
    model: str,
    results: list[dict],
    aggregation: dict,
    diff: dict,
    timing: float,
    run_id: str | None = None,
    commit: str | None = None,
    embedding_prefilter_meta: dict | None = None,
    max_chars_per_file: int | None = None,
    full_scan: bool = False,
) -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    report_path = REPORTS_DIR / f"repo-check-{timestamp}.md"
    meta_path = REPORTS_DIR / f"repo-check-{timestamp}_meta.json"
    _max_chars = max_chars_per_file if max_chars_per_file is not None else get_max_chars_per_file()

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
        f"run_id: {run_id or 'default'}",
        f"files_total: {files_total}",
        f"files_checked: {files_checked}",
        f"files_skipped: {len(diff.get('skipped', []))}",
        f"findings_count: {findings_count}",
        f"run_time_s: {timing:.1f}",
    ]
    if full_scan:
        lines.append("full_scan: true")
    _so_model = aggregation.get("second_opinion_stats", {}).get("model")
    if _so_model:
        lines.append(f"second_opinion_model: {_so_model}")
    if embedding_prefilter_meta:
        ep = embedding_prefilter_meta
        lines.append(
            f"embedding_prefilter: model={ep['model']},"
            f" top_k={ep['top_k']},"
            f" threshold={ep['threshold']},"
            f" files_before={ep['files_before']},"
            f" files_after={ep['files_after']}"
        )
    lines += [
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
    ]
    if full_scan:
        lines.append("- **Modus:** Full Scan (kein Cache, kein Limit)")
    if embedding_prefilter_meta:
        ep = embedding_prefilter_meta
        lines.append(
            f"- **Embedding-Vorfilter:** {ep['model']}"
            f" · Top-{ep['top_k']}"
            + (f" · Threshold {ep['threshold']}" if ep['threshold'] is not None else "")
            + f" · Query: _{ep['query']}_"
            + f" · {ep['files_before']} → {ep['files_after']} Dateien"
        )
    adv_stats = aggregation.get("adversarial_stats", {})
    if adv_stats:
        lines.append(
            f"- **Adversarial:** {adv_stats.get('validated', 0)} BESTÄTIGT"
            f" · {adv_stats.get('rejected', 0)} ABGELEHNT"
            f" · {adv_stats.get('errors', 0)} Fehler"
        )
    so_stats = aggregation.get("second_opinion_stats", {})
    if so_stats:
        so_model_label = f" (`{so_stats['model']}`)" if so_stats.get("model") else ""
        lines.append(
            f"- **Second Opinion{so_model_label}:** {so_stats.get('confirmed', 0)} BESTÄTIGT"
            f" · {so_stats.get('rejected', 0)} ABGELEHNT"
            f" · {so_stats.get('refined', 0)} PRÄZISIERT"
            f" · {so_stats.get('errors', 0)} Fehler"
        )
    lines.append("")

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

    # Detect social-research mode: any result has 'codierungen' instead of 'potential_smells'
    is_sozial = any("codierungen" in r for r in results if "error" not in r)

    for r in results:
        if "error" in r:
            lines += [f"### ⚠ {r['filename']}", "", f"FEHLER: {r['error']}", "", "---", ""]
            continue

        lines.append(f"### {r['filename']}")
        if r.get("category"):
            lines.append(f"*Kategorie: {r['category']}*")
        if r.get("truncated"):
            lines.append(f"*⚠ Inhalt gekürzt auf {_max_chars} Zeichen*")
        lines.append("")

        if r.get("file_summary"):
            lines.append(r["file_summary"])
            lines.append("")

        if is_sozial:
            codierungen = r.get("codierungen", [])
            if not codierungen:
                lines.append("Keine Codierungen gefunden.")
            else:
                for c in codierungen:
                    conf = c.get("confidence", "?")
                    explik_note = " *(Explikation empfohlen)*" if c.get("needs_explikation") else ""
                    lines.append(
                        f"- **{c.get('category', '?')}**{explik_note}"
                        f" (Zeile ~{c.get('line_hint', '?')}"
                        f", Konfidenz: {conf})"
                    )
                    if c.get("evidence_excerpt"):
                        lines.append(f"  > {c['evidence_excerpt'][:200]}")
                    if c.get("reasoning"):
                        lines.append(f"  → {c['reasoning']}")
        else:
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

    # --- Induktiv: Category Summary (aggregiert über alle Dateien) ---
    all_cat_summaries: list[dict] = []
    for r in results:
        all_cat_summaries.extend(r.get("category_summary", []))
    if all_cat_summaries:
        # Merge counts for same category across files
        merged: dict[str, dict] = {}
        for cs in all_cat_summaries:
            cat = cs.get("category", "?")
            if cat in merged:
                merged[cat]["count"] += cs.get("count", 0)
            else:
                merged[cat] = {
                    "definition": cs.get("definition", ""),
                    "count": cs.get("count", 0),
                }
        lines += [
            "## Induktiv entwickelte Kategorien",
            "",
            "| Kategorie | Definition | Anzahl |",
            "|---|---|---:|",
        ]
        for cat, info in sorted(merged.items(), key=lambda x: -x[1]["count"]):
            lines.append(f"| {cat} | {info['definition']} | {info['count']} |")
        lines += [""]

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
            excerpt = (e.get("evidence_excerpt") or "")[:100]
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
        "run_id": run_id or "default",
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
        "full_scan": full_scan,
        "embedding_prefilter": embedding_prefilter_meta,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return str(report_path)


def generate_overview_report(
    repo_url: str,
    model: str,
    results: list[dict],
    diff: dict,
    timing: float,
    run_id: str | None = None,
    full_scan: bool = False,
) -> str:
    """Generate a lightweight overview-only report (no findings / aggregation)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    report_path = REPORTS_DIR / f"overview-{timestamp}.md"

    files_total = len(results)
    errors = [r for r in results if "error" in r]

    lines: list[str] = [
        "---",
        f"repo: {repo_url}",
        f"date: {datetime.now().isoformat()}",
        f"model: {model}",
        f"run_id: {run_id or 'default'}",
        f"mode: overview",
        f"files_total: {files_total}",
        f"errors: {len(errors)}",
        f"run_time_s: {timing:.1f}",
    ]
    if full_scan:
        lines.append("full_scan: true")
    lines += [
        "---",
        "",
        f"# Funktions-Übersicht — {timestamp}",
        "",
        f"- **Repo:** {repo_url}",
        f"- **Dateien:** {files_total}",
        f"- **Fehler:** {len(errors)}",
        f"- **Laufzeit:** {timing:.1f}s",
    ]
    if full_scan:
        lines.append("- **Modus:** Full Scan")
    lines.append("")

    # Group by category
    cat_groups: dict[str, list[dict]] = {}
    for r in results:
        cat = r.get("category", "uncategorized")
        cat_groups.setdefault(cat, []).append(r)

    for cat, items in sorted(cat_groups.items()):
        lines += [f"## {cat}", ""]
        for r in items:
            lines.append(f"### {r['filename']}")
            if "error" in r:
                lines.append(f"⚠ FEHLER: {r['error']}")
            elif r.get("file_summary"):
                lines.append(r["file_summary"])
            else:
                lines.append("*(keine Zusammenfassung)*")
            lines += ["", "---", ""]

    report_text = "\n".join(lines)
    print(report_text)
    report_path.write_text(report_text, encoding="utf-8")
    return str(report_path)
