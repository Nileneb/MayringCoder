"""Aggregate findings across all analyzed files — Mayring Stufe 4.

Produces:
- total_findings count
- by_severity breakdown
- top_findings (Top-5 by severity × confidence)
- next_steps (Top-5 deduplicated fix suggestions)
- needs_explikation (findings marked for manual re-run)
- parse_errors (files where JSON parsing failed)
"""

_SEVERITY_RANK = {"critical": 0, "warning": 1, "info": 2}
_CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}


def _sort_key(smell: dict) -> tuple[int, int]:
    sev = _SEVERITY_RANK.get(smell.get("severity", "info").lower(), 2)
    conf = _CONFIDENCE_RANK.get(smell.get("confidence", "medium").lower(), 1)
    return (sev, conf)


def aggregate_findings(results: list[dict]) -> dict:
    all_smells: list[dict] = []
    needs_explikation: list[dict] = []
    parse_errors: list[str] = []

    for r in results:
        if "error" in r:
            continue
        for smell in r.get("potential_smells", []):
            enriched = {**smell, "_filename": r["filename"]}
            all_smells.append(enriched)
            if smell.get("needs_explikation"):
                needs_explikation.append({
                    "filename": r["filename"],
                    "type": smell.get("type", ""),
                    "evidence_excerpt": smell.get("evidence_excerpt", ""),
                })
        if r.get("_parse_error"):
            parse_errors.append(r["filename"])

    by_severity: dict[str, int] = {"critical": 0, "warning": 0, "info": 0}
    for s in all_smells:
        key = s.get("severity", "info").lower()
        if key in by_severity:
            by_severity[key] += 1

    ranked = sorted(all_smells, key=_sort_key)
    top_findings = ranked[:5]

    # Deduplicated next steps, ordered by severity
    seen: set[str] = set()
    next_steps: list[str] = []
    for s in ranked:
        fix = s.get("fix_suggestion", "").strip()
        if fix and fix not in seen:
            seen.add(fix)
            next_steps.append(fix)
        if len(next_steps) >= 5:
            break

    return {
        "total_findings": len(all_smells),
        "by_severity": by_severity,
        "top_findings": top_findings,
        "next_steps": next_steps,
        "needs_explikation": needs_explikation,
        "parse_errors": parse_errors,
    }
