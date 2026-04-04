"""Aggregate findings across all analyzed files — Mayring Stufe 4.

Produces:
- total_findings count
- by_severity breakdown
- top_findings (Top-5 by severity × confidence)
- next_steps (Top-5 deduplicated fix suggestions)
- needs_explikation (findings marked for manual re-run)
- parse_errors (files where JSON parsing failed)
- stage2_extracted_count (findings recovered via Stage-2 extraction from freetext)
- redundancy_candidates (potential redundancies found via name similarity)
- adversarial_stats (Advocatus Diaboli: validated/rejected/error counts)
- _below_confidence_filtered (count of findings dropped by min_confidence threshold)
"""

_SEVERITY_RANK = {"critical": 0, "warning": 1, "info": 2}
_VALID_CONFIDENCE = {"high": 0, "medium": 1, "low": 2}


def _sort_key(smell: dict) -> tuple[int, int]:
    sev = _SEVERITY_RANK.get(smell.get("severity", "info").lower(), 2)
    conf = _VALID_CONFIDENCE.get(smell.get("confidence", "medium").lower(), 1)
    return (sev, conf)


def aggregate_findings(
    results: list[dict],
    min_confidence: str = "low",
    adversarial_stats: dict | None = None,
) -> dict:
    """Aggregate findings from all analyzed files.

    Applies the following filters before ranking:
    1. min_confidence threshold: drops findings below the given confidence level
       ("high" → only high, "medium" → high+medium, "low" → all)
    2. Adversarial validation: findings marked ABGELEHNT by validate_findings()
       are already excluded; only BESTÄTIGT findings enter the aggregation.

    Args:
        results:          List of result dicts from analyze_files.
        min_confidence:   Minimum confidence to keep ("high", "medium", "low").
        adversarial_stats: Stats dict from validate_findings().

    Returns a summary dict with totals, severity breakdown, top findings, and lists.
    """
    raw_smells: list[dict] = []
    needs_explikation: list[dict] = []
    parse_errors: list[str] = []
    stage2_count = 0

    for r in results:
        if "error" in r:
            continue

        if "codierungen" in r:
            # Social-research mode
            for cod in r.get("codierungen", []):
                enriched = {**cod, "_filename": r["filename"]}
                enriched.setdefault("severity", "info")
                raw_smells.append(enriched)
                if cod.get("needs_explikation"):
                    needs_explikation.append({
                        "filename": r["filename"],
                        "type": cod.get("category", ""),
                        "evidence_excerpt": cod.get("evidence_excerpt", ""),
                    })
        else:
            for smell in r.get("potential_smells", []):
                enriched = {**smell, "_filename": r["filename"]}
                raw_smells.append(enriched)
                if smell.get("needs_explikation"):
                    needs_explikation.append({
                        "filename": r["filename"],
                        "type": smell.get("type", ""),
                        "evidence_excerpt": smell.get("evidence_excerpt", ""),
                    })

        if r.get("_stage2_extracted"):
            stage2_count += len(r.get("potential_smells", []))

        if r.get("_parse_error"):
            parse_errors.append(r["filename"])

    # ── Confidence threshold filter ──────────────────────────────────────────
    min_rank = _VALID_CONFIDENCE.get(min_confidence.lower(), 2)
    all_smells: list[dict] = []
    below_confidence_count = 0

    for s in raw_smells:
        conf_rank = _VALID_CONFIDENCE.get(s.get("confidence", "medium").lower(), 1)
        if conf_rank > min_rank:
            below_confidence_count += 1
            continue
        all_smells.append(s)

    # ── Severity breakdown ────────────────────────────────────────────────────
    by_severity: dict[str, int] = {"critical": 0, "warning": 0, "info": 0}
    for s in all_smells:
        key = s.get("severity", "info").lower()
        if key in by_severity:
            by_severity[key] += 1

    # ── Ranking ───────────────────────────────────────────────────────────────
    ranked = sorted(all_smells, key=_sort_key)
    top_findings = ranked[:5]

    # ── Deduplicated next steps ───────────────────────────────────────────────
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
        "stage2_extracted_count": stage2_count,
        "redundancy_candidates": [],
        "adversarial_stats": adversarial_stats or {},
        "_below_confidence_filtered": below_confidence_count,
    }


def aggregate_with_redundancy(
    results: list[dict],
    overview_results: list[dict],
    threshold: float = 0.80,
    min_confidence: str = "low",
    adversarial_stats: dict | None = None,
) -> dict:
    """Like aggregate_findings, but also runs the name-redundancy check.

    Requires overview_results with _signatures (from overview_files + extract_python_signatures).
    Redundancy candidates are appended to the aggregation dict under
    ``redundancy_candidates`` and added to the findings list for ranking.
    """
    agg = aggregate_findings(
        results, min_confidence=min_confidence, adversarial_stats=adversarial_stats
    )

    try:
        from src.extractor import check_redundancy_by_names

        candidates = check_redundancy_by_names(overview_results, threshold=threshold)
    except Exception:
        candidates = []

    agg["redundancy_candidates"] = candidates

    for candidate in candidates:
        candidate["_filename"] = "cross-file"
    agg["total_findings"] = len(agg.get("top_findings", [])) + len(candidates)

    return agg
