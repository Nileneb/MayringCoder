"""Gemeinsame Helper für alle Workflow-Module.

is_test_file / load_prompt / load_turbulence_cache — klein genug, um an einer
Stelle zu wohnen, und unabhängig von einem bestimmten Workflow.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.config import CACHE_DIR, repo_slug as _repo_slug


_TEST_FILE_PATTERNS = [
    re.compile(r"(?:^|/)(?:test[s]?[_\-].*|tests?|spec|__tests?__)", re.IGNORECASE),
    re.compile(r"_test\.(?:py|php|js|ts|go|java)$", re.IGNORECASE),
    re.compile(r"(?:^|/)(?:test)\.\w+$", re.IGNORECASE),
]


def is_test_file(filename: str) -> bool:
    """True wenn der Dateiname einem verbreiteten Test-Pattern folgt."""
    return any(p.search(filename) for p in _TEST_FILE_PATTERNS)


def load_prompt(path: Path | str) -> str:
    """Read a prompt file as UTF-8."""
    return Path(path).read_text(encoding="utf-8")


def load_turbulence_cache(
    repo_url: str,
) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    """Liest den Turbulence-Cache und liefert (hot_zone_map, tier_map).

    `hot_zone_map` mapped Dateinamen auf einen Markdown-Ausschnitt mit Hot-Zone-
    Details (Zeilen, Kategorien, betroffene Funktionen). `tier_map` mapped
    Dateinamen auf Stabilitäts-Tier ("stable", "moderate", "turbulent" etc.).
    Beide sind None wenn kein Cache existiert.
    """
    cache_path = CACHE_DIR / f"{_repo_slug(repo_url)}_turbulence.json"
    if not cache_path.exists():
        return None, None
    try:
        report = json.loads(cache_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None, None

    hot_zone_map: dict[str, str] = {}
    tier_map: dict[str, str] = {}

    for cf in report.get("all_files", report.get("critical_files", [])):
        path = cf.get("path", "")
        tier = cf.get("tier", "")
        tier_map[path] = tier

        zones = cf.get("hot_zones", [])
        if not zones:
            continue

        lines = ["## Hot-Zone-Kontext (aus Turbulenz-Analyse)"]
        for zone in zones:
            start = zone.get("start_line", "?")
            end = zone.get("end_line", "?")
            cats = zone.get("categories", [])
            peak = zone.get("peak_score", 0)
            cats_str = " × ".join(cats) if isinstance(cats, list) else str(cats)
            lines.append(
                f"ACHTUNG: Hot-Zone bei Zeile {start}-{end} "
                f"({cats_str}, Peak-Score: {peak:.0%})"
            )
            affected = zone.get("affected_functions", [])
            for fn_info in affected[:5]:
                if isinstance(fn_info, dict):
                    name = fn_info.get("name", "")
                    inputs = ", ".join(fn_info.get("inputs", []))
                    calls = ", ".join(fn_info.get("calls", []))
                    lines.append(
                        f"  Betroffene Funktion: {name}({inputs}) → calls: {calls}"
                    )

        hot_zone_map[path] = "\n".join(lines)

    return hot_zone_map, tier_map
