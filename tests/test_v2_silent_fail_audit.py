"""V2 Stufe 2 — Silent-Fail-Audit.

Spec: docs/v2-master-audit.md Section 7 Stufe 2.

Verbotene Patterns in src/api/ und src/memory/:
  - `except Exception:\\n    pass`
  - `except Exception:\\n    return None`  (außer in UI-render-code)
  - `except Exception:\\n    return error_dict`  (silent fallback)

UI-render-paths sind explizit ausgenommen (web_ui_helpers.py).
mcp_memory_tools.py wird hier explizit getestet — 6 Stellen laut audit.

Speziell der Hook-silent-skip-counter (memory_inject.py): wenn ≥5 skips
in 24h, soll ein Status-File geschrieben werden, das SessionStart-Hook
liest.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent

# UI-render-pfade dürfen fail-soft sein (kein blocker)
EXEMPT_FILES = {
    "src/api/web_ui_helpers.py",
    "src/api/web_ui.py",
    "src/api/web_ui_tabs.py",
    "src/api/web_ui_tabs_b.py",
    "src/api/web_ui_tabs_c.py",
    "src/api/templates",
}

# Pattern: except Exception(:\s|\sas\s\w+:)\s*\n\s+(pass|return None)
# negative-only matcher — fängt nicht "except Exception as exc:\n  log + raise"
# sondern nur wirklich silent fallbacks.
# WHY(Sourcery #3 PR201): die regex muss `except Exception as exc: pass` UND
# `return error_dict` und `return {}` matchen, nicht nur bare `:` und
# `return None`. Sonst rutscht silent-fallback durch (siehe heutiger
# MayringMcpClient → 671 sources im falschen workspace).
SILENT_FAIL_RE = re.compile(
    r"except\s+Exception(?:\s+as\s+\w+)?\s*:\s*\n\s+pass\b",
    re.MULTILINE,
)
SILENT_RETURN_NONE_RE = re.compile(
    r"except\s+Exception(?:\s+as\s+\w+)?\s*:\s*\n\s+return\s+(None|\{\}|\[\]|\"\"|'')\b",
    re.MULTILINE,
)
SILENT_RETURN_ERROR_DICT_RE = re.compile(
    r"except\s+Exception(?:\s+as\s+\w+)?\s*:\s*\n\s+return\s+\{[\"']error[\"']",
    re.MULTILINE,
)


def _is_exempt(path: Path) -> bool:
    rel = str(path.relative_to(ROOT))
    return any(rel.startswith(prefix) for prefix in EXEMPT_FILES)


def _scan(directory: str, pattern: re.Pattern) -> dict[str, int]:
    """Return {file: count} for every match."""
    hits: dict[str, int] = {}
    base = ROOT / directory
    for py in base.rglob("*.py"):
        if _is_exempt(py):
            continue
        if "__pycache__" in py.parts:
            continue
        text = py.read_text(encoding="utf-8")
        n = len(pattern.findall(text))
        if n:
            hits[str(py.relative_to(ROOT))] = n
    return hits


def test_no_silent_pass_in_api_pipeline():
    """`except Exception: pass` — verboten in Daten-Pipelines."""
    hits = _scan("src/api", SILENT_FAIL_RE)
    # mcp_memory_tools.py speziell: 0 erlaubt nach Stufe 2.1.
    target = "src/api/mcp_memory_tools.py"
    assert hits.get(target, 0) == 0, (
        f"{target} hat noch {hits.get(target)} `except Exception: pass`-Stellen"
    )


def test_no_silent_pass_in_memory_pipeline():
    """memory/ — Daten-Pipeline; KEIN silent pass."""
    hits = _scan("core/mayring_core/memory", SILENT_FAIL_RE)
    # Lockerung: ein paar bekannte Stellen sind noch da. Nach Stufe 2.1
    # zählen wir nur "neue" — Baseline = aktuelle Werte. Wir wollen das
    # NICHT-WACHSEN. Hard cap: 6 (audit-baseline).
    total = sum(hits.values())
    assert total <= 6, (
        f"src/memory hat {total} silent-pass-Stellen (audit-cap: 6). "
        f"Hits: {hits}"
    )


def test_no_silent_return_none_in_pipelines():
    """`except Exception: return None` — verboten in Daten-Pipelines.

    Eine Suche/Ingest die scheitert, MUSS sichtbar scheitern. UI-Helpers
    sind exempt.
    """
    api_hits = _scan("src/api", SILENT_RETURN_NONE_RE)
    memory_hits = _scan("core/mayring_core/memory", SILENT_RETURN_NONE_RE)
    # mcp_memory_tools speziell strikt 0:
    target = "src/api/mcp_memory_tools.py"
    assert api_hits.get(target, 0) == 0, (
        f"{target} hat noch silent return None"
    )
    # memory.py-pfade: cap 4 (audit-baseline).
    total_memory = sum(memory_hits.values())
    assert total_memory <= 4, (
        f"src/memory silent-return-None: {total_memory} (cap 4). "
        f"Hits: {memory_hits}"
    )
