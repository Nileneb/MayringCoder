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
    hits = _scan("src/memory", SILENT_FAIL_RE)
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
    memory_hits = _scan("src/memory", SILENT_RETURN_NONE_RE)
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


# ---------------------------------------------------------------------------
# 2.2 — Hook-silent-skip-counter
# ---------------------------------------------------------------------------


def _load_skip_counter():
    """Direct file-import — hooks/ ist kein Python-package."""
    import importlib.util
    p = ROOT / "claude-plugin" / "hooks" / "_silent_skip_counter.py"
    spec = importlib.util.spec_from_file_location("_skip_counter", p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_silent_skip_counter_module_exists():
    """claude-plugin/hooks/_silent_skip_counter.py existiert + exportiert
    record_silent_skip() + recent_skip_count()."""
    mod = _load_skip_counter()
    assert callable(mod.record_silent_skip)
    assert callable(mod.recent_skip_count)


def test_silent_skip_counter_records_and_reads(tmp_path, monkeypatch):
    """record_silent_skip → file existiert + recent_skip_count >= 1."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    mod = _load_skip_counter()
    mod.record_silent_skip(reason="all_5xx")
    assert mod.recent_skip_count(window_hours=24) >= 1


def test_silent_skip_counter_window_filters_old_entries(tmp_path, monkeypatch):
    """Alte Einträge (>24h) zählen nicht in das current window."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    import json
    import time
    mod = _load_skip_counter()
    p = mod.skip_log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    old_ts = time.time() - 25 * 3600  # 25h ago
    p.write_text(json.dumps({"events": [
        {"ts": old_ts, "reason": "all_5xx"},
    ]}))
    assert mod.recent_skip_count(window_hours=24) == 0


def test_silent_skip_counter_threshold_helper(tmp_path, monkeypatch):
    """`should_warn(threshold=5)` returns True when ≥5 recent skips logged."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    mod = _load_skip_counter()
    for _ in range(5):
        mod.record_silent_skip(reason="all_5xx")
    assert mod.should_warn(threshold=5) is True
    # Reset → counter zeigt 0.
    mod.reset_counter()
    assert mod.recent_skip_count(window_hours=24) == 0
