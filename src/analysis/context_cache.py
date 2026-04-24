"""Phase 1 Overview-Cache — JSON persistence + context string builders."""

import json
from pathlib import Path

from src.config import (
    CACHE_DIR,
    MAX_CONTEXT_CHARS,
    repo_slug as _repo_slug,
)

_ACTIVE_MAX_CONTEXT_CHARS = MAX_CONTEXT_CHARS


def set_max_context_chars(limit: int) -> None:
    global _ACTIVE_MAX_CONTEXT_CHARS
    _ACTIVE_MAX_CONTEXT_CHARS = max(1, int(limit))


def _overview_path(repo_url: str) -> Path:
    return CACHE_DIR / f"{_repo_slug(repo_url)}_overview.json"


def save_overview_context(results: list[dict], repo_url: str) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for r in results:
        if "error" in r:
            continue
        entry = {
            "filename": r["filename"],
            "category": r.get("category", "uncategorized"),
            "file_summary": r.get("file_summary", ""),
        }
        for field in ("file_type", "key_responsibilities", "dependencies", "purpose_keywords",
                       "functions", "external_deps"):
            if field in r:
                entry[field] = r[field]
        if "_signatures" in r:
            entry["_signatures"] = r["_signatures"]
        entries.append(entry)
    path = _overview_path(repo_url)
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def load_overview_cache_raw(repo_url: str) -> dict[str, dict] | None:
    path = _overview_path(repo_url)
    if not path.exists():
        return None
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not entries:
        return None
    return {e["filename"]: e for e in entries if "filename" in e}


def load_overview_context(repo_url: str) -> str | None:
    path = _overview_path(repo_url)
    if not path.exists():
        return None

    entries = json.loads(path.read_text(encoding="utf-8"))
    if not entries:
        return None

    lines = [
        "## Projektkontext",
        f"Das Projekt enthält {len(entries)} Dateien. Übersicht:",
        "",
    ]
    char_budget = _ACTIVE_MAX_CONTEXT_CHARS - sum(len(l) + 1 for l in lines)

    for e in entries:
        summary = e.get("file_summary", "").replace("\n", " ").strip()
        if not summary:
            summary = "(keine Zusammenfassung)"
        if len(summary) > 120:
            summary = summary[:117] + "..."
        line = f"- [{e.get('category', '?')}] {e['filename']} → {summary}"
        if len(line) + 1 > char_budget:
            lines.append(f"... und {len(entries) - len(lines) + 3} weitere Dateien")
            break
        lines.append(line)
        char_budget -= len(line) + 1

    return "\n".join(lines)


def build_inventory_context(repo_url: str) -> str | None:
    path = _overview_path(repo_url)
    if not path.exists():
        return None
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not entries:
        return None

    lines = [
        "## Projekt-Inventar (alle Dateien)",
        "",
    ]
    char_budget = _ACTIVE_MAX_CONTEXT_CHARS - sum(len(l) + 1 for l in lines)

    for e in entries:
        fn = e.get("filename", "")
        cat = e.get("category", "?")
        ftype = e.get("file_type", "")
        responsibilities = e.get("key_responsibilities", [])

        summary = e.get("file_summary", "").replace("\n", " ").strip()
        if len(summary) > 100:
            summary = summary[:97] + "..."

        parts = [f"- [{cat}]"]
        if ftype:
            parts.append(f"type={ftype}")
        parts.append(fn)
        if summary:
            parts.append(f"→ {summary}")

        line = " ".join(parts)

        if responsibilities:
            resp_str = "  Verantwortlichkeiten: " + "; ".join(responsibilities[:3])
            if len(line) + len(resp_str) + 1 <= char_budget:
                line += resp_str
                char_budget -= len(resp_str)

        if len(line) + 1 > char_budget:
            lines.append(f"... und {len(entries) - len(lines) + 1} weitere Dateien")
            break

        lines.append(line)
        char_budget -= len(line) + 1

    return "\n".join(lines)


def build_dependency_context(repo_url: str, file: dict) -> str | None:
    path = _overview_path(repo_url)
    if not path.exists():
        return None
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    entry_map: dict[str, dict] = {e["filename"]: e for e in entries}
    current = entry_map.get(file.get("filename", ""))
    if current is None:
        return None

    lines = [
        "## Referenzierte Dateien",
        "",
    ]
    char_budget = _ACTIVE_MAX_CONTEXT_CHARS

    self_summary = current.get("file_summary", "").replace("\n", " ").strip()
    lines.append(f"### {file.get('filename')} (diese Datei)")
    if self_summary:
        lines.append(f"  {self_summary}")
    deps = current.get("dependencies", [])
    if deps:
        lines.append(f"  Abhängigkeiten: {', '.join(deps[:6])}")
    char_budget -= sum(len(l) + 1 for l in lines)

    referenced: list[dict] = []
    for dep in deps:
        dep_key = dep.split("\\")[-1].split(".")[-1]
        for fn, entry in entry_map.items():
            if dep_key.lower() in fn.lower() or dep_key.lower() in entry.get("file_summary", "").lower():
                referenced.append(entry)
                break

    seen: set[str] = {file.get("filename", "")}
    for ref in referenced:
        if ref["filename"] in seen:
            continue
        seen.add(ref["filename"])
        ref_summary = ref.get("file_summary", "").replace("\n", " ").strip()
        if len(ref_summary) > 100:
            ref_summary = ref_summary[:97] + "..."
        line = f"- {ref['filename']}: {ref_summary}"
        if len(line) + 1 > char_budget:
            break
        lines.append(line)
        char_budget -= len(line) + 1

    return "\n".join(lines)
