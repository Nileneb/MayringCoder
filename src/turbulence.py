"""Turbulenz-Analyse: Chunking, Kategorisierung, Score, Report.

Öffentliche API:
    analyze_repo(repo_path, use_llm, model, overview_cache) -> dict
    build_markdown(report, repo_url, model, elapsed, full_scan) -> str

Merged from: turbulence_analyzer.py, turbulence_calculator.py, turbulence_report.py
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from src.ollama_client import generate as _ollama_generate

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def _default_model() -> str:
    from src.model_router import ModelRouter
    return ModelRouter(_OLLAMA_URL).resolve("analysis") or "mayring-qwen3:2b"

THRESHOLD_SKIP = 0.20
THRESHOLD_HIGH_ONLY = 0.50
THRESHOLD_DEEP = 0.50
MIN_CHUNKS_FOR_TRIAGE = 3
SIMILARITY_THRESHOLD = 0.70
CATEGORIES = ["Daten", "UI", "Sicherheit", "KI", "Logik", "Config"]

_LARAVEL_KONTEXT = """
Laravel-Konventionen (NICHT als Smell melden):
- Relationships gehören ins Eloquent Model
- DB::getDriverName()-Guards in Migrations = gewollt (Multi-DB-Support)
- Test-Fixtures (::create/::factory in Tests) = kein Zombie-Code
- app(Service::class) = Laravel Service Container = korrekt
"""

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    file: str
    start_line: int
    end_line: int
    code: str
    category: str = ""
    functional_name: str = ""


@dataclass
class FileAnalysis:
    path: str
    total_lines: int
    chunks: list = field(default_factory=list)
    turbulence_score: float = 0.0
    hot_zones: list = field(default_factory=list)
    findings: list = field(default_factory=list)
    tier: str = ""


@dataclass
class Redundancy:
    name_a: str
    file_a: str
    name_b: str
    file_b: str
    similarity: float
    verdict: str = ""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunkify(filepath: str, chunk_size: int = 15) -> list[Chunk]:
    path = Path(filepath)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return []

    chunks = []
    current_start = 0
    while current_start < len(lines):
        end = min(current_start + chunk_size, len(lines))
        if end < len(lines):
            for look in range(end, max(current_start + 5, end - 5), -1):
                line = lines[look].strip()
                if (line == "" or
                        line.startswith(("function ", "public ", "private ",
                                         "protected ", "class ", "def ", "<?php"))):
                    end = look
                    break
        chunks.append(Chunk(
            file=filepath,
            start_line=current_start + 1,
            end_line=end,
            code="\n".join(lines[current_start:end]),
        ))
        current_start = end
    return chunks


# ---------------------------------------------------------------------------
# Categorization
# ---------------------------------------------------------------------------

_CATEGORIZE_PROMPT = """Analysiere diesen Code-Abschnitt. Antworte NUR mit einem JSON-Objekt:
{{
  "category": "<eine von: Daten, UI, Sicherheit, KI, Logik, Config>",
  "functional_name": "<kurzer deutscher Name>"
}}

Code ({file}, Zeile {start}-{end}):
```
{code}
```"""


def categorize_chunk_llm(chunk: Chunk, model: str | None = None) -> Chunk:
    _model = model or _default_model()
    prompt = _CATEGORIZE_PROMPT.format(
        file=Path(chunk.file).name, start=chunk.start_line,
        end=chunk.end_line, code=chunk.code[:2000],
    )
    try:
        text = _ollama_generate(
            _OLLAMA_URL, _model, prompt,
            stream=False, timeout=30.0,
        )
        m = re.search(r'\{[^}]+\}', text)
        if m:
            data = json.loads(m.group())
            chunk.category = data.get("category", "Logik")
            chunk.functional_name = data.get("functional_name", "unbekannt")
            return chunk
    except Exception:
        pass
    chunk.category = "Logik"
    chunk.functional_name = "nicht_erkannt"
    return chunk


def categorize_chunk_heuristic(chunk: Chunk) -> Chunk:
    code = chunk.code.lower()
    scores = {cat: 0 for cat in CATEGORIES}
    for kw in ["::create", "::find", "::where", "->save", "->delete",
                "migration", "schema::", "$fillable", "belongsto", "hasmany"]:
        if kw in code:
            scores["Daten"] += 2
    for kw in ["blade", "<div", "<form", "wire:", "@if", "@foreach",
                "class=", "x-data"]:
        if kw in code:
            scores["UI"] += 2
    for kw in ["auth::", "validate", "authorize", "policy", "gate::",
                "middleware", "sanctum", "abort(403"]:
        if kw in code:
            scores["Sicherheit"] += 2
    for kw in ["langdock", "agent", "ollama", "embedding", "llm",
                "openai", "anthropic", "chat/completions"]:
        if kw in code:
            scores["KI"] += 3
    for kw in ["route::", "config(", "env(", "return ["]:
        if kw in code:
            scores["Config"] += 1
    scores["Logik"] += 1
    chunk.category = max(scores, key=scores.get)
    m = re.search(r'(?:function|public|private|protected)\s+(\w+)\s*\(', chunk.code)
    chunk.functional_name = m.group(1) if m else f"block_{chunk.start_line}"
    return chunk


# ---------------------------------------------------------------------------
# Turbulence score
# ---------------------------------------------------------------------------


def calculate_turbulence(chunks: list[Chunk], window: int = 5) -> tuple[float, list]:
    if len(chunks) < 2:
        return 0.0, []
    categories = [c.category for c in chunks]
    changes = []
    for i in range(len(categories)):
        start = max(0, i - window // 2)
        end = min(len(categories), i + window // 2 + 1)
        win = categories[start:end]
        change_count = sum(1 for a, b in zip(win, win[1:]) if a != b)
        unique_count = len(set(win))
        changes.append((change_count / max(1, len(win) - 1)) * (unique_count / len(CATEGORIES)))

    overall = sum(changes) / len(changes)
    hot_zones = []
    in_zone = False
    zone_start = None
    for i, score in enumerate(changes):
        if score > THRESHOLD_DEEP and not in_zone:
            in_zone, zone_start = True, chunks[i].start_line
        elif score <= THRESHOLD_DEEP and in_zone:
            in_zone = False
            hot_zones.append({
                "start_line": zone_start,
                "end_line": chunks[i - 1].end_line,
                "peak_score": max(changes[max(0, i - 5):i]),
            })
    if in_zone:
        hot_zones.append({
            "start_line": zone_start,
            "end_line": chunks[-1].end_line,
            "peak_score": max(changes[-5:]),
        })
    return round(overall, 3), hot_zones


# ---------------------------------------------------------------------------
# Redundancy detection
# ---------------------------------------------------------------------------


def find_redundancies(all_chunks: list[Chunk]) -> list[Redundancy]:
    named = [
        c for c in all_chunks
        if c.functional_name
        and not c.functional_name.startswith(("block_", "fehler_"))
        and c.functional_name not in ("unbekannt", "nicht_erkannt")
    ]
    redundancies = []
    seen: set[tuple] = set()
    for i, a in enumerate(named):
        for b in named[i + 1:]:
            if a.file == b.file:
                continue
            key = tuple(sorted([f"{a.file}:{a.functional_name}", f"{b.file}:{b.functional_name}"]))
            if key in seen:
                continue
            seen.add(key)
            name_sim = SequenceMatcher(None, a.functional_name.lower(), b.functional_name.lower()).ratio()
            if name_sim < SIMILARITY_THRESHOLD:
                continue
            code_sim = SequenceMatcher(None, a.code[:600].lower(), b.code[:600].lower()).ratio()
            redundancies.append(Redundancy(
                name_a=a.functional_name, file_a=f"{a.file}:{a.start_line}",
                name_b=b.functional_name, file_b=f"{b.file}:{b.start_line}",
                similarity=round(0.35 * name_sim + 0.65 * code_sim, 2),
            ))
    redundancies.sort(key=lambda r: -r.similarity)
    return redundancies


# ---------------------------------------------------------------------------
# Hot-zone deep analysis
# ---------------------------------------------------------------------------

_DEEP_PROMPT = """Du bist ein Code-Reviewer. Analysiere diesen turbulenten Code-Abschnitt.
{konventionen}
Antworte NUR mit JSON:
{{"problem": "...", "refactoring": "...", "severity": "low|medium|high", "confidence": "low|medium|high"}}

Datei: {file} (Zeile {start}-{end}), Turbulenz: {score}
```
{code}
```"""


def deep_analyze_hotzone(
    filepath: str, start: int, end: int, score: float,
    use_llm: bool = True, model: str | None = None,
) -> Optional[dict]:
    if not use_llm:
        return {
            "problem": f"Hohe Turbulenz ({score:.0%}) – vermischte Verantwortlichkeiten",
            "refactoring": "Manuelle Prüfung empfohlen",
            "severity": "high" if score > 0.7 else "medium",
            "confidence": "high",
        }
    _model = model or _default_model()
    lines = Path(filepath).read_text(encoding="utf-8", errors="replace").splitlines()
    code = "\n".join(lines[max(0, start - 1):end])
    prompt = _DEEP_PROMPT.format(
        konventionen=_LARAVEL_KONTEXT, file=Path(filepath).name,
        start=start, end=end, score=f"{score:.0%}", code=code[:3000],
    )
    try:
        text = _ollama_generate(
            _OLLAMA_URL, _model, prompt,
            stream=False, timeout=60.0,
        )
        m = re.search(r'\{[^}]+\}', text)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def build_report(analyses: list[FileAnalysis], redundancies: list[Redundancy]) -> dict:
    analyses.sort(key=lambda a: -a.turbulence_score)
    stable = [a for a in analyses if a.tier == "stable"]
    skipped = [a for a in analyses if a.tier == "skip"]
    light = [a for a in analyses if a.tier == "light"]
    deep = [a for a in analyses if a.tier == "deep"]
    stable_total = stable + skipped
    total_findings = sum(len(a.findings) for a in analyses)

    print("\n" + "=" * 60)
    print("TURBULENZ-REPORT")
    print("=" * 60)
    print(f"  Dateien gesamt: {len(analyses)}")
    print(f"  Kritisch (>50%): {len(deep)}")
    print(f"  Mittel (20-50%): {len(light)}")
    print(f"  Stabil: {len(stable_total)}")
    print(f"  Findings: {total_findings}")
    print(f"  Redundanzen: {len(redundancies)}")

    return {
        "summary": {
            "total_files": len(analyses),
            "critical": len(deep), "medium": len(light),
            "stable": len(stable_total),
            "findings": total_findings, "redundancies": len(redundancies),
        },
        "critical_files": [
            {"path": a.path, "turbulence": a.turbulence_score,
             "hot_zones": a.hot_zones, "findings": a.findings}
            for a in deep
        ],
        "all_files": [
            {"path": a.path, "tier": a.tier, "turbulence": a.turbulence_score,
             "hot_zones": a.hot_zones, "findings": a.findings}
            for a in analyses
        ],
        "redundancies": [
            {"name_a": r.name_a, "file_a": r.file_a, "name_b": r.name_b,
             "file_b": r.file_b, "similarity": r.similarity}
            for r in redundancies[:20]
        ],
    }


def build_markdown(report: dict, repo_url: str, model: str, elapsed: float,
                   full_scan: bool = False) -> str:
    s = report["summary"]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "---", f"repo: {repo_url}", f"date: {datetime.now().isoformat()}",
        f"model: {model}", "mode: turbulence",
        f"files_total: {s.get('total_files', 0)}", f"run_time_s: {elapsed:.1f}",
    ]
    if full_scan:
        lines.append("full_scan: true")
    lines += [
        "---", "",
        f"# Turbulenz-Analyse — {ts}", "",
        "## Zusammenfassung", "",
        "| Metrik | Wert |", "|--------|------|",
        f"| Dateien gesamt | {s.get('total_files', 0)} |",
        f"| Kritisch (>50%) | {s.get('critical', 0)} |",
        f"| Mittel (20-50%) | {s.get('medium', 0)} |",
        f"| Stabil (<20%) | {s.get('stable', 0)} |",
        f"| Findings | {s.get('findings', 0)} |",
        f"| Redundanzen | {s.get('redundancies', 0)} |", "",
    ]
    for f in report.get("critical_files", []):
        pct = round(f["turbulence"] * 100)
        lines.append(f"### `{f['path']}` — {pct}% Turbulenz")
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
    for r in report.get("redundancies", []):
        lines.append(
            f"- `{r['name_a']}` ≈ `{r['name_b']}` "
            f"({round(r['similarity'] * 100)}% Ähnlichkeit)"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def analyze_repo(
    repo_path: str,
    use_llm: bool = False,
    model: str | None = None,
    overview_cache: dict | None = None,
) -> dict:
    repo = Path(repo_path)
    files: list[Path] = []
    for ext in ["**/*.php", "**/*.blade.php", "**/*.js", "**/*.ts"]:
        files.extend(repo.glob(ext))

    seen: set[Path] = set()
    filtered: list[Path] = []
    for f in files:
        resolved = f.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        rel = str(f.relative_to(repo))
        if any(s in rel for s in ["vendor/", "node_modules/", ".git/", "storage/"]):
            continue
        filtered.append(f)

    print(f"\n{len(filtered)} Dateien gefunden in {repo_path}")
    categorize_fn = categorize_chunk_llm if use_llm else categorize_chunk_heuristic
    _ov_hits = 0
    all_analyses: list[FileAnalysis] = []
    all_chunks: list[Chunk] = []

    for filepath in sorted(filtered):
        rel_path = str(filepath.relative_to(repo))
        chunks = chunkify(str(filepath))
        if not chunks:
            continue

        ov_entry = (overview_cache or {}).get(rel_path)
        for chunk in chunks:
            if ov_entry and ov_entry.get("category"):
                categorize_chunk_heuristic(chunk)
                chunk.category = ov_entry["category"]
                _ov_hits += 1
            elif use_llm:
                categorize_chunk_llm(chunk, model=model)
            else:
                categorize_chunk_heuristic(chunk)

        all_chunks.extend(chunks)

        if len(chunks) < MIN_CHUNKS_FOR_TRIAGE:
            tier, turb_score, hot_zones = "stable", 0.0, []
        else:
            turb_score, hot_zones = calculate_turbulence(chunks)
            tier = "skip" if turb_score < THRESHOLD_SKIP else (
                "light" if turb_score < THRESHOLD_DEEP else "deep"
            )

        analysis = FileAnalysis(
            path=rel_path,
            total_lines=sum(c.end_line - c.start_line + 1 for c in chunks),
            chunks=chunks, turbulence_score=turb_score,
            hot_zones=hot_zones, tier=tier,
        )
        if ov_entry and hot_zones:
            for zone in hot_zones:
                zone["affected_functions"] = ov_entry.get("functions", [])

        if tier == "deep" and hot_zones:
            for zone in hot_zones[:3]:
                finding = deep_analyze_hotzone(
                    str(filepath), zone["start_line"], zone["end_line"],
                    zone["peak_score"], use_llm=use_llm, model=model,
                )
                if finding:
                    analysis.findings.append({"zone": zone, **finding})

        icon = {"stable": ".", "skip": ".", "light": "~", "deep": "!"}[tier]
        print(f"  {icon} {rel_path:50s} {turb_score:.0%} [{tier}]")
        all_analyses.append(analysis)

    redundancies = find_redundancies(all_chunks)
    return build_report(all_analyses, redundancies)
