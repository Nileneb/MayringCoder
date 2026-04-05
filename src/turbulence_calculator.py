"""turbulence_calculator.py — Reine Berechnungslogik für Turbulenz-Analyse.

Enthält:
- Konstanten und Schwellenwerte
- Dataklassen: Chunk, FileAnalysis, Redundancy
- Datei → Chunks zerlegen (chunkify)
- Chunk-Kategorisierung: heuristisch und per LLM
- Turbulenz-Score + Hot-Zone-Erkennung
- Redundanz-Erkennung über Datei-Grenzen
- Hot-Zone-Tiefenanalyse (heuristisch oder per LLM)

Keine I/O-Seiteneffekte außer Ollama-HTTP-Aufrufen.
Kein print(). Kein SQLite. Kein Markdown.
"""

import json
import os
import re
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

# ── Konfiguration ──────────────────────────────────────────────

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("TURB_MODEL", "mistral:7b-instruct")

# Turbulenz-Schwellen
THRESHOLD_SKIP = 0.20       # < 20% → Datei überspringen
THRESHOLD_HIGH_ONLY = 0.50  # 20-50% → nur high-confidence Findings
THRESHOLD_DEEP = 0.50       # > 50% → volle Tiefenanalyse + Refactoring

# Mindestanzahl Chunks für sinnvollen Turbulenz-Score.
# Dateien mit weniger Chunks haben zu wenig Datenpunkte — der Score wäre
# ein mathematisches Artefakt (z. B. exakt 50% bei 2 verschiedenen Chunks).
MIN_CHUNKS_FOR_TRIAGE = 3

# Ähnlichkeitsschwelle für Redundanz-Erkennung
SIMILARITY_THRESHOLD = 0.70

# Kategorien (angelehnt an die Farbcodierung)
CATEGORIES = ["Daten", "UI", "Sicherheit", "KI", "Logik", "Config"]

# Framework-Konventionen (NICHT als Problem melden)
LARAVEL_KONTEXT = """
Laravel-Konventionen (NICHT als Smell melden):
- Relationships gehören ins Eloquent Model (kein Overengineering)
- DB::getDriverName()-Guards in Migrations = gewollt (Multi-DB-Support)
- Test-Fixtures (::create/::factory in Tests) = kein Zombie-Code
- belongsTo() mit/ohne explizitem FK = beides korrekt
- ViewRecord/ListRecords = Filament-Standard-Klassen
- $timestamps = false = bewusste Entscheidung
- app(Service::class) = Laravel Service Container = korrekt
- if ($x === null) { return; } in Jobs = Standard-Guard, kein Overengineering
"""


# ── Datenstrukturen ────────────────────────────────────────────

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
    tier: str = ""  # "skip" | "light" | "deep"


@dataclass
class Redundancy:
    name_a: str
    file_a: str
    name_b: str
    file_b: str
    similarity: float
    verdict: str = ""  # "echte_redundanz" | "aehnlicher_zweck" | "unklar"


# ── Schritt 1: Datei in Chunks zerlegen ────────────────────────

def chunkify(filepath: str, chunk_size: int = 15) -> list[Chunk]:
    """Zerlegt eine Datei in Blöcke à ~chunk_size Zeilen.

    Versucht an logischen Grenzen zu trennen (Leerzeilen,
    Funktions-/Methodendefinitionen), fällt aber auf feste
    Größe zurück wenn keine gefunden werden.
    """
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

        # Versuche an einer logischen Grenze zu trennen
        if end < len(lines):
            for look in range(end, max(current_start + 5, end - 5), -1):
                line = lines[look].strip()
                if (line == "" or
                        line.startswith("function ") or
                        line.startswith("public ") or
                        line.startswith("private ") or
                        line.startswith("protected ") or
                        line.startswith("class ") or
                        line.startswith("def ") or
                        line.startswith("<?php")):
                    end = look
                    break

        chunk_code = "\n".join(lines[current_start:end])
        chunks.append(Chunk(
            file=filepath,
            start_line=current_start + 1,
            end_line=end,
            code=chunk_code,
        ))
        current_start = end

    return chunks


# ── Schritt 2: LLM-Kategorisierung ────────────────────────────

CATEGORIZE_PROMPT = """Analysiere diesen Code-Abschnitt. Antworte NUR mit einem JSON-Objekt:
{{
  "category": "<eine von: Daten, UI, Sicherheit, KI, Logik, Config>",
  "functional_name": "<kurzer deutscher Name für die Funktion, z.B. 'chat_nachricht_speichern'>"
}}

Kategorien:
- Daten: Datenbankzugriffe, Models, Queries, Migrations
- UI: Templates, HTML, CSS, Blade-Direktiven, Formulare
- Sicherheit: Auth, Validierung, Guards, Policies, null-checks
- KI: API-Aufrufe an LLMs/Agenten, Embeddings, externe KI-Services
- Logik: Geschäftslogik, Berechnungen, Transformationen
- Config: Konfigurationsdateien, Routen, Service-Provider

Code-Abschnitt ({file}, Zeile {start}-{end}):
```
{code}
```

Antworte NUR mit dem JSON-Objekt, kein weiterer Text."""


def categorize_chunk_llm(chunk: Chunk, model: str | None = None) -> Chunk:
    """Ruft das LLM auf, um einen Chunk zu kategorisieren.

    Args:
        chunk: Der zu kategorisierende Chunk.
        model: Ollama-Modellname. Wenn None, wird der Modulwert MODEL verwendet.
    """
    import urllib.request

    _model = model if model is not None else MODEL

    prompt = CATEGORIZE_PROMPT.format(
        file=Path(chunk.file).name,
        start=chunk.start_line,
        end=chunk.end_line,
        code=chunk.code[:2000],
    )

    payload = json.dumps({
        "model": _model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 100},
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            text = result.get("response", "")

            match = re.search(r'\{[^}]+\}', text)
            if match:
                data = json.loads(match.group())
                chunk.category = data.get("category", "Logik")
                chunk.functional_name = data.get("functional_name", "unbekannt")
            else:
                chunk.category = "Logik"
                chunk.functional_name = "nicht_erkannt"
    except Exception as e:
        chunk.category = "Logik"
        chunk.functional_name = f"fehler_{e.__class__.__name__}"

    return chunk


def categorize_chunk_heuristic(chunk: Chunk) -> Chunk:
    """Schnelle regelbasierte Kategorisierung (Fallback ohne LLM)."""
    code = chunk.code.lower()

    scores = {cat: 0 for cat in CATEGORIES}

    # Daten-Signale
    for kw in ["::create", "::find", "::where", "->save", "->delete",
                "migration", "schema::", "->table", "->column",
                "$fillable", "$casts", "belongsto", "hasmany", "factory"]:
        if kw in code:
            scores["Daten"] += 2

    # UI-Signale
    for kw in ["blade", "<div", "<form", "wire:", "@if", "@foreach",
                "@forelse", "class=", "wire:click", "wire:model",
                "$this->dispatch", "loading", "x-data"]:
        if kw in code:
            scores["UI"] += 2

    # Sicherheit-Signale
    for kw in ["auth::", "validate", "authorize", "policy", "gate::",
                "middleware", "sanctum", "guard", "permission",
                "=== null", "!== null", "abort(403"]:
        if kw in code:
            scores["Sicherheit"] += 2

    # KI-Signale
    for kw in ["langdock", "agent", "ollama", "embedding", "llm",
                "openai", "anthropic", "mistral", "chat/completions"]:
        if kw in code:
            scores["KI"] += 3

    # Config-Signale
    for kw in ["route::", "config(", "env(", "'key' =>",
                "return [", "service"]:
        if kw in code:
            scores["Config"] += 1

    # Logik als Default
    scores["Logik"] += 1

    chunk.category = max(scores, key=scores.get)

    func_match = re.search(
        r'(?:function|public|private|protected)\s+(\w+)\s*\(',
        chunk.code,
    )
    if func_match:
        chunk.functional_name = func_match.group(1)
    else:
        chunk.functional_name = f"block_{chunk.start_line}"

    return chunk


# ── Schritt 3: Turbulenz berechnen ─────────────────────────────

def calculate_turbulence(chunks: list[Chunk], window: int = 5) -> tuple[float, list]:
    """Berechnet den Turbulenz-Score einer Datei.

    Turbulenz = gewichteter Durchschnitt der Kategorie-Wechsel
    in einem gleitenden Fenster.

    Returns: (score 0.0-1.0, liste der Hot-Zones)
    """
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

        score = (change_count / max(1, len(win) - 1)) * (unique_count / len(CATEGORIES))
        changes.append(score)

    overall = sum(changes) / len(changes)

    hot_zones = []
    in_zone = False
    zone_start = None

    for i, score in enumerate(changes):
        if score > THRESHOLD_DEEP and not in_zone:
            in_zone = True
            zone_start = chunks[i].start_line
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


# ── Schritt 4: Redundanz-Erkennung ─────────────────────────────

def find_redundancies(all_chunks: list[Chunk]) -> list[Redundancy]:
    """Vergleicht funktionale Namen und Code-Inhalt über alle Dateien hinweg.

    Score-Berechnung (zweistufig):
    1. Name-Ähnlichkeit (SequenceMatcher) als Vorfilter — nur Paare >= SIMILARITY_THRESHOLD
    2. Finaler Score = 35% Name-Ähnlichkeit + 65% Code-Ähnlichkeit (erste 600 Zeichen)

    Damit sind Werte wie 0.42 / 0.78 möglich statt immer 1.0 bei identischen Namen.
    """
    named = [
        c for c in all_chunks
        if c.functional_name
        and not c.functional_name.startswith("block_")
        and not c.functional_name.startswith("fehler_")
        and c.functional_name not in ("unbekannt", "nicht_erkannt")
    ]

    redundancies = []
    seen_pairs: set[tuple] = set()

    for i, a in enumerate(named):
        for b in named[i + 1:]:
            if a.file == b.file:
                continue

            pair_key = tuple(sorted([
                f"{a.file}:{a.functional_name}",
                f"{b.file}:{b.functional_name}",
            ]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            name_sim = SequenceMatcher(
                None,
                a.functional_name.lower(),
                b.functional_name.lower(),
            ).ratio()

            if name_sim < SIMILARITY_THRESHOLD:
                continue

            code_sim = SequenceMatcher(
                None,
                a.code[:600].lower(),
                b.code[:600].lower(),
            ).ratio()

            sim = round(0.35 * name_sim + 0.65 * code_sim, 2)

            redundancies.append(Redundancy(
                name_a=a.functional_name,
                file_a=f"{a.file}:{a.start_line}",
                name_b=b.functional_name,
                file_b=f"{b.file}:{b.start_line}",
                similarity=sim,
            ))

    redundancies.sort(key=lambda r: -r.similarity)
    return redundancies


# ── Schritt 5: Tiefenanalyse (nur für Hot-Zones) ──────────────

DEEP_ANALYSIS_PROMPT = """Du bist ein Code-Reviewer. Analysiere diesen Code-Abschnitt.
Er wurde als "turbulent" erkannt (viele vermischte Verantwortlichkeiten).

{konventionen}

Beantworte NUR diese Fragen im JSON-Format:
{{
  "problem": "<1 Satz: Was genau ist hier vermischt?>",
  "refactoring": "<1 Satz: Wie könnte man es trennen?>",
  "severity": "<low|medium|high>",
  "confidence": "<low|medium|high>"
}}

Datei: {file} (Zeile {start}-{end})
Turbulenz-Score: {score}

```
{code}
```

Antworte NUR mit dem JSON-Objekt."""


def deep_analyze_hotzone(
    filepath: str,
    start: int,
    end: int,
    score: float,
    use_llm: bool = True,
    model: str | None = None,
) -> Optional[dict]:
    """Analysiert eine Hot-Zone heuristisch oder per LLM.

    Args:
        filepath: Pfad zur Quelldatei.
        start:    Startzeile der Hot-Zone.
        end:      Endzeile der Hot-Zone.
        score:    Turbulenz-Score der Zone.
        use_llm:  True = LLM-Analyse, False = Heuristik.
        model:    Ollama-Modellname. Wenn None, wird MODEL verwendet.
    """
    if not use_llm:
        return {
            "problem": f"Hohe Turbulenz ({score:.0%}) – vermischte Verantwortlichkeiten",
            "refactoring": "Manuelle Prüfung empfohlen",
            "severity": "high" if score > 0.7 else "medium",
            "confidence": "high",
        }

    import urllib.request

    _model = model if model is not None else MODEL

    path = Path(filepath)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    code = "\n".join(lines[max(0, start - 1):end])

    prompt = DEEP_ANALYSIS_PROMPT.format(
        konventionen=LARAVEL_KONTEXT,
        file=path.name,
        start=start,
        end=end,
        score=f"{score:.0%}",
        code=code[:3000],
    )

    payload = json.dumps({
        "model": _model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 200},
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            text = result.get("response", "")
            match = re.search(r'\{[^}]+\}', text)
            if match:
                return json.loads(match.group())
    except Exception:
        pass

    return None
