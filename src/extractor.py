"""Extraction of structured findings from raw (non-JSON) LLM output.

Strategy: Two-stage output (separates analysis from format compliance).
  Stage 1 — Primary analysis: returns freetext (any format is acceptable).
  Stage 2 — Extraction: a second, simple prompt extracts only findings with
             mandatory fields (file, line, type, reasoning, recommendation).
             Everything that lacks these fields is discarded.

This makes the format problem orthogonal to the analysis problem.
"""

import json
import re
from pathlib import Path

from src.analyzer import _load_prompt

try:
    _EXTRACT_PROMPT = _load_prompt(Path(__file__).parent.parent / "prompts" / "extract_findings.md")
except Exception:
    _EXTRACT_PROMPT = """Du erhältst eine rohe LLM-Antwort (Freitext oder teilweise formatiert).

Extrahiere daraus ALLE Findings (= kodierte Erkenntnisse), die alle 5 Pflichtfelder haben:
  1. datei:      Welche Datei ist betroffen? (Dateiname oder "gesamte Datei")
  2. zeile:      Ungefähre Zeilennummer (z.B. "~42") oder leer ("")
  3. typ:        Einer von: zombie_code, redundanz, inkonsistenz,
                 fehlerbehandlung, overengineering, sicherheit, unklar, freitext
  4. begründung: Kurze Erklärung (1–2 Sätze), WARUM das ein Problem ist
  5. empfehlung: Konkrete Handlungsempfehlung (1 Satz)

Wenn ein Finding nicht alle 5 Pflichtfelder hat → IGNORIEREN (nicht erfinden).
Wenn keine gültigen Findings vorhanden → leeres Array zurückgeben.

Antworte NUR mit diesem JSON-Format, keine Prosa:

{
  "findings": [
    {
      "datei": "src/UserController.php",
      "zeile": "~23",
      "typ": "redundanz",
      "begründung": "Dieselbe Validierungslogik existiert bereits in...",
      "empfehlung": "Auslagern in gemeinsame Validierungsklasse"
    }
  ]
}
"""


# ---------------------------------------------------------------------------
# Signature extraction — static analysis helpers
# ---------------------------------------------------------------------------

_SMELL_KEYWORDS_RE = re.compile(
    r"\b(zombie[_\s]?code|redundan[zt]|inkonsistenz|fehlerbehandlung|overengineering"
    r"|sicherheit|security|dead[_\s]?code|performance|problem|issue|bug|smell|finding"
    r"|code[_\s]?qualit[äa]t|verbesserung|refactor)\b",
    re.IGNORECASE,
)
_LINE_HINT_RE = re.compile(r"[Zz]eile\s*:?\s*~?(\d+)|[Ll]ine\s*:?\s*~?(\d+)|[Ll]ine\s+(\d+)", re.IGNORECASE)


def _regex_extract_findings(raw: str, filename: str) -> list[dict]:
    """Fast heuristic extraction without a second LLM call.

    Splits the raw output on bullet/numbered list markers and looks for
    chunks that contain smell keywords. Each matching chunk becomes a
    low-confidence finding. Returns an empty list when no keywords are found —
    in that case the caller may fall back to the LLM extractor.
    """
    # Split on bullet/numbered-list markers
    chunks = re.split(r"\n+(?=\s*[-*•#]|\s*\d+[.)]\s)", raw.strip())
    findings: list[dict] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) < 30:
            continue
        if not _SMELL_KEYWORDS_RE.search(chunk):
            continue
        m = _LINE_HINT_RE.search(chunk)
        line_hint = ""
        if m:
            line_hint = "~" + (m.group(1) or m.group(2) or m.group(3))
        findings.append({
            "type":             "freitext",
            "line_hint":        line_hint,
            "evidence_excerpt": chunk[:200].replace("\n", " "),
            "fix_suggestion":   "",
            "confidence":       "low",
            "severity":         "info",
            "source":           "regex_extraction",
        })
        if len(findings) >= 10:
            break
    return findings


_IMPORT_RE = re.compile(
    r"^(?:from\s+([\w.]+)\s+import\s+|import\s+)(.+?)$", re.MULTILINE
)
_METHOD_RE = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(", re.MULTILINE)
_CLASS_RE = re.compile(r"^\s*class\s+(\w+)", re.MULTILINE)


def extract_python_signatures(content: str) -> dict:
    """Extract function names, class names, and imports from Python source.

    Returns:
        {
            "functions": ["handle_request", "create_user", ...],
            "classes":   ["UserController", "Service", ...],
            "imports":   ["django.http", "rest_framework", ...],
        }
    """
    imports: list[str] = []
    for m in _IMPORT_RE.finditer(content):
        if m.group(1):
            imports.append(m.group(1).strip())
        else:
            for part in m.group(2).split(","):
                imports.append(part.strip().split(" as ")[0].strip())

    return {
        "functions": _METHOD_RE.findall(content),
        "classes": _CLASS_RE.findall(content),
        "imports": [i for i in imports if i],
    }


# ---------------------------------------------------------------------------
# Stage 2: Freetext → Structured findings extraction
# ---------------------------------------------------------------------------

def extract_freetext_findings(
    raw_response: str,
    ollama_url: str,
    model: str,
    filename: str,
    category: str,
) -> list[dict]:
    """Stage-2 extraction: extract structured findings from unstructured LLM output.

    Strategy (fastest first):
    1. Regex extraction — no network call, covers list/bullet-formatted output.
    2. LLM extraction — only if regex yields nothing; costs one more API call.

    Falls back to an empty list if both strategies fail or Ollama is unreachable.
    """
    # Fast path: regex-based extraction (no second LLM call)
    quick = _regex_extract_findings(raw_response, filename)
    if quick:
        return quick

    # Slow path: second LLM call only when regex found nothing
    prompt = (
        _EXTRACT_PROMPT
        + "\n\n## Rohe LLM-Antwort\n\n"
        + raw_response.strip()
    )

    try:
        from src.analyzer import _ollama_generate

        raw = _ollama_generate(prompt, ollama_url, model, f"[EXTRACT] {filename}")
    except Exception:
        return []

    # Try JSON fences first, then raw JSON, then fall back
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if not m:
        m = re.search(r"(\{.*\})", raw, re.DOTALL)
    candidate = m.group(1) if m else raw

    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return []

    findings_raw: list[dict] = parsed if isinstance(parsed, list) else parsed.get("findings", [])

    mandatory_keys = {"datei", "typ", "begründung", "empfehlung"}
    result: list[dict] = []

    for f in findings_raw:
        # All mandatory fields must be non-empty strings
        if not all(str(v).strip() for k, v in f.items() if k in mandatory_keys):
            continue
        result.append({
            "type":            f.get("typ", "freitext"),
            "line_hint":       f.get("zeile", ""),
            "evidence_excerpt": f.get("begründung", "")[:200],
            "fix_suggestion":  f.get("empfehlung", ""),
            "confidence":      "low",
            "severity":        "info",
            "source":          "freetext_extraction",
        })

    return result


# ---------------------------------------------------------------------------
# Redundancy detection via function-name similarity
# ---------------------------------------------------------------------------

def levenshtein_ratio(a: str, b: str) -> float:
    """Return normalised Levenshtein similarity in [0, 1]. 1 = identical."""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    # Simple DP Levenshtein
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i]
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return 1.0 - prev[lb] / max(la, lb)


def check_redundancy_by_names(
    overview_results: list[dict],
    threshold: float = 0.80,
) -> list[dict]:
    """Detect potential redundancy via function-name similarity.

    Step 1 — Build a flat list of all function names across all files.
    Step 2 — Compute pairwise Levenshtein similarity.
    Step 3 — Flag pairs above *threshold* as candidate redundancies.

    Returns a list of candidate redundancy findings. True redundancy requires
    LLM re-verification (caller may use ``needs_llm_review: True`` flag).
    """
    import itertools

    findings: list[dict] = []
    seen: set[tuple[str, str]] = set()

    # Collect all (filename, function_name) pairs
    items: list[tuple[str, str]] = []
    for r in overview_results:
        fn = r.get("filename", "")
        sigs = r.get("_signatures", {})
        for fname in sigs.get("functions", []):
            name = fname.strip().lower()
            if name and not name.startswith("_"):
                items.append((fn, name))

    # Pairwise similarity
    for (fa, na), (fb, nb) in itertools.combinations(items, 2):
        ratio = levenshtein_ratio(na, nb)
        if ratio < threshold:
            continue
        key = tuple(sorted([f"{fa}::{na}", f"{fb}::{nb}"]))
        if key in seen:
            continue
        seen.add(key)
        findings.append({
            "type":             "redundanz",
            "severity":         "info",
            "confidence":       "low",
            "line_hint":        "",
            "evidence_excerpt": (
                f"'{na}' und '{nb}' sind funktional ähnlich "
                f"(Levenshtein-Ähnlichkeit {ratio:.0%}). "
                f"Dateien: {fa}, {fb}"
            ),
            "fix_suggestion": (
                f"Prüfen ob '{na}' und '{nb}' dieselbe Logik implementieren. "
                f"Wenn ja: Zusammenführen oder eine Funktion eliminieren."
            ),
            "needs_llm_review": True,
            "source":           "name_redundancy_check",
        })

    return findings


# ---------------------------------------------------------------------------
# Adversarial validation — "Advocatus Diaboli"
# Every finding must pass a second LLM call before entering the report.
# This catches false positives where the recommendation would actually
# make the code worse or where the original assessment is wrong.
# ---------------------------------------------------------------------------

_ADV_PROMPT_TEMPLATE = """Du bist ein kritischer Code-Reviewer, der Fehlalarme aufdeckt.

Gegeben sei ein {finding_type} in einer {file_type}-Datei.

BEWERTUNGSKRITERIEN (nur eines von beidem):
- BESTÄTIGT → Der bestehende Code hat tatsächlich ein Problem und die Empfehlung würde es beheben.
- ABGELEHNT → Der bestehende Code ist bereits korrekt / die Empfehlung ist falsch / das "Problem" ist ein Framewor­k-Pattern / eine bewusste Designentscheidung / würde den Code verschlimmbessern.

CODE:
```
{evidence}
```

EMPFEHLUNG: {suggestion}

Antworte exakt in diesem Format (keine Prosa):
BESTÄTIGT: <ein kurzer Satz>
oder
ABGELEHNT: <ein kurzer Satz mit Begründung>"""


def _is_test_file(filename: str) -> bool:
    """Return True if *filename* is a test file (heuristic)."""
    import re as _re
    patterns = [
        _re.compile(r"(?:^|/)test[s]?[_\-].*\.(?:py|php|js|ts|go|java)$", _re.IGNORECASE),
        _re.compile(r"(?:^|/)(?:tests?|spec|__tests?__|test)", _re.IGNORECASE),
        _re.compile(r"_test\.(?:py|php|js|ts|go|java)$", _re.IGNORECASE),
    ]
    return any(p.search(filename) for p in patterns)


def _file_type_label(filename: str) -> str:
    """Return a human-readable file-type label."""
    if _is_test_file(filename):
        return "Test"
    import re as _re
    if _re.search(r"\.(?:php|blade\.php)$", filename, _re.IGNORECASE):
        return "PHP/Laravel"
    if _re.search(r"\.py$", filename, _re.IGNORECASE):
        return "Python/Django"
    if _re.search(r"\.(?:js|ts)$", filename, _re.IGNORECASE):
        return "JavaScript/TypeScript"
    if _re.search(r"\.go$", filename, _re.IGNORECASE):
        return "Go"
    return "Source"


def validate_findings(
    findings: list[dict],
    files: list[dict],
    ollama_url: str,
    model: str,
    min_confidence: str = "low",
) -> tuple[list[dict], dict]:
    """Adversarial validation: re-examine each finding through a second LLM call.

    Each finding is given to the "Advocatus Diaboli" prompt:
    "Does this finding hold, or is the existing code already correct?"

    Findings that are ABGELEHNT are marked with _adversarial_verdict = "ABGELEHNT"
    and can be filtered out by the caller (default: keep only BESTÄTIGT).

    Findings from test files are validated with relaxed criteria:
    Test fixtures, redundant assertions, and helper methods are explicitly
    NOT considered problems (test logic has different conventions).

    Args:
        findings:    List of findings from analyze_files.
        files:      File dicts (needed to look up content for evidence).
        ollama_url: Ollama base URL.
        model:      Model name.
        min_confidence: Minimum confidence to keep ("high", "medium", "low").

    Returns:
        (validated_findings, stats_dict) where stats_dict contains:
          - validated: count of BESTÄTIGT findings
          - rejected:  count of ABGELEHNT findings
          - errors:    count of validation failures (Ollama unreachable etc.)
    """
    from src.analyzer import _ollama_generate

    VALID_CONFIDENCE = {"high": 0, "medium": 1, "low": 2}
    min_rank = VALID_CONFIDENCE.get(min_confidence.lower(), 2)

    file_map = {f["filename"]: f for f in files}
    validated: list[dict] = []
    stats = {"validated": 0, "rejected": 0, "errors": 0, "below_confidence": 0}

    for f in findings:
        fn = f.get("_filename", "")
        conf_rank = VALID_CONFIDENCE.get(f.get("confidence", "medium").lower(), 1)
        if conf_rank > min_rank:
            stats["below_confidence"] += 1
            continue

        # Look up the file content for the evidence excerpt
        file_entry = file_map.get(fn, {})
        file_content = file_entry.get("content", "")
        is_test = _is_test_file(fn)
        ftype = _file_type_label(fn)
        finding_type = f.get("type", "Unknown")
        evidence = f.get("evidence_excerpt", "")[:500]
        suggestion = f.get("fix_suggestion", "")[:300]

        prompt = (
            _ADV_PROMPT_TEMPLATE
            .replace("{finding_type}", finding_type)
            .replace("{file_type}", ftype)
            .replace("{evidence}", evidence)
            .replace("{suggestion}", suggestion)
        )
        if is_test:
            prompt += (
                "\n\nWICHTIG FÜR TESTS: "
                "Test-Fixtures (::create/factory/make), redundante Assertions, "
                "und Helper-Methoden in Tests sind KEINE Smells. "
                "Nur echte Probleme (fehlende Assertions, unerreichbarer Code) bestätigen."
            )

        try:
            raw = _ollama_generate(prompt, ollama_url, model, f"[ADV] {fn}")
        except Exception:
            stats["errors"] += 1
            f["_adversarial_verdict"] = "ERROR"
            validated.append(f)
            stats["validated"] += 1
            continue

        verdict_raw = raw.strip()
        is_confirmed = verdict_raw.upper().startswith("BESTÄTIGT")
        f["_adversarial_verdict"] = "BESTÄTIGT" if is_confirmed else "ABGELEHNT"
        f["_adversarial_reason"] = verdict_raw[:200]

        if is_confirmed:
            validated.append(f)
            stats["validated"] += 1
        else:
            stats["rejected"] += 1

    return validated, stats


def filter_by_confidence(
    findings: list[dict],
    min_confidence: str = "low",
) -> list[dict]:
    """Drop findings below the minimum confidence threshold.

    Confidence levels (from highest to lowest):
        high → medium → low

    min_confidence="high"   → keeps only high
    min_confidence="medium" → keeps high + medium
    min_confidence="low"    → keeps all (default)
    """
    VALID_CONFIDENCE = {"high": 0, "medium": 1, "low": 2}
    min_rank = VALID_CONFIDENCE.get(min_confidence.lower(), 2)

    return [
        f for f in findings
        if VALID_CONFIDENCE.get(f.get("confidence", "medium").lower(), 1) <= min_rank
    ]
