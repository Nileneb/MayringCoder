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

# Load the extraction prompt from file; fall back to an inline default so the
# module works without the prompts directory (e.g. in isolated unit tests).
_EXTRACT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "extract_findings.md"
try:
    EXTRACT_PROMPT: str = _EXTRACT_PROMPT_PATH.read_text(encoding="utf-8")
except Exception:
    EXTRACT_PROMPT = """Du erhältst eine rohe LLM-Antwort (Freitext oder teilweise formatiert).

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

# Backward-compat alias — callers that imported the private name still work.
_EXTRACT_PROMPT = EXTRACT_PROMPT


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

def parse_freetext_findings(raw_response: str, filename: str) -> list[dict]:
    """Stage-2 fast path: extract findings from unstructured LLM output via regex.

    No network calls. Covers list/bullet-formatted output.
    Returns an empty list when no smell keywords are found — the caller should
    then decide whether to make a second LLM call (see analyzer.py Stage-2 block).
    """
    return _regex_extract_findings(raw_response, filename)


def parse_llm_extraction(raw: str, filename: str) -> list[dict]:
    """Parse the JSON response from an LLM extraction call (extract_findings.md).

    No network calls. Expects the raw string output of an Ollama generate call
    that was prompted with EXTRACT_PROMPT. Returns structured findings or [].
    """
    # Try JSON fences first, then bare JSON object
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if not m:
        m = re.search(r"(\{.*\})", raw, re.DOTALL)
    candidate = m.group(1) if m else raw

    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return []

    findings_raw: list[dict] = (
        parsed if isinstance(parsed, list) else parsed.get("findings", [])
    )

    mandatory_keys = {"datei", "typ", "begründung", "empfehlung"}
    result: list[dict] = []
    for f in findings_raw:
        # All mandatory keys must be present AND contain non-empty strings.
        if not mandatory_keys.issubset(f.keys()):
            continue
        if not all(str(f[k]).strip() for k in mandatory_keys):
            continue
        result.append({
            "type":             f.get("typ", "freitext"),
            "line_hint":        f.get("zeile", ""),
            "evidence_excerpt": f.get("begründung", "")[:200],
            "fix_suggestion":   f.get("empfehlung", ""),
            "confidence":       "low",
            "severity":         "info",
            "source":           "freetext_extraction",
        })
    return result


def extract_freetext_findings(
    raw_response: str,
    ollama_url: str,
    model: str,
    filename: str,
    category: str,
) -> list[dict]:
    """Deprecated: regex-only extraction (LLM fallback moved to analyzer.py).

    Kept for backward compatibility. Only the fast regex path runs here;
    the LLM fallback is now orchestrated by analyzer.analyze_file().
    """
    return parse_freetext_findings(raw_response, filename)


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
        # Inject finding-reactive RAG context (Issue #18)
        rag_ctx = f.get("_rag_context", "")
        if rag_ctx:
            prompt += f"\n\nPROJEKTKONTEXT:\n{rag_ctx}"
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


# ---------------------------------------------------------------------------
# Second-opinion validation — different model reviews primary findings
# ---------------------------------------------------------------------------

_SECOND_OPINION_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "second_opinion.md"
try:
    _SECOND_OPINION_PROMPT: str = _SECOND_OPINION_PROMPT_PATH.read_text(encoding="utf-8")
except Exception:
    _SECOND_OPINION_PROMPT = ""


def second_opinion_validate(
    findings: list[dict],
    files: list[dict],
    ollama_url: str,
    second_opinion_model: str,
) -> tuple[list[dict], dict]:
    """Validate findings using a second, independent LLM model.

    Sends each finding to *second_opinion_model* together with the relevant
    code snippet. The model returns one of three verdicts:

      - BESTÄTIGT  → finding is correct, keep as-is
      - ABGELEHNT  → false positive, drop from results
      - PRÄZISIERT → true core but severity/description adjusted

    Args:
        findings:             Flat list of findings (each has ``_filename``).
        files:                File dicts (needed to look up content).
        ollama_url:           Ollama base URL.
        second_opinion_model: Model name for the second opinion.

    Returns:
        (validated_findings, stats_dict) where stats_dict contains:
          - confirmed: count of BESTÄTIGT findings kept
          - rejected:  count of ABGELEHNT findings dropped
          - refined:   count of PRÄZISIERT findings kept (with adjusted severity)
          - errors:    count of LLM call failures (kept in output, not counted as confirmed)
    """
    from src.analyzer import _ollama_generate

    if not _SECOND_OPINION_PROMPT:
        # Prompt file missing — pass all findings through unchanged
        return findings, {"confirmed": len(findings), "rejected": 0, "refined": 0, "errors": 0}

    file_map = {f["filename"]: f for f in files}
    validated: list[dict] = []
    stats = {"confirmed": 0, "rejected": 0, "refined": 0, "errors": 0}

    for finding in findings:
        fn = finding.get("_filename", "")
        file_entry = file_map.get(fn, {})
        code_snippet = file_entry.get("content", "")[:500]

        def _str(val, default: str = "") -> str:
            if isinstance(val, list):
                return " ".join(str(v) for v in val)
            return str(val) if val is not None else default

        prompt = (
            _SECOND_OPINION_PROMPT
            .replace("{type}", _str(finding.get("type"), "unknown"))
            .replace("{severity}", _str(finding.get("severity"), "info"))
            .replace("{filename}", _str(fn))
            .replace("{line_hint}", _str(finding.get("line_hint")))
            .replace("{evidence_excerpt}", _str(finding.get("evidence_excerpt"))[:300])
            .replace("{fix_suggestion}", _str(finding.get("fix_suggestion"))[:200])
            .replace("{code_snippet}", code_snippet)
        )
        # Inject finding-reactive RAG context (Issue #18)
        rag_ctx = finding.get("_rag_context", "")
        prompt = prompt.replace(
            "{rag_context}",
            rag_ctx if rag_ctx else "(kein zusätzlicher Projektkontext verfügbar)",
        )

        try:
            raw = _ollama_generate(
                prompt, ollama_url, second_opinion_model, f"[2ND] {fn}"
            )
        except Exception:
            stats["errors"] += 1
            finding["_second_opinion_verdict"] = "ERROR"
            validated.append(finding)
            continue

        # Parse JSON response
        verdict = "BESTÄTIGT"
        reasoning = ""
        adjusted_severity: str | None = None
        additional_note: str | None = None

        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if not m:
            m = re.search(r"(\{[^{}]*\})", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
                verdict = data.get("verdict", "BESTÄTIGT").strip().upper()
                reasoning = data.get("reasoning", "")
                adjusted_severity = data.get("adjusted_severity")
                additional_note = data.get("additional_note")
            except (json.JSONDecodeError, ValueError):
                pass

        finding["_second_opinion_verdict"] = verdict
        if reasoning:
            finding["_second_opinion_reasoning"] = reasoning
        if additional_note:
            finding["_second_opinion_note"] = additional_note

        if verdict == "ABGELEHNT":
            stats["rejected"] += 1
        elif verdict == "PRÄZISIERT":
            stats["refined"] += 1
            if adjusted_severity in ("critical", "warning", "info"):
                finding["severity"] = adjusted_severity
            validated.append(finding)
        else:
            # BESTÄTIGT or unrecognised → keep
            stats["confirmed"] += 1
            validated.append(finding)

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
