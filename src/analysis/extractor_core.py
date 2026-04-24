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


def _coerce_str(val: object) -> str:
    """LLM-JSON-Felder können Liste, int oder None sein — immer zu str normalisieren."""
    if val is None:
        return ""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val if v)
    return str(val)


_EXTRACT_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "extract_findings.md"
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

_EXTRACT_PROMPT = EXTRACT_PROMPT

_SMELL_KEYWORDS_RE = re.compile(
    r"\b(zombie[_\s]?code|redundan[zt]|inkonsistenz|fehlerbehandlung|overengineering"
    r"|sicherheit|security|dead[_\s]?code|performance|problem|issue|bug|smell|finding"
    r"|code[_\s]?qualit[äa]t|verbesserung|refactor)\b",
    re.IGNORECASE,
)
_LINE_HINT_RE = re.compile(r"[Zz]eile\s*:?\s*~?(\d+)|[Ll]ine\s*:?\s*~?(\d+)|[Ll]ine\s+(\d+)", re.IGNORECASE)


def _regex_extract_findings(raw: str, filename: str) -> list[dict]:
    """Fast heuristic extraction without a second LLM call."""
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
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if not m:
        m = re.search(r"(\{.*\})", raw, re.DOTALL)
    candidate = m.group(1) if m else raw

    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return []

    if isinstance(parsed, list):
        findings_raw: list[dict] = parsed
    elif isinstance(parsed, dict):
        findings_raw = parsed.get("findings", [])
    else:
        findings_raw = []

    mandatory_keys = {"datei", "typ", "begründung", "empfehlung"}
    result: list[dict] = []
    for f in findings_raw:
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
    """Deprecated: regex-only extraction (LLM fallback moved to analyzer.py)."""
    return parse_freetext_findings(raw_response, filename)
