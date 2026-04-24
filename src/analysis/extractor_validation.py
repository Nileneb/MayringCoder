import re
import json
from pathlib import Path

from src.analysis.extractor_core import _coerce_str


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
    patterns = [
        re.compile(r"(?:^|/)test[s]?[_\-].*\.(?:py|php|js|ts|go|java)$", re.IGNORECASE),
        re.compile(r"(?:^|/)(?:tests?|spec|__tests?__|test)", re.IGNORECASE),
        re.compile(r"_test\.(?:py|php|js|ts|go|java)$", re.IGNORECASE),
    ]
    return any(p.search(filename) for p in patterns)


def _file_type_label(filename: str) -> str:
    if _is_test_file(filename):
        return "Test"
    if re.search(r"\.(?:php|blade\.php)$", filename, re.IGNORECASE):
        return "PHP/Laravel"
    if re.search(r"\.py$", filename, re.IGNORECASE):
        return "Python/Django"
    if re.search(r"\.(?:js|ts)$", filename, re.IGNORECASE):
        return "JavaScript/TypeScript"
    if re.search(r"\.go$", filename, re.IGNORECASE):
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
    from src.analysis.analyzer import _ollama_generate

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

        file_entry = file_map.get(fn, {})
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


_SECOND_OPINION_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "second_opinion.md"
try:
    _SECOND_OPINION_PROMPT: str = _SECOND_OPINION_PROMPT_PATH.read_text(encoding="utf-8")
except Exception:
    _SECOND_OPINION_PROMPT = ""


_QUESTION_TEMPLATES: dict[str, str] = {
    "redundanz": (
        "Existiert die beschriebene Funktionalitaet ({evidence_short}) tatsaechlich "
        "an anderer Stelle im Projekt mit gleicher oder sehr aehnlicher Logik?"
    ),
    "sicherheit": (
        "Ist der beschriebene Input ({evidence_short}) tatsaechlich unsanitisiert "
        "und von aussen erreichbar, oder wird er bereits durch Framework/Middleware validiert?"
    ),
    "zombie_code": (
        "Wird die beschriebene Funktion/der Code ({evidence_short}) wirklich nirgendwo "
        "im Projekt aufgerufen oder referenziert?"
    ),
    "dead_code": (
        "Ist dieser Code ({evidence_short}) tatsaechlich unerreichbar, oder gibt es "
        "Ausfuehrungspfade (Exceptions, Bedingungen), die ihn doch erreichen?"
    ),
    "inkonsistenz": (
        "Weicht dieses Pattern ({evidence_short}) wirklich vom etablierten "
        "Projekt-Standard ab, oder ist es eine bewusste Designentscheidung?"
    ),
    "fehlerbehandlung": (
        "Fehlt hier tatsaechlich eine Fehlerbehandlung, oder wird der Fehler "
        "bereits auf einer uebergeordneten Ebene (Middleware, Framework) abgefangen?"
    ),
    "overengineering": (
        "Ist die beschriebene Abstraktion ({evidence_short}) wirklich ueberfluessig, "
        "oder dient sie einem erkennbaren Erweiterungszweck?"
    ),
}

_FALLBACK_QUESTION = (
    "Ist dieses Finding ({evidence_short}) korrekt und actionable, "
    "oder handelt es sich um ein Framework-Pattern bzw. eine bewusste Designentscheidung?"
)


def _build_second_opinion_question(finding: dict) -> str:
    """Generate a targeted yes/no question for second-opinion validation."""
    ftype = (finding.get("type") or "").lower().strip()
    evidence = (finding.get("evidence_excerpt") or "")[:80]
    if not evidence:
        evidence = (finding.get("fix_suggestion") or "")[:80]

    template = _QUESTION_TEMPLATES.get(ftype, _FALLBACK_QUESTION)
    return template.format(evidence_short=evidence or "siehe Code")


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
    from src.analysis.analyzer import _ollama_generate

    if not _SECOND_OPINION_PROMPT:
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
        rag_ctx = finding.get("_rag_context", "")
        prompt = prompt.replace(
            "{rag_context}",
            rag_ctx if rag_ctx else "(kein zusätzlicher Projektkontext verfügbar)",
        )
        targeted_q = _build_second_opinion_question(finding)
        prompt = prompt.replace("{targeted_question}", targeted_q)

        try:
            raw = _ollama_generate(
                prompt, ollama_url, second_opinion_model, f"[2ND] {fn}"
            )
        except Exception:
            stats["errors"] += 1
            finding["_second_opinion_verdict"] = "ERROR"
            validated.append(finding)
            continue

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
                verdict = _coerce_str(data.get("verdict", "BESTÄTIGT")).strip().upper() or "BESTÄTIGT"
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
