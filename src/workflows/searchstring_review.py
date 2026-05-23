"""Pi-Task: P4-Suchstring-Optimierung via Ollama (#261).

Kostengünstige Überarbeitung bestehender P4-Suchstrings über den Pi-Agent
(lokales Ollama, kein Cloud-API-Call). Läuft über die PI_AGENT_URL-Boundary
(Cloud-Distributor wenn gesetzt, sonst in-process) — KEIN eigener
Direct-Ollama-Pfad (v2.0-Auflösung). Telemetrie wird im #260-Format geloggt,
damit die Daten später fürs Fine-Tuning verwertbar sind.

Pure functions hier (Prompt-Bau + Parsing) sind ohne Ollama testbar; der
Routing-/Telemetrie-Teil sitzt im MCP-Tool ``pi_optimize_searchstring``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_TEMPLATE_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "pi_searchstring_review.md"
)

_FALLBACK_TEMPLATE = (
    "Überarbeite den Suchstring für Datenbank {{database}} zur Forschungsfrage:\n"
    "{{forschungsfrage}}\n\nSuchstring:\n{{searchstring}}\n\n"
    'Antworte als JSON: {"revised": "...", "reasoning": "..."}'
)


def load_template() -> str:
    if _TEMPLATE_PATH.exists():
        return _TEMPLATE_PATH.read_text(encoding="utf-8")
    return _FALLBACK_TEMPLATE


def build_prompt(searchstring: str, database: str, forschungsfrage: str) -> str:
    """Fill the review template. Empty inputs are passed through as markers."""
    return (
        load_template()
        .replace("{{database}}", database.strip() or "(unbekannt)")
        .replace("{{forschungsfrage}}", forschungsfrage.strip() or "(keine angegeben)")
        .replace("{{searchstring}}", searchstring.strip())
    )


def parse_response(raw: str) -> dict:
    """Extract {revised, reasoning} from the Pi response.

    Robust gegen code-fences und Prosa drumherum: erstes JSON-Objekt mit einem
    ``revised``-Key wird genommen. Fällt sonst auf den getrimmten Rohtext als
    ``revised`` zurück (reasoning leer), damit der Aufrufer nie None bekommt.
    """
    if not raw or not raw.strip():
        return {"revised": "", "reasoning": "", "parsed": False}

    text = raw.strip()
    # strip ```json fences
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()

    for match in re.finditer(r"\{.*?\}", text, re.DOTALL):
        try:
            obj = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and "revised" in obj:
            return {
                "revised": str(obj.get("revised", "")).strip(),
                "reasoning": str(obj.get("reasoning", "")).strip(),
                "parsed": True,
            }

    return {"revised": text, "reasoning": "", "parsed": False}
