"""LLM analysis via Ollama — Mayring Stufen 2 + 3.

Stufe 2 (Reduktion): LLM analysiert jede geänderte Datei, gibt strukturiertes JSON zurück.
Stufe 3 (Explikation): Findings mit confidence=low werden als needs_explikation=True markiert
                       (kein automatischer 2. LLM-Call — manueller Re-Run mit explainer.md).
"""

import json
import re
import time
from pathlib import Path

import httpx

from src.config import (
    BATCH_DELAY_SECONDS,
    BATCH_SIZE,
    MAX_CHARS_PER_FILE,
    MAX_FINDINGS_PER_FILE,
    OLLAMA_TIMEOUT,
)


def _load_prompt(path: Path | str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _truncate(content: str) -> tuple[str, bool]:
    if len(content) > MAX_CHARS_PER_FILE:
        trimmed = content[:MAX_CHARS_PER_FILE]
        suffix = f"\n\n[... gekürzt, Original {len(content)} Zeichen]"
        return trimmed + suffix, True
    return content, False


def _parse_llm_json(raw: str) -> dict | None:
    """Extract a JSON object from the LLM response; handles markdown code fences."""
    raw = raw.strip()
    # Try to strip ```json ... ``` fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        # Find the first {...} block
        block = re.search(r"(\{.*\})", raw, re.DOTALL)
        candidate = block.group(1) if block else raw
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None


def _freetext_fallback(raw: str) -> dict:
    return {
        "file_summary": "",
        "potential_smells": [
            {
                "type": "freetext",
                "severity": "info",
                "confidence": "low",
                "line_hint": "",
                "evidence_excerpt": raw[:500],
                "fix_suggestion": "",
            }
        ],
        "_parse_error": True,
    }


def analyze_file(
    file: dict,
    prompt_template: str,
    ollama_url: str,
    model: str,
) -> dict:
    filename = file["filename"]
    category = file.get("category", "uncategorized")
    content, truncated = _truncate(file["content"])

    prompt = (
        f"{prompt_template}\n\n"
        f"Datei: {filename}\n"
        f"Kategorie: {category}\n"
        f"```\n{content}\n```"
    )

    try:
        resp = httpx.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        raw_response = resp.json()["response"]
    except httpx.ConnectError:
        return {
            "filename": filename,
            "error": f"Ollama nicht erreichbar unter {ollama_url}. Läuft `ollama serve`?",
        }
    except httpx.TimeoutException:
        return {
            "filename": filename,
            "error": f"Timeout ({OLLAMA_TIMEOUT}s) bei Analyse von {filename}.",
        }
    except (httpx.HTTPStatusError, KeyError) as exc:
        return {"filename": filename, "error": str(exc)}

    parsed = _parse_llm_json(raw_response) or _freetext_fallback(raw_response)

    smells = parsed.get("potential_smells", [])[:MAX_FINDINGS_PER_FILE]
    for smell in smells:
        if smell.get("confidence", "high").lower() == "low":
            smell["needs_explikation"] = True

    return {
        "filename": filename,
        "category": category,
        "truncated": truncated,
        "file_summary": parsed.get("file_summary", ""),
        "potential_smells": smells,
        "_parse_error": parsed.get("_parse_error", False),
    }


def analyze_files(
    files: list[dict],
    filenames_to_check: list[str],
    prompt_path: Path | str,
    ollama_url: str,
    model: str,
) -> list[dict]:
    prompt_template = _load_prompt(prompt_path)
    file_map = {f["filename"]: f for f in files}
    results = []
    total = len(filenames_to_check)
    for i, fn in enumerate(filenames_to_check, 1):
        print(f"  [{i}/{total}] {fn} ...", flush=True)
        results.append(analyze_file(file_map[fn], prompt_template, ollama_url, model))
        if BATCH_SIZE > 0 and i % BATCH_SIZE == 0 and i < total:
            print(f"  ⏸ GPU-Pause ({BATCH_DELAY_SECONDS}s nach {i} Dateien) ...", flush=True)
            time.sleep(BATCH_DELAY_SECONDS)
    return results


# ---------------------------------------------------------------------------
# Overview mode — only file summaries, no findings
# ---------------------------------------------------------------------------

def overview_file(
    file: dict,
    prompt_template: str,
    ollama_url: str,
    model: str,
) -> dict:
    """Ask the LLM for a max-10-line summary of what the file does — no findings."""
    filename = file["filename"]
    category = file.get("category", "uncategorized")
    content, truncated = _truncate(file["content"])

    prompt = (
        f"{prompt_template}\n\n"
        f"Datei: {filename}\n"
        f"Kategorie: {category}\n"
        f"```\n{content}\n```"
    )

    try:
        resp = httpx.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        raw_response = resp.json()["response"]
    except httpx.ConnectError:
        return {"filename": filename, "error": f"Ollama nicht erreichbar unter {ollama_url}"}
    except httpx.TimeoutException:
        return {"filename": filename, "error": f"Timeout ({OLLAMA_TIMEOUT}s)"}
    except (httpx.HTTPStatusError, KeyError) as exc:
        return {"filename": filename, "error": str(exc)}

    parsed = _parse_llm_json(raw_response)
    summary = parsed.get("file_summary", raw_response.strip()) if parsed else raw_response.strip()

    return {
        "filename": filename,
        "category": category,
        "truncated": truncated,
        "file_summary": summary,
    }


def overview_files(
    files: list[dict],
    filenames: list[str],
    prompt_path: Path | str,
    ollama_url: str,
    model: str,
) -> list[dict]:
    prompt_template = _load_prompt(prompt_path)
    file_map = {f["filename"]: f for f in files}
    results = []
    total = len(filenames)
    for i, fn in enumerate(filenames, 1):
        print(f"  [{i}/{total}] {fn} ...", flush=True)
        results.append(overview_file(file_map[fn], prompt_template, ollama_url, model))
        if BATCH_SIZE > 0 and i % BATCH_SIZE == 0 and i < total:
            print(f"  ⏸ GPU-Pause ({BATCH_DELAY_SECONDS}s nach {i} Dateien) ...", flush=True)
            time.sleep(BATCH_DELAY_SECONDS)
    return results
