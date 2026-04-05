"""LLM analysis via Ollama — Mayring Stufen 2 + 3.

Stufe 2 (Reduktion): LLM analysiert jede geänderte Datei, gibt strukturiertes JSON zurück.
Stufe 3 (Explikation): Findings mit confidence=low werden als needs_explikation=True markiert.
"""

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx

from src.config import (
    MAX_FINDINGS_PER_FILE,
    OLLAMA_TIMEOUT,
    get_batch_delay,
    get_batch_size,
    get_max_chars_per_file,
)

# ---------------------------------------------------------------------------
# Training-Data Logger (opt-in via configure_training_log())
# ---------------------------------------------------------------------------

_training_log_path: Path | None = None
_training_run_id: str = "default"


def configure_training_log(path: Path | str, run_id: str = "default") -> None:
    """Enable training-data logging. Call once from checker.py when --log-training-data is set.

    Each LLM call in analyze_file() and overview_file() will append a JSONL
    entry to *path*. The file is created if it does not exist.
    """
    global _training_log_path, _training_run_id
    _training_log_path = Path(path)
    _training_run_id = run_id
    _training_log_path.parent.mkdir(parents=True, exist_ok=True)


def _log_training_entry(
    model: str,
    label: str,
    prompt: str,
    raw_response: str,
    parsed_ok: bool,
    findings_count: int,
    call_type: str = "analyze",
) -> None:
    """Append one JSONL entry to the training log. No-op if logging is disabled."""
    if _training_log_path is None:
        return
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": _training_run_id,
        "model": model,
        "label": label,
        "call_type": call_type,          # "analyze" | "overview" | "extract" | "adversarial" | "second_opinion"
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed_ok": parsed_ok,
        "findings_count": findings_count,
    }
    with _training_log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_str(value: object) -> str:
    """Coerce an LLM-returned value to a plain string."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value) if value is not None else ""

# Retry settings for Ollama connection failures
_MAX_RETRIES = 3
_RETRY_DELAYS = (2, 5, 10)  # seconds between retries


def _load_prompt(path: Path | str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _truncate(content: str) -> tuple[str, bool]:
    limit = get_max_chars_per_file()
    if len(content) > limit:
        trimmed = content[:limit]
        suffix = f"\n\n[... gekürzt, Original {len(content)} Zeichen]"
        return trimmed + suffix, True
    return content, False


# ---------------------------------------------------------------------------
# Stage 2: Structured JSON extraction from LLM response
# ---------------------------------------------------------------------------

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
                "type": "freitext",
                "severity": "info",
                "confidence": "low",
                "line_hint": "",
                "evidence_excerpt": raw[:500],
                "fix_suggestion": "",
            }
        ],
        "_parse_error": True,
    }


def _freetext_fallback_sozial(raw: str) -> dict:
    return {
        "file_summary": "",
        "codierungen": [
            {
                "category": "unklar",
                "confidence": "low",
                "line_hint": "",
                "evidence_excerpt": raw[:500],
                "reasoning": "Freetext-Fallback — LLM lieferte kein valides JSON",
            }
        ],
        "_parse_error": True,
    }


# ---------------------------------------------------------------------------
# Ollama core
# ---------------------------------------------------------------------------

def _ollama_generate(prompt: str, ollama_url: str, model: str, label: str) -> str:
    """Send a prompt to Ollama and collect the streamed response.

    Uses stream=True so httpx's read_timeout fires if the model stops producing
    tokens for OLLAMA_TIMEOUT seconds — prevents the 19-minute hangs that occur
    with stream=False (server holds connection open until generation is complete).
    """
    for attempt in range(_MAX_RETRIES):
        try:
            chunks: list[str] = []
            with httpx.stream(
                "POST",
                f"{ollama_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": True},
                timeout=OLLAMA_TIMEOUT,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunks.append(data.get("response", ""))
                    if data.get("done"):
                        break
            return "".join(chunks)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                print(f"    ⟳ Retry {attempt + 1}/{_MAX_RETRIES} für {label} (in {delay}s) …", flush=True)
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------

def analyze_file(
    file: dict,
    prompt_template: str,
    ollama_url: str,
    model: str,
    project_context: str | None = None,
    hot_zone_context: str | None = None,
) -> dict:
    """Analyze one file. Uses a two-stage output strategy:

    Stage 1: Primary LLM call — accepts any output format.
    Stage 2: If Stage 1 returns no valid JSON → Stage-2 extractor parses the
             freetext response for structured findings (requires mandatory fields).

    Args:
        file:             File dict with at least 'filename', 'content', 'category'.
        prompt_template:  The main analysis prompt (file_inspector.md or similar).
        ollama_url:       Ollama base URL.
        model:            Model name.
        project_context:  Optional project-wide context injected before the file block.
        hot_zone_context: Optional turbulence hot-zone info injected before file (Issue #17).

    Returns:
        Dict with 'filename', 'category', 'truncated', 'file_summary',
        'potential_smells' / 'codierungen', '_parse_error'.
    """
    filename = file["filename"]
    category = file.get("category", "uncategorized")
    content, truncated = _truncate(file["content"])

    parts = [prompt_template]
    if project_context:
        parts.append(f"\n{project_context}\n")
    if hot_zone_context:
        parts.append(f"\n{hot_zone_context}\n")
    parts.append(
        f"\nDatei: {filename}\n"
        f"Kategorie: {category}\n"
        f"```\n{content}\n```"
    )
    prompt = "\n".join(parts)

    try:
        raw_response = _ollama_generate(prompt, ollama_url, model, filename)
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

    parsed = _parse_llm_json(raw_response)

    # Detect social-research mode (codierungen) vs code-review mode (potential_smells)
    is_sozial = parsed is not None and "codierungen" in parsed

    if parsed is None:
        # Stage 2: Try to extract structured findings from unstructured output.
        # Fast path — pure regex, no network call.
        try:
            from src.extractor import parse_freetext_findings, parse_llm_extraction, EXTRACT_PROMPT

            extracted = parse_freetext_findings(raw_response, filename)

            # Slow path — second LLM call only when regex found nothing.
            if not extracted:
                extraction_prompt = (
                    EXTRACT_PROMPT
                    + "\n\n## Rohe LLM-Antwort\n\n"
                    + raw_response.strip()
                )
                try:
                    raw_extraction = _ollama_generate(
                        extraction_prompt, ollama_url, model, f"[EXTRACT] {filename}"
                    )
                    extracted = parse_llm_extraction(raw_extraction, filename)
                    _log_training_entry(model, filename, extraction_prompt, raw_extraction,
                                        bool(extracted), len(extracted), call_type="extract")
                except Exception:
                    extracted = []
        except Exception:
            extracted = []

        if extracted:
            _log_training_entry(model, filename, prompt, raw_response,
                                False, len(extracted), call_type="analyze")
            result: dict = {
                "filename": filename,
                "category": category,
                "truncated": truncated,
                "file_summary": "",
                "_parse_error": True,
                "_stage2_extracted": True,
                "potential_smells": extracted,
            }
            for smell in result["potential_smells"]:
                if smell.get("confidence", "high").lower() == "low":
                    smell["needs_explikation"] = True
            return result
        # No structured findings extracted — fall back to raw dump.
        parsed = _freetext_fallback_sozial(raw_response) if is_sozial else _freetext_fallback(raw_response)

    # Log the primary analyze call
    findings_in_parsed = len(parsed.get("potential_smells") or parsed.get("codierungen") or [])
    _log_training_entry(model, filename, prompt, raw_response,
                        not parsed.get("_parse_error", False), findings_in_parsed,
                        call_type="analyze")

    result = {
        "filename": filename,
        "category": category,
        "truncated": truncated,
        "file_summary": _ensure_str(parsed.get("file_summary", "")),
        "_parse_error": parsed.get("_parse_error", False),
    }

    if is_sozial:
        codierungen = parsed.get("codierungen", [])[:MAX_FINDINGS_PER_FILE]
        for c in codierungen:
            for key in ("evidence_excerpt", "reasoning", "type", "kategorie"):
                if c.get(key) is None:
                    c[key] = ""
            if c.get("confidence", "high").lower() == "low":
                c["needs_explikation"] = True
        result["codierungen"] = codierungen
        if "category_summary" in parsed:
            result["category_summary"] = parsed["category_summary"]
    else:
        smells = parsed.get("potential_smells", [])[:MAX_FINDINGS_PER_FILE]
        for smell in smells:
            for key in ("evidence_excerpt", "reasoning", "type", "severity"):
                if smell.get(key) is None:
                    smell[key] = ""
            if smell.get("confidence", "high").lower() == "low":
                smell["needs_explikation"] = True
        result["potential_smells"] = smells

    return result


def analyze_files(
    files: list[dict],
    filenames_to_check: list[str],
    prompt_path: Path | str,
    ollama_url: str,
    model: str,
    project_context: str | None = None,
    context_fn: Callable[[dict], str | None] | None = None,
    hot_zone_context_map: dict[str, str] | None = None,
    time_budget: float | None = None,
) -> tuple[list[dict], bool]:
    """Analyze multiple files, optionally enriching each with per-file context.

    Args:
        time_budget: Optional wall-clock budget in seconds. When set, the loop
                     stops gracefully after the current file once the budget is
                     exceeded.  The second return value indicates whether the
                     budget was hit (True) or all files were processed (False).

    Returns:
        (results, time_budget_hit)
    """
    prompt_template = _load_prompt(prompt_path)
    file_map = {f["filename"]: f for f in files}
    results = []
    total = len(filenames_to_check)
    time_budget_hit = False
    _start = time.perf_counter()
    for i, fn in enumerate(filenames_to_check, 1):
        print(f"  [{i}/{total}] {fn} ...", flush=True)
        file = file_map[fn]
        ctx = None
        if context_fn is not None:
            ctx = context_fn(file)
        if ctx is None:
            ctx = project_context
        hz_ctx = hot_zone_context_map.get(fn) if hot_zone_context_map else None
        results.append(analyze_file(file, prompt_template, ollama_url, model, ctx, hz_ctx))
        _bs = get_batch_size()
        if _bs > 0 and i % _bs == 0 and i < total:
            _bd = get_batch_delay()
            print(f"  ⏸ GPU-Pause ({_bd}s nach {i} Dateien) ...", flush=True)
            time.sleep(_bd)
        if time_budget is not None and (time.perf_counter() - _start) >= time_budget:
            print(
                f"  ⏱ Zeit-Budget ({time_budget:.0f}s) erreicht — stoppe nach {i}/{total} Dateien.",
                flush=True,
            )
            time_budget_hit = True
            break
    return results, time_budget_hit


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
        raw_response = _ollama_generate(prompt, ollama_url, model, filename)
    except httpx.ConnectError:
        return {"filename": filename, "error": f"Ollama nicht erreichbar unter {ollama_url}"}
    except httpx.TimeoutException:
        return {"filename": filename, "error": f"Timeout ({OLLAMA_TIMEOUT}s)"}
    except (httpx.HTTPStatusError, KeyError) as exc:
        return {"filename": filename, "error": str(exc)}

    parsed = _parse_llm_json(raw_response)
    summary = parsed.get("file_summary", raw_response.strip()) if parsed else raw_response.strip()
    summary = _ensure_str(summary)

    _log_training_entry(model, filename, prompt, raw_response,
                        parsed is not None, 0, call_type="overview")

    result = {
        "filename": filename,
        "category": category,
        "truncated": truncated,
        "file_summary": summary,
    }

    # Structured I/O fields for feed-forward pipeline (Issue #17)
    if parsed:
        for field in ("functions", "external_deps"):
            if field in parsed:
                result[field] = parsed[field]

    # Enrich with signature extraction for Phase 3 (redundancy check).
    try:
        from src.extractor import extract_python_signatures

        if filename.endswith(".py"):
            result["_signatures"] = extract_python_signatures(content)
    except Exception:
        pass

    return result


def overview_files(
    files: list[dict],
    filenames: list[str],
    prompt_path: Path | str,
    ollama_url: str,
    model: str,
    time_budget: float | None = None,
) -> tuple[list[dict], bool]:
    """Summarize multiple files.

    Returns:
        (results, time_budget_hit)
    """
    prompt_template = _load_prompt(prompt_path)
    file_map = {f["filename"]: f for f in files}
    results = []
    total = len(filenames)
    time_budget_hit = False
    _start = time.perf_counter()
    for i, fn in enumerate(filenames, 1):
        print(f"  [{i}/{total}] {fn} ...", flush=True)
        results.append(overview_file(file_map[fn], prompt_template, ollama_url, model))
        _bs = get_batch_size()
        if _bs > 0 and i % _bs == 0 and i < total:
            _bd = get_batch_delay()
            print(f"  ⏸ GPU-Pause ({_bd}s nach {i} Dateien) ...", flush=True)
            time.sleep(_bd)
        if time_budget is not None and (time.perf_counter() - _start) >= time_budget:
            print(
                f"  ⏱ Zeit-Budget ({time_budget:.0f}s) erreicht — stoppe nach {i}/{total} Dateien.",
                flush=True,
            )
            time_budget_hit = True
            break
    return results, time_budget_hit
