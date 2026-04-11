#!/usr/bin/env python3
"""Batch-Annotation von Training-Samples via LLM-Review.

Liest training_log JSONL, lässt jedes Finding von einem Review-LLM bewerten,
schreibt annotierte JSONL mit Qualitätslabels.

Usage:
    # Standard (qwen3.5:9b, 2s Delay)
    python tools/annotate_training_data.py

    # Schneller, anderes Modell
    python tools/annotate_training_data.py --model qwen2.5-coder:7b --delay 1.0

    # Nur die ersten 50 Samples (Test)
    python tools/annotate_training_data.py --limit 50

    # Fortsetzen nach Abbruch (überspringt bereits annotierte)
    python tools/annotate_training_data.py --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

_REVIEW_SYSTEM = """You are a code review quality assessor. Rate automated findings as true or false positives.
Reply ONLY with JSON: {"overall_quality":"good|partial|bad","true_positives":N,"false_positives":N,"total_findings":N,"precision":0.0,"reasoning":"1 sentence"}"""

_REVIEW_PROMPT = """Rate this automated code analysis. How many findings are real vs false positives?

CODE (truncated):
{code}

LLM OUTPUT:
{response}"""


def _ollama_generate(
    prompt: str,
    ollama_url: str,
    model: str,
    system_prompt: str = "",
    timeout: float = 300.0,
) -> str:
    """Single Ollama generate call."""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
    }
    resp = httpx.post(
        f"{ollama_url.rstrip('/')}/api/generate",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _parse_review(raw: str) -> dict | None:
    """Parse JSON from LLM review response — tolerates markdown fences and surrounding text."""
    import re
    raw = raw.strip()
    # Strip markdown fences
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw = part
                break
    # Try direct parse
    try:
        d = json.loads(raw)
        if isinstance(d, dict) and "overall_quality" in d:
            return d
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: find first JSON object in text
    m = re.search(r'\{[^{}]*"overall_quality"[^{}]*\}', raw)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _warmup_model(ollama_url: str, model: str) -> bool:
    """Send a trivial prompt to ensure the model is loaded in VRAM."""
    print(f"[warmup] Lade {model} in VRAM...", end=" ", flush=True)
    try:
        resp = _ollama_generate("Say OK.", ollama_url, model, timeout=300.0)
        print(f"OK ({len(resp)} chars)")
        return True
    except Exception as exc:
        print(f"FEHLER: {exc}")
        return False


def annotate_batch(
    input_path: Path,
    output_path: Path,
    ollama_url: str,
    model: str,
    delay: float,
    limit: int,
    resume: bool,
) -> dict:
    """Annotate training samples with LLM review."""
    # Warmup: ensure model is hot in VRAM before starting
    if not _warmup_model(ollama_url, model):
        print("[WARNUNG] Model warmup fehlgeschlagen, versuche trotzdem...")

    lines = input_path.read_text().splitlines()
    total = len(lines)

    if limit > 0:
        lines = lines[:limit]

    # Load already-annotated IDs for resume
    done_hashes: set[str] = set()
    if resume and output_path.exists():
        for line in output_path.read_text().splitlines():
            try:
                d = json.loads(line)
                done_hashes.add(d.get("prompt_hash", ""))
            except (json.JSONDecodeError, ValueError):
                pass
        print(f"[resume] {len(done_hashes)} bereits annotiert, überspringe diese")

    stats = {"total": len(lines), "annotated": 0, "skipped": 0, "errors": 0}
    mode = "a" if resume else "w"

    with output_path.open(mode, encoding="utf-8") as out:
        for i, line in enumerate(lines):
            try:
                sample = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                stats["errors"] += 1
                continue

            prompt_hash = sample.get("prompt_hash", f"line_{i}")

            # Skip if already done (resume mode)
            if prompt_hash in done_hashes:
                stats["skipped"] += 1
                continue

            code = sample.get("prompt", "")[:600]
            response = sample.get("raw_response", "")[:800]

            if not code or not response or len(response.strip()) < 20:
                stats["skipped"] += 1
                continue

            # Progress
            pct = (i + 1) / len(lines) * 100
            label = sample.get("label", "?")
            print(f"[{i+1}/{len(lines)}] ({pct:.0f}%) {label[:30]}...", end=" ", flush=True)

            try:
                review_prompt = _REVIEW_PROMPT.format(code=code, response=response)
                raw_review = _ollama_generate(
                    review_prompt, ollama_url, model, _REVIEW_SYSTEM
                )
                annotation = _parse_review(raw_review)

                if annotation:
                    sample["annotation"] = annotation
                    sample["annotation_model"] = model
                    sample["annotation_ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    stats["annotated"] += 1
                    quality = annotation.get("overall_quality", "?")
                    precision = annotation.get("precision", 0)
                    print(f"→ {quality} (precision: {precision:.2f})")
                else:
                    sample["annotation"] = {"overall_quality": "parse_error", "raw": raw_review[:500]}
                    sample["annotation_model"] = model
                    stats["errors"] += 1
                    print("→ parse_error")

            except Exception as exc:
                sample["annotation"] = {"overall_quality": "error", "error": str(exc)[:200]}
                stats["errors"] += 1
                print(f"→ ERROR: {exc}")

            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            out.flush()

            # Delay between calls (GPU cooldown + rate limit protection)
            if delay > 0 and i < len(lines) - 1:
                time.sleep(delay)

    return stats


def main() -> None:
    p = argparse.ArgumentParser(
        description="Batch-Annotation von Training-Samples via LLM-Review"
    )
    p.add_argument(
        "--input",
        default="cache/nileneb-applinngames_training_log.jsonl",
        help="Input JSONL (default: training_log)",
    )
    p.add_argument(
        "--output",
        default="cache/training_annotated.jsonl",
        help="Output JSONL mit Annotations",
    )
    p.add_argument("--model", default="qwen3.5:9b", help="Ollama Review-Modell")
    p.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama URL",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Sekunden Pause zwischen Calls (GPU-Cooldown, default: 2.0)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max Samples (0 = alle)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Bereits annotierte Samples überspringen",
    )
    args = p.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: {input_path} nicht gefunden", file=sys.stderr)
        sys.exit(1)

    total_lines = sum(1 for _ in input_path.open())
    effective = min(total_lines, args.limit) if args.limit > 0 else total_lines
    est_minutes = effective * (args.delay + 3) / 60  # ~3s per LLM call + delay

    print(f"=== Batch-Annotation ===")
    print(f"  Input:    {input_path} ({total_lines} Samples)")
    print(f"  Output:   {output_path}")
    print(f"  Modell:   {args.model}")
    print(f"  Delay:    {args.delay}s")
    print(f"  Limit:    {effective}")
    print(f"  Est. Zeit: ~{est_minutes:.0f} min")
    print()

    stats = annotate_batch(
        input_path, output_path, args.ollama_url, args.model, args.delay, args.limit, args.resume
    )

    print()
    print(f"=== Ergebnis ===")
    print(f"  Annotiert: {stats['annotated']}")
    print(f"  Übersprungen: {stats['skipped']}")
    print(f"  Fehler: {stats['errors']}")
    print(f"  Output: {output_path}")

    # Quick quality distribution
    if output_path.exists():
        dist: dict[str, int] = {}
        for line in output_path.read_text().splitlines():
            try:
                d = json.loads(line)
                q = d.get("annotation", {}).get("overall_quality", "?")
                dist[q] = dist.get(q, 0) + 1
            except (json.JSONDecodeError, ValueError):
                pass
        if dist:
            print(f"\n  Qualitätsverteilung:")
            for q, c in sorted(dist.items(), key=lambda x: -x[1]):
                print(f"    {q}: {c}")


if __name__ == "__main__":
    main()
