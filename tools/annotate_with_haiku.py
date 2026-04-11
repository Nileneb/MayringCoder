#!/usr/bin/env python3
"""Batch-Annotation von Training-Samples via Claude Haiku API.

Liest training_log JSONL, lässt jedes Finding von Claude Haiku bewerten,
schreibt annotierte JSONL mit Qualitätslabels — als Basisdatensatz für
Fine-Tuning von qwen3.5:2b.

Usage:
    # Basis: 75 Samples (Standardlimit)
    ANTHROPIC_API_KEY=sk-ant-... python tools/annotate_with_haiku.py

    # Custom Limit + Output
    python tools/annotate_with_haiku.py --limit 100 --output cache/haiku_annotations.jsonl

    # Fortsetzen nach Abbruch
    python tools/annotate_with_haiku.py --resume

    # Nur parsed_ok=True Samples (höhere Datenqualität)
    python tools/annotate_with_haiku.py --only-parsed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REVIEW_SYSTEM = """You are a code review quality assessor. Rate automated findings as true or false positives.
Reply ONLY with valid JSON (no markdown, no explanation):
{"overall_quality":"good|partial|bad","true_positives":N,"false_positives":N,"total_findings":N,"precision":0.0,"reasoning":"1 sentence max"}

Criteria:
- good: precision >= 0.8, findings are real issues in the code
- partial: precision 0.4-0.79, mixed real and noise
- bad: precision < 0.4, mostly false positives or irrelevant"""

_REVIEW_PROMPT = """Rate this automated code analysis. How many findings are real vs false positives?

FILE: {label}

CODE (truncated to 600 chars):
{code}

LLM ANALYSIS OUTPUT:
{response}

Reply with JSON only."""


def _get_client():
    """Get Anthropic client, with helpful error if key missing."""
    try:
        import anthropic
    except ImportError:
        print("Error: anthropic SDK nicht installiert. Bitte: pip install anthropic", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Try .env file
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    if not api_key:
        print(
            "Error: ANTHROPIC_API_KEY nicht gefunden.\n"
            "Setze die Variable: export ANTHROPIC_API_KEY=sk-ant-...\n"
            "Oder füge sie zur .env Datei hinzu: ANTHROPIC_API_KEY=sk-ant-...",
            file=sys.stderr,
        )
        sys.exit(1)

    return anthropic.Anthropic(api_key=api_key)


def _haiku_review(client, label: str, code: str, response: str) -> dict | None:
    """Single Haiku API call for annotation."""
    prompt = _REVIEW_PROMPT.format(
        label=label,
        code=code[:600],
        response=response[:800],
    )

    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=_REVIEW_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = msg.content[0].text.strip()

    # Parse JSON
    try:
        d = json.loads(raw)
        if isinstance(d, dict) and "overall_quality" in d:
            return d
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: strip markdown fences
    import re
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                try:
                    d = json.loads(part)
                    if "overall_quality" in d:
                        return d
                except (json.JSONDecodeError, ValueError):
                    pass

    # Last resort: regex
    m = re.search(r'\{[^{}]*"overall_quality"[^{}]*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def annotate_batch(
    input_path: Path,
    output_path: Path,
    limit: int,
    resume: bool,
    only_parsed: bool,
    delay: float,
) -> dict:
    """Annotate training samples via Haiku API."""
    client = _get_client()

    lines = input_path.read_text().splitlines()

    # Filter: skip empty lines
    samples = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            samples.append(d)
        except (json.JSONDecodeError, ValueError):
            pass

    # Filter: only parsed_ok samples if requested (better data quality)
    if only_parsed:
        before = len(samples)
        samples = [s for s in samples if s.get("parsed_ok", False)]
        print(f"[filter] parsed_ok=True: {len(samples)}/{before} Samples")

    # Filter: skip samples with empty/trivial responses
    before = len(samples)
    samples = [
        s for s in samples
        if len(s.get("raw_response", "").strip()) >= 20
    ]
    print(f"[filter] response>=20 chars: {len(samples)}/{before} Samples")

    # Apply limit
    if limit > 0:
        samples = samples[:limit]

    # Load already-annotated hashes for resume
    done_hashes: set[str] = set()
    if resume and output_path.exists():
        for line in output_path.read_text().splitlines():
            try:
                d = json.loads(line)
                done_hashes.add(d.get("prompt_hash", ""))
            except (json.JSONDecodeError, ValueError):
                pass
        print(f"[resume] {len(done_hashes)} bereits annotiert, überspringe diese")

    stats = {"total": len(samples), "annotated": 0, "skipped": 0, "errors": 0, "cost_tokens": 0}
    mode = "a" if resume else "w"

    print(f"\n[start] {len(samples)} Samples → {output_path}")
    print(f"[model] claude-haiku-4-5-20251001")
    print()

    with output_path.open(mode, encoding="utf-8") as out:
        for i, sample in enumerate(samples):
            prompt_hash = sample.get("prompt_hash", f"line_{i}")

            # Skip if already done
            if prompt_hash in done_hashes:
                stats["skipped"] += 1
                continue

            label = sample.get("label", "?")
            code = sample.get("prompt", "")
            response = sample.get("raw_response", "")

            pct = (i + 1) / len(samples) * 100
            print(f"[{i+1}/{len(samples)}] ({pct:.0f}%) {label[:40]}...", end=" ", flush=True)

            try:
                annotation = _haiku_review(client, label, code, response)

                if annotation:
                    sample["annotation"] = annotation
                    sample["annotation_model"] = "claude-haiku-4-5-20251001"
                    sample["annotation_ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    stats["annotated"] += 1
                    quality = annotation.get("overall_quality", "?")
                    precision = annotation.get("precision", 0)
                    tp = annotation.get("true_positives", "?")
                    fp = annotation.get("false_positives", "?")
                    print(f"→ {quality} (prec: {precision:.2f}, tp:{tp} fp:{fp})")
                else:
                    sample["annotation"] = {"overall_quality": "parse_error"}
                    sample["annotation_model"] = "claude-haiku-4-5-20251001"
                    stats["errors"] += 1
                    print("→ parse_error")

            except Exception as exc:
                sample["annotation"] = {"overall_quality": "error", "error": str(exc)[:200]}
                stats["errors"] += 1
                print(f"→ ERROR: {exc}")

            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            out.flush()

            # Small delay to stay within rate limits (Haiku: 1000 RPM on free tier)
            if delay > 0 and i < len(samples) - 1:
                time.sleep(delay)

    return stats


def main() -> None:
    p = argparse.ArgumentParser(
        description="Batch-Annotation via Claude Haiku API (Basisdatensatz für Fine-Tuning)"
    )
    p.add_argument(
        "--input",
        default="cache/nileneb-applinngames_training_log.jsonl",
        help="Input JSONL (default: training_log)",
    )
    p.add_argument(
        "--output",
        default="cache/haiku_annotations.jsonl",
        help="Output JSONL mit Annotations",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=75,
        help="Max Samples (0 = alle, default: 75)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Bereits annotierte Samples überspringen",
    )
    p.add_argument(
        "--only-parsed",
        action="store_true",
        help="Nur Samples mit parsed_ok=True (höhere Datenqualität)",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Pause zwischen API-Calls in Sekunden (default: 0.1)",
    )
    args = p.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: {input_path} nicht gefunden", file=sys.stderr)
        sys.exit(1)

    total_lines = sum(1 for _ in input_path.open() if _.strip())
    effective = min(total_lines, args.limit) if args.limit > 0 else total_lines

    print(f"=== Haiku Batch-Annotation ===")
    print(f"  Input:       {input_path} ({total_lines} Samples)")
    print(f"  Output:      {output_path}")
    print(f"  Limit:       {effective}")
    print(f"  Only-parsed: {args.only_parsed}")
    print(f"  Resume:      {args.resume}")
    print(f"  Delay:       {args.delay}s")
    print(f"  Est. Zeit:   ~{effective * (args.delay + 0.5) / 60:.1f} min")
    print()

    stats = annotate_batch(
        input_path,
        output_path,
        args.limit,
        args.resume,
        args.only_parsed,
        args.delay,
    )

    print()
    print(f"=== Ergebnis ===")
    print(f"  Annotiert:    {stats['annotated']}")
    print(f"  Übersprungen: {stats['skipped']}")
    print(f"  Fehler:       {stats['errors']}")
    print(f"  Output:       {output_path}")

    # Quality distribution
    if output_path.exists():
        dist: dict[str, int] = {}
        prec_sum = 0.0
        prec_count = 0
        for line in output_path.read_text().splitlines():
            try:
                d = json.loads(line)
                ann = d.get("annotation", {})
                q = ann.get("overall_quality", "?")
                dist[q] = dist.get(q, 0) + 1
                prec = ann.get("precision")
                if isinstance(prec, (int, float)):
                    prec_sum += prec
                    prec_count += 1
            except (json.JSONDecodeError, ValueError):
                pass

        if dist:
            print(f"\n  Qualitätsverteilung:")
            for q, c in sorted(dist.items(), key=lambda x: -x[1]):
                pct = c / sum(dist.values()) * 100
                print(f"    {q}: {c} ({pct:.0f}%)")
            if prec_count:
                print(f"\n  Avg Precision: {prec_sum / prec_count:.2f}")

        # Export good+partial for fine-tuning
        finetuning_path = output_path.parent / "haiku_finetuning_ready.jsonl"
        good_samples = []
        for line in output_path.read_text().splitlines():
            try:
                d = json.loads(line)
                q = d.get("annotation", {}).get("overall_quality", "")
                if q in ("good", "partial"):
                    good_samples.append(d)
            except (json.JSONDecodeError, ValueError):
                pass

        if good_samples:
            finetuning_path.write_text(
                "\n".join(json.dumps(s, ensure_ascii=False) for s in good_samples),
                encoding="utf-8",
            )
            print(f"\n  Fine-tuning set: {len(good_samples)} Samples → {finetuning_path}")


if __name__ == "__main__":
    main()
