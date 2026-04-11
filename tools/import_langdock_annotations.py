#!/usr/bin/env python3
"""Importiert Langdock-Annotations-Antworten zurück ins Training-JSONL.

Usage:
    # Füge Langdock-Antworten als Textdateien ein:
    python tools/import_langdock_annotations.py \
        --responses cache/langdock_batches/response_batch_1.txt \
                    cache/langdock_batches/response_batch_2.txt \
                    cache/langdock_batches/response_batch_3.txt

    # Output-Datei (default: cache/haiku_annotations.jsonl)
    python tools/import_langdock_annotations.py --responses ... --output cache/haiku_annotations.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


def parse_annotations_from_text(text: str) -> dict[str, dict]:
    """Extract JSON annotations from Langdock response text."""
    annotations: dict[str, dict] = {}

    # Find all JSON objects with sample_id
    pattern = r'\{[^{}]*"sample_id"[^{}]*\}'
    for m in re.finditer(pattern, text, re.DOTALL):
        try:
            d = json.loads(m.group())
            sid = d.get("sample_id")
            if sid and "overall_quality" in d:
                annotations[sid] = d
        except (json.JSONDecodeError, ValueError):
            pass

    # Also try line-by-line
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
            sid = d.get("sample_id")
            if sid and "overall_quality" in d:
                annotations[sid] = d
        except (json.JSONDecodeError, ValueError):
            pass

    return annotations


def merge_annotations(
    training_path: Path,
    response_files: list[Path],
    output_path: Path,
    limit: int = 75,
) -> dict:
    """Merge Langdock annotations back into training JSONL."""

    # Load all response annotations
    all_annotations: dict[str, dict] = {}
    for rfile in response_files:
        if not rfile.exists():
            print(f"[WARN] Response-Datei nicht gefunden: {rfile}")
            continue
        text = rfile.read_text(encoding="utf-8")
        parsed = parse_annotations_from_text(text)
        all_annotations.update(parsed)
        print(f"[load] {rfile.name}: {len(parsed)} Annotations")

    print(f"[total] {len(all_annotations)} Annotations geladen")

    # Load training samples (same filter as batch export)
    samples = []
    with training_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if d.get("parsed_ok") and len(d.get("raw_response", "").strip()) >= 20:
                    samples.append(d)
            except (json.JSONDecodeError, ValueError):
                pass

    if limit > 0:
        samples = samples[:limit]

    stats = {"total": len(samples), "matched": 0, "unmatched": 0}

    with output_path.open("w", encoding="utf-8") as out:
        for sample in samples:
            ph = sample.get("prompt_hash", "")
            ann = all_annotations.get(ph)

            if ann:
                # Remove sample_id from annotation (redundant)
                ann_clean = {k: v for k, v in ann.items() if k != "sample_id"}
                sample["annotation"] = ann_clean
                sample["annotation_model"] = "claude-haiku-4-5-20251001-via-langdock"
                sample["annotation_ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                stats["matched"] += 1
            else:
                stats["unmatched"] += 1

            out.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Importiert Langdock-Annotations in Training-JSONL")
    p.add_argument(
        "--responses",
        nargs="+",
        required=True,
        help="Langdock-Antwort-Textdateien (eine oder mehrere)",
    )
    p.add_argument(
        "--input",
        default="cache/nileneb-applinngames_training_log.jsonl",
        help="Training-JSONL (Input)",
    )
    p.add_argument(
        "--output",
        default="cache/haiku_annotations.jsonl",
        help="Annotiertes JSONL (Output)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=75,
        help="Anzahl Samples (muss mit Export übereinstimmen)",
    )
    args = p.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    response_files = [Path(r) for r in args.responses]

    if not input_path.exists():
        print(f"Error: {input_path} nicht gefunden")
        return

    stats = merge_annotations(input_path, response_files, output_path, args.limit)

    print()
    print(f"=== Import-Ergebnis ===")
    print(f"  Gesamt:      {stats['total']}")
    print(f"  Gematched:   {stats['matched']}")
    print(f"  Unmatched:   {stats['unmatched']}")
    print(f"  Output:      {output_path}")

    # Quality distribution
    if output_path.exists():
        dist: dict[str, int] = {}
        prec_vals: list[float] = []
        for line in output_path.read_text().splitlines():
            try:
                d = json.loads(line)
                ann = d.get("annotation", {})
                q = ann.get("overall_quality", "unannotated")
                dist[q] = dist.get(q, 0) + 1
                prec = ann.get("precision")
                if isinstance(prec, (int, float)):
                    prec_vals.append(float(prec))
            except (json.JSONDecodeError, ValueError):
                pass

        print(f"\n  Qualitätsverteilung:")
        for q, c in sorted(dist.items(), key=lambda x: -x[1]):
            pct = c / sum(dist.values()) * 100
            print(f"    {q}: {c} ({pct:.0f}%)")
        if prec_vals:
            print(f"\n  Avg Precision: {sum(prec_vals)/len(prec_vals):.2f}")

    # Export fine-tuning ready set
    finetuning_path = output_path.parent / "haiku_finetuning_ready.jsonl"
    good = []
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            try:
                d = json.loads(line)
                q = d.get("annotation", {}).get("overall_quality", "")
                if q in ("good", "partial"):
                    good.append(d)
            except (json.JSONDecodeError, ValueError):
                pass
    if good:
        finetuning_path.write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in good),
            encoding="utf-8",
        )
        print(f"\n  Fine-tuning set: {len(good)} Samples → {finetuning_path}")


if __name__ == "__main__":
    main()
