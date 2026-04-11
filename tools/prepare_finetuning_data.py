#!/usr/bin/env python3
"""Konvertiert annotierte Training-Samples in Unsloth/SFT-Format.

Nur "good"-Quality-Samples werden verwendet (precision >= 0.8).
Output: train.jsonl + val.jsonl im messages-Format für Qwen3-Chat-Template.

Usage:
    python tools/prepare_finetuning_data.py
    python tools/prepare_finetuning_data.py --include-partial --val-split 0.1
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

_SYSTEM_MARKER = "Du bist ein erfahrener Code-Reviewer"
_FILE_MARKERS = ["## Datei:", "---\n\n## ", "```"]


def _split_prompt(prompt: str) -> tuple[str, str]:
    """Split full prompt into (system_part, user_content)."""
    # Find where the file content starts
    best_idx = len(prompt)
    for marker in _FILE_MARKERS:
        idx = prompt.find(marker)
        if 0 < idx < best_idx:
            best_idx = idx

    system_part = prompt[:best_idx].strip()
    user_content = prompt[best_idx:].strip()

    # If no split found, treat everything as user content with default system
    if best_idx == len(prompt) or not user_content:
        system_part = "Du bist ein erfahrener Code-Reviewer mit Fokus auf nachhaltige Software-Qualität."
        user_content = prompt.strip()

    return system_part, user_content


def prepare(
    input_path: Path,
    output_dir: Path,
    include_partial: bool,
    val_split: float,
    seed: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    accept_quality = {"good"}
    if include_partial:
        accept_quality.add("partial")

    samples = []
    skipped = 0
    for line in input_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue

        ann = d.get("annotation", {})
        quality = ann.get("overall_quality", "")
        if quality not in accept_quality:
            skipped += 1
            continue

        prompt = d.get("prompt", "").strip()
        response = d.get("raw_response", "").strip()
        if not prompt or not response or len(response) < 10:
            skipped += 1
            continue

        system_part, user_content = _split_prompt(prompt)

        samples.append({
            "messages": [
                {"role": "system", "content": system_part},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response},
            ],
            # Metadata (not used for training, useful for debugging)
            "_meta": {
                "prompt_hash": d.get("prompt_hash", ""),
                "label": d.get("label", ""),
                "quality": quality,
                "precision": ann.get("precision", 0),
            },
        })

    # Shuffle + split
    random.seed(seed)
    random.shuffle(samples)

    val_n = max(1, int(len(samples) * val_split))
    val_samples = samples[:val_n]
    train_samples = samples[val_n:]

    def write_jsonl(path: Path, data: list) -> None:
        path.write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in data),
            encoding="utf-8",
        )

    write_jsonl(output_dir / "train.jsonl", train_samples)
    write_jsonl(output_dir / "val.jsonl", val_samples)

    # Stats
    quality_dist: dict[str, int] = {}
    for s in samples:
        q = s["_meta"]["quality"]
        quality_dist[q] = quality_dist.get(q, 0) + 1

    return {
        "total": len(samples),
        "train": len(train_samples),
        "val": len(val_samples),
        "skipped": skipped,
        "quality_dist": quality_dist,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-Tuning Daten vorbereiten")
    p.add_argument("--input", default="cache/haiku_finetuning_ready.jsonl")
    p.add_argument("--output-dir", default="cache/finetuning")
    p.add_argument("--include-partial", action="store_true",
                   help="Auch 'partial' Quality-Samples einbeziehen")
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Anteil Validation-Set (default: 0.1)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    stats = prepare(
        Path(args.input),
        Path(args.output_dir),
        args.include_partial,
        args.val_split,
        args.seed,
    )

    print(f"=== Fine-Tuning Daten ===")
    print(f"  Gesamt:    {stats['total']} Samples")
    print(f"  Train:     {stats['train']}")
    print(f"  Val:       {stats['val']}")
    print(f"  Skipped:   {stats['skipped']}")
    print(f"  Qualität:  {stats['quality_dist']}")
    print(f"  Output:    {args.output_dir}/")
    print(f"    → train.jsonl ({stats['train']} samples)")
    print(f"    → val.jsonl  ({stats['val']} samples)")


if __name__ == "__main__":
    main()
