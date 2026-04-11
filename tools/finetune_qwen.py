#!/usr/bin/env python3
"""Fine-Tuning von Qwen3-2B mit Unsloth + QLoRA auf annotierten Code-Review-Samples.

Voraussetzungen:
    pip install unsloth
    pip install trl datasets

Modell: Qwen/Qwen3-2B-Instruct (HuggingFace)
Hardware: RTX 3060 12GB (QLoRA 4-bit: ~5-6GB VRAM)

Usage:
    # Standard (Qwen3-2B, 3 Epochen)
    python tools/finetune_qwen.py

    # Mit partial-Samples, mehr Epochen
    python tools/finetune_qwen.py --epochs 5 --include-partial

    # Checkpoint fortsetzen
    python tools/finetune_qwen.py --resume-from cache/finetuning/checkpoints/
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_dataset_from_jsonl(path: Path):
    """Load JSONL as HuggingFace Dataset."""
    from datasets import Dataset
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            # Remove _meta for training (not needed by model)
            d.pop("_meta", None)
            records.append(d)
        except (json.JSONDecodeError, ValueError):
            pass
    return Dataset.from_list(records)


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-Tune Qwen3-2B mit QLoRA")
    p.add_argument("--model", default="unsloth/Qwen3-2B-Instruct-bnb-4bit",
                   help="HuggingFace Modell-ID")
    p.add_argument("--data-dir", default="cache/finetuning",
                   help="Verzeichnis mit train.jsonl + val.jsonl")
    p.add_argument("--output-dir", default="cache/finetuning/output",
                   help="Ausgabeverzeichnis für LoRA-Adapter")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2,
                   help="Batch size (2 für RTX 3060 12GB)")
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Gradient accumulation steps (effektiv: batch*accum=8)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=2048,
                   help="Max Sequenzlänge (2048 spart VRAM)")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--include-partial", action="store_true",
                   help="Muss mit prepare_finetuning_data.py --include-partial übereinstimmen")
    p.add_argument("--resume-from", default=None,
                   help="Checkpoint-Pfad zum Fortsetzen")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"

    if not train_path.exists():
        print(f"Error: {train_path} nicht gefunden.")
        print("Zuerst ausführen: python tools/prepare_finetuning_data.py")
        return

    print(f"=== Qwen3 Fine-Tuning ===")
    print(f"  Modell:    {args.model}")
    print(f"  Epochen:   {args.epochs}")
    print(f"  Batch:     {args.batch_size} × {args.grad_accum} (effektiv: {args.batch_size * args.grad_accum})")
    print(f"  LR:        {args.lr}")
    print(f"  LoRA r:    {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"  Max seq:   {args.max_seq_len}")
    print(f"  Output:    {output_dir}")
    print()

    # --- Imports (deferred so errors are clear) ---
    try:
        from unsloth import FastLanguageModel
        from unsloth import is_bfloat16_supported
    except ImportError:
        print("Error: unsloth nicht installiert.")
        print("Installieren: pip install unsloth")
        return

    try:
        from trl import SFTTrainer, SFTConfig
        from datasets import concatenate_datasets
    except ImportError:
        print("Error: trl oder datasets nicht installiert.")
        print("Installieren: pip install trl datasets")
        return

    # --- Model laden ---
    print("[1/4] Modell laden...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    # --- LoRA konfigurieren ---
    print("[2/4] LoRA konfigurieren...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    model.print_trainable_parameters()

    # --- Dataset laden ---
    print("[3/4] Dataset laden...")
    train_ds = load_dataset_from_jsonl(train_path)
    val_ds = load_dataset_from_jsonl(val_path)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Apply chat template
    def format_sample(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_ds = train_ds.map(format_sample, remove_columns=["messages"])
    val_ds = val_ds.map(format_sample, remove_columns=["messages"])

    # --- Training ---
    print("[4/4] Training starten...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
            dataset_text_field="text",
            max_seq_length=args.max_seq_len,
            resume_from_checkpoint=args.resume_from,
        ),
    )

    trainer_stats = trainer.train()

    print(f"\n=== Training abgeschlossen ===")
    print(f"  Steps:      {trainer_stats.global_step}")
    print(f"  Train Loss: {trainer_stats.training_loss:.4f}")
    runtime_min = trainer_stats.metrics.get("train_runtime", 0) / 60
    print(f"  Dauer:      {runtime_min:.1f} min")

    # --- Adapter speichern ---
    adapter_path = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n  LoRA-Adapter gespeichert: {adapter_path}")
    print(f"\nNächster Schritt: python tools/export_to_ollama.py")


if __name__ == "__main__":
    main()
