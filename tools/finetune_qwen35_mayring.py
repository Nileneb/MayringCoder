#!/usr/bin/env python3
"""QLoRA-Fine-Tune von Qwen3.5-2B auf Mayring-Kategorisierung (vanilla PEFT, KEIN unsloth).

WHY(2026-06-06): unsloth 2026.6.1 verlangt torch<2.11 + transformers<=5.5, der lokale
Env hat torch 2.12+cu130 + transformers 5.9 → unsloth-Training bricht. Vanilla
transformers+peft+bitsandbytes (alle schon installiert, torch-kompatibel) trainieren
Qwen3.5-2B (arch qwen3_5, von transformers 5.9 unterstützt) genauso, nur ohne unsloth-
Speedup. Distillation-Targets von mistral:7b-instruct (build_dataset.py) lehren die
richtige Label-Granularität + Dedup-Verhalten, das untuned qwen3.5:2b fehlt (Duell 2:5).

Hardware: RTX 3060 12GB. 4-bit QLoRA ~6GB VRAM → mistral vorher entladen.

Usage:
    python tools/finetune_qwen35_mayring.py --data /tmp/distill_full.json --epochs 3
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

BASE = "unsloth/Qwen3.5-2B"

SYSTEM = ("Du bildest eine Kategorie nach qualitativer Inhaltsanalyse (Mayring): "
          "Paraphrase → Generalisierung → Reduktion zu EINER prägnanten Kategorie "
          "(snake_case, max 4 Wörter), die den Bezug zum Ziel wahrt.")


def _user_prompt(goal: str, text: str) -> str:
    return (
        f"ZIEL/Aufgabe (obligatorischer Bezug): {goal[:300]}\n"
        f"Textstelle:\n{text[:1200]}\n\n"
        'Antworte NUR mit JSON: {"paraphrase":"...","generalization":"...",'
        '"label":"<snake_case>"} — kein Markdown, keine Prosa.'
    )


def build_examples(records, tokenizer):
    out = []
    for r in records:
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _user_prompt(r["goal"], r["text"])},
            {"role": "assistant", "content": r["target"]},
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        out.append({"text": text})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/tmp/distill_full.json")
    p.add_argument("--base", default=BASE)
    p.add_argument("--out", default=str(Path.home() / ".cache/mayring_finetune/qwen35-mayring"))
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--bsz", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--val-split", type=float, default=0.1)
    args = p.parse_args()

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    records = json.load(open(args.data))
    print(f"records: {len(records)}")

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    examples = build_examples(records, tok)
    ds = Dataset.from_list(examples).train_test_split(test_size=args.val_split, seed=7)

    cfg = SFTConfig(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, logging_steps=5, save_strategy="epoch",
        eval_strategy="no", bf16=True, optim="adafactor",
        warmup_ratio=0.05, lr_scheduler_type="cosine", max_length=512,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[], dataset_text_field="text", packing=False)

    trainer = SFTTrainer(
        model=model, args=cfg,
        train_dataset=ds["train"],
        processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"\nLoRA-Adapter gespeichert: {args.out}")


if __name__ == "__main__":
    main()
