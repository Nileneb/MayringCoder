#!/usr/bin/env python3
"""Exportiert den fine-getunten LoRA-Adapter als GGUF → Ollama-Modell.

Schritte:
  1. LoRA-Adapter + Basismodell zusammenführen (merge)
  2. Als GGUF exportieren (Q8_0 wie das Original)
  3. Ollama Modelfile erstellen
  4. Ollama-Modell registrieren

Voraussetzungen:
    pip install unsloth
    ollama (im PATH)

Usage:
    python tools/export_to_ollama.py
    python tools/export_to_ollama.py --quant q4_k_m --name mayring-qwen3:2b
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


_MODELFILE_TEMPLATE = """\
FROM {gguf_path}

RENDERER qwen3.5
PARSER qwen3.5

PARAMETER temperature 0.3
PARAMETER top_k 5
PARAMETER top_p 0.95
PARAMETER presence_penalty 1.5
PARAMETER num_predict 1024

SYSTEM \"\"\"Du bist Pi, ein automatischer Code-Analyse-Agent. Deine Antwort MUSS immer und ausschließlich ein valides JSON-Objekt sein — keine Erklärungen, kein Markdown, kein Fließtext.

PFLICHTFORMAT (exakt so, kein anderes Format erlaubt):
{{\"file_summary\":\"...\",\"potential_smells\":[]}}

WICHTIG:
- Nutze search_memory bevor du ein Finding schreibst — prüfe ob es eine bekannte Framework-Konvention ist
- Im Zweifel KEIN Finding (false positives sind teurer als false negatives)
- Antworte NUR mit dem JSON-Objekt, kein Text davor oder danach\"\"\"
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-Tuned Qwen3 → Ollama exportieren")
    p.add_argument("--adapter-dir", default="cache/finetuning/output/lora_adapter",
                   help="Pfad zum gespeicherten LoRA-Adapter")
    p.add_argument("--output-dir", default="cache/finetuning/output",
                   help="Ausgabeverzeichnis für GGUF")
    p.add_argument("--quant", default="q8_0",
                   choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
                   help="GGUF-Quantisierung (default: q8_0, wie Original)")
    p.add_argument("--name", default="mayring-qwen3:2b",
                   help="Ollama Modell-Name")
    p.add_argument("--base-model", default="unsloth/Qwen3-2B-Instruct-bnb-4bit",
                   help="HF Basis-Modell-ID (muss mit finetune_qwen.py übereinstimmen)")
    p.add_argument("--skip-merge", action="store_true",
                   help="Merge überspringen (wenn bereits gemergt)")
    args = p.parse_args()

    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_dir = output_dir / "merged_model"
    gguf_path = output_dir / f"qwen3_finetuned_{args.quant}.gguf"
    modelfile_path = output_dir / "Modelfile"

    # --- Step 1: Merge LoRA → full model ---
    if not args.skip_merge:
        if not adapter_dir.exists():
            print(f"Error: Adapter nicht gefunden: {adapter_dir}")
            print("Zuerst: python tools/finetune_qwen.py")
            sys.exit(1)

        print("[1/3] LoRA-Adapter mit Basismodell zusammenführen...")
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            print("Error: pip install unsloth")
            sys.exit(1)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_dir),
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        merged_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        print(f"  Merged model: {merged_dir}")
    else:
        print(f"[1/3] Merge übersprungen, verwende: {merged_dir}")

    # --- Step 2: Convert to GGUF ---
    print(f"[2/3] Als GGUF exportieren ({args.quant})...")
    try:
        from unsloth import FastLanguageModel
        # Unsloth kann direkt als GGUF exportieren
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_dir) if not args.skip_merge else str(merged_dir),
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        model.save_pretrained_gguf(
            str(output_dir / "qwen3_finetuned"),
            tokenizer,
            quantization_method=args.quant,
        )
        # Unsloth saves to a subdirectory — search recursively for Q8_0/q8_0 GGUF
        quant_upper = args.quant.upper()
        candidates = (
            list(output_dir.glob(f"**/*.{quant_upper}.gguf"))
            or list(output_dir.glob(f"**/*.{args.quant}.gguf"))
            or list(output_dir.glob("**/*.gguf"))
        )
        if candidates:
            gguf_path = candidates[0]
        print(f"  GGUF: {gguf_path}")
    except Exception as exc:
        print(f"  Unsloth GGUF-Export fehlgeschlagen: {exc}")
        print("  Fallback: llama.cpp convert_hf_to_gguf.py nutzen")
        print(f"  python llama.cpp/convert_hf_to_gguf.py {merged_dir} --outfile {gguf_path} --outtype {args.quant}")
        sys.exit(1)

    # --- Step 3: Ollama Modelfile + Register ---
    print(f"[3/3] Ollama Modell erstellen: {args.name}")
    modelfile_content = _MODELFILE_TEMPLATE.format(gguf_path=gguf_path.resolve())
    modelfile_path.write_text(modelfile_content, encoding="utf-8")
    print(f"  Modelfile: {modelfile_path}")

    result = subprocess.run(
        ["ollama", "create", args.name, "-f", str(modelfile_path)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"\n  Ollama-Modell registriert: {args.name}")
        print(f"\nTesten:")
        print(f'  ollama run {args.name} "Analysiere diese PHP-Datei: ..."')
        print(f"\nIn MayringCoder nutzen:")
        print(f"  OLLAMA_MODEL={args.name} .venv/bin/python checker.py --repo ...")
    else:
        print(f"  Ollama-Fehler: {result.stderr}")
        print(f"\n  Manuell registrieren:")
        print(f"  ollama create {args.name} -f {modelfile_path}")


if __name__ == "__main__":
    main()
