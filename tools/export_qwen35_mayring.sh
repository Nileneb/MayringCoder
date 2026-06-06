#!/usr/bin/env bash
# Merge LoRA-Adapter → GGUF → ollama-Modell qwen3.5-mayring:2b.
# WHY(2026-06-06): vanilla-PEFT-Training (finetune_qwen35_mayring.py) liefert einen
# LoRA-Adapter; ollama braucht ein eigenständiges Modell.
#
# FALLEN (teuer gelernt, siehe [[project-qwen35-mayring-finetune]]):
#  - Weder unsloth-llama.cpp noch ollama-0.21-intern können qwen3_5 konvertieren →
#    NUR die NEUESTE llama.cpp master (arch in conversion/qwen.py).
#  - ollama 0.21 lädt das GGUF sonst nicht ("qwen3next: layer 24 missing
#    attn_qkv/attn_gate") → die MTP-Layer müssen raus: --no-mtp.
#  - Stop-Tokens nötig, sonst generiert das Modell über <|endoftext|> hinaus.
set -euo pipefail

ADAPTER="${1:-$HOME/.cache/mayring_finetune/qwen35-mayring}"
BASE="unsloth/Qwen3.5-2B"
MERGED="$HOME/.cache/mayring_finetune/qwen35-mayring-merged"
GGUF="$HOME/.cache/mayring_finetune/qwen35-mayring-nomtp.gguf"
LLAMACPP_DIR="${LLAMACPP_DIR:-/tmp/llamacpp_latest}"
TAG="qwen3.5-mayring:2b"

if [ ! -f "$LLAMACPP_DIR/convert_hf_to_gguf.py" ]; then
  echo "=== clone neueste llama.cpp (qwen3_5-Support) ==="
  git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMACPP_DIR"
  pip install -q -r "$LLAMACPP_DIR/requirements/requirements-convert_hf_to_gguf.txt"
fi

echo "=== 1. Merge LoRA → fp16 (CPU) ==="
python3 - "$BASE" "$ADAPTER" "$MERGED" <<'PY'
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base, adapter, out = sys.argv[1], sys.argv[2], sys.argv[3]
tok = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16,
        device_map="cpu", trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
model.save_pretrained(out, safe_serialization=True); tok.save_pretrained(out)
print("merged ->", out)
PY

echo "=== 2. HF → GGUF q8_0, OHNE MTP-Head ==="
python3 "$LLAMACPP_DIR/convert_hf_to_gguf.py" "$MERGED" --outfile "$GGUF" --outtype q8_0 --no-mtp

echo "=== 3. ollama create $TAG (mit Stop-Tokens) ==="
MF="$HOME/.cache/mayring_finetune/Modelfile_final"
{ printf 'FROM %s\n' "$GGUF"
  printf 'PARAMETER stop "<|im_end|>"\n'
  printf 'PARAMETER stop "<|endoftext|>"\n'; } > "$MF"
ollama create "$TAG" -f "$MF"
echo "=== fertig: $TAG ===" && ollama list | grep -i mayring || true
