#!/usr/bin/env bash
set -e

PYTHON=".venv/bin/python"
COMMON="--no-limit --max-chars 190000 --cache-by-model"

MODELS=(
    "llama3.1:8b"
    "codellama:latest"
    "qwen2.5-coder:7b"
    "mistral:7b-instruct"
    "deepseek-coder:6.7b-instruct"
    "starcoder2:7b"
)

for model in "${MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "  Modell: $model"
    echo "========================================"

    echo "--- Stufe 1: Overview ($model) ---"
    "$PYTHON" checker.py --mode overview $COMMON --model "$model"

    echo "--- Stufe 2: Fehlersuche ($model) ---"
    "$PYTHON" checker.py $COMMON --model "$model"
done

echo ""
echo "--- Stufe 3: Turbulenz-Analyse ---"
"$PYTHON" turbulence_run.py --llm

echo ""
echo "Alle Modelle und Turbulenz-Analyse abgeschlossen."