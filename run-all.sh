#!/usr/bin/env bash
set -e

PYTHON=".venv/bin/python"
COMMON="--no-limit --max-chars 9000 --cache-by-model"

MODELS=(
    "llama3.1:8b"
    "llama3.2:latest"
    "codellama:latest"
    "qwen2.5-coder:7b"
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
echo "Alle Modelle abgeschlossen."