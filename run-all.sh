#!/usr/bin/env bash
# run-all.sh — Alle Modelle durchlaufen + Turbulenz-Analyse
#
# Jedes Modell bekommt eine eigene --run-id (cache-isoliert), sodass
# Ergebnisse danach mit --compare verglichen werden können:
#
#   python checker.py --compare llama3.1-8b qwen2.5-coder-7b
#
set -e

PYTHON=".venv/bin/python"
COMMON="--no-limit --max-chars 190000"

MODELS=(
    "llama3.1:8b"
    "codellama:latest"
    "qwen2.5-coder:7b"
    "mistral:7b-instruct"
    "deepseek-coder:6.7b-instruct"
    "starcoder2:7b"
)

for model in "${MODELS[@]}"; do
    # run-id: Modellname ohne Sonderzeichen, z.B. "llama3.1:8b" → "llama3.1-8b"
    run_id="${model//[:.\/]/-}"

    echo ""
    echo "========================================"
    echo "  Modell: $model  (run-id: $run_id)"
    echo "========================================"

    echo "--- Stufe 1: Overview ($model) ---"
    "$PYTHON" checker.py --mode overview $COMMON --model "$model" --run-id "$run_id"

    echo "--- Stufe 2: Fehlersuche ($model) ---"
    "$PYTHON" checker.py $COMMON --model "$model" --run-id "$run_id"
done

echo ""
echo "--- Stufe 3: Turbulenz-Analyse (einmalig, Heuristik) ---"
"$PYTHON" checker.py --mode turbulence --llm

echo ""
echo "========================================"
echo "  Alle Modelle abgeschlossen."
echo ""
echo "  Runs vergleichen z.B.:"
for i in "${!MODELS[@]}"; do
    run_id_a="${MODELS[$i]//[:.\/]/-}"
    if [ $((i+1)) -lt ${#MODELS[@]} ]; then
        run_id_b="${MODELS[$((i+1))]//[:.\/]/-}"
        echo "    $PYTHON checker.py --compare $run_id_a $run_id_b"
    fi
done
echo "========================================"
