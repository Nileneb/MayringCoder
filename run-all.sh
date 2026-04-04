#!/usr/bin/env bash
# run-all.sh — Alle Modell-Kombinationen durchlaufen + Turbulenz-Analyse
#
# Jede Kombination aus Primär- und Second-Opinion-Modell bekommt eine eigene
# --run-id (cache-isoliert), sodass Ergebnisse danach verglichen werden können:
#
#   python checker.py --compare qwen2.5-coder-7b__deepseek-coder-6.7b llama3.1-8b__none
#
# Konfiguration: MODELS und SECOND_OPINION_MODELS unten anpassen.
# Leerer String "" in SECOND_OPINION_MODELS = kein Second-Opinion-Lauf.
set -e

PYTHON=".venv/bin/python"
COMMON="--no-limit --max-chars 190000"

# ── Primäre Analysemodelle ────────────────────────────────────────────────────
MODELS=(
    "llama3.1:8b"
    "codellama:latest"
    "qwen2.5-coder:7b"
    "mistral:7b-instruct"
    "deepseek-coder:6.7b-instruct"
    "starcoder2:7b"
)

# ── Second-Opinion-Modelle ────────────────────────────────────────────────────
# "" = kein Second-Opinion-Lauf (nur Primärmodell)
# Jeder Eintrag wird mit jedem Primärmodell kombiniert (Kreuzprodukt).
SECOND_OPINION_MODELS=(
    ""
    "deepseek-coder:6.7b-instruct"
    "qwen2.5-coder:7b"
)

# ── Hilfsfunktion: Modellname → sicherer Slug ─────────────────────────────────
slugify() { echo "${1//[:.\/]/-}"; }

# ── Kombinationen durchlaufen ─────────────────────────────────────────────────
for model in "${MODELS[@]}"; do
    primary_slug=$(slugify "$model")

    for so_model in "${SECOND_OPINION_MODELS[@]}"; do
        # Second-Opinion-Modell darf nicht identisch mit dem Primärmodell sein
        # (ein Modell validiert sich nicht selbst — dafür gibt es --adversarial).
        if [ -n "$so_model" ] && [ "$so_model" = "$model" ]; then
            echo ""
            echo "  ⏭  Übersprungen: $model + $so_model (identisch)"
            continue
        fi

        if [ -n "$so_model" ]; then
            so_slug=$(slugify "$so_model")
            run_id="${primary_slug}__${so_slug}"
            so_flag="--second-opinion ${so_model}"
            combo_label="${model} + Second Opinion: ${so_model}"
        else
            run_id="${primary_slug}__none"
            so_flag=""
            combo_label="${model} (kein Second Opinion)"
        fi

        echo ""
        echo "========================================"
        echo "  Primär:         $model"
        echo "  Second Opinion: ${so_model:-—}"
        echo "  run-id:         $run_id"
        echo "========================================"

        echo "--- Stufe 1: Overview ($model) ---"
        "$PYTHON" checker.py --mode overview $COMMON --model "$model" --run-id "$run_id"

        echo "--- Stufe 2: Fehlersuche ($combo_label) ---"
        # shellcheck disable=SC2086
        "$PYTHON" checker.py $COMMON --model "$model" --run-id "$run_id" $so_flag
    done
done

echo ""
echo "--- Stufe 3: Turbulenz-Analyse (einmalig, Heuristik) ---"
"$PYTHON" checker.py --mode turbulence --llm

echo ""
echo "========================================"
echo "  Alle Kombinationen abgeschlossen."
echo ""
echo "  Beispiel-Vergleiche:"
for model in "${MODELS[@]}"; do
    primary_slug=$(slugify "$model")
    # Vergleich: mit vs. ohne Second Opinion (erste SO-Kombo)
    first_so="${SECOND_OPINION_MODELS[1]:-}"
    if [ -n "$first_so" ] && [ "$first_so" != "$model" ]; then
        so_slug=$(slugify "$first_so")
        echo "    $PYTHON checker.py --compare ${primary_slug}__none ${primary_slug}__${so_slug}"
    fi
done
echo ""
echo "  Vollständigen Vergleich zweier Runs:"
echo "    $PYTHON checker.py --compare <run-id-1> <run-id-2>"
echo "========================================"
