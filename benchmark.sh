#!/usr/bin/env bash
# benchmark.sh — Vergleichbarer Benchmark aller verfügbaren Ollama-Modelle
#
# Jedes Modell bekommt dasselbe Zeit-Budget (Standard: 600s = 10 Minuten).
# Stufe 1 (Overview) und Stufe 2 (Turbulenz) laufen einmalig vorab —
# sie sind modellunabhängig und werden gecacht.
#
# Danach: benchmark_summary.py zeigt Vergleichsmatrix (Dateien, Findings,
# Parse-Fehler, Laufzeit) — wer kam am weitesten, wer lieferte die besten Ergebnisse.
#
# Verwendung:
#   bash benchmark.sh                    # alle Ollama-Modelle, 10min je Modell
#   TIME_BUDGET=300 bash benchmark.sh    # 5 Minuten je Modell
#   MODELS="qwen2.5-coder:7b mistral:7b-instruct" bash benchmark.sh  # nur diese Modelle
set -e

PYTHON=".venv/bin/python"
TIME_BUDGET="${TIME_BUDGET:-600}"
COMMON="--no-limit --max-chars 190000"

# ── Modelle: MODELS-Env überschreibt automatische Erkennung ──────────────────
if [ -z "$MODELS" ]; then
    if ! command -v ollama &>/dev/null; then
        echo "FEHLER: ollama nicht gefunden. Bitte MODELS='model1 model2' setzen."
        exit 1
    fi
    MODELS=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | grep -v '^$')
    if [ -z "$MODELS" ]; then
        echo "FEHLER: Keine Ollama-Modelle gefunden."
        exit 1
    fi
fi

slugify() { echo "${1//[:.\/]/-}"; }

echo ""
echo "========================================"
echo "  MayringCoder — Benchmark"
echo "  Zeit-Budget: ${TIME_BUDGET}s je Modell"
echo "  Modelle:"
for m in $MODELS; do echo "    - $m"; done
echo "========================================"

# ── Stufe 1: Overview (einmalig mit erstem Modell) ───────────────────────────
FIRST_MODEL=$(echo "$MODELS" | awk 'NR==1')
echo ""
echo "========================================"
echo "  Stufe 1: Overview (einmalig, $FIRST_MODEL)"
echo "========================================"
# shellcheck disable=SC2086
"$PYTHON" checker.py --mode overview $COMMON --model "$FIRST_MODEL" --cache-by-model

# ── Stufe 2: Turbulenz (einmalig, mit Overview-Cache) ────────────────────────
echo ""
echo "========================================"
echo "  Stufe 2: Turbulenz-Analyse (einmalig)"
echo "========================================"
"$PYTHON" checker.py --mode turbulence --use-overview-cache --model "$FIRST_MODEL"

# ── Stufe 3: Analyse — jedes Modell mit Zeit-Budget ──────────────────────────
echo ""
echo "========================================"
echo "  Stufe 3: Analyse pro Modell (${TIME_BUDGET}s je)"
echo "========================================"
for model in $MODELS; do
    slug=$(slugify "$model")
    run_id="bench_${slug}"

    echo ""
    echo "----------------------------------------"
    echo "  Modell:   $model"
    echo "  run-id:   $run_id"
    echo "  Budget:   ${TIME_BUDGET}s"
    echo "----------------------------------------"

    # shellcheck disable=SC2086
    "$PYTHON" checker.py $COMMON \
        --model "$model" \
        --run-id "$run_id" \
        --cache-by-model \
        --use-overview-cache \
        --use-turbulence-cache \
        --time-budget "$TIME_BUDGET" \
        --log-training-data \
        || echo "  ⚠ Modell $model fehlgeschlagen — weiter mit nächstem"
done

# ── Zusammenfassung ───────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Benchmark abgeschlossen."
echo ""
echo "  Auswertung:"
echo "    $PYTHON benchmark_summary.py"
echo ""
echo "  Runs vergleichen:"
for m in $MODELS; do
    echo "    $PYTHON checker.py --history  # alle bench_* Runs anzeigen"
    break
done
echo "========================================"
