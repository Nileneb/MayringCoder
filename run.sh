#!/usr/bin/env bash
set -e

PYTHON=".venv/bin/python"

# CLI-Flags parsen
FULL_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --full) FULL_FLAG="--full" ;;
    esac
done

# Primäres Modell auflösen (einmalig, damit beide Stufen dasselbe Modell nutzen).
# Wenn OLLAMA_MODEL bereits gesetzt ist, wird kein interaktiver Prompt gezeigt.
if [ -z "${OLLAMA_MODEL:-}" ]; then
    OLLAMA_MODEL=$("$PYTHON" -m src.pipeline --resolve-model-only)
    export OLLAMA_MODEL
fi

# Second-Opinion-Modell auflösen (optional).
# Wenn SECOND_OPINION_MODEL bereits gesetzt ist, wird kein interaktiver Prompt gezeigt.
# Leer lassen → kein Second-Opinion-Lauf.
if [ -z "${SECOND_OPINION_MODEL:-}" ]; then
    echo ""
    echo "Second-Opinion-Modell (Enter = überspringen, Name oder Nummer eingeben):" >&2
    # Verfügbare Modelle zur Orientierung anzeigen
    "$PYTHON" -c "
from src.model_selector import fetch_ollama_models
import os, sys
models = fetch_ollama_models(os.getenv('OLLAMA_URL', 'http://localhost:11434'))
if models:
    for i, m in enumerate(models, 1):
        print(f'  {i}. {m}', file=sys.stderr)
" 2>&1 >&2 || true
    read -r _so_input </dev/tty || _so_input=""
    if [ -n "$_so_input" ]; then
        # Zahl eingegeben → in Modellnamen umwandeln
        if echo "$_so_input" | grep -qE '^[0-9]+$'; then
            SECOND_OPINION_MODEL=$("$PYTHON" -c "
from src.model_selector import fetch_ollama_models
import os, sys
models = fetch_ollama_models(os.getenv('OLLAMA_URL', 'http://localhost:11434')) or []
idx = int('$_so_input') - 1
print(models[idx] if 0 <= idx < len(models) else '')
")
        else
            SECOND_OPINION_MODEL="$_so_input"
        fi
    fi
    export SECOND_OPINION_MODEL
fi

# Second-Opinion-Flag zusammenbauen (leer wenn kein Modell gewählt)
_SO_FLAG=""
if [ -n "${SECOND_OPINION_MODEL:-}" ]; then
    _SO_FLAG="--second-opinion ${SECOND_OPINION_MODEL}"
    echo "Second Opinion: ${SECOND_OPINION_MODEL}"
fi

# Stufe 1: Strukturiertes Overview (Funktionen + I/O pro Datei)
# shellcheck disable=SC2086
"$PYTHON" -m src.pipeline --mode overview --no-limit --max-chars 190000 --cache-by-model ${FULL_FLAG}

# Stufe 2: Turbulenz-Analyse mit Overview-Daten (Hot-Zones + betroffene Funktionen)
# shellcheck disable=SC2086
"$PYTHON" -m src.pipeline --mode turbulence --llm --use-overview-cache ${FULL_FLAG}

# Stufe 3: Gezielte Fehleranalyse (Hot-Zones + I/O als Kontext, Second Opinion hier)
# shellcheck disable=SC2086
"$PYTHON" -m src.pipeline --no-limit --max-chars 190000 --cache-by-model \
    --use-turbulence-cache ${_SO_FLAG} ${FULL_FLAG}