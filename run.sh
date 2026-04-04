#!/usr/bin/env bash
set -e

PYTHON=".venv/bin/python"

# Modell auflösen (einmalig, damit beide Stufen dasselbe Modell nutzen).
# Wenn OLLAMA_MODEL bereits gesetzt ist, wird kein interaktiver Prompt gezeigt.
if [ -z "${OLLAMA_MODEL:-}" ]; then
    OLLAMA_MODEL=$("$PYTHON" checker.py --resolve-model-only)
    export OLLAMA_MODEL
fi

# Stufe 1: Alle Dateien kartieren (was macht jede Datei?)
"$PYTHON" checker.py --mode overview --no-limit --max-chars 190000 --cache-by-model

# Stufe 2: Fehlersuche über alle/ausgewählte Dateien
"$PYTHON" checker.py --no-limit --max-chars 190000 --cache-by-model

# Stufe 3: Turbulenz-Analyse (vermischte Verantwortlichkeiten / Hot-Zones)
# Heuristik-Modus (kein Ollama nötig). Für LLM-Modus: --llm anhängen. -second-opinion MODEL
"$PYTHON" checker.py --mode turbulence --llm