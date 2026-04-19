#!/usr/bin/env bash
# Lokaler Start: API + WebUI in einem Rutsch (für Development/Testing).
#
# Was passiert:
#   1. MAYRING_MCP_AUTH_TOKEN wird auf "dev-local-token" gesetzt, falls .env den Wert nicht liefert
#      → damit kannst du in der UI als Login-Token denselben Wert eintragen und die API akzeptiert ihn
#   2. API (uvicorn src.api.server) startet auf Port 8080
#   3. WebUI (src.api.web_ui) startet auf Port 7861 und zeigt auf die lokale API
#
# Stop:  Ctrl+C  — beide Prozesse werden sauber beendet
#
# Voraussetzungen:
#   - .venv vorhanden
#   - Ollama läuft (OLLAMA_URL aus .env, default http://localhost:11434)
#   - mind. 2 Ollama-Modelle für das Duell-Tab

set -euo pipefail

PYTHON=".venv/bin/python"
API_PORT="${API_PORT:-8080}"
UI_PORT="${UI_PORT:-7861}"
API_URL="${API_URL:-http://localhost:${API_PORT}}"

if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source ./.env
    set +a
fi

export MAYRING_MCP_AUTH_TOKEN="${MAYRING_MCP_AUTH_TOKEN:-dev-local-token}"
export OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

API_PID=""
UI_PID=""

cleanup() {
    echo ""
    echo "[run_ui] Shutdown…"
    [ -n "$API_PID" ] && kill "$API_PID" 2>/dev/null || true
    [ -n "$UI_PID" ] && kill "$UI_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[run_ui] Ollama:     $OLLAMA_URL"
echo "[run_ui] API:        http://localhost:${API_PORT}"
echo "[run_ui] WebUI:      http://localhost:${UI_PORT}"
echo "[run_ui] Dev-Token:  $MAYRING_MCP_AUTH_TOKEN"
echo ""
echo "  → in der UI als 'Personal Access Token' eintragen und 'Einloggen' klicken."
echo ""

"$PYTHON" -m uvicorn src.api.server:app --host 127.0.0.1 --port "$API_PORT" --log-level warning &
API_PID=$!

sleep 2

"$PYTHON" -m src.api.web_ui --port "$UI_PORT" --api-url "$API_URL" &
UI_PID=$!

wait "$API_PID" "$UI_PID"
