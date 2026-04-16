#!/usr/bin/env bash
set -euo pipefail

# ── Konfiguration (als Env-Variablen oder hier eintragen) ─────────────────────
PROD_HOST="${PROD_HOST:-nileneb@u-server}"
PROD_PORT="${PROD_PORT:-22}"
SSH_KEY="${SSH_KEY:-}"                              # optional: Pfad zu SSH-Key
PROD_CACHE_PATH="${PROD_CACHE_PATH:-}"             # leer = automatisch suchen
PROD_COMPOSE_PATH="${PROD_COMPOSE_PATH:-~/app.linn.games/docker/docker-compose.production.yml}"
DRY_RUN="${DRY_RUN:-0}"
RESTART="${RESTART:-1}"

# ── SSH-Optionen zusammenbauen ─────────────────────────────────────────────────
SSH_BASE="-p ${PROD_PORT}"
[[ -n "$SSH_KEY" ]] && SSH_BASE="${SSH_BASE} -i ${SSH_KEY}"

# ── Lokale Quelldateien prüfen ─────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ ! -f "${PROJECT_ROOT}/cache/memory.db" ]]; then
  echo "FEHLER: cache/memory.db nicht gefunden. Erst lokal ingestieren:"
  echo "  .venv/bin/python tools/ingest_claude_memory.py"
  exit 1
fi
if [[ ! -d "${PROJECT_ROOT}/cache/memory_chroma" ]]; then
  echo "FEHLER: cache/memory_chroma/ nicht gefunden."
  exit 1
fi

LOCAL_CHUNKS=$(sqlite3 "${PROJECT_ROOT}/cache/memory.db" \
  "SELECT COUNT(*) FROM chunks WHERE is_active=1" 2>/dev/null || echo "?")
echo "Lokale Chunks (aktiv): ${LOCAL_CHUNKS}"

# ── Cache-Pfad auf Server ermitteln ───────────────────────────────────────────
if [[ -z "$PROD_CACHE_PATH" ]]; then
  echo "Suche cache-Pfad auf Server ${PROD_HOST}..."
  FOUND=$(ssh ${SSH_BASE} "$PROD_HOST" \
    "find ~ -name 'memory.db' -maxdepth 6 2>/dev/null | head -1 || true")
  if [[ -n "$FOUND" ]]; then
    PROD_CACHE_PATH=$(ssh ${SSH_BASE} "$PROD_HOST" "dirname '${FOUND}'")
    echo "Gefunden: ${PROD_CACHE_PATH}"
  else
    PROD_CACHE_PATH="~/app.linn.games/cache"
    echo "Nicht gefunden — Fallback: ${PROD_CACHE_PATH}"
    ssh ${SSH_BASE} "$PROD_HOST" "mkdir -p ${PROD_CACHE_PATH}"
  fi
fi

# ── Rsync ─────────────────────────────────────────────────────────────────────
RSYNC_OPTS="-avz --checksum -e \"ssh ${SSH_BASE}\""
[[ "$DRY_RUN" == "1" ]] && RSYNC_OPTS="${RSYNC_OPTS} --dry-run" && echo "[DRY-RUN]"

echo ""
echo "Sync memory.db → ${PROD_HOST}:${PROD_CACHE_PATH}/memory.db"
eval rsync ${RSYNC_OPTS} \
  "${PROJECT_ROOT}/cache/memory.db" \
  "${PROD_HOST}:${PROD_CACHE_PATH}/memory.db"

echo ""
echo "Sync memory_chroma/ → ${PROD_HOST}:${PROD_CACHE_PATH}/memory_chroma/"
eval rsync ${RSYNC_OPTS} \
  "${PROJECT_ROOT}/cache/memory_chroma/" \
  "${PROD_HOST}:${PROD_CACHE_PATH}/memory_chroma/"

# ── Container-Restart ─────────────────────────────────────────────────────────
if [[ "$RESTART" == "1" && "$DRY_RUN" != "1" ]]; then
  echo ""
  echo "Restart mcp-server + api-server auf ${PROD_HOST}..."
  ssh ${SSH_BASE} "$PROD_HOST" \
    "docker compose -f ${PROD_COMPOSE_PATH} restart mcp-server api-server 2>/dev/null || true"
fi

# ── Smoke-Test: Chunk-Anzahl auf Server ───────────────────────────────────────
if [[ "$DRY_RUN" != "1" ]]; then
  echo ""
  echo "Smoke-Test — Chunks auf Server:"
  ssh ${SSH_BASE} "$PROD_HOST" \
    "sqlite3 ${PROD_CACHE_PATH}/memory.db 'SELECT COUNT(*) FROM chunks WHERE is_active=1' 2>/dev/null || echo 'sqlite3 nicht verfügbar — manuelle Prüfung nötig'"
fi

echo ""
echo "Fertig."
