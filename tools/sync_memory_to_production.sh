#!/usr/bin/env bash
set -euo pipefail

# ── Konfiguration (als Env-Variablen oder hier eintragen) ─────────────────────
PROD_HOST="${PROD_HOST:-nileneb@u-server}"
PROD_PORT="${PROD_PORT:-22}"
SSH_KEY="${SSH_KEY:-}"                              # optional: Pfad zu SSH-Key
PROD_COMPOSE_PATH="${PROD_COMPOSE_PATH:-~/app.linn.games/docker-compose.yml}"
DRY_RUN="${DRY_RUN:-0}"
RESTART="${RESTART:-1}"

# Staging-Dir auf Server (reguläres Dateisystem — kein /tmp: Docker-Daemon (root)
# kann tmpfs-gemountete /tmp Verzeichnisse oft nicht per lstat erreichen)
PROD_HOME="${PROD_HOME:-/home/nileneb}"
STAGING="${STAGING:-${PROD_HOME}/memory_sync_staging}"

# ── SSH-Optionen zusammenbauen ─────────────────────────────────────────────────
SSH_BASE="-p ${PROD_PORT}"
[[ -n "$SSH_KEY" ]] && SSH_BASE="${SSH_BASE} -i ${SSH_KEY}"

# ── Lokale Quelldateien prüfen ─────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ ! -f "${PROJECT_ROOT}/cache/memory.db" ]]; then
  echo "FEHLER: cache/memory.db nicht gefunden."
  exit 1
fi
if [[ ! -d "${PROJECT_ROOT}/cache/memory_chroma" ]]; then
  echo "FEHLER: cache/memory_chroma/ nicht gefunden."
  exit 1
fi

LOCAL_CHUNKS=$(sqlite3 "${PROJECT_ROOT}/cache/memory.db" \
  "SELECT COUNT(*) FROM chunks WHERE is_active=1" 2>/dev/null || echo "?")
echo "Lokale Chunks (aktiv): ${LOCAL_CHUNKS}"

# ── WAL-Checkpoint ────────────────────────────────────────────────────────────
echo "WAL-Checkpoint..."
sqlite3 "${PROJECT_ROOT}/cache/memory.db" "PRAGMA wal_checkpoint(TRUNCATE);" 2>/dev/null || true

# ── Staging-Dir auf Server anlegen ────────────────────────────────────────────
ssh ${SSH_BASE} "$PROD_HOST" "mkdir -p ${STAGING}/memory_chroma"

# ── Rsync → Staging ───────────────────────────────────────────────────────────
RSYNC_OPTS="-avz --checksum -e \"ssh ${SSH_BASE}\""
[[ "$DRY_RUN" == "1" ]] && RSYNC_OPTS="${RSYNC_OPTS} --dry-run" && echo "[DRY-RUN]"

echo ""
echo "Sync memory.db → ${PROD_HOST}:${STAGING}/memory.db"
eval rsync ${RSYNC_OPTS} \
  "${PROJECT_ROOT}/cache/memory.db" \
  "${PROD_HOST}:${STAGING}/memory.db"

echo ""
echo "Sync memory_chroma/ → ${PROD_HOST}:${STAGING}/memory_chroma/"
eval rsync ${RSYNC_OPTS} \
  "${PROJECT_ROOT}/cache/memory_chroma/" \
  "${PROD_HOST}:${STAGING}/memory_chroma/"

# ── docker cp ins Volume + Container-Restart ──────────────────────────────────
if [[ "$RESTART" == "1" && "$DRY_RUN" != "1" ]]; then
  echo ""
  echo "docker cp + Neustart auf ${PROD_HOST}..."
  ssh ${SSH_BASE} "$PROD_HOST" "
    set -e
    cd ~/app.linn.games
    docker compose -f ${PROD_COMPOSE_PATH} stop mayring-mcp mayring-api
    docker cp ${STAGING}/memory.db applinngames-mayring-mcp-1:/app/cache/memory.db
    docker cp ${STAGING}/memory.db applinngames-mayring-api-1:/app/cache/memory.db
    docker cp -a ${STAGING}/memory_chroma/. applinngames-mayring-mcp-1:/app/cache/memory_chroma/
    docker cp -a ${STAGING}/memory_chroma/. applinngames-mayring-api-1:/app/cache/memory_chroma/
    docker compose -f ${PROD_COMPOSE_PATH} start mayring-mcp mayring-api
    sleep 3
  "
fi

# ── Smoke-Test: Chunk-Anzahl im Container ─────────────────────────────────────
if [[ "$DRY_RUN" != "1" ]]; then
  echo ""
  echo "Smoke-Test — Chunks im Container:"
  ssh ${SSH_BASE} "$PROD_HOST" \
    "docker exec applinngames-mayring-mcp-1 python3 -c \"
import sqlite3
c = sqlite3.connect('/app/cache/memory.db').cursor()
c.execute('SELECT COUNT(*) FROM chunks WHERE is_active=1')
print('Aktive Chunks (Cloud):', c.fetchone()[0])
\""
fi

echo ""
echo "Fertig."
