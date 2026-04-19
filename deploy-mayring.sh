#!/usr/bin/env bash
# Deploy MayringCoder standalone stack (docker-compose.mayring.yml).
# Läuft unabhängig vom app.linn.games Stack.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Pulling latest MayringCoder source..."
git pull origin master

echo "==> Ensuring linn-shared network exists..."
docker network inspect linn-shared &>/dev/null || docker network create linn-shared

echo "==> Ensuring linn-mayring-cache volume exists..."
docker volume inspect linn-mayring-cache &>/dev/null || docker volume create linn-mayring-cache

echo "==> Building + starting MayringCoder stack..."
docker compose -f docker-compose.mayring.yml up -d --build

echo "==> Status:"
docker compose -f docker-compose.mayring.yml ps

echo ""
echo "Done. WebUI: http://localhost:7860  API: http://localhost:8090  MCP: http://localhost:8092"
