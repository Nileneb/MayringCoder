#!/usr/bin/env bash
# Deploy MayringCoder standalone stack (docker-compose.mayring.yml).
# Image wird aus GHCR gepullt — kein lokaler Build.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Ensuring linn-shared network exists..."
docker network inspect linn-shared &>/dev/null || docker network create linn-shared

echo "==> Ensuring linn-mayring-cache volume exists..."
docker volume inspect linn-mayring-cache &>/dev/null || docker volume create linn-mayring-cache

echo "==> Pulling latest image from GHCR..."
docker pull nileneb/mayring:latest

echo "==> Starting MayringCoder stack..."
docker compose -f docker-compose.mayring.yml up -d

echo "==> Status:"
docker compose -f docker-compose.mayring.yml ps

echo ""
echo "Done. WebUI: http://localhost:7860  API: http://localhost:8090  MCP: http://localhost:8092"
