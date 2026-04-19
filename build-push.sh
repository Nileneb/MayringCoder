#!/usr/bin/env bash
# Lokal builden (auf BigOne) + nach Docker Hub pushen.
# Danach GitHub Actions "Deploy to u-server" manuell triggern oder deploy-mayring.sh auf u-server.
set -euo pipefail

TAG=${1:-latest}
IMAGE="nileneb/mayring:$TAG"

echo "==> Building $IMAGE ..."
docker build -t "$IMAGE" -f docker/Dockerfile .

echo "==> Pushing $IMAGE ..."
docker push "$IMAGE"

echo ""
echo "Done: $IMAGE ist auf Docker Hub."
echo "Deploy: gh workflow run 'Deploy to u-server' --repo Nileneb/MayringCoder"
