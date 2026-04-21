#!/usr/bin/env bash
# Dev-Helper: lokaler Build für schnelle Iteration auf Entwicklungsmaschine.
#
# Production-Builds laufen ab sofort über GitHub Actions (.github/workflows/build-and-push.yml):
#   - Push auf master → CI baut :sha + :latest, triggert deploy-mayring.yml in app.linn.games
#   - Image landet auf Docker Hub, u-server zieht es beim Deploy
#
# Lokaler Build default auf :dev-local → kein Race-Konflikt gegen CI-:latest.
# Manuell :latest pushen: ./build-push.sh latest   (überschreibt CI-Image!)
set -euo pipefail

TAG=${1:-dev-local}
IMAGE="nileneb/mayring:$TAG"

echo "==> Building $IMAGE ..."
docker build -t "$IMAGE" -f docker/Dockerfile .

echo "==> Pushing $IMAGE ..."
docker push "$IMAGE"

echo ""
echo "Done: $IMAGE ist auf Docker Hub."
echo ""
echo "Deploy mit diesem Tag:"
echo "  gh workflow run 'Deploy MayringCoder' --repo Nileneb/app.linn.games -f tag=$TAG"
