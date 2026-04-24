#!/bin/bash
set -e
DEST="$HOME/.claude/plugins/mayring-coder"
mkdir -p "$DEST"
cp -r "$(cd "$(dirname "$0")" && pwd)/." "$DEST/"
echo "Installed to $DEST"
echo "Setze MAYRING_JWT in deiner Shell oder in ~/.claude/env:"
echo "  export MAYRING_JWT=<dein Token von app.linn.games/mayring/watcher>"
