#!/usr/bin/env bash
# Installiert mayring-log-ingester systemd-units auf u-server.
#
# Aufruf einmalig auf u-server (zuerst MayringCoder-clone updaten):
#   ssh nileneb@u-server '
#     cd ~/app.linn.games/MayringCoder && git pull origin master &&
#     bash tools/systemd/install.sh
#   '
#
# Wichtig: Der MayringCoder-Clone unter ~/app.linn.games/MayringCoder/
# ist eine separate working tree, NICHT vom app.linn.games-checkout
# mit `git pull` mitgepullt. Manueller pull dort nötig.
#
# Idempotent: re-running aktualisiert die unit-files + reload.

set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="/etc/systemd/system"

sudo cp "$SRC/mayring-log-ingester.service" "$DEST/"
sudo cp "$SRC/mayring-log-ingester.timer"   "$DEST/"

sudo systemctl daemon-reload
sudo systemctl enable --now mayring-log-ingester.timer

echo
echo "Installed. Status:"
systemctl status mayring-log-ingester.timer --no-pager -l | head -20
echo
echo "Letzte Runs:"
journalctl --user -u mayring-log-ingester.service --since "1h ago" --no-pager | tail -20 || true
