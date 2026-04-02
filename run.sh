#!/usr/bin/env bash
set -e

# Stufe 1: Alle Dateien kartieren (was macht jede Datei?)
python checker.py --mode overview --no-limit --max-chars 9000

# Stufe 2: Fehlersuche über alle/ausgewählte Dateien
python checker.py --no-limit --max-chars 9000