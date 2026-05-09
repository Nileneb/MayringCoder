#!/usr/bin/env python3
"""JSON-Log-File-Tail-Ingester → bene:logs Workspace.

Statt `docker logs` zu pollen lesen wir die strukturierten JSON-
Logfiles, die die Apps selbst schreiben:

  MayringCoder:  $MAYRING_LOG_FILE          (default cache/logs/mayring.json)
  Laravel:       storage/logs/laravel-*.json (per daily JSON-channel)

Vorteile:
  - line-orientiert: Tail via byte-offset, kein --since-Polling
  - strukturiert: keine Multi-Line-Block-Reconstruction nötig
  - PII-clean: was nicht ins logger-Statement geht, leaked nicht

Pro file ist State (last_offset) in
``$LOG_INGEST_STATE_FILE`` (default ~/.local/state/mayring-log-ingester.json).
Wenn die Datei vom logger rotiert wird (size > maxBytes), erkennen
wir das (offset > new file size) und reseten auf 0.

Usage:
    python tools/log_ingest_cron.py
    python tools/log_ingest_cron.py --dry-run  # filter only

Env:
    MAYRING_API_URL          default https://mcp.linn.games
    MCP_SERVICE_TOKEN        bearer für service-Token
    LOG_INGEST_TARGET_SLUG   default bene:logs
    LOG_INGEST_STATE_FILE    default ~/.local/state/mayring-log-ingester.json
    LOG_INGEST_FILES         comma-separated; default mayring + laravel paths
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ─── Default file-Set ──────────────────────────────────────────────

DEFAULT_FILES = [
    # MayringCoder API+Pi+MCP, alle in linn-mayring-cache volume
    "/var/lib/docker/volumes/linn-mayring-cache/_data/logs/mayring.json",
    # Laravel daily-JSON-channel
    "/home/nileneb/app.linn.games/storage/logs/app.json",
]

# Severity ab WARNING ingesten — INFO bleibt außen vor (sonst MB-Mengen)
ACCEPTED_LEVELS = {"WARNING", "ERROR", "CRITICAL", "EXCEPTION", "FATAL"}


def parse_json_line(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def scrub_pii(text: str) -> str:
    """Defensive scrub auch wenn der Logger schon clean ist — falls
    jemand `logger.info("Bearer xxx")` einbaut. KISS regex-set."""
    import re
    patterns = [
        (re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.I), "Bearer <REDACTED>"),
        (re.compile(r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),
         "<JWT_REDACTED>"),
        (re.compile(r"\b[a-f0-9]{32,128}\b"), "<HEX_REDACTED>"),
        (re.compile(r"(password|passwd|secret|token|api_key)=[^&\s]+", re.I),
         r"\1=<REDACTED>"),
    ]
    out = text
    for pat, repl in patterns:
        out = pat.sub(repl, out)
    return out


def line_to_chunk(record: dict) -> str:
    """Strukturiertes JSON → menschenlesbarer chunk-text. Sucht-feld-
    optimiert: timestamp + level + logger + msg + stack-trace."""
    parts = []
    if "ts" in record or "time" in record:
        parts.append(record.get("ts", record.get("time", "")))
    if "level" in record or "level_name" in record:
        parts.append(record.get("level", record.get("level_name", "")))
    if "logger" in record or "channel" in record:
        parts.append(f"[{record.get('logger', record.get('channel', ''))}]")
    if "msg" in record or "message" in record:
        parts.append(record.get("msg", record.get("message", "")))
    out = " ".join(str(p) for p in parts if p)
    if record.get("exc"):
        out += "\n" + record["exc"]
    if record.get("trace"):
        out += "\n" + json.dumps(record["trace"])[:2000]
    return scrub_pii(out)


# ─── State (offset pro file) ───────────────────────────────────────

def state_path() -> Path:
    p = os.environ.get("LOG_INGEST_STATE_FILE", "")
    if p:
        return Path(p)
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local/state")
    return Path(base) / "mayring-log-ingester.json"


def load_state() -> dict[str, int]:
    p = state_path()
    if not p.exists():
        return {}
    try:
        return {k: int(v) for k, v in json.loads(p.read_text()).items()}
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def save_state(state: dict[str, int]) -> None:
    p = state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


# ─── File-tail ────────────────────────────────────────────────────

def tail_new_lines(path: Path, last_offset: int) -> tuple[list[str], int]:
    """Liest NEUE Zeilen seit ``last_offset``. Erkennt Rotation
    (file kleiner als offset → reset auf 0). Returns (lines, new_offset).
    """
    if not path.exists():
        return [], last_offset
    size = path.stat().st_size
    if size < last_offset:
        # File rotated → start over
        last_offset = 0
    if size == last_offset:
        return [], last_offset
    with path.open("rb") as f:
        f.seek(last_offset)
        chunk = f.read(size - last_offset)
    text = chunk.decode("utf-8", errors="replace")
    lines = text.splitlines()
    # Falls die letzte Zeile abgeschnitten ist (Logger schreibt grad),
    # behalten wir sie für den nächsten Run.
    if not text.endswith("\n") and lines:
        last_partial = lines.pop()
        new_offset = size - len(last_partial.encode("utf-8"))
    else:
        new_offset = size
    return lines, new_offset


# ─── Memory-API-POST ───────────────────────────────────────────────

def post_chunk(api: str, token: str, target_slug: str, source_path: str,
               text: str, dry_run: bool) -> bool:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    source_id = f"log:{Path(source_path).name}:{h}"
    body = {
        "source_id": source_id,
        "source_type": "log_event",
        "repo": "",
        "path": Path(source_path).name,
        "content": text,
        "categorize": False,
    }
    if dry_run:
        print(f"# DRY-RUN POST {source_id}: {text[:120]}…")
        return True
    req = urllib.request.Request(
        api.rstrip("/") + "/memory/put",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "X-Workspace-Id": target_slug,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return 200 <= r.status < 300
    except urllib.error.HTTPError as e:
        print(f"# {source_id}: HTTP {e.code} {e.read()[:200]}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"# {source_id}: {e}", file=sys.stderr)
        return False


# ─── Main ─────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="parse + filter, kein POST, kein state-save")
    args = parser.parse_args()

    api = os.environ.get("MAYRING_API_URL", "https://mcp.linn.games")
    token = os.environ.get("MCP_SERVICE_TOKEN", "").strip()
    if not token and not args.dry_run:
        sys.exit("MCP_SERVICE_TOKEN required (or --dry-run)")
    target = os.environ.get("LOG_INGEST_TARGET_SLUG", "bene:logs")

    files_env = (os.environ.get("LOG_INGEST_FILES") or "").strip()
    file_list = (
        [Path(p.strip()) for p in files_env.split(",") if p.strip()]
        if files_env else [Path(p) for p in DEFAULT_FILES]
    )

    state = load_state()
    new_state = dict(state)
    total = 0
    posted = 0
    accepted = 0

    for path in file_list:
        last_offset = state.get(str(path), 0)
        lines, new_offset = tail_new_lines(path, last_offset)
        if not lines:
            continue

        for line in lines:
            total += 1
            record = parse_json_line(line)
            if record is None:
                continue  # nicht-JSON-Zeile ignorieren (z.B. uvicorn-Stdout)
            level = str(record.get("level", record.get("level_name", ""))).upper()
            if level not in ACCEPTED_LEVELS:
                continue
            accepted += 1
            text = line_to_chunk(record)
            if post_chunk(api, token, target, str(path), text, args.dry_run):
                posted += 1

        new_state[str(path)] = new_offset
        print(f"# {path.name}: tailed {len(lines)} lines, "
              f"{accepted} accepted, posted {posted}", file=sys.stderr)

    if not args.dry_run:
        save_state(new_state)

    print(f"# done: {total} lines total, {accepted} accepted, "
          f"{posted} posted, target={target}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
