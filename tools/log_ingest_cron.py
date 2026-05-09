#!/usr/bin/env python3
"""Container-Log-Ingester: ERROR/WARNING/Traceback → bene:logs Workspace.

Ein 5-Minuten-Cron der von einem systemd-Timer auf u-server gefeuert
wird. Liest seit der letzten Run-Zeit `docker logs <container>` für die
relevanten Stack-Container, filtert auf Severity, scrubbed PII (Bearer/
JWT/Email/IP), und ingestet als source_type='log_event' in den
`bene:logs`-Sub-Workspace.

Usage:
    python tools/log_ingest_cron.py
    python tools/log_ingest_cron.py --once --since 1h    # ad-hoc
    python tools/log_ingest_cron.py --dry-run            # filter+scrub
                                                          # ohne POST

Env:
    MAYRING_API_URL          default https://mcp.linn.games
    MCP_SERVICE_TOKEN        bearer für service-Token
    LOG_INGEST_TARGET_SLUG   default bene:logs
    LOG_INGEST_STATE_FILE    default ~/.local/state/mayring-log-ingester.json
    LOG_INGEST_CONTAINERS    comma-separated; default mayring/applinngames
                             stack-container-namen
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ─── Severity / Multiline-Block-Detection ──────────────────────────

SEVERITY_RE = re.compile(
    r"\bERROR\b|\bCRITICAL\b|\bWARNING\b|\bEXCEPTION\b|"
    r"\bFATAL\b|\bFAIL(?:ED)?\b|"
    # Python-Exceptions: ValueError, RuntimeError, etc.
    r"\b\w*Error\b|"
    # Traceback-Header (kein \b am Ende — Doppelpunkt nach `)`).
    r"Traceback \(most recent call last\)"
)

# Beginn einer neuen Log-Zeile (timestamp am Zeilenanfang). Wenn die
# Zeile NICHT mit timestamp beginnt, ist sie Continuation des
# vorherigen blocks (z.B. Stacktrace-Frame).
LOG_LINE_START = re.compile(
    r"^(?:\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}|"
    r"\d{2}:\d{2}:\d{2}|"
    r"INFO:|DEBUG:|WARNING:|ERROR:|"
    r"\[\d{2}:\d{2}:\d{2})"
)

DEFAULT_CONTAINERS = [
    "mayring-mayring-api-1",
    "mayring-mayring-mcp-1",
    "mayring-mayring-pi-1",
    "mayring-mayring-webui-1",
    "applinngames-php-fpm-1",
    "applinngames-queue-worker-1",
    "applinngames-mcp-paper-search-1",
]

# ─── PII-Scrubber ──────────────────────────────────────────────────

PII_PATTERNS = [
    # Bearer-Tokens / JWTs
    (re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.I), "Bearer <REDACTED>"),
    (re.compile(r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),
     "<JWT_REDACTED>"),
    # API-Keys-Form: 32+ hex, sk-..., usw.
    (re.compile(r"\b[a-f0-9]{32,128}\b"), "<HEX_REDACTED>"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "<APIKEY_REDACTED>"),
    # Email
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
     "<EMAIL>"),
    # IPv4 (außer 127.x und 0.x)
    (re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
                r"(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), "<IP>"),
    # passwort=... query-strings
    (re.compile(r"(password|passwd|secret|token|api_key)=[^&\s]+", re.I),
     r"\1=<REDACTED>"),
]


def scrub_pii(text: str) -> str:
    out = text
    for pattern, replacement in PII_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


# ─── Block-Extraction ──────────────────────────────────────────────

def extract_severity_blocks(raw: str) -> list[str]:
    """Splittet logs in Multi-Line-Blöcke und behält nur Blöcke die
    Severity matchen. Stack-Traces (Folgezeilen ohne timestamp) werden
    als Teil des vorherigen Blocks erhalten.
    """
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in raw.splitlines():
        if LOG_LINE_START.match(line) and current:
            blocks.append(current)
            current = [line]
        elif not current:
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)

    matching: list[str] = []
    for block in blocks:
        joined = "\n".join(block)
        if SEVERITY_RE.search(joined):
            matching.append(joined)
    return matching


# ─── State-Tracking (last-run-timestamps pro Container) ────────────

def state_path() -> Path:
    p = os.environ.get("LOG_INGEST_STATE_FILE", "")
    if p:
        return Path(p)
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local/state")
    return Path(base) / "mayring-log-ingester.json"


def load_state() -> dict[str, str]:
    p = state_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: dict[str, str]) -> None:
    p = state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


# ─── docker logs reader ────────────────────────────────────────────

def fetch_container_logs(container: str, since: str) -> str:
    try:
        proc = subprocess.run(
            ["docker", "logs", container, "--since", since,
             "--timestamps", "--tail", "5000"],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"# {container}: docker logs failed: {e}", file=sys.stderr)
        return ""
    if proc.returncode != 0:
        print(f"# {container}: docker rc={proc.returncode}: "
              f"{proc.stderr[:200]}", file=sys.stderr)
        return ""
    # docker logs schreibt sowohl auf stdout als auch stderr.
    return (proc.stdout or "") + (proc.stderr or "")


# ─── Memory-API-POST ───────────────────────────────────────────────

def post_chunk(api: str, token: str, target_slug: str, container: str,
               block: str, captured_at: str, dry_run: bool) -> bool:
    # source_id stable für dedup: hash des Block-Texts + Container
    import hashlib
    h = hashlib.sha1(block.encode("utf-8")).hexdigest()[:16]
    source_id = f"log:{container}:{h}"
    body = {
        "source_id": source_id,
        "source_type": "log_event",
        "repo": "",
        "path": container,
        "content": block,
        "categorize": False,
    }
    if dry_run:
        print(f"# DRY-RUN POST {source_id}: {block[:120]}…")
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


# ─── Main ──────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--since", default=None,
                        help="docker --since override; default: last-run "
                             "aus state-file oder '5m'")
    parser.add_argument("--once", action="store_true",
                        help="ein Durchlauf, kein Loop")
    parser.add_argument("--dry-run", action="store_true",
                        help="filter+scrub, kein POST, kein state-save")
    args = parser.parse_args()

    api = os.environ.get("MAYRING_API_URL", "https://mcp.linn.games")
    token = os.environ.get("MCP_SERVICE_TOKEN", "").strip()
    if not token and not args.dry_run:
        sys.exit("MCP_SERVICE_TOKEN required (or --dry-run)")
    target = os.environ.get("LOG_INGEST_TARGET_SLUG", "bene:logs")
    containers = (os.environ.get("LOG_INGEST_CONTAINERS") or "").strip()
    container_list = (
        containers.split(",") if containers else DEFAULT_CONTAINERS
    )

    state = load_state()
    new_state = dict(state)
    total_blocks = 0
    posted = 0

    for container in container_list:
        since = args.since or state.get(container) or "5m"
        raw = fetch_container_logs(container, since)
        if not raw.strip():
            continue
        blocks = extract_severity_blocks(raw)
        if not blocks:
            continue
        captured_at = datetime.now(timezone.utc).isoformat()
        for block in blocks:
            scrubbed = scrub_pii(block)
            total_blocks += 1
            if post_chunk(api, token, target, container, scrubbed,
                          captured_at, args.dry_run):
                posted += 1
        new_state[container] = captured_at
        print(f"# {container}: {len(blocks)} blocks, posted {posted}",
              file=sys.stderr)

    if not args.dry_run:
        save_state(new_state)

    print(f"# done: {total_blocks} blocks total, {posted} posted, "
          f"target={target}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
