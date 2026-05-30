"""Server-managed repo-watch state (per workspace).

WHY(2026-05-30): the watch list was hardcoded in the plugin's ci_security_warner
(_DEFAULT_WATCH) → only editable in code. This moves it to a shared server store
so the app.linn.games dashboard can toggle repos active/inactive and the local
hook fetches the live list. Small low-write config → a single JSON file in the
shared cache volume (atomic write + flock) instead of a mayring-core schema change.

Shape: { workspace_id: { repo_slug: {active: bool, alerts: [str], ingested_at: str|None} } }
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any

_VALID_ALERTS = ("ci", "code_scanning", "dependabot", "pulls")


def _store_path() -> Path:
    from mayring_core.config import CACHE_DIR
    return CACHE_DIR / "watched_repos.json"


def _read_all() -> dict[str, Any]:
    p = _store_path()
    try:
        with p.open(encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def get_watched(workspace_id: str) -> list[dict[str, Any]]:
    """All watched repos for the workspace (active + inactive), for the dashboard."""
    ws = _read_all().get(workspace_id, {})
    return [
        {"repo_slug": slug, "active": bool(v.get("active")),
         "alerts": v.get("alerts", []), "ingested_at": v.get("ingested_at")}
        for slug, v in sorted(ws.items())
    ]


def active_watch_map(workspace_id: str) -> dict[str, list[str]]:
    """slug → alert-types for ACTIVE repos only — the shape the watcher hook wants."""
    ws = _read_all().get(workspace_id, {})
    return {slug: v.get("alerts", []) for slug, v in ws.items() if v.get("active")}


def set_watched(workspace_id: str, repo_slug: str, *, active: bool,
                alerts: list[str], ingested_at: str | None = None) -> dict[str, Any]:
    """Upsert one repo's watch state. flock-guarded read-modify-write (multi-worker)."""
    alerts = [a for a in (alerts or []) if a in _VALID_ALERTS] or ["ci"]
    p = _store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # Open r+ (create if missing) and hold an exclusive lock across read+write.
    fd = os.open(str(p), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        with os.fdopen(fd, "r+", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            raw = f.read().strip()
            data = json.loads(raw) if raw else {}
            if not isinstance(data, dict):
                data = {}
            entry = data.setdefault(workspace_id, {}).setdefault(repo_slug, {})
            entry["active"] = bool(active)
            entry["alerts"] = alerts
            if ingested_at is not None:
                entry["ingested_at"] = ingested_at
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2, sort_keys=True)
        return {"repo_slug": repo_slug, "active": bool(active), "alerts": alerts,
                "ingested_at": entry.get("ingested_at")}
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
