"""Per-agent API keys for external A2A clients (Langdock).

WHY(2026-05-30): Langdock's A2A integration sends the credential as a SHORT
`X-API-Key` header (<=1000 chars), not an Authorization Bearer JWT — and a single
workspace JWT would over-share (full memory + every agent). So: short opaque keys,
one per agent, hashed at rest, independently revocable. The /a2a auth gate accepts
either a JWT (Bearer) or one of these keys (X-API-Key).

Shape: { key_id: {key_hash, workspace_id, label, created_at, revoked_at} }
Only the sha256 hash is stored — the plaintext is shown once at mint time.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PREFIX = "mca_"  # mayring-coder-agent key


def _store_path() -> Path:
    from mayring_core.config import CACHE_DIR
    return CACHE_DIR / "agent_keys.json"


def _hash(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_all() -> dict[str, Any]:
    try:
        with _store_path().open(encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _mutate(fn) -> Any:
    """flock-guarded read-modify-write of the whole store (multi-worker safe)."""
    p = _store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(p), os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "r+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        raw = f.read().strip()
        data = json.loads(raw) if raw else {}
        if not isinstance(data, dict):
            data = {}
        result = fn(data)
        f.seek(0)
        f.truncate()
        json.dump(data, f, indent=2, sort_keys=True)
    return result


def mint(workspace_id: str, label: str) -> tuple[str, dict[str, Any]]:
    """Create a new agent key. Returns (plaintext_shown_once, record_without_secret)."""
    plaintext = _PREFIX + secrets.token_urlsafe(24)
    key_id = secrets.token_hex(8)
    rec = {
        "key_hash": _hash(plaintext),
        "workspace_id": workspace_id,
        "label": (label or "agent")[:80],
        "created_at": _now(),
        "revoked_at": None,
    }
    _mutate(lambda data: data.__setitem__(key_id, rec))
    return plaintext, {"key_id": key_id, "label": rec["label"], "workspace_id": workspace_id}


def verify(plaintext: str) -> dict[str, Any] | None:
    """Resolve a presented key → {workspace_id, label, key_id} if valid + not revoked."""
    if not plaintext or not plaintext.startswith(_PREFIX):
        return None
    h = _hash(plaintext)
    for key_id, rec in _read_all().items():
        if rec.get("key_hash") == h and not rec.get("revoked_at"):
            return {"workspace_id": rec["workspace_id"], "label": rec.get("label"), "key_id": key_id}
    return None


def list_keys(workspace_id: str) -> list[dict[str, Any]]:
    """Keys for the workspace — never includes the hash or plaintext."""
    return [
        {"key_id": kid, "label": rec.get("label"), "created_at": rec.get("created_at"),
         "revoked": bool(rec.get("revoked_at"))}
        for kid, rec in sorted(_read_all().items(), key=lambda kv: kv[1].get("created_at", ""))
        if rec.get("workspace_id") == workspace_id
    ]


def revoke(workspace_id: str, key_id: str) -> bool:
    """Revoke a key — only within its own workspace (foreign revoke is a no-op)."""
    def _do(data: dict) -> bool:
        rec = data.get(key_id)
        if rec and rec.get("workspace_id") == workspace_id and not rec.get("revoked_at"):
            rec["revoked_at"] = _now()
            return True
        return False
    return _mutate(_do)
